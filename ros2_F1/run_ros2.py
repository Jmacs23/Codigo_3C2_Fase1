"""
run_ros2.py --- Script principal del Bloque 4
Bloque 4: ros2_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

La LOGICA de este bloque esta verificada con pruebas automatizadas. Los
NODOS ROS2 no, porque ROS2 no estaba disponible en el entorno donde se
escribieron.

Si algo falla en el robot:
  1. Comprueba primero si el problema esta en la logica o en la plomeria.
     `python run_ros2.py mision` ejercita la logica sin ROS2.
  2. Si la logica funciona, el problema esta en el nodo: revisa nombres de
     topics, tipos de mensaje y QoS.
  3. Reportalo con el traceback completo y la version de ROS2.
===========================================================================

COMANDOS
--------
    python run_ros2.py config      Configuracion de la capa ROS2
    python run_ros2.py grafo       Estructura de nodos y topics
    python run_ros2.py seguridad   Histeresis del override de emergencia
    python run_ros2.py navegacion  Modulacion por campo potencial
    python run_ros2.py receptor    Robustez ante mensajes corruptos
    python run_ros2.py mision      COMPUERTA: arbitraje completo
    python run_ros2.py integracion Simulacion del grafo sin ROS2
    python run_ros2.py despliegue  Instrucciones de compilacion
    python run_ros2.py test        Bateria de tests

LA COMPUERTA DEL BLOQUE
-----------------------
`python run_ros2.py mision` verifica el orden de prioridades del arbitraje,
que es lo que garantiza que la seguridad nunca quede supeditada al estado
cognitivo.

En el robot hay que verificar ademas que los topics se publican a la
frecuencia esperada (`ros2 topic hz /tasm/state`) y que la latencia real
entre las dos maquinas se parece a los 20 ms presupuestados.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "logica"))

from config import CONFIG
from logica.nodos_logica import (
    LogicaSeguridad, LogicaNavegacion, LogicaControl, LogicaMision,
    ReceptorTASM, MensajeTASMRecibido, ModoOperacion, Etapa, Velocidad,
)


def titulo(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def seccion(t: str) -> None:
    print()
    print("-" * 72)
    print(t)
    print("-" * 72)


def veredicto(ok: bool, msg: str) -> bool:
    print(f"  {'[ OK ]' if ok else '[FALLO]'} {msg}")
    return ok


# ===========================================================================

def cmd_config() -> int:
    titulo("CONFIGURACION DE LA CAPA ROS2")
    r = CONFIG.ros2
    b = CONFIG.bci

    print()
    print("  Frecuencias de los nodos")
    print(f"    bci_node       : {r.hz_bci} Hz "
          f"(tasa de TASM: {1/b.paso_tasm:.1f} Hz)")
    print(f"    mission_node   : {r.hz_control} Hz")
    print(f"    ctrl_node      : {r.hz_control} Hz "
          f"(ts_control: {1/CONFIG.robot.ts_control:.1f} Hz)")
    print(f"    safety_node    : {r.hz_seguridad} Hz")
    print(f"    interface_node : {r.hz_interfaz} Hz")
    print()
    print("  Comunicacion con Windows")
    print(f"    Estado TASM    : puerto {r.tcp_puerto_bci} (Windows -> Linux)")
    print(f"    Video          : puerto {r.tcp_puerto_video} (Linux -> Windows)")
    print(f"    Resolucion     : {r.video_ancho}x{r.video_alto} "
          f"@ {r.video_fps} fps, JPEG {r.video_calidad_jpeg}%")

    ancho_banda = (r.video_ancho * r.video_alto * 3 *
                   (r.video_calidad_jpeg / 100) * 0.03 * r.video_fps)
    print(f"    Ancho de banda : ~{ancho_banda/1e6:.1f} MB/s estimado")
    print()
    print("  Calidad de servicio")
    print(f"    Profundidad    : {r.qos_profundidad}")
    print(f"    Transporte     : "
          f"{'fiable' if r.qos_fiable else 'best-effort'}")
    print()
    print("    En control en tiempo real un mensaje viejo es peor que")
    print("    ninguno. Por eso best-effort y cola corta. La excepcion es el")
    print("    comando de emergencia, que va por transporte fiable.")

    seccion("VALIDACION")
    probs = r.verificar_coherencia(b.paso_tasm, CONFIG.robot.ts_control)
    if probs:
        for p in probs:
            print(f"  - {p}")
        return 1
    print("  Las frecuencias de los nodos son coherentes con el resto del")
    print("  sistema.")
    return 0


def cmd_grafo() -> int:
    titulo("GRAFO DE NODOS Y TOPICS")

    print()
    print("  MAQUINA WINDOWS                    MAQUINA LINUX")
    print("  ---------------                    -------------")
    print()
    print("  g.USBamp")
    print("     |")
    print("     v")
    print("  filtrado + FBCCA + TASM")
    print("     |")
    print("     |  TCP :5556 (JSON)")
    print("     +-------------------------->  bci_node")
    print("                                      |")
    print("                                      | /tasm/state")
    print("                                      v")
    print("  LiDAR ---> /scan ------------->  mission_node  <--- /bci/etapa")
    print("                                      |")
    print("                                      | /bci/command")
    print("                                      v")
    print("                                   ctrl_node")
    print("                                      |")
    print("                                      | /cmd_vel")
    print("                                      v")
    print("                                   TurtleBot3")
    print("                                      ^")
    print("                                      | /cmd_vel (prioritario)")
    print("  camara ---> /camera/image_raw    safety_node <--- /scan")
    print("     |                                |")
    print("     |                                | /bci/emergencia")
    print("     v                                v")
    print("  video_publisher                 interface_node")
    print("     |                                |")
    print("     |  TCP :5555 (JPEG)              |  TCP :5557 (JSON)")
    print("     v                                v")
    print("  PsychoPy <---------------------------+")

    seccion("POR QUE safety_node PUBLICA DIRECTAMENTE EN /cmd_vel")
    print("  Es redundancia deliberada. mission_node ya evalua la seguridad,")
    print("  pero si se colgara o tuviera un error de logica, el robot")
    print("  quedaria sin proteccion.")
    print()
    print("  safety_node es simple a proposito: lee el LiDAR, compara con un")
    print("  umbral, publica. Cuanto menos codigo tenga, menos puede fallar.")
    print()
    print("  Y NO se suscribe a /tasm/state, deliberadamente. Un obstaculo a")
    print("  15 cm es igual de peligroso si el usuario esta en control")
    print("  intencional, en transicion de mirada o en reposo.")

    seccion("TOPICS")
    r = CONFIG.ros2
    filas = [
        ("/tasm/state", "TASMState", "bci_node", f"{r.hz_bci} Hz"),
        ("/bci/command", "BciCommand", "mission_node", f"{r.hz_control} Hz"),
        ("/cmd_vel", "Twist", "ctrl_node, safety_node", f"{r.hz_control} Hz"),
        ("/scan", "LaserScan", "LiDAR", "~5 Hz"),
        ("/odom", "Odometry", "robot", "~30 Hz"),
        ("/bci/emergencia", "Bool", "safety_node", f"{r.hz_seguridad} Hz"),
        ("/bci/etapa", "Int8", "e2_node (Bloque 5)", "por evento"),
        ("/bci/objetos", "String", "e2_node (Bloque 5)", "por evento"),
    ]
    print(f"  {'topic':<20} {'tipo':<14} {'publica':<24} {'tasa'}")
    print("  " + "-" * 68)
    for t, tip, pub, hz in filas:
        print(f"  {t:<20} {tip:<14} {pub:<24} {hz}")
    return 0


def cmd_seguridad() -> int:
    titulo("HISTERESIS DEL OVERRIDE DE EMERGENCIA")
    r = CONFIG.robot

    print()
    print(f"  Entrada : rho <= {r.rho_safe} m")
    print(f"  Salida  : rho > {r.rho_safe + r.rho_hist} m durante "
          f"{r.n_hist} ciclos consecutivos")
    print()
    print("  La histeresis no es un adorno. Sin ella, un obstaculo justo en")
    print("  el umbral haria oscilar el sistema varias veces por segundo,")
    print("  produciendo un movimiento a tirones.")

    seg = LogicaSeguridad(CONFIG)
    n = CONFIG.robot.lidar_n_rayos

    seccion("Secuencia de acercamiento y alejamiento")
    print(f"  {'ciclo':>6} {'rho_min':>9} {'emergencia':>12} "
          f"{'ciclos seguros':>16}  evento")

    secuencia = [0.60, 0.40, 0.25, 0.18, 0.15, 0.13, 0.12, 0.14,
                 0.17, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26, 0.35]
    previo = False
    ok = True
    for i, rho in enumerate(secuencia):
        d = [rho] + [5.0] * (n - 1)
        e = seg.evaluar(d)
        evento = ""
        if e.emergencia and not previo:
            evento = "<-- ENTRA"
        elif previo and not e.emergencia:
            evento = "<-- SALE"
        previo = e.emergencia
        print(f"  {i:>6} {rho:>8.2f}m {'SI' if e.emergencia else 'no':>12} "
              f"{e.ciclos_seguros:>16}  {evento}")

    seccion("Verificaciones")
    seg2 = LogicaSeguridad(CONFIG)
    d_cerca = [0.10] + [5.0] * (n - 1)
    d_lejos = [5.0] * n

    e = seg2.evaluar(d_cerca)
    ok &= veredicto(e.emergencia, "entra en emergencia al detectar 0.10 m")

    e = seg2.evaluar(d_lejos)
    ok &= veredicto(e.emergencia,
                    "NO sale inmediatamente aunque el camino este libre")

    for _ in range(r.n_hist):
        e = seg2.evaluar(d_lejos)
    ok &= veredicto(not e.emergencia,
                    f"sale tras {r.n_hist} ciclos por encima del umbral")

    seg3 = LogicaSeguridad(CONFIG)
    e = seg3.evaluar([])
    ok &= veredicto(e.emergencia,
                    "sin datos de LiDAR asume emergencia (conservador)")

    return 0 if ok else 1


def cmd_navegacion() -> int:
    titulo("MODULACION POR CAMPO POTENCIAL")
    c, r = CONFIG.control, CONFIG.robot

    print()
    print(f"  Radio de influencia : {c.apf_rho0} m")
    print(f"  Ganancia repulsiva  : {c.apf_eta}")
    print()
    print("  El campo potencial NO sustituye al usuario. No hay destino")
    print("  autonomo: el usuario decide adonde va y el campo solo corrige")
    print("  la trayectoria para que no roce.")

    nav = LogicaNavegacion(CONFIG)
    n = r.lidar_n_rayos
    cmd = Velocidad(r.u_max, 0.0)

    seccion("Obstaculo frontal")
    print(f"  Comando del usuario: avanzar a {r.u_max} m/s")
    print()
    print(f"  {'rho_min':>9} {'u resultante':>14} {'factor':>9}")
    for rho in (1.00, 0.60, 0.40, 0.35, 0.30, 0.25, 0.20, 0.16):
        d = [rho] + [5.0] * (n - 1)
        out, _ = nav.asistir(cmd, d, 0.0)
        print(f"  {rho:>8.2f}m {out.lineal:>13.3f}m/s "
              f"{out.lineal/r.u_max:>9.2f}")

    seccion("Obstaculo lateral")
    print("  Un obstaculo a un lado corrige el rumbo sin frenar tanto.")
    print()
    print(f"  {'posicion':>12} {'u':>10} {'omega':>10}")
    for nombre, idx in (("izquierda", n // 4), ("derecha", 3 * n // 4)):
        d = [5.0] * n
        d[idx] = 0.25
        out, _ = nav.asistir(cmd, d, 0.0)
        print(f"  {nombre:>12} {out.lineal:>9.3f} {out.angular:>10.3f}")

    seccion("Verificacion de signo")
    print("  El giro debe alejar al robot del obstaculo. Si observas que se")
    print("  PEGA a las paredes en el robot real, el signo del gradiente")
    print("  esta invertido: es un error facil de cometer al derivarlo.")
    d = [5.0] * n
    d[n // 4] = 0.25          # obstaculo a la izquierda
    out, _ = nav.asistir(cmd, d, 0.0)
    ok = veredicto(out.angular < 0,
                   f"obstaculo a la izquierda -> gira a la derecha "
                   f"(omega={out.angular:.3f})")
    return 0 if ok else 1


def cmd_receptor() -> int:
    titulo("ROBUSTEZ DEL RECEPTOR DE MENSAJES")
    print()
    print("  Un mensaje corrupto NO debe tumbar el nodo. Se descarta y se")
    print("  cuenta. Si se descartaran muchos, el watchdog detendria el")
    print("  robot, que es la respuesta correcta.")

    rec = ReceptorTASM(CONFIG)
    casos = [
        ('{"estado":"IC","freq_idx":2,"p_max":0.9,"lambda_bci":0.95,'
         '"valido":true,"timestamp":1.0,"secuencia":1}',
         "mensaje valido", True),
        ('{"estado":"TR","freq_idx":-1,"p_max":0.3,"lambda_bci":0.2,'
         '"valido":true,"timestamp":1.05,"secuencia":2}',
         "estado TR valido", True),
        ('{"estado":"Idle","freq_idx":-1,"p_max":0.1,"lambda_bci":0.05,'
         '"valido":false,"timestamp":1.1,"secuencia":3}',
         "ventana invalida", True),
        ('{roto', "JSON malformado", False),
        ('{"estado":"XX","freq_idx":0,"p_max":0.5,"lambda_bci":0.5,'
         '"valido":true,"timestamp":1.2,"secuencia":4}',
         "estado desconocido", False),
        ('{"estado":"IC","freq_idx":99,"p_max":0.5,"lambda_bci":0.5,'
         '"valido":true,"timestamp":1.25,"secuencia":5}',
         "indice fuera de rango", False),
        ('{"estado":"IC"}', "campos faltantes", False),
        ('', "linea vacia", False),
        ('   \n  ', "solo espacios", False),
    ]

    seccion("Casos")
    print(f"  {'caso':>24} {'esperado':>12} {'obtenido':>12}")
    ok = True
    for linea, desc, esperado in casos:
        m = rec.procesar_linea(linea, t=1.0)
        obtenido = m is not None
        ok &= (obtenido == esperado)
        marca = "" if obtenido == esperado else "  <-- DISCREPANCIA"
        print(f"  {desc:>24} "
              f"{'acepta' if esperado else 'descarta':>12} "
              f"{'acepta' if obtenido else 'descarta':>12}{marca}")

    seccion("Estadisticas")
    st = rec.estadisticas
    print(f"  Recibidos    : {st['recibidos']}")
    print(f"  Descartados  : {st['descartados']}")
    print(f"  Tasa validos : {st['tasa_validos']*100:.0f}%")

    seccion("Watchdog")
    rec2 = ReceptorTASM(CONFIG)
    ok &= veredicto(rec2.watchdog_expirado(t=0.0),
                    "expirado antes del primer mensaje")
    rec2.procesar_linea(casos[0][0], t=10.0)
    ok &= veredicto(not rec2.watchdog_expirado(t=10.1),
                    "no expirado justo tras un mensaje")
    ok &= veredicto(rec2.watchdog_expirado(t=10.0 + CONFIG.bci.watchdog_s + 0.1),
                    f"expirado tras {CONFIG.bci.watchdog_s*1000:.0f} ms "
                    f"sin mensajes")

    return 0 if ok else 1


def cmd_mision() -> int:
    """Compuerta del bloque."""
    titulo("COMPUERTA DEL BLOQUE 4 --- Arbitraje de mision")
    b, r = CONFIG.bci, CONFIG.robot
    n = r.lidar_n_rayos
    ok = True

    libre = [5.0] * n
    cerca = [0.10] + [5.0] * (n - 1)
    msg_ic = MensajeTASMRecibido("IC", 2, 0.9, 0.95, True, 0.0, 1)
    msg_tr = MensajeTASMRecibido("TR", -1, 0.3, 0.2, True, 0.0, 2)

    # --- 1. El comando se enclava tras la racha ---
    seccion("1. Enclavamiento tras la racha de confirmacion")
    mis = LogicaMision(CONFIG)
    t = 0.0
    for i in range(b.n_conf):
        d = mis.ciclo(msg_ic, libre, t)
        t += b.paso_tasm
    ok &= veredicto(d.velocidad.lineal > 0,
                    f"avanza tras {b.n_conf} ventanas de IC "
                    f"({b.t_confirmacion*1000:.0f} ms)")

    # --- 2. La emergencia se impone ---
    seccion("2. La emergencia se impone sobre el comando")
    d = mis.ciclo(msg_ic, cerca, t)
    t += b.paso_tasm
    ok &= veredicto(d.modo == ModoOperacion.EMERGENCIA,
                    "modo emergencia con obstaculo a 0.10 m")
    ok &= veredicto(d.velocidad.lineal < 0,
                    f"retrocede a {d.velocidad.lineal:.2f} m/s")

    # --- 3. La histeresis retiene el control ---
    seccion("3. Histeresis a la salida")
    d = mis.ciclo(msg_ic, libre, t)
    t += b.paso_tasm
    ok &= veredicto(d.modo == ModoOperacion.EMERGENCIA,
                    "sigue en emergencia el primer ciclo tras despejarse")
    for _ in range(r.n_hist):
        d = mis.ciclo(msg_ic, libre, t)
        t += b.paso_tasm
    ok &= veredicto(d.modo == ModoOperacion.BCI_MANUAL,
                    f"devuelve el control tras {r.n_hist} ciclos")

    # --- 4. El watchdog detiene ---
    seccion("4. Watchdog")
    mis2 = LogicaMision(CONFIG)
    d = mis2.ciclo(None, libre, 0.0, watchdog_expirado=True)
    ok &= veredicto(d.modo == ModoOperacion.DETENIDO_SEGURO,
                    "parada segura sin mensajes de TASM")
    ok &= veredicto(d.velocidad.es_cero(),
                    "velocidad exactamente cero")

    # --- 5. TR no transiciona ---
    seccion("5. La transicion de mirada no enclava comandos")
    mis3 = LogicaMision(CONFIG)
    t = 0.0
    for _ in range(b.n_conf * 3):
        d = mis3.ciclo(msg_tr, libre, t)
        t += b.paso_tasm
    ok &= veredicto(d.velocidad.es_cero(),
                    f"tras {b.n_conf*3} ventanas de TR el robot sigue parado")

    # --- 6. Etapa 2: robot estatico ---
    seccion("6. Etapa 2 con robot estatico")
    mis4 = LogicaMision(CONFIG)
    mis4.cambiar_etapa(Etapa.SELECCION)
    t = 0.0
    for _ in range(b.n_conf + 2):
        d = mis4.ciclo(msg_ic, libre, t)
        t += b.paso_tasm
    ok &= veredicto(d.velocidad.es_cero(),
                    "el robot no se mueve durante la seleccion")
    ok &= veredicto(d.etapa == Etapa.SELECCION, "etapa correcta")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    print()
    print("  RECORDATORIO: esto verifica la LOGICA de decision. En el robot")
    print("  hay que comprobar ademas:")
    print("    - ros2 topic hz /tasm/state    (debe dar ~20 Hz)")
    print("    - ros2 topic echo /bci/command (comandos coherentes)")
    print("    - latencia real entre las dos maquinas")
    return 0 if ok else 1


def cmd_integracion() -> int:
    """Simula el grafo completo sin ROS2."""
    titulo("SIMULACION DEL GRAFO SIN ROS2")
    print()
    print("  Se encadenan las clases de logica igual que lo harian los nodos,")
    print("  pero llamandolas directamente en vez de por topics.")
    print()
    print("  Sirve para verificar que el comportamiento conjunto es correcto")
    print("  antes de arrancar ROS2, donde depurar es mucho mas lento.")

    b, r = CONFIG.bci, CONFIG.robot
    n = r.lidar_n_rayos

    receptor = ReceptorTASM(CONFIG)
    mision = LogicaMision(CONFIG)
    control = LogicaControl(CONFIG)

    seccion("Escenario: avanzar, encontrar obstaculo, evadir, continuar")
    print(f"  {'t (s)':>7} {'estado':>7} {'rho':>7} {'modo':>17} "
          f"{'u_ref':>8} {'u_cmd':>8}")

    medida = Velocidad(0.0, 0.0)
    t = 0.0
    TAU = 0.2   # constante de tiempo de la planta simulada

    guion = (
        [("IC", 2, 5.0)] * 12 +      # el usuario pide avanzar
        [("Idle", -1, 5.0)] * 8 +   # deja de mirar, el comando sigue
        [("Idle", -1, 0.60)] * 4 +  # aparece un obstaculo
        [("Idle", -1, 0.30)] * 4 +
        [("Idle", -1, 0.12)] * 6 +  # emergencia
        [("Idle", -1, 0.35)] * 8 +  # se despeja
        [("Idle", -1, 5.0)] * 6
    )

    for i, (estado, idx, rho) in enumerate(guion):
        linea = (f'{{"estado":"{estado}","freq_idx":{idx},"p_max":0.9,'
                 f'"lambda_bci":0.9,"valido":true,"timestamp":{t},'
                 f'"secuencia":{i}}}')
        msg = receptor.procesar_linea(linea, t=t)

        d = [rho] + [5.0] * (n - 1)
        dec = mision.ciclo(msg, d, t)
        u = control.actualizar(dec.velocidad, medida)

        # Planta de primer orden
        alpha = min(1.0, b.paso_tasm / TAU)
        medida = Velocidad(
            medida.lineal + (u.lineal - medida.lineal) * alpha,
            medida.angular + (u.angular - medida.angular) * alpha,
        )

        if i % 3 == 0 or i in (12, 20, 28, 34):
            print(f"  {t:>7.2f} {estado:>7} {rho:>6.2f}m "
                  f"{dec.modo.value:>17} {dec.velocidad.lineal:>8.3f} "
                  f"{u.lineal:>8.3f}")
        t += b.paso_tasm

    seccion("Lectura")
    print("  Tres cosas que se ven en la traza:")
    print()
    print("  1. El robot sigue avanzando cuando el usuario pasa a Idle: el")
    print("     comando esta enclavado y su intencion previa sigue vigente.")
    print()
    print("  2. La emergencia se activa sola al bajar de 0.15 m, sin que el")
    print("     usuario haga nada.")
    print()
    print("  3. Al despejarse, el control no vuelve de inmediato: la")
    print("     histeresis exige varios ciclos por encima del umbral.")
    return 0


def cmd_despliegue() -> int:
    titulo("DESPLIEGUE EN EL ROBOT")

    print()
    print("  1. COPIAR EL WORKSPACE")
    print()
    print("     Copia ros2_ws/ a la maquina Linux, y el directorio logica/")
    print("     junto a el:")
    print()
    print("       ~/bci3c2/")
    print("         ros2_ws/")
    print("         logica/")
    print("         config.py")
    print()
    print("     Si prefieres otra disposicion, define la variable de entorno:")
    print("       export BCI3C2_LOGICA=/ruta/a/logica")

    print()
    print("  2. COMPILAR")
    print()
    print("       cd ~/bci3c2/ros2_ws")
    print("       colcon build --packages-select turtlebot3_bci_3c2")
    print("       source install/setup.bash")

    print()
    print("  3. VERIFICAR LOS MENSAJES")
    print()
    print("       ros2 interface show turtlebot3_bci_3c2/msg/TASMState")
    print()
    print("     Si esto falla, la generacion de mensajes no funciono y nada")
    print("     mas va a arrancar.")

    print()
    print("  4. PROBAR SIN ROBOT")
    print()
    print("       ros2 launch turtlebot3_bci_3c2 test_sin_robot.launch.py")
    print()
    print("     En otra terminal:")
    print("       ros2 topic hz /tasm/state      # deberia dar ~20 Hz")
    print("       ros2 topic echo /bci/command")

    print()
    print("  5. ARRANCAR EL SISTEMA COMPLETO")
    print()
    print("       ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py \\")
    print("           windows_ip:=192.168.1.50")

    seccion("ORDEN DE ARRANQUE")
    print("  safety_node arranca PRIMERO y sin retardo. Si el robot ya esta")
    print("  encendido y hay un obstaculo cerca, queremos proteccion desde el")
    print("  primer instante, antes de que nada pueda mandarle un avance.")

    seccion("PROBLEMAS FRECUENTES")
    print("  'No module named turtlebot3_bci_3c2.msg'")
    print("     Falta hacer source install/setup.bash, o la compilacion de")
    print("     mensajes fallo.")
    print()
    print("  'No se encontro la logica del sistema'")
    print("     El directorio logica/ no esta donde logica_bridge.py lo")
    print("     busca. Usa la variable BCI3C2_LOGICA.")
    print()
    print("  El robot no se mueve")
    print("     Comprueba en este orden:")
    print("       ros2 topic hz /tasm/state     ~20 Hz?")
    print("       ros2 topic echo /bci/command  que modo reporta?")
    print("     Si el modo es detenido_seguro, no llegan mensajes de Windows.")
    print("     Si es emergencia, el LiDAR ve algo cerca.")
    print()
    print("  El robot se pega a las paredes en vez de evitarlas")
    print("     El signo del gradiente del campo potencial esta invertido.")
    print("     Ver LogicaNavegacion.fuerza_repulsiva.")
    return 0


def cmd_test() -> int:
    import subprocess
    titulo("BATERIA DE TESTS")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v",
                        "--tb=short", "-W", "ignore"])
    return r.returncode


# ===========================================================================

COMANDOS = {
    "config": cmd_config,
    "grafo": cmd_grafo,
    "seguridad": cmd_seguridad,
    "navegacion": cmd_navegacion,
    "receptor": cmd_receptor,
    "mision": cmd_mision,
    "integracion": cmd_integracion,
    "despliegue": cmd_despliegue,
    "test": cmd_test,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print(__doc__)
        print("Comandos:")
        for k in COMANDOS:
            print(f"  {k}")
        return 1
    try:
        return COMANDOS[sys.argv[1]]()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

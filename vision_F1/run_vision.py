"""
run_vision.py --- Script principal del Bloque 5
Bloque 5: vision_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este bloque tiene tres partes con distinto grado de verificacion:

  ArUco          VERIFICADO por ejecucion. OpenCV con soporte aruco estaba
                 disponible, asi que las pruebas detectan marcadores reales
                 sobre imagenes sinteticas y comprueban la pose.

  Fusion,        VERIFICADAS con pruebas automatizadas.
  transicion,
  aproximacion

  YOLO           NO verificado. ultralytics no estaba disponible. La clase
                 es un esqueleto con las llamadas marcadas para completar.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

COMANDOS
--------
    python run_vision.py config       Configuracion de vision
    python run_vision.py marcadores   Genera los marcadores para imprimir
    python run_vision.py deteccion    Detecta sobre escena sintetica
    python run_vision.py alcance      Alcance y precision segun distancia
    python run_vision.py fusion       Fusion ArUco + YOLO
    python run_vision.py transicion   COMPUERTA: las tres condiciones
    python run_vision.py aproximacion Control de aproximacion final
    python run_vision.py calibracion  Como calibrar la camara
    python run_vision.py test         Bateria de tests

LA COMPUERTA DEL BLOQUE
-----------------------
`python run_vision.py transicion` verifica que las tres condiciones se
evaluan correctamente y que ninguna se puede saltar. Es lo que impide el
falso positivo mas probable: que el usuario pare a mitad de camino y el
sistema lo interprete como que ha llegado.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "logica"))

from config import CONFIG


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
    titulo("CONFIGURACION DE VISION")
    v = CONFIG.vision

    print()
    print("  Camara")
    print(f"    Resolucion      : {v.ancho_px} x {v.alto_px}")
    print(f"    Campo visual    : {v.fov_horizontal} grados")
    print(f"    Focal (fx, fy)  : {v.fx:.0f}, {v.fy:.0f} px")
    print(f"    Centro (cx, cy) : {v.cx:.0f}, {v.cy:.0f} px")
    print(f"    Calibracion     : "
          f"{'VERIFICADA' if v.calibracion_verificada else 'nominal, SIN CALIBRAR'}")

    print()
    print("  Marcadores")
    print(f"    Diccionario     : {v.diccionario_aruco}")
    print(f"    Lado impreso    : {v.lado_marcador*1000:.0f} mm")
    print(f"    Minimo legible  : {v.px_minimo_marcador} px")
    print(f"    Alcance         : {v.distancia_maxima_deteccion():.2f} m")

    print()
    print("  Transicion de etapa")
    print(f"    Distancia       : < {v.umbral_distancia_transicion} m")
    print(f"    Objetos minimos : {v.min_objetos_transicion}")
    print(f"    Permanencia     : {v.permanencia_detenido} s detenido")
    print(f"    Sector frontal  : {v.sector_frontal} grados")
    print(f"    Margen obstaculo: {v.margen_obstaculo_frontal} m")

    print()
    print("  Aproximacion")
    print(f"    Distancia final : {v.distancia_objetivo} m")
    print(f"    Velocidad max   : {v.velocidad_aproximacion} m/s")
    print(f"    Tolerancias     : {v.tolerancia_posicion*100:.0f} cm, "
          f"{math.degrees(v.tolerancia_angulo):.0f} grados")

    seccion("COHERENCIA CON LA ETAPA 2")
    e2 = CONFIG.etapa2
    print(f"  {'N':>4} {'D necesaria':>13} {'cabe en umbral':>16} "
          f"{'marcador px':>13}")
    ok = True
    for n in e2.niveles_n:
        d = e2.distancia_minima(n)
        cabe = d <= v.umbral_distancia_transicion
        px = v.px_marcador_a(d)
        legible = px >= v.px_minimo_marcador
        ok &= (cabe and legible)
        print(f"  {n:>4} {d:>12.2f}m {'si' if cabe else 'NO':>16} "
              f"{px:>10.0f} {'ok' if legible else 'NO'}")

    seccion("VALIDACION")
    probs = [p for p in CONFIG.validar()
             if any(k in p.lower() for k in
                    ("vision", "marcador", "calibra", "transicion", "fov",
                     "resolucion"))]
    if probs:
        for p in probs:
            print(f"  - {p}")
    else:
        print("  Sin problemas.")
    return 0 if ok else 1


def cmd_marcadores() -> int:
    from aruco_detector import generar_hoja_marcadores, _HAY_ARUCO
    import cv2

    titulo("GENERACION DE MARCADORES PARA IMPRIMIR")
    if not _HAY_ARUCO:
        print("\n  OpenCV sin soporte aruco. Instala opencv-contrib-python.")
        return 1

    v = CONFIG.vision
    n_max = max(CONFIG.etapa2.niveles_n)
    ids = list(range(n_max))

    print()
    print(f"  Se generan {n_max} marcadores del diccionario "
          f"{v.diccionario_aruco}")
    print(f"  Lado configurado: {v.lado_marcador*1000:.0f} mm")

    hoja = generar_hoja_marcadores(CONFIG, ids, lado_px=400)
    ruta = "marcadores_para_imprimir.png"
    cv2.imwrite(ruta, hoja)

    print()
    print(f"  Guardado en: {ruta}")
    print(f"  Dimensiones: {hoja.shape[1]} x {hoja.shape[0]} px")

    seccion("INSTRUCCIONES DE IMPRESION")
    print("  1. Abre el archivo e imprime a TAMANO REAL.")
    print("     En el dialogo de impresion elige '100%' o 'tamano real',")
    print("     NUNCA 'ajustar a pagina': las impresoras escalan por defecto")
    print("     y eso falsearia todas las distancias.")
    print()
    print("  2. MIDE con una regla el lado del marcador impreso, sin contar")
    print("     el borde blanco.")
    print()
    print(f"  3. Si NO mide {v.lado_marcador*1000:.0f} mm, corrige")
    print("     config.vision.lado_marcador con el valor real. Un error del")
    print("     5% aqui da un error del 5% en TODAS las distancias.")
    print()
    print("  4. Pega cada marcador en el FRENTE del objeto, sobre superficie")
    print("     plana. Si el objeto es curvo, pega antes un carton pequeno.")
    print()
    print("  5. Los OBSTACULOS no llevan marcador. Esa ausencia es lo que")
    print("     los discrimina de los objetos seleccionables.")
    return 0


def cmd_deteccion() -> int:
    from aruco_detector import DetectorArUco, renderizar_escena

    titulo("DETECCION SOBRE ESCENA SINTETICA")
    v, e2 = CONFIG.vision, CONFIG.etapa2
    det = DetectorArUco(CONFIG)
    ok = True

    for n in e2.niveles_n:
        d = e2.distancia_minima(n)
        seccion(f"N = {n} objetos a {d:.2f} m")

        sep = e2.objeto_separacion
        x0 = -(n - 1) * sep / 2
        escena = [(i, x0 + i * sep, 0.0, d) for i in range(n)]

        img = renderizar_escena(CONFIG, escena)
        detectados = det.filtrar_fiables(det.detectar(img))

        ok &= veredicto(len(detectados) == n,
                        f"detectados {len(detectados)} de {n}")

        if detectados:
            errs = []
            for m in detectados:
                real = next((e for e in escena if e[0] == m.id), None)
                if real:
                    errs.append(abs(m.profundidad - real[3]))
            err = np.mean(errs) * 1000
            print(f"         error medio de distancia: {err:.1f} mm")

            ids = [m.id for m in detectados]
            ok &= veredicto(ids == sorted(ids),
                            "ordenados de izquierda a derecha")

    seccion("RESULTADO")
    print(f"  {'DETECCION CORRECTA' if ok else 'HAY FALLOS'}")
    return 0 if ok else 1


def cmd_alcance() -> int:
    from aruco_detector import DetectorArUco, renderizar_escena

    titulo("ALCANCE Y PRECISION SEGUN DISTANCIA")
    v = CONFIG.vision
    det = DetectorArUco(CONFIG)

    print()
    print(f"  Marcador de {v.lado_marcador*1000:.0f} mm, "
          f"camara de {v.ancho_px} px")
    print(f"  Alcance teorico: {v.distancia_maxima_deteccion():.2f} m")

    seccion("Medida")
    print(f"  {'distancia':>11} {'lado px':>10} {'detectado':>11} "
          f"{'medida':>10} {'error':>11}")

    for d in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
        img = renderizar_escena(CONFIG, [(0, 0.0, 0.0, d)])
        res = det.detectar(img)
        px = v.px_marcador_a(d)
        marca = ("  <-- umbral"
                 if abs(d - v.umbral_distancia_transicion) < 0.05 else "")
        if res:
            z = res[0].profundidad
            err = abs(z - d)
            print(f"  {d:>10.1f}m {px:>10.0f} {'si':>11} {z:>9.3f}m "
                  f"{err*1000:>9.1f}mm{marca}")
        else:
            print(f"  {d:>10.1f}m {px:>10.0f} {'no':>11} {'---':>10} "
                  f"{'---':>11}{marca}")

    seccion("LECTURA")
    print("  El error crece con la distancia, y no es casual: a mayor")
    print("  distancia el marcador ocupa menos pixeles, y un error de un")
    print("  pixel en la deteccion de esquinas se traduce en mas milimetros")
    print("  de error en la profundidad.")
    print()
    print("  Esto REFUERZA la necesidad de calibrar la camara: con")
    print("  intrinsecos nominales hay ademas un error sistematico que se")
    print("  suma al aleatorio.")
    print()
    print("  Tiene una consecuencia util: la fisica de la deteccion actua")
    print("  como filtro de distancia. Mas alla del alcance el marcador")
    print("  simplemente no se detecta, antes de aplicar ningun umbral.")
    return 0


def cmd_fusion() -> int:
    from aruco_detector import DetectorArUco, renderizar_escena
    from vision_fusion import (DetectorYOLO, FusionVision, DeteccionYOLO)

    titulo("FUSION ArUco + YOLO")

    print()
    print("  No es redundancia: resuelven problemas distintos.")
    print()
    print("    ArUco  identidad univoca y pose 3D. Discrimina objetos de")
    print("           obstaculos sin ambiguedad (los obstaculos no llevan")
    print("           marcador).")
    print()
    print("    YOLO   recuadros de objetos reales, que es lo que el usuario")
    print("           ve parpadear. Conserva la validez ecologica.")

    det = DetectorArUco(CONFIG)
    fusion = FusionVision(CONFIG)
    yolo = DetectorYOLO(CONFIG)

    escena = [(i, -0.375 + i * 0.25, 0.0, 1.20) for i in range(4)]
    img = renderizar_escena(CONFIG, escena)
    marcadores = det.detectar(img)

    seccion("Estado de los detectores")
    print(f"  ArUco : disponible, {len(marcadores)} marcadores detectados")
    print(f"  YOLO  : "
          f"{'disponible' if yolo.disponible else 'ESQUELETO (sin modelo)'}")

    # --- Sin YOLO ---
    seccion("Fusion sin YOLO (situacion actual)")
    objetos = fusion.fusionar(marcadores, [])
    print(f"  {'id':>4} {'distancia':>11} {'con YOLO':>10} "
          f"{'origen del recuadro':>22}")
    for o in objetos:
        print(f"  {o.id:>4} {o.distancia:>10.2f}m "
              f"{'si' if o.tiene_yolo else 'no':>10} "
              f"{'estimado del marcador':>22}")
    print()
    print("  El sistema funciona igual: la identidad y la pose las da el")
    print("  marcador. Lo que se pierde es que el recuadro encuadre el")
    print("  objeto completo en vez de una estimacion.")

    # --- Con YOLO simulado ---
    seccion("Fusion con recuadros simulados de YOLO")
    simuladas = []
    for m in marcadores:
        cx, cy = m.centro_px
        w = m.lado_px * 2.2
        h = m.lado_px * 3.0
        simuladas.append(DeteccionYOLO(
            clase="objeto", confianza=0.9,
            bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))

    objetos2 = fusion.fusionar(marcadores, simuladas)
    print(f"  {'id':>4} {'con YOLO':>10} {'recuadro (x0,y0,x1,y1)':>34}")
    for o in objetos2:
        b = o.bbox_interfaz()
        print(f"  {o.id:>4} {'si' if o.tiene_yolo else 'no':>10} "
              f"({b[0]:>7.0f},{b[1]:>6.0f},{b[2]:>7.0f},{b[3]:>6.0f})")

    ok = all(o.tiene_yolo for o in objetos2)
    print()
    veredicto(ok, "todos los marcadores asociados a su recuadro")

    # --- Objeto sin marcador ---
    seccion("Un recuadro sin marcador se descarta")
    print("  Es la discriminacion entre objetos y obstaculos: si YOLO detecta")
    print("  algo que no lleva marcador, no es seleccionable.")
    print()
    extra = DeteccionYOLO(clase="obstaculo", confianza=0.95,
                          bbox=(50, 50, 150, 250))
    objetos3 = fusion.fusionar(marcadores, simuladas + [extra])
    veredicto(len(objetos3) == len(marcadores),
              f"{len(objetos3)} objetos de {len(simuladas)+1} recuadros")
    return 0


def cmd_transicion() -> int:
    """Compuerta del bloque."""
    from aruco_detector import DetectorArUco, renderizar_escena
    from vision_fusion import FusionVision, LogicaTransicion

    titulo("COMPUERTA DEL BLOQUE 5 --- Transicion de etapa")
    v = CONFIG.vision
    n_rayos = CONFIG.robot.lidar_n_rayos
    ok = True

    print()
    print("  Las tres condiciones deben cumplirse SIMULTANEAMENTE:")
    print(f"    1. Objetos a menos de {v.umbral_distancia_transicion} m")
    print(f"       (al menos {v.min_objetos_transicion})")
    print(f"    2. Sin obstaculos en el sector frontal de "
          f"{v.sector_frontal} grados")
    print(f"    3. Robot detenido durante {v.permanencia_detenido} s")

    det = DetectorArUco(CONFIG)
    fusion = FusionVision(CONFIG)

    def objetos_a(dist, n=4):
        esc = [(i, -0.375 + i * 0.25, 0.0, dist) for i in range(n)]
        return fusion.fusionar(det.detectar(renderizar_escena(CONFIG, esc)), [])

    libre = [5.0] * n_rayos
    con_obst = [5.0] * n_rayos
    con_obst[0] = 0.60

    # --- Condicion 1 ---
    seccion("Condicion 1: distancia y numero de objetos")
    log = LogicaTransicion(CONFIG)
    est = log.evaluar(objetos_a(3.0), libre, True, 5.0)
    ok &= veredicto(not est.procede, "objetos a 3.0 m: no procede")

    log.reiniciar()
    est = log.evaluar([], libre, True, 5.0)
    ok &= veredicto(not est.procede, "sin objetos: no procede")

    log.reiniciar()
    est = log.evaluar(objetos_a(1.2, n=1), libre, True, 5.0)
    ok &= veredicto(not est.procede,
                    f"un solo objeto: no procede "
                    f"(hacen falta {v.min_objetos_transicion})")

    # --- Condicion 2 ---
    seccion("Condicion 2: obstaculo intermedio")
    log.reiniciar()
    log.evaluar(objetos_a(1.2), con_obst, True, 0.0)
    est = log.evaluar(objetos_a(1.2), con_obst, True, 5.0)
    ok &= veredicto(not est.procede,
                    "obstaculo a 0.60 m con objetos a 1.2 m: no procede")

    # --- Condicion 3 ---
    seccion("Condicion 3: permanencia")
    log.reiniciar()
    est = log.evaluar(objetos_a(1.2), libre, False, 0.0)
    ok &= veredicto(not est.procede, "robot en movimiento: no procede")

    log.reiniciar()
    log.evaluar(objetos_a(1.2), libre, True, 0.0)
    est = log.evaluar(objetos_a(1.2), libre, True, 0.5)
    ok &= veredicto(not est.procede,
                    f"0.5 s detenido: no procede "
                    f"(hacen falta {v.permanencia_detenido})")
    print()
    print("  Esta condicion evita el falso positivo mas probable: que el")
    print("  usuario pare a mitad de camino para evaluar el entorno y el")
    print("  sistema lo interprete como que ha llegado.")

    # --- Las tres juntas ---
    seccion("Las tres condiciones cumplidas")
    log.reiniciar()
    log.evaluar(objetos_a(1.2), libre, True, 0.0)
    est = log.evaluar(objetos_a(1.2), libre, True, 5.0)
    ok &= veredicto(est.procede, "procede la transicion")
    print()
    print(est.resumen())

    # --- Contador se reinicia al moverse ---
    seccion("El contador se reinicia si el robot se mueve")
    log.reiniciar()
    log.evaluar(objetos_a(1.2), libre, True, 0.0)
    log.evaluar(objetos_a(1.2), libre, True, 1.5)
    log.evaluar(objetos_a(1.2), libre, False, 2.0)     # se mueve
    est = log.evaluar(objetos_a(1.2), libre, True, 2.5)
    ok &= veredicto(not est.procede,
                    "tras moverse, el contador vuelve a empezar")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    print()
    print("  RECORDATORIO: esto verifica la LOGICA. En el robot hay que")
    print("  comprobar ademas:")
    print("    - que los marcadores se detectan con la camara real")
    print("    - que la distancia medida coincide con la real (calibrar)")
    print("    - que la altura del marcador cae en el campo visual")
    return 0 if ok else 1


def cmd_aproximacion() -> int:
    from vision_fusion import ControlAproximacion

    titulo("CONTROL DE APROXIMACION")
    v = CONFIG.vision

    print()
    print(f"  Distancia objetivo : {v.distancia_objetivo} m")
    print(f"  Velocidad maxima   : {v.velocidad_aproximacion} m/s")
    print(f"  Tolerancias        : {v.tolerancia_posicion*100:.0f} cm, "
          f"{math.degrees(v.tolerancia_angulo):.0f} grados")
    print()
    print("  Es un control PROPORCIONAL, no un PID. La maniobra es corta y")
    print("  termina en reposo: un termino integral acumularia error durante")
    print("  la aproximacion y daria sobreimpulso justo al final, que es lo")
    print("  peor cuando el robot se esta acercando a un objeto.")

    ctrl = ControlAproximacion(CONFIG)
    dt = CONFIG.robot.ts_control

    for d0, a0_deg in ((1.20, 15.0), (0.80, -25.0), (0.35, 2.0)):
        seccion(f"Desde {d0} m y {a0_deg} grados")
        d, a = d0, math.radians(a0_deg)
        n_align = n_avance = 0
        completada = False

        for i in range(400):
            c = ctrl.actualizar(d, a)
            if c.completada:
                completada = True
                print(f"  Completada en {i*dt:.2f} s")
                print(f"    error de distancia: {c.error_distancia*100:+.1f} cm")
                print(f"    error de angulo   : "
                      f"{math.degrees(c.error_angulo):+.1f} grados")
                break
            if abs(c.lineal) < 1e-6:
                n_align += 1
            else:
                n_avance += 1
            d -= c.lineal * dt
            a -= c.angular * dt

        if not completada:
            print(f"  NO completada en {400*dt:.1f} s")
        print(f"    ciclos alineando: {n_align}, avanzando: {n_avance}")

    seccion("NOTA SOBRE LA REALIMENTACION")
    print("  La pose del objetivo se mide UNA VEZ, al seleccionarlo, y se")
    print("  navega hacia ese punto con realimentacion de ODOMETRIA.")
    print()
    print("  Refinar continuamente con realimentacion VISUAL (visual")
    print("  servoing) compensaria la deriva odometrica y daria mejor")
    print("  precision final. Se identifica como extension natural y NO se")
    print("  implementa aqui: el trabajo versa sobre arbitracion cognitiva,")
    print("  y anadirlo mezclaria dos contribuciones distintas.")
    return 0


def cmd_calibracion() -> int:
    titulo("CALIBRACION DE LA CAMARA")
    v = CONFIG.vision

    print()
    print("  ESTADO ACTUAL: "
          f"{'calibrada' if v.calibracion_verificada else 'SIN CALIBRAR'}")
    print()
    print("  Los intrinsecos por defecto son NOMINALES, derivados del campo")
    print("  visual declarado. Sirven para desarrollar, pero la distorsion de")
    print("  lente de las camaras pequenas es apreciable y produce error")
    print("  SISTEMATICO en la distancia estimada.")
    print()
    print("  Y de esa distancia depende la condicion de transicion de etapa.")

    seccion("PROCEDIMIENTO")
    print("  1. Imprime un patron de ajedrez (9x6 esquinas internas es lo")
    print("     habitual) y pegalo sobre una superficie RIGIDA y plana. Un")
    print("     papel curvado falsea la calibracion.")
    print()
    print("  2. Toma entre 15 y 25 fotos con la camara del robot, variando:")
    print("       - la distancia (de 0.3 a 2 m)")
    print("       - el angulo (inclinando el patron)")
    print("       - la posicion en el encuadre (centro, esquinas)")
    print()
    print("     Las fotos con el patron en las ESQUINAS son las que mas")
    print("     informacion dan sobre la distorsion, que es maxima ahi.")
    print()
    print("  3. Ejecuta cv2.calibrateCamera con esas imagenes.")
    print()
    print("  4. Copia los resultados a config.vision:")
    print("       fx, fy, cx, cy      de la matriz de camara")
    print("       coef_distorsion     los cinco coeficientes")
    print("       calibracion_verificada = True")

    seccion("CODIGO DE REFERENCIA")
    print("""
    import cv2, numpy as np, glob

    patron = (9, 6)          # esquinas internas
    lado = 0.025             # lado de cada cuadro en metros

    objp = np.zeros((patron[0]*patron[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2)
    objp *= lado

    puntos_3d, puntos_2d = [], []
    for ruta in glob.glob("calib/*.jpg"):
        img = cv2.imread(ruta)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, esq = cv2.findChessboardCorners(gris, patron, None)
        if ok:
            puntos_3d.append(objp)
            puntos_2d.append(cv2.cornerSubPix(
                gris, esq, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 30, 0.001)))

    ok, K, dist, _, _ = cv2.calibrateCamera(
        puntos_3d, puntos_2d, gris.shape[::-1], None, None)

    print("fx =", K[0,0], " fy =", K[1,1])
    print("cx =", K[0,2], " cy =", K[1,2])
    print("distorsion =", dist.ravel())
    """)

    seccion("COMO SABER SI QUEDO BIEN")
    print("  Coloca un marcador a una distancia MEDIDA con cinta metrica")
    print("  (por ejemplo 1.00 m exactos) y comprueba que el sistema reporta")
    print("  esa distancia con un error menor a 2 cm.")
    print()
    print("  Repitelo a 0.5, 1.0 y 1.5 m. Si el error crece de forma")
    print("  sistematica con la distancia, el valor de lado_marcador esta")
    print("  mal: mide el marcador impreso otra vez.")
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
    "marcadores": cmd_marcadores,
    "deteccion": cmd_deteccion,
    "alcance": cmd_alcance,
    "fusion": cmd_fusion,
    "transicion": cmd_transicion,
    "aproximacion": cmd_aproximacion,
    "calibracion": cmd_calibracion,
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

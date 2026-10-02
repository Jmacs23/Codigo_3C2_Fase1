"""
simulator2d.py --- Simulador 2D headless
Bloque 1: nucleo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Integra todos los modulos del bloque en un trial completo: navegacion con
obstaculos (Etapa 1) mas seleccion de objeto (Etapa 2), sin ROS2, sin
Gazebo, sin hardware y sin EEG.

POR QUE UN SIMULADOR PROPIO Y NO GAZEBO DIRECTAMENTE
----------------------------------------------------
Gazebo simula fisica 3D con fidelidad, pero es lento: un trial de tres
minutos tarda tres minutos. Para responder preguntas de diseno hacen falta
cientos de trials.

Este simulador corre un trial de tres minutos en decimas de segundo. Eso
permite barrer parametros (n_conf, tasas de error, geometrias) y detectar
problemas de diseno en horas en lugar de meses.

El mismo controlador y la misma FSM corren despues en Gazebo y en el robot
real sin cambios: lo unico que cambia es de donde vienen las lecturas del
LiDAR y a donde va el comando de velocidad.

QUE SIMULA Y QUE NO
-------------------
SI simula: cinematica diferencial, raycasting 2D sobre obstaculos
rectangulares, campo potencial repulsivo, override de emergencia, la FSM
completa y la seleccion binaria.

NO simula: dinamica (masas, inercias, deslizamiento), ruido de odometria,
latencia de red, ni el aspecto visual de la escena. Para eso esta Gazebo.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

from config import Config, Escenario
from command_fsm import CommandFSM, EstadoTASM, EstadoFSM, MotivoNoTransicion
from tasm_mock import TASMMock
from metrics import CalculadorMetricas, RegistroCiclo, MetricasTrial
from binary_search import (BusquedaBinaria, LadoSeleccion,
                           generar_objetos_en_fila)


# ===========================================================================
# GEOMETRIA
# ===========================================================================

@dataclass
class Obstaculo:
    """Obstaculo rectangular alineado a los ejes."""
    x: float
    y: float
    ancho: float
    alto: float

    def contiene(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.ancho and
                self.y <= py <= self.y + self.alto)

    def distancia_a(self, px: float, py: float) -> float:
        """Distancia euclidea del punto al rectangulo (0 si esta dentro)."""
        dx = max(self.x - px, 0.0, px - (self.x + self.ancho))
        dy = max(self.y - py, 0.0, py - (self.y + self.alto))
        return math.hypot(dx, dy)


@dataclass
class Mundo:
    """Entorno de un escenario."""
    ancho: float
    alto: float
    obstaculos: List[Obstaculo] = field(default_factory=list)
    inicio: Tuple[float, float, float] = (0.5, 2.0, 0.0)
    zona_e2: Tuple[float, float] = (5.0, 2.0)
    """Posicion de la zona de seleccion (x, y)."""


def construir_mundo(config: Config, escenario: Escenario) -> Mundo:
    """
    Genera la geometria de un escenario a partir de su definicion.

    Los obstaculos se colocan de modo que el paso libre entre ellos sea
    exactamente el `gap` declarado en la configuracion del escenario. Asi la
    dificultad queda controlada por un solo numero.
    """
    W, H = escenario.sala
    obs: List[Obstaculo] = []

    lado = 0.30   # los obstaculos son cajas de 30 x 30 cm en planta
    gap = escenario.gap

    if escenario.n_obstaculos == 2:
        # Corredor recto: dos cajas alternadas en paredes opuestas
        obs.append(Obstaculo(x=W * 0.30, y=0.0, ancho=lado,
                             alto=(H - gap) / 2))
        obs.append(Obstaculo(x=W * 0.55, y=H - (H - gap) / 2, ancho=lado,
                             alto=(H - gap) / 2))

    elif escenario.n_obstaculos == 3:
        # Corredor en L
        obs.append(Obstaculo(x=W * 0.25, y=0.0, ancho=lado,
                             alto=(H - gap) / 2))
        obs.append(Obstaculo(x=W * 0.45, y=H - (H - gap) / 2, ancho=lado,
                             alto=(H - gap) / 2))
        obs.append(Obstaculo(x=W * 0.72, y=H * 0.45, ancho=lado, alto=lado))

    else:
        # Sala abierta: obstaculos dispersos
        posiciones = [(0.22, 0.25), (0.35, 0.68), (0.50, 0.42),
                      (0.64, 0.75), (0.76, 0.28)]
        for fx, fy in posiciones[:escenario.n_obstaculos]:
            obs.append(Obstaculo(x=W * fx, y=H * fy, ancho=lado, alto=lado))

    return Mundo(
        ancho=W, alto=H, obstaculos=obs,
        inicio=(0.5, H / 2, 0.0),
        zona_e2=(W - 0.8, H / 2),
    )


# ===========================================================================
# SENSOR
# ===========================================================================

class LidarSimulado:
    """
    LiDAR 2D por raycasting.

    Publica el mismo formato que un LiDAR real (un array de distancias por
    angulo), de modo que el nodo de control no distinga entre simulacion y
    hardware.
    """

    def __init__(self, config: Config, mundo: Mundo):
        self.cfg = config
        self.mundo = mundo
        self.n_rayos = config.robot.lidar_n_rayos
        self.rango_max = config.robot.lidar_rango_max

    def escanear(self, x: float, y: float, theta: float) -> List[float]:
        """Devuelve las distancias de los rayos, en orden angular."""
        dists = []
        paso_ang = 2 * math.pi / self.n_rayos
        # Resolucion del muestreo a lo largo del rayo. 2 cm es suficiente
        # para los margenes que maneja este sistema y mantiene el coste bajo.
        paso_r = 0.02

        for i in range(self.n_rayos):
            ang = theta + i * paso_ang
            dx, dy = math.cos(ang), math.sin(ang)
            d = self.rango_max
            r = paso_r
            while r < self.rango_max:
                px, py = x + dx * r, y + dy * r
                # Paredes de la sala
                if not (0 <= px <= self.mundo.ancho and
                        0 <= py <= self.mundo.alto):
                    d = r
                    break
                # Obstaculos
                if any(o.contiene(px, py) for o in self.mundo.obstaculos):
                    d = r
                    break
                r += paso_r
            dists.append(d)
        return dists


# ===========================================================================
# CONTROL
# ===========================================================================

class CampoPotencial:
    """
    Campo potencial repulsivo para asistencia de evasion.

    No sustituye al usuario: modula su comando cuando hay obstaculos cerca.
    El usuario sigue decidiendo hacia donde va.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def fuerza_repulsiva(self, distancias: List[float]) -> Tuple[float, float]:
        """
        Fuerza repulsiva total en el marco del robot.

        IMPORTANTE --- SIGNO: el vector resultante apunta ALEJANDOSE de los
        obstaculos. Si al implementar esto en el robot real observas que se
        pega a las paredes en lugar de evitarlas, el signo esta invertido.
        Es un error facil de cometer al derivar el gradiente.
        """
        c = self.cfg.control
        n = len(distancias)
        fx = fy = 0.0
        paso_ang = 2 * math.pi / n

        for i, d in enumerate(distancias):
            if d >= c.apf_rho0 or d < 1e-6:
                continue
            # Magnitud del gradiente del potencial de Khatib
            mag = c.apf_eta * (1.0 / d - 1.0 / c.apf_rho0) / (d * d)
            ang = i * paso_ang
            # Signo negativo: la fuerza empuja en direccion OPUESTA al rayo
            fx -= mag * math.cos(ang)
            fy -= mag * math.sin(ang)

        return (fx, fy)


class ControladorPID:
    """PID discreto con antiwindup para seguimiento de velocidad."""

    def __init__(self, config: Config):
        self.cfg = config
        self.reiniciar()

    def reiniciar(self) -> None:
        self._int_u = 0.0
        self._int_w = 0.0
        self._e_u_prev = 0.0
        self._e_w_prev = 0.0

    def actualizar(self, u_ref: float, w_ref: float,
                   u_act: float, w_act: float) -> Tuple[float, float]:
        c = self.cfg.control
        ts = self.cfg.robot.ts_control

        e_u = u_ref - u_act
        e_w = w_ref - w_act

        self._int_u += e_u * ts
        self._int_w += e_w * ts

        # Antiwindup por saturacion del termino integral
        lim = c.antiwindup
        self._int_u = max(-lim, min(lim, self._int_u))
        self._int_w = max(-lim, min(lim, self._int_w))

        du = (e_u - self._e_u_prev) / ts if ts > 0 else 0.0
        dw = (e_w - self._e_w_prev) / ts if ts > 0 else 0.0
        self._e_u_prev = e_u
        self._e_w_prev = e_w

        u = c.kp * e_u + c.ki * self._int_u + c.kd * du
        w = c.kp * e_w + c.ki * self._int_w + c.kd * dw

        r = self.cfg.robot
        return (max(-r.u_max, min(r.u_max, u)),
                max(-r.omega_max, min(r.omega_max, w)))



# ===========================================================================
# USUARIO SIMULADO
# ===========================================================================

class UsuarioSimulado:
    """
    Modelo del piloto humano.

    Decide QUE COMANDO quiere dar en cada instante, mirando el video (aqui,
    la posicion del robot respecto al objetivo). Es la pieza que cierra el
    lazo: sin ella el robot avanzaria en linea recta indefinidamente.

    No modela al usuario perfectamente. Modela lo suficiente para que el
    trial tenga la estructura correcta: alternancia de comandos, y por tanto
    transiciones de mirada en los momentos en que un usuario real las haria.

    ESTRATEGIA
    ----------
    Sigue una politica de alineacion en dos fases, que es como conduce una
    persona con un BCI de cuatro comandos:

      1. Si el rumbo apunta lejos del objetivo, girar hasta alinearse.
      2. Si esta alineado, avanzar.
      3. Al llegar, parar.

    Y respeta la regla de la FSM: para cambiar de accion hay que pasar por
    DETENIDO. Si esta avanzando y necesita girar, primero pide parar.

    HISTERESIS
    ----------
    Los umbrales de alineacion son distintos para entrar y salir
    (`tol_alinear` < `tol_desviado`). Sin esa histeresis, un usuario simulado
    oscilaria indefinidamente entre girar y avanzar cerca del umbral,
    generando muchas mas transiciones de las que haria una persona.
    """

    # Umbrales de rumbo (rad)
    TOL_ALINEAR = math.radians(12)
    """Por debajo de este error angular se considera alineado y avanza."""

    TOL_DESVIADO = math.radians(28)
    """Por encima de este error angular decide corregir el rumbo."""

    def __init__(self, config: Config, meta: Tuple[float, float]):
        self.cfg = config
        self.meta = meta

    def comando_deseado(self,
                        x: float, y: float, th: float,
                        estado_fsm: EstadoFSM,
                        distancias: Optional[List[float]] = None) -> int:
        """
        Indice de frecuencia que el usuario quiere mirar ahora.

        Devuelve un indice del conjunto config.bci.frecuencias:
            0 = giro izquierda, 1 = giro derecha, 2 = avanzar, 3 = parar

        El orden lo fija config.bci.frecuencias y no debe asumirse aqui de
        otra forma.
        """
        IZQ, DER, AVANZAR, PARAR = 0, 1, 2, 3

        dx = self.meta[0] - x
        dy = self.meta[1] - y
        dist = math.hypot(dx, dy)

        # Ya llego: parar
        if dist <= self.cfg.etapa2.umbral_transicion:
            return PARAR

        # Error de rumbo, normalizado a [-pi, pi]
        rumbo = math.atan2(dy, dx)
        err = math.atan2(math.sin(rumbo - th), math.cos(rumbo - th))

        # --- Decision segun el estado actual de la FSM ---
        if estado_fsm == EstadoFSM.AVANZANDO:
            # Solo interrumpe el avance si se desvio de verdad
            if abs(err) > self.TOL_DESVIADO:
                return PARAR      # primero parar, luego girara
            return AVANZAR        # sigue: el comando ya esta enclavado

        if estado_fsm in (EstadoFSM.ROTANDO_IZQ, EstadoFSM.ROTANDO_DER):
            # Deja de girar cuando ya esta apuntando al objetivo
            if abs(err) < self.TOL_ALINEAR:
                return PARAR
            return IZQ if err > 0 else DER

        # DETENIDO: decide que hacer
        if abs(err) > self.TOL_ALINEAR:
            return IZQ if err > 0 else DER
        return AVANZAR


# ===========================================================================
# SIMULADOR
# ===========================================================================

@dataclass
class ResultadoTrial:
    """Resultado completo de un trial simulado."""
    metricas: MetricasTrial
    llego_a_zona: bool
    seleccion_correcta: bool
    n_decisiones_e2: int
    trayectoria: List[Tuple[float, float]] = field(default_factory=list)


class Simulador2D:
    """
    Simulador headless de un trial completo.

    Uso:
        sim = Simulador2D(CONFIG, escenario, modo="tasm", semilla=0)
        res = sim.correr(n_objetos=4, objetivo=2)
        print(res.metricas.resumen())
    """

    TAU_PLANTA: float = 0.20
    """
    Constante de tiempo de la respuesta de velocidad (s).

    Modela que el robot no alcanza la velocidad comandada de forma
    instantanea. El valor es orientativo: para el TurtleBot3 real habria que
    identificarlo experimentalmente con un escalon de velocidad.

    Vive aqui y no en config.robot porque es un detalle de SIMULACION: en el
    robot real la dinamica la impone el hardware, no un parametro.
    """

    def __init__(self,
                 config: Config,
                 escenario: Escenario,
                 modo: Optional[str] = None,
                 semilla: int = 0):
        self.cfg = config
        self.escenario = escenario
        self.modo = modo or config.experimento.modo_bci
        self.semilla = semilla

        self.mundo = construir_mundo(config, escenario)
        self.lidar = LidarSimulado(config, self.mundo)
        self.apf = CampoPotencial(config)
        self.pid = ControladorPID(config)

    # -------------------------------------------------------------------
    def correr(self, n_objetos: int = 4,
               objetivo: Optional[int] = None) -> ResultadoTrial:
        """
        Ejecuta un trial completo: Etapa 1 (navegacion) y Etapa 2 (seleccion).

        Devuelve las metricas y el resultado de la seleccion.
        """
        cfg = self.cfg
        if objetivo is None:
            objetivo = n_objetos // 2

        fsm = CommandFSM(cfg)
        usuario = UsuarioSimulado(cfg, self.mundo.zona_e2)
        # El objetivo del mock se actualiza en cada ciclo con lo que el
        # usuario simulado quiere mirar en ese instante.
        mock = TASMMock(cfg, objetivo=2, modo=self.modo, semilla=self.semilla)
        calc = CalculadorMetricas(cfg)

        x, y, th = self.mundo.inicio
        u_act = w_act = 0.0
        t = 0.0
        paso = cfg.bci.paso_tasm

        registros: List[RegistroCiclo] = []
        trayectoria: List[Tuple[float, float]] = [(x, y)]

        llego = False
        n_iter = int(self.escenario.timeout_s / paso)

        # ---------------- ETAPA 1: navegacion ----------------
        for _ in range(n_iter):
            dists = self.lidar.escanear(x, y, th)

            # El usuario mira el video y decide que comando quiere
            mock.objetivo = usuario.comando_deseado(x, y, th, fsm.estado,
                                                    dists)

            msg = mock.siguiente()
            salida = fsm.actualizar(msg.estado, msg.freq_idx, t, msg.valido)
            rho_min = min(dists)

            # Override de emergencia: prioridad absoluta, ajeno a TASM
            emergencia = rho_min <= cfg.robot.rho_safe
            if emergencia:
                u_ref = cfg.robot.u_emergencia
                fx, fy = self.apf.fuerza_repulsiva(dists)
                w_ref = math.copysign(cfg.robot.omega_max * 0.6,
                                      fy if abs(fy) > 1e-9 else 1.0)
            else:
                u_ref, w_ref = salida.u, salida.omega
                # Asistencia del campo potencial cerca de obstaculos
                if rho_min < cfg.control.apf_rho0:
                    fx, fy = self.apf.fuerza_repulsiva(dists)
                    w_ref += 0.5 * fy
                    w_ref = max(-cfg.robot.omega_max,
                                min(cfg.robot.omega_max, w_ref))

            u_cmd, w_cmd = self.pid.actualizar(u_ref, w_ref, u_act, w_act)

            # Respuesta de la planta: primer orden con constante TAU_PLANTA.
            #
            # Es esencial que la planta NO responda instantaneamente. Si se
            # hiciera u_act = u_cmd, el error del PID se anularia en el
            # ciclo siguiente, el termino proporcional caeria a cero y el
            # robot se detendria. Modelar la inercia es lo que da al
            # controlador algo real que seguir.
            alpha = min(1.0, paso / self.TAU_PLANTA)
            u_act += (u_cmd - u_act) * alpha
            w_act += (w_cmd - w_act) * alpha

            # Integracion cinematica. Se usa el paso de TASM como paso de
            # integracion: en el sistema real el control corre mas rapido
            # (ts_control), pero para la fidelidad de este simulador la
            # diferencia no es relevante.
            x += u_act * math.cos(th) * paso
            y += u_act * math.sin(th) * paso
            th += w_act * paso
            th = math.atan2(math.sin(th), math.cos(th))

            # Contencion dentro de la sala
            x = max(0.05, min(self.mundo.ancho - 0.05, x))
            y = max(0.05, min(self.mundo.alto - 0.05, y))

            registros.append(RegistroCiclo(
                t=t,
                estado_tasm_reportado=msg.estado,
                estado_tasm_real=msg.estado_real,
                estado_fsm=salida.estado,
                transiciono=salida.transiciono,
                por_comando=(salida.motivo == MotivoNoTransicion.TRANSICIONO),
                u=u_cmd, omega=w_cmd,
                lambda_bci=msg.lambda_bci,
                freq_idx=msg.freq_idx,
                rho_min=rho_min,
                emergencia=emergencia,
                x=x, y=y, theta=th,
            ))
            trayectoria.append((x, y))

            # Condicion de llegada a la zona de seleccion
            zx, zy = self.mundo.zona_e2
            if math.hypot(x - zx, y - zy) <= cfg.etapa2.umbral_transicion:
                llego = True
                break

            t += paso

        # ---------------- ETAPA 2: seleccion ----------------
        seleccion_ok = False
        n_dec = 0

        if llego:
            objetos = generar_objetos_en_fila(cfg, n_objetos)
            bb = BusquedaBinaria(cfg, objetos)
            mock.reiniciar()

            while not bb.terminada:
                lado_correcto = bb.lado_correcto(objetivo)
                # Frecuencia que el usuario debe mirar segun el lado
                idx_deseado = 0 if lado_correcto == LadoSeleccion.IZQUIERDA else 1
                mock.objetivo = idx_deseado

                ventanas = 0
                ventanas_tr = 0
                voto: Optional[LadoSeleccion] = None
                max_v = int(cfg.etapa2.t_max_iteracion / paso)

                while ventanas < max_v:
                    msg = mock.siguiente()
                    salida = fsm.actualizar(msg.estado, msg.freq_idx, t,
                                            msg.valido)
                    if msg.estado_real == EstadoTASM.TR:
                        ventanas_tr += 1

                    registros.append(RegistroCiclo(
                        t=t,
                        estado_tasm_reportado=msg.estado,
                        estado_tasm_real=msg.estado_real,
                        estado_fsm=salida.estado,
                        transiciono=salida.transiciono,
                        por_comando=(salida.motivo ==
                                     MotivoNoTransicion.TRANSICIONO),
                        u=0.0, omega=0.0,   # el robot esta estatico en E2
                        lambda_bci=msg.lambda_bci,
                        freq_idx=msg.freq_idx,
                        rho_min=float("inf"),
                        x=x, y=y, theta=th,
                    ))

                    ventanas += 1
                    t += paso

                    # El voto se registra cuando la racha confirma un comando
                    if salida.transiciono and salida.por_comando if hasattr(
                            salida, "por_comando") else salida.transiciono:
                        voto = (LadoSeleccion.IZQUIERDA if msg.freq_idx == 0
                                else LadoSeleccion.DERECHA)
                        break

                if voto is None:
                    bb.registrar_timeout(duracion=ventanas * paso)
                    # Si acumula demasiados timeouts se aborta el trial
                    if bb._n_timeouts > 3 * bb.decisiones_teoricas:
                        break
                else:
                    bb.votar(voto, n_ventanas=ventanas,
                             n_ventanas_tr=ventanas_tr,
                             duracion=ventanas * paso)

            r_sel = bb.resultado(objetivo=objetivo)
            seleccion_ok = r_sel.acierto
            n_dec = r_sel.n_decisiones

        metricas = calc.calcular(
            registros,
            exito=(llego and seleccion_ok),
            escenario=self.escenario.nombre,
            n_objetos=n_objetos,
            modo_bci=self.modo,
        )

        return ResultadoTrial(
            metricas=metricas,
            llego_a_zona=llego,
            seleccion_correcta=seleccion_ok,
            n_decisiones_e2=n_dec,
            trayectoria=trayectoria,
        )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 70)
    print("DEMOSTRACION DEL SIMULADOR 2D")
    print("=" * 70)
    print()

    esc = CONFIG.escenarios.lista[0]
    print(f"Escenario: {esc.nombre}")
    print(f"  Sala           : {esc.sala[0]} x {esc.sala[1]} m")
    print(f"  Obstaculos     : {esc.n_obstaculos}")
    print(f"  Gap            : {esc.gap} m")
    print(f"  Margen libre   : "
          f"{esc.margen_libre(CONFIG.robot.ancho):.3f} m")
    print()

    mundo = construir_mundo(CONFIG, esc)
    print(f"  Inicio         : ({mundo.inicio[0]:.2f}, {mundo.inicio[1]:.2f})")
    print(f"  Zona E2        : ({mundo.zona_e2[0]:.2f}, "
          f"{mundo.zona_e2[1]:.2f})")
    print(f"  Obstaculos generados: {len(mundo.obstaculos)}")
    for i, o in enumerate(mundo.obstaculos):
        print(f"    {i}: x=[{o.x:.2f}, {o.x+o.ancho:.2f}] "
              f"y=[{o.y:.2f}, {o.y+o.alto:.2f}]")
    print()

    print("-" * 70)
    print("TRIAL COMPLETO en ambos modos")
    print("-" * 70)
    for modo in ("binary", "tasm"):
        sim = Simulador2D(CONFIG, esc, modo=modo, semilla=5)
        res = sim.correr(n_objetos=4, objetivo=2)
        print()
        print(f"  MODO: {modo}")
        print(f"    Llego a zona E2   : {'SI' if res.llego_a_zona else 'NO'}")
        print(f"    Seleccion correcta: "
              f"{'SI' if res.seleccion_correcta else 'NO'}")
        print(f"    Decisiones E2     : {res.n_decisiones_e2}")
        print(f"    N_FP_TR           : {res.metricas.n_fp_tr}")
        print(f"    Colisiones        : {res.metricas.n_colisiones}")
        print(f"    Emergencias       : {res.metricas.n_emergencias}")
        print(f"    rho_min global    : {res.metricas.rho_min_global:.3f} m")
        print(f"    Distancia         : "
              f"{res.metricas.distancia_recorrida:.2f} m")
        print(f"    Tiempo            : {res.metricas.tiempo_total:.1f} s")

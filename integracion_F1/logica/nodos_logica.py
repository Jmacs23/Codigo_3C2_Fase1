"""
logica/nodos_logica.py --- Logica de los nodos, sin dependencia de ROS2
Bloque 4: ros2_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado sobre ROS2 ni en el robot real.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

POR QUE LA LOGICA ESTA SEPARADA DE ROS2
----------------------------------------
Cada nodo de ROS2 hace dos cosas: hablar con otros nodos (suscribirse,
publicar, temporizar) y decidir algo (que velocidad mandar, si activar la
emergencia).

Aqui esta solo la segunda parte. Los nodos de ros2_ws/ son envoltorios finos
que llaman a estas clases.

La razon es practica: la logica se puede probar sin arrancar ROS2, sin
Gazebo y sin robot. Las 40 pruebas de este bloque corren en segundos en
cualquier maquina. Si la logica estuviera dentro de los nodos, probarla
requeriria levantar todo el sistema.

Y hay una razon de fondo: los errores dificiles no estan en la plomeria de
ROS2, que es codigo repetitivo y bien documentado. Estan en las decisiones,
que es lo que vive aqui.


LOS CINCO BLOQUES DE LOGICA
---------------------------
  LogicaSeguridad    Cuando activar el override de emergencia
  LogicaNavegacion   Como modula el campo potencial el comando del usuario
  LogicaControl      PID discreto con antiwindup
  LogicaMision       Arbitraje entre etapas y estados
  ReceptorTASM       Interpretacion de los mensajes que llegan de Windows
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict
import json
import math
import time
import numpy as np

# Permite importar config estando dentro del paquete logica/
import sys as _sys
import os as _os
_aqui = _os.path.dirname(_os.path.abspath(__file__))
_raiz = _os.path.dirname(_aqui)
for _ruta in (_raiz, _aqui):
    if _ruta not in _sys.path:
        _sys.path.insert(0, _ruta)

from config import Config


# ===========================================================================
# TIPOS COMPARTIDOS
# ===========================================================================

class ModoOperacion(Enum):
    """Quien tiene el control del robot en cada momento."""
    BCI_MANUAL = "bci_manual"
    """El usuario comanda a traves del BCI, con asistencia del campo
    potencial."""

    EMERGENCIA = "emergencia"
    """Override automatico por proximidad de obstaculo. Prioridad absoluta:
    ignora por completo el estado cognitivo."""

    DETENIDO_SEGURO = "detenido_seguro"
    """Parada por watchdog o por fallo. El robot no se mueve hasta que se
    restablezca la comunicacion."""


class Etapa(Enum):
    NAVEGACION = 1
    SELECCION = 2


@dataclass
class Velocidad:
    """Comando de velocidad para un robot diferencial."""
    lineal: float = 0.0
    angular: float = 0.0

    def limitar(self, u_max: float, w_max: float) -> "Velocidad":
        return Velocidad(
            lineal=max(-u_max, min(u_max, self.lineal)),
            angular=max(-w_max, min(w_max, self.angular)),
        )

    def es_cero(self, tol: float = 1e-6) -> bool:
        return abs(self.lineal) < tol and abs(self.angular) < tol


# ===========================================================================
# SEGURIDAD
# ===========================================================================

@dataclass
class EstadoSeguridad:
    """Resultado de una evaluacion de seguridad."""
    emergencia: bool
    rho_min: float
    angulo_obstaculo: float
    """Direccion del obstaculo mas cercano, en el marco del robot (rad)."""
    ciclos_seguros: int
    """Ciclos consecutivos por encima del umbral de recuperacion."""


class LogicaSeguridad:
    """
    Decide cuando activar y desactivar el override de emergencia.

    ES DELIBERADAMENTE AJENA A TASM
    -------------------------------
    Esta clase no sabe nada del estado cognitivo del usuario, y no debe
    saberlo. El override tiene que activarse igual si el usuario esta en
    control intencional, en transicion de mirada o en reposo: un obstaculo a
    15 cm es igual de peligroso en los tres casos.

    Mezclar seguridad fisica con intencion cognitiva seria un error de
    diseno, y ademas introduciria un modo de fallo nuevo: un error de TASM
    podria desactivar la proteccion.

    LA HISTERESIS NO ES OPCIONAL
    ----------------------------
    Se entra en emergencia a rho_safe (0.15 m) pero se sale a
    rho_safe + rho_hist (0.20 m), y ademas hay que mantenerse por encima
    durante n_hist ciclos consecutivos.

    Sin histeresis, un obstaculo justo en el umbral haria oscilar el sistema
    entre emergencia y control normal varias veces por segundo, produciendo
    un movimiento a tirones y confundiendo al usuario.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        self._emergencia = False
        self._ciclos_seguros = 0

    # -------------------------------------------------------------------
    @property
    def en_emergencia(self) -> bool:
        return self._emergencia

    # -------------------------------------------------------------------
    def evaluar(self, distancias: List[float]) -> EstadoSeguridad:
        """
        Evalua el estado de seguridad a partir del escaneo LiDAR.

        Parametros
        ----------
        distancias : lista de distancias por rayo, en orden angular
        """
        r = self.cfg.robot

        if not distancias:
            # Sin datos de LiDAR se asume lo peor. Es la opcion conservadora:
            # es preferible una emergencia espuria a una colision.
            return EstadoSeguridad(True, 0.0, 0.0, 0)

        d = np.asarray(distancias, dtype=float)
        # Los ceros suelen indicar lectura invalida, no un obstaculo pegado
        d = np.where(d <= 1e-6, r.lidar_rango_max, d)

        i_min = int(np.argmin(d))
        rho_min = float(d[i_min])
        angulo = 2 * math.pi * i_min / len(d)

        umbral_salida = r.rho_safe + r.rho_hist

        if self._emergencia:
            # Ya en emergencia: comprobar si se puede salir
            if rho_min > umbral_salida:
                self._ciclos_seguros += 1
                if self._ciclos_seguros >= r.n_hist:
                    self._emergencia = False
                    self._ciclos_seguros = 0
            else:
                self._ciclos_seguros = 0
        else:
            # En operacion normal: comprobar si hay que entrar
            if rho_min <= r.rho_safe:
                self._emergencia = True
                self._ciclos_seguros = 0

        return EstadoSeguridad(
            emergencia=self._emergencia,
            rho_min=rho_min,
            angulo_obstaculo=angulo,
            ciclos_seguros=self._ciclos_seguros,
        )

    # -------------------------------------------------------------------
    def comando_escape(self, estado: EstadoSeguridad) -> Velocidad:
        """
        Comando de evasion durante la emergencia.

        Retrocede y gira alejandose del obstaculo. La velocidad es
        deliberadamente baja: el objetivo es salir de la situacion, no
        maniobrar con agilidad.
        """
        r = self.cfg.robot

        # El giro es hacia el lado contrario del obstaculo. El seno del
        # angulo da el signo correcto: si el obstaculo esta a la izquierda
        # (angulo positivo), se gira a la derecha.
        giro = -math.copysign(r.omega_max * 0.6,
                              math.sin(estado.angulo_obstaculo)
                              if abs(math.sin(estado.angulo_obstaculo)) > 1e-6
                              else 1.0)

        return Velocidad(lineal=r.u_emergencia, angular=giro)


# ===========================================================================
# NAVEGACION
# ===========================================================================

class LogicaNavegacion:
    """
    Campo potencial artificial para asistir la evasion de obstaculos.

    NO SUSTITUYE AL USUARIO
    -----------------------
    No hay destino autonomo. El usuario decide adonde va mirando el video; el
    campo potencial solo corrige la trayectoria para que no roce.

    Esa distincion importa para el planteamiento: el sistema es de teleoperacion
    asistida, no de navegacion autonoma con supervision.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._t_estancado = 0.0
        self._ultimo_t = None

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        self._t_estancado = 0.0
        self._ultimo_t = None

    # -------------------------------------------------------------------
    def fuerza_repulsiva(self, distancias: List[float]) -> Tuple[float, float]:
        """
        Fuerza repulsiva total en el marco del robot.

        VERIFICACION DE SIGNO
        ---------------------
        El vector resultante apunta ALEJANDOSE de los obstaculos.

        Si al probar en el robot real observas que se PEGA a las paredes en
        lugar de evitarlas, el signo esta invertido. Es un error facil de
        cometer al derivar el gradiente del potencial, y la Linea 2 lo tuvo
        en su formulacion publicada.
        """
        c = self.cfg.control
        if not distancias:
            return (0.0, 0.0)

        n = len(distancias)
        paso_ang = 2 * math.pi / n
        fx = fy = 0.0

        for i, d in enumerate(distancias):
            if d >= c.apf_rho0 or d < 1e-6:
                continue
            # Magnitud del gradiente del potencial de Khatib
            mag = c.apf_eta * (1.0 / d - 1.0 / c.apf_rho0) / (d * d)
            ang = i * paso_ang
            # El signo negativo empuja en direccion OPUESTA al rayo
            fx -= mag * math.cos(ang)
            fy -= mag * math.sin(ang)

        return (fx, fy)

    # -------------------------------------------------------------------
    def asistir(self, comando: Velocidad,
                distancias: List[float],
                t: float) -> Tuple[Velocidad, bool]:
        """
        Modula el comando del usuario segun la proximidad de obstaculos.

        Devuelve (comando_modificado, estancado).

        El indicador de estancamiento sirve para detectar minimos locales del
        campo potencial: el robot recibe orden de avanzar pero la repulsion
        la cancela y no se mueve.
        """
        c = self.cfg.control
        r = self.cfg.robot

        if not distancias:
            return (comando, False)

        rho_min = min(d for d in distancias if d > 1e-6) \
            if any(d > 1e-6 for d in distancias) else r.lidar_rango_max

        # Fuera del radio de influencia no se toca el comando
        if rho_min >= c.apf_rho0:
            self._t_estancado = 0.0
            self._ultimo_t = t
            return (comando, False)

        fx, fy = self.fuerza_repulsiva(distancias)

        # La componente lateral de la repulsion corrige el rumbo
        w = comando.angular + 0.5 * fy

        # La componente frontal reduce la velocidad al acercarse de frente
        factor = max(0.0, min(1.0, (rho_min - r.rho_safe) /
                              max(c.apf_rho0 - r.rho_safe, 1e-6)))
        u = comando.lineal * factor if comando.lineal > 0 else comando.lineal

        salida = Velocidad(u, w).limitar(r.u_max, r.omega_max)

        # Deteccion de estancamiento
        estancado = False
        if self._ultimo_t is not None:
            dt = t - self._ultimo_t
            if dt > 0:
                if (abs(salida.lineal) < c.apf_epsilon_stall and
                        abs(comando.lineal) > c.apf_epsilon_stall):
                    self._t_estancado += dt
                    estancado = self._t_estancado >= c.apf_t_stall
                else:
                    self._t_estancado = 0.0
        self._ultimo_t = t

        return (salida, estancado)


# ===========================================================================
# CONTROL
# ===========================================================================

class LogicaControl:
    """
    PID discreto con antiwindup.

    SOBRE EL ANTIWINDUP
    -------------------
    Si el robot no puede alcanzar la velocidad pedida (por saturacion o
    porque el campo potencial lo frena), el error se mantiene y el termino
    integral crece sin limite. Cuando la situacion se resuelve, ese termino
    acumulado produce un sobreimpulso brusco.

    La saturacion del integrador lo evita. Es una linea de codigo que se
    olvida con facilidad y cuyo efecto solo aparece en situaciones concretas.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        self._int_u = 0.0
        self._int_w = 0.0
        self._e_u_prev = 0.0
        self._e_w_prev = 0.0

    # -------------------------------------------------------------------
    def actualizar(self, referencia: Velocidad,
                   medida: Velocidad) -> Velocidad:
        """Calcula el comando de control para seguir la referencia."""
        c = self.cfg.control
        r = self.cfg.robot
        ts = r.ts_control

        e_u = referencia.lineal - medida.lineal
        e_w = referencia.angular - medida.angular

        self._int_u += e_u * ts
        self._int_w += e_w * ts

        # Antiwindup por saturacion
        lim = c.antiwindup
        self._int_u = max(-lim, min(lim, self._int_u))
        self._int_w = max(-lim, min(lim, self._int_w))

        du = (e_u - self._e_u_prev) / ts if ts > 0 else 0.0
        dw = (e_w - self._e_w_prev) / ts if ts > 0 else 0.0
        self._e_u_prev = e_u
        self._e_w_prev = e_w

        u = c.kp * e_u + c.ki * self._int_u + c.kd * du
        w = c.kp * e_w + c.ki * self._int_w + c.kd * dw

        return Velocidad(u, w).limitar(r.u_max, r.omega_max)

    # -------------------------------------------------------------------
    @property
    def integrales(self) -> Tuple[float, float]:
        """Estado del integrador. Solo para diagnostico."""
        return (self._int_u, self._int_w)


# ===========================================================================
# RECEPTOR DE MENSAJES TASM
# ===========================================================================

@dataclass
class MensajeTASMRecibido:
    """Mensaje procedente de la maquina Windows."""
    estado: str
    freq_idx: int
    p_max: float
    lambda_bci: float
    valido: bool
    timestamp: float
    secuencia: int = 0
    rho: Optional[List[float]] = None
    rho_grad: Optional[List[float]] = None


class ReceptorTASM:
    """
    Interpreta los mensajes JSON que llegan de Windows.

    Se separa del socket a proposito: asi la interpretacion se puede probar
    sin red, alimentandola con cadenas.

    ROBUSTEZ ANTE MENSAJES CORRUPTOS
    --------------------------------
    Un mensaje mal formado NO debe tumbar el nodo. Se descarta y se cuenta.
    Si se descartan muchos, el watchdog detendra el robot, que es la
    respuesta correcta.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._ultimo: Optional[MensajeTASMRecibido] = None
        self._t_ultimo: Optional[float] = None
        self._recibidos = 0
        self._descartados = 0
        self._fuera_de_orden = 0
        self._ultima_secuencia = -1

    # -------------------------------------------------------------------
    def procesar_linea(self, linea: str,
                       t: Optional[float] = None
                       ) -> Optional[MensajeTASMRecibido]:
        """
        Interpreta una linea JSON.

        Devuelve el mensaje, o None si estaba mal formado.
        """
        linea = linea.strip()
        if not linea:
            return None

        try:
            d = json.loads(linea)
        except (json.JSONDecodeError, ValueError):
            self._descartados += 1
            return None

        try:
            msg = MensajeTASMRecibido(
                estado=str(d["estado"]),
                freq_idx=int(d["freq_idx"]),
                p_max=float(d["p_max"]),
                lambda_bci=float(d["lambda_bci"]),
                valido=bool(d["valido"]),
                timestamp=float(d.get("timestamp", 0.0)),
                secuencia=int(d.get("secuencia", 0)),
                rho=d.get("rho"),
                rho_grad=d.get("rho_grad"),
            )
        except (KeyError, TypeError, ValueError):
            self._descartados += 1
            return None

        # Validacion semantica: un estado desconocido es tan malo como un
        # JSON roto, y es mas facil que ocurra si alguien cambia el contrato
        # en un lado y no en el otro.
        if msg.estado not in ("IC", "TR", "Idle"):
            self._descartados += 1
            return None

        if msg.estado == "IC" and not (0 <= msg.freq_idx <
                                       len(self.cfg.bci.frecuencias)):
            self._descartados += 1
            return None

        # Deteccion de mensajes fuera de orden. TCP garantiza el orden, asi
        # que si esto ocurre indica un problema de reconexion o de dos
        # emisores simultaneos.
        if msg.secuencia <= self._ultima_secuencia:
            self._fuera_de_orden += 1
        self._ultima_secuencia = msg.secuencia

        self._recibidos += 1
        self._ultimo = msg
        self._t_ultimo = t if t is not None else time.perf_counter()
        return msg

    # -------------------------------------------------------------------
    def watchdog_expirado(self, t: Optional[float] = None) -> bool:
        """
        True si ha pasado demasiado tiempo sin mensaje valido.

        Cuando expira, el consumidor debe degradar a estado seguro: mantener
        el comando enclavado no es aceptable si no se sabe si el usuario
        sigue conectado.
        """
        if self._t_ultimo is None:
            return True
        ahora = t if t is not None else time.perf_counter()
        return (ahora - self._t_ultimo) > self.cfg.bci.watchdog_s

    # -------------------------------------------------------------------
    @property
    def ultimo(self) -> Optional[MensajeTASMRecibido]:
        return self._ultimo

    @property
    def estadisticas(self) -> Dict[str, float]:
        total = self._recibidos + self._descartados
        return {
            "recibidos": self._recibidos,
            "descartados": self._descartados,
            "fuera_de_orden": self._fuera_de_orden,
            "tasa_validos": self._recibidos / total if total else 0.0,
        }


# ===========================================================================
# MISION
# ===========================================================================

@dataclass
class DecisionMision:
    """Salida de un ciclo del arbitro de mision."""
    modo: ModoOperacion
    etapa: Etapa
    velocidad: Velocidad
    transiciono_fsm: bool
    motivo: str = ""


class LogicaMision:
    """
    Arbitra entre modos y etapas, y decide la velocidad de referencia.

    ORDEN DE PRIORIDADES
    --------------------
      1. EMERGENCIA        Tiene prioridad absoluta. Ignora TASM.
      2. WATCHDOG          Si no llegan mensajes, parada segura.
      3. ETAPA 2           El robot esta estatico; la seleccion no mueve.
      4. ETAPA 1           Comando del usuario, compuertado por TASM.

    El orden importa. Si la emergencia no fuera lo primero, un error de TASM
    podria impedir que se activara.
    """

    def __init__(self, config: Config):
        self.cfg = config

        # La maquina de estados viene del Bloque 1: es la misma clase que se
        # verifico alli, no una reimplementacion.
        from command_fsm import CommandFSM, EstadoTASM
        self._CommandFSM = CommandFSM
        self._EstadoTASM = EstadoTASM

        self.fsm = CommandFSM(config)
        self.seguridad = LogicaSeguridad(config)
        self.navegacion = LogicaNavegacion(config)

        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        self.fsm.reiniciar()
        self.seguridad.reiniciar()
        self.navegacion.reiniciar()
        self._etapa = Etapa.NAVEGACION
        self._t_detenido: Optional[float] = None

    # -------------------------------------------------------------------
    @property
    def etapa(self) -> Etapa:
        return self._etapa

    # -------------------------------------------------------------------
    def cambiar_etapa(self, etapa: Etapa) -> None:
        """
        Cambia de etapa.

        Al cambiar se reinicia la FSM: las frecuencias activas pasan de
        cuatro a dos, y el estado de comando anterior deja de tener sentido.
        """
        if etapa != self._etapa:
            self._etapa = etapa
            self.fsm.reiniciar()

    # -------------------------------------------------------------------
    def ciclo(self,
              mensaje: Optional[MensajeTASMRecibido],
              distancias: List[float],
              t: float,
              watchdog_expirado: bool = False) -> DecisionMision:
        """Ejecuta un ciclo completo de decision."""
        r = self.cfg.robot

        # --- 1. Seguridad, siempre primero ---
        est_seg = self.seguridad.evaluar(distancias)
        if est_seg.emergencia:
            return DecisionMision(
                modo=ModoOperacion.EMERGENCIA,
                etapa=self._etapa,
                velocidad=self.seguridad.comando_escape(est_seg),
                transiciono_fsm=False,
                motivo=f"obstaculo a {est_seg.rho_min:.3f} m",
            )

        # --- 2. Watchdog ---
        if watchdog_expirado or mensaje is None:
            return DecisionMision(
                modo=ModoOperacion.DETENIDO_SEGURO,
                etapa=self._etapa,
                velocidad=Velocidad(0.0, 0.0),
                transiciono_fsm=False,
                motivo="sin mensajes de TASM",
            )

        # --- 3. Traducir el estado a la FSM ---
        try:
            estado_tasm = self._EstadoTASM(mensaje.estado) \
                if mensaje.estado != "Idle" else self._EstadoTASM.IDLE
        except ValueError:
            estado_tasm = self._EstadoTASM.IDLE

        salida = self.fsm.actualizar(estado_tasm, mensaje.freq_idx, t,
                                     mensaje.valido)

        # --- 4. Etapa 2: el robot no se mueve ---
        if self._etapa == Etapa.SELECCION:
            return DecisionMision(
                modo=ModoOperacion.BCI_MANUAL,
                etapa=self._etapa,
                velocidad=Velocidad(0.0, 0.0),
                transiciono_fsm=salida.transiciono,
                motivo="seleccion con robot estatico",
            )

        # --- 5. Etapa 1: comando asistido ---
        comando = Velocidad(salida.u, salida.omega)
        asistido, estancado = self.navegacion.asistir(comando, distancias, t)

        return DecisionMision(
            modo=ModoOperacion.BCI_MANUAL,
            etapa=self._etapa,
            velocidad=asistido,
            transiciono_fsm=salida.transiciono,
            motivo="minimo local detectado" if estancado else "",
        )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DE LA LOGICA DE LOS NODOS")
    print("=" * 72)

    r = CONFIG.robot

    # --- Seguridad ---
    print()
    print("-" * 72)
    print("SEGURIDAD: histeresis de entrada y salida")
    print("-" * 72)
    print(f"  Entrada en emergencia : rho <= {r.rho_safe} m")
    print(f"  Salida                : rho > {r.rho_safe + r.rho_hist} m "
          f"durante {r.n_hist} ciclos")
    print()

    seg = LogicaSeguridad(CONFIG)
    n = 72
    print(f"  {'ciclo':>6} {'rho_min':>9} {'emergencia':>12} "
          f"{'ciclos seguros':>16}")
    secuencia = [0.50, 0.30, 0.16, 0.14, 0.12, 0.14, 0.17, 0.19,
                 0.21, 0.21, 0.22, 0.23, 0.25, 0.30]
    for i, rho in enumerate(secuencia):
        d = [rho] + [5.0] * (n - 1)
        e = seg.evaluar(d)
        marca = ""
        if i > 0 and rho <= r.rho_safe and not seg.en_emergencia:
            marca = ""
        print(f"  {i:>6} {rho:>8.2f}m {'SI' if e.emergencia else 'no':>12} "
              f"{e.ciclos_seguros:>16}")
    print()
    print("  Notese que la salida no es inmediata: hace falta superar 0.20 m")
    print("  durante 5 ciclos seguidos. Sin esa histeresis, un obstaculo")
    print("  justo en el umbral haria oscilar el sistema varias veces por")
    print("  segundo.")

    # --- Navegacion ---
    print()
    print("-" * 72)
    print("NAVEGACION: modulacion por campo potencial")
    print("-" * 72)
    nav = LogicaNavegacion(CONFIG)
    cmd = Velocidad(r.u_max, 0.0)
    print(f"  Comando del usuario: avanzar a {r.u_max} m/s")
    print()
    print(f"  {'rho_min':>9} {'u resultante':>14} {'factor':>9}")
    for rho in (1.00, 0.50, 0.40, 0.30, 0.20, 0.16):
        d = [rho] + [5.0] * (n - 1)
        out, _ = nav.asistir(cmd, d, 0.0)
        print(f"  {rho:>8.2f}m {out.lineal:>13.3f}m/s "
              f"{out.lineal/r.u_max:>9.2f}")
    print()
    print("  El campo potencial no cancela el comando: lo escala segun la")
    print("  proximidad. El usuario sigue decidiendo adonde va.")

    # --- Receptor ---
    print()
    print("-" * 72)
    print("RECEPTOR: robustez ante mensajes corruptos")
    print("-" * 72)
    rec = ReceptorTASM(CONFIG)

    casos = [
        ('{"estado":"IC","freq_idx":2,"p_max":0.9,"lambda_bci":0.95,'
         '"valido":true,"timestamp":1.0,"secuencia":1}', "valido"),
        ('{"estado":"TR","freq_idx":-1,"p_max":0.3,"lambda_bci":0.2,'
         '"valido":true,"timestamp":1.05,"secuencia":2}', "valido"),
        ('{roto', "JSON invalido"),
        ('{"estado":"XX","freq_idx":0,"p_max":0.5,"lambda_bci":0.5,'
         '"valido":true,"timestamp":1.1,"secuencia":3}', "estado desconocido"),
        ('{"estado":"IC","freq_idx":99,"p_max":0.5,"lambda_bci":0.5,'
         '"valido":true,"timestamp":1.15,"secuencia":4}', "indice fuera"),
        ('', "linea vacia"),
    ]

    print(f"  {'caso':>22} {'resultado':>12}")
    for linea, desc in casos:
        m = rec.procesar_linea(linea, t=1.0)
        print(f"  {desc:>22} {'aceptado' if m else 'descartado':>12}")

    print()
    st = rec.estadisticas
    print(f"  Recibidos   : {st['recibidos']}")
    print(f"  Descartados : {st['descartados']}")
    print(f"  Tasa validos: {st['tasa_validos']*100:.0f}%")
    print()
    print("  Un mensaje corrupto no tumba el nodo: se descarta y se cuenta.")
    print("  Si se descartaran muchos, el watchdog detendria el robot.")

    # --- Mision ---
    print()
    print("-" * 72)
    print("MISION: orden de prioridades")
    print("-" * 72)
    mis = LogicaMision(CONFIG)
    libre = [5.0] * n
    cerca = [0.10] + [5.0] * (n - 1)

    msg_ic = MensajeTASMRecibido("IC", 2, 0.9, 0.95, True, 0.0, 1)

    print(f"  {'situacion':>36} {'modo':>18} {'u':>8}")

    # Caso 1: comando normal. Hacen falta n_conf ventanas para enclavar.
    t = 0.0
    for _ in range(CONFIG.bci.n_conf + 1):
        d = mis.ciclo(msg_ic, libre, t)
        t += CONFIG.bci.paso_tasm
    print(f"  {'IC sostenido + camino libre':>36} {d.modo.value:>18} "
          f"{d.velocidad.lineal:>7.2f}")

    # Caso 2: aparece un obstaculo. La emergencia se impone.
    d = mis.ciclo(msg_ic, cerca, t)
    t += CONFIG.bci.paso_tasm
    print(f"  {'IC + obstaculo a 0.10 m':>36} {d.modo.value:>18} "
          f"{d.velocidad.lineal:>7.2f}")

    # Caso 3: el obstaculo desaparece, pero la histeresis exige n_hist
    # ciclos por encima del umbral antes de devolver el control.
    for i in range(CONFIG.robot.n_hist):
        d = mis.ciclo(msg_ic, libre, t)
        t += CONFIG.bci.paso_tasm
        if i == 0:
            print(f"  {'camino libre, 1er ciclo':>36} {d.modo.value:>18} "
                  f"{d.velocidad.lineal:>7.2f}")
    print(f"  {'camino libre, tras histeresis':>36} {d.modo.value:>18} "
          f"{d.velocidad.lineal:>7.2f}")

    # Caso 4: se pierde la comunicacion
    mis2 = LogicaMision(CONFIG)
    d = mis2.ciclo(None, libre, 0.0, watchdog_expirado=True)
    print(f"  {'sin mensajes (watchdog)':>36} {d.modo.value:>18} "
          f"{d.velocidad.lineal:>7.2f}")

    print()
    print("  Tres cosas que se ven aqui:")
    print()
    print("  1. El comando tarda en enclavarse: hacen falta "
          f"{CONFIG.bci.n_conf} ventanas")
    print(f"     consecutivas de IC ({CONFIG.bci.t_confirmacion*1000:.0f} ms).")
    print()
    print("  2. La emergencia se impone sobre el comando del usuario, aunque")
    print("     TASM reporte control intencional con alta confianza.")
    print()
    print("  3. Salir de la emergencia NO es inmediato: la histeresis exige")
    print(f"     {CONFIG.robot.n_hist} ciclos por encima de "
          f"{CONFIG.robot.rho_safe + CONFIG.robot.rho_hist} m. Sin eso, un")
    print("     obstaculo en el umbral haria oscilar el sistema.")

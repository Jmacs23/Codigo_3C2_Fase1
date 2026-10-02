"""
command_fsm.py --- Maquina de estados de comando con enclavamiento
Bloque 1: nucleo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.

El debugging es parte normal del trabajo de investigacion.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Implementa la maquina de estados que traduce comandos BCI en velocidades del
robot, con dos propiedades centrales:

1. ENCLAVAMIENTO (latching). El usuario mira un estimulo una sola vez, el
   comando queda enganchado, y el robot lo mantiene hasta recibir otro. El
   usuario NO tiene que mantener la mirada fija mientras el robot se mueve:
   deja de mirar los estimulos y observa el video de la camara para navegar.

   Consecuencia: durante la mayor parte de un trial el usuario esta
   legitimamente en estado Idle. Eso es operacion normal, no un fallo.

2. COMPUERTA COGNITIVA. La FSM solo cambia de estado cuando TASM confirma
   estado IC durante N_conf ventanas consecutivas. Este es el mecanismo
   central del trabajo: impide que un comando espurio detectado durante una
   transicion de mirada se convierta en el nuevo comando enclavado.

TRANSICION OBLIGATORIA POR DETENIDO
-----------------------------------
Desde cualquier estado de movimiento, el unico comando aceptado es PARAR.
Para cambiar de tipo de movimiento hay que pasar por DETENIDO.

Esto tiene dos efectos:
  - Elimina cambios bruscos de regimen (de avanzar a girar sin pausa).
  - Hace que todo cambio de accion requiera DOS desplazamientos de mirada,
    lo que duplica las ventanas TR observables y favorece la potencia
    estadistica del experimento.

    Esto no es casual: es consecuencia directa del diseno, y conviene
    declararlo asi en el manuscrito.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import math

from config import Config


# ===========================================================================
# TIPOS
# ===========================================================================

class EstadoFSM(Enum):
    """Estados de la maquina de comando."""
    DETENIDO = "DETENIDO"
    AVANZANDO = "AVANZANDO"
    ROTANDO_IZQ = "ROTANDO_IZQ"
    ROTANDO_DER = "ROTANDO_DER"


class EstadoTASM(Enum):
    """
    Estados cognitivos que entrega TASM.

    IC   : control intencional. El usuario mira deliberadamente un estimulo.
    TR   : transicion de mirada. Se esta desplazando entre estimulos, o entre
           la escena y un estimulo. NO hay intencion de comando que respetar.
    IDLE : reposo genuino. El usuario no atiende a ningun estimulo. Bajo el
           paradigma de enclavamiento este es el estado DOMINANTE.
    """
    IC = "IC"
    TR = "TR"
    IDLE = "Idle"


class MotivoNoTransicion(Enum):
    """
    Por que la FSM no cambio de estado en un ciclo dado.

    Se registra para diagnostico: permite distinguir si el sistema no
    responde porque TASM esta suprimiendo (correcto) o porque hay un
    problema de configuracion (incorrecto).
    """
    TRANSICIONO = "transiciono"
    ESTADO_NO_IC = "estado_no_ic"
    RACHA_INSUFICIENTE = "racha_insuficiente"
    COMANDO_IGNORADO = "comando_ignorado_en_movimiento"
    MENSAJE_INVALIDO = "mensaje_invalido"
    WATCHDOG = "watchdog_expirado"
    TOPE_SEGURIDAD = "tope_de_seguridad"


@dataclass
class SalidaFSM:
    """Resultado de un ciclo de la maquina de estados."""
    estado: EstadoFSM
    u: float                    # velocidad lineal (m/s)
    omega: float                # velocidad angular (rad/s)
    transiciono: bool
    motivo: MotivoNoTransicion
    racha_ic: int
    estado_anterior: Optional[EstadoFSM] = None


# ===========================================================================
# MAQUINA DE ESTADOS
# ===========================================================================

class CommandFSM:
    """
    Maquina de estados de comando con enclavamiento y compuerta TASM.

    Uso tipico:
        fsm = CommandFSM(CONFIG)
        salida = fsm.actualizar(EstadoTASM.IC, freq_idx=2, t=1.25)
        # salida.u, salida.omega -> se publican en /cmd_vel

    IMPORTANTE: todos los parametros vienen de `config`. Este modulo no
    define ningun valor numerico propio. Si quieres cambiar la racha de
    confirmacion o los topes de seguridad, hazlo en config.py.
    """

    def __init__(self, config: Config):
        self.cfg = config

        # Mapeo de indice de frecuencia a comando. El orden es el de
        # config.bci.frecuencias: (giro_izq, giro_der, avanzar, parar)
        self._comandos = (
            EstadoFSM.ROTANDO_IZQ,
            EstadoFSM.ROTANDO_DER,
            EstadoFSM.AVANZANDO,
            EstadoFSM.DETENIDO,
        )

        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Vuelve al estado inicial. Llamar al empezar cada trial."""
        self._estado = EstadoFSM.DETENIDO
        self._racha_ic = 0
        self._freq_racha: Optional[int] = None

        # Acumuladores para las salvaguardas de terminacion
        self._rotacion_acumulada = 0.0      # grados
        self._tiempo_en_movimiento = 0.0    # segundos
        self._t_ultimo = None               # marca de tiempo previa

        # Watchdog de comunicacion
        self._t_ultimo_mensaje: Optional[float] = None

        # Historial para diagnostico
        self.historial: List[SalidaFSM] = []

    # -------------------------------------------------------------------
    @property
    def estado(self) -> EstadoFSM:
        return self._estado

    @property
    def racha_ic(self) -> int:
        """Ventanas consecutivas en el mismo estado IC."""
        return self._racha_ic

    # -------------------------------------------------------------------
    def _velocidades(self, estado: EstadoFSM) -> Tuple[float, float]:
        """Traduce un estado de la FSM a velocidades del robot."""
        r = self.cfg.robot
        if estado == EstadoFSM.AVANZANDO:
            return (r.u_max, 0.0)
        if estado == EstadoFSM.ROTANDO_IZQ:
            return (0.0, +r.omega_max)
        if estado == EstadoFSM.ROTANDO_DER:
            return (0.0, -r.omega_max)
        return (0.0, 0.0)   # DETENIDO

    # -------------------------------------------------------------------
    def _en_movimiento(self) -> bool:
        return self._estado != EstadoFSM.DETENIDO

    # -------------------------------------------------------------------
    def _actualizar_acumuladores(self, t: float) -> Optional[MotivoNoTransicion]:
        """
        Actualiza los contadores de las salvaguardas y comprueba si alguna se
        ha disparado.

        Devuelve TOPE_SEGURIDAD si hay que forzar DETENIDO, None si no.

        Estas salvaguardas cubren modos de fallo (el BCI queda enganchado en
        un comando) y no deberian activarse nunca en operacion normal.
        """
        if self._t_ultimo is None:
            self._t_ultimo = t
            return None

        dt = t - self._t_ultimo
        self._t_ultimo = t

        if dt <= 0:
            return None

        if self._estado in (EstadoFSM.ROTANDO_IZQ, EstadoFSM.ROTANDO_DER):
            grados = math.degrees(self.cfg.robot.omega_max) * dt
            self._rotacion_acumulada += grados
            if self._rotacion_acumulada >= self.cfg.fsm.tope_rotacion_grados:
                return MotivoNoTransicion.TOPE_SEGURIDAD

        if self._estado == EstadoFSM.AVANZANDO:
            self._tiempo_en_movimiento += dt
            if self._tiempo_en_movimiento >= self.cfg.fsm.tope_traslacion_s:
                return MotivoNoTransicion.TOPE_SEGURIDAD

        return None

    # -------------------------------------------------------------------
    def _forzar_detenido(self, motivo: MotivoNoTransicion) -> SalidaFSM:
        """Transicion forzada a DETENIDO por salvaguarda o watchdog."""
        anterior = self._estado
        self._estado = EstadoFSM.DETENIDO
        self._racha_ic = 0
        self._freq_racha = None
        self._rotacion_acumulada = 0.0
        self._tiempo_en_movimiento = 0.0
        u, w = self._velocidades(self._estado)
        return SalidaFSM(
            estado=self._estado, u=u, omega=w,
            transiciono=(anterior != self._estado),
            motivo=motivo, racha_ic=0, estado_anterior=anterior,
        )

    # -------------------------------------------------------------------
    def actualizar(self,
                   estado_tasm: EstadoTASM,
                   freq_idx: int,
                   t: float,
                   valido: bool = True) -> SalidaFSM:
        """
        Procesa una ventana de TASM y devuelve las velocidades resultantes.

        Parametros
        ----------
        estado_tasm : estado cognitivo reportado por TASM
        freq_idx    : indice de frecuencia detectada (0..K-1), o -1 si no hay
        t           : marca de tiempo actual (s)
        valido      : False si TASM rechazo la ventana por artefactos

        Devuelve
        --------
        SalidaFSM con el estado resultante y las velocidades a publicar.

        LOGICA
        ------
        La FSM solo transiciona si se cumplen TRES condiciones:
          (a) el mensaje es valido y el watchdog no ha expirado
          (b) el estado TASM es IC
          (c) la racha de IC consecutivas alcanza N_conf

        En cualquier otro caso se mantiene el comando enclavado. Ese
        mantenimiento es deliberado: durante TR e Idle la intencion previa
        del usuario sigue vigente.
        """
        # --- Watchdog de comunicacion -----------------------------------
        if valido:
            self._t_ultimo_mensaje = t
        elif self._t_ultimo_mensaje is not None:
            if (t - self._t_ultimo_mensaje) > self.cfg.bci.watchdog_s:
                salida = self._forzar_detenido(MotivoNoTransicion.WATCHDOG)
                self.historial.append(salida)
                return salida

        # --- Salvaguardas de terminacion --------------------------------
        tope = self._actualizar_acumuladores(t)
        if tope is not None:
            salida = self._forzar_detenido(tope)
            self.historial.append(salida)
            return salida

        # --- Mensaje invalido: se mantiene el comando --------------------
        if not valido:
            self._racha_ic = 0
            self._freq_racha = None
            salida = self._construir_salida(
                False, MotivoNoTransicion.MENSAJE_INVALIDO)
            self.historial.append(salida)
            return salida

        # --- Actualizacion de la racha ----------------------------------
        # La racha solo crece si el estado es IC Y la frecuencia es la misma
        # que en la ventana anterior. Un cambio de frecuencia reinicia el
        # contador: si el usuario salta de un estimulo a otro sin pasar por
        # una fijacion estable, no se acumula evidencia.
        if estado_tasm == EstadoTASM.IC and freq_idx >= 0:
            if self._freq_racha == freq_idx:
                self._racha_ic += 1
            else:
                self._freq_racha = freq_idx
                self._racha_ic = 1
        else:
            self._racha_ic = 0
            self._freq_racha = None

        # --- Compuerta: no es IC ----------------------------------------
        if estado_tasm != EstadoTASM.IC:
            salida = self._construir_salida(
                False, MotivoNoTransicion.ESTADO_NO_IC)
            self.historial.append(salida)
            return salida

        # --- Compuerta: racha insuficiente ------------------------------
        if self._racha_ic < self.cfg.bci.n_conf:
            salida = self._construir_salida(
                False, MotivoNoTransicion.RACHA_INSUFICIENTE)
            self.historial.append(salida)
            return salida

        # --- IC confirmado: intentar transicion -------------------------
        if not (0 <= freq_idx < len(self._comandos)):
            salida = self._construir_salida(
                False, MotivoNoTransicion.MENSAJE_INVALIDO)
            self.historial.append(salida)
            return salida

        destino = self._comandos[freq_idx]
        anterior = self._estado

        # Regla de transicion obligatoria por DETENIDO:
        # en movimiento, el unico comando admitido es PARAR.
        if self._en_movimiento() and destino != EstadoFSM.DETENIDO:
            salida = self._construir_salida(
                False, MotivoNoTransicion.COMANDO_IGNORADO)
            self.historial.append(salida)
            return salida

        # Transicion efectiva
        self._estado = destino
        self._racha_ic = 0
        self._freq_racha = None

        # Al entrar en DETENIDO se limpian los acumuladores de salvaguarda
        if destino == EstadoFSM.DETENIDO:
            self._rotacion_acumulada = 0.0
            self._tiempo_en_movimiento = 0.0

        u, w = self._velocidades(self._estado)
        salida = SalidaFSM(
            estado=self._estado, u=u, omega=w,
            transiciono=(anterior != self._estado),
            motivo=MotivoNoTransicion.TRANSICIONO,
            racha_ic=0, estado_anterior=anterior,
        )
        self.historial.append(salida)
        return salida

    # -------------------------------------------------------------------
    def _construir_salida(self, transiciono: bool,
                          motivo: MotivoNoTransicion) -> SalidaFSM:
        """Salida sin cambio de estado: se mantiene el comando enclavado."""
        u, w = self._velocidades(self._estado)
        return SalidaFSM(
            estado=self._estado, u=u, omega=w,
            transiciono=transiciono, motivo=motivo,
            racha_ic=self._racha_ic, estado_anterior=self._estado,
        )

    # -------------------------------------------------------------------
    def resumen_historial(self) -> str:
        """Vuelca el historial de forma legible, para diagnostico."""
        if not self.historial:
            return "(sin historial)"
        L = [f"{'n':>4} {'estado':<13} {'u':>6} {'w':>6} {'racha':>6} motivo"]
        L.append("-" * 68)
        for i, s in enumerate(self.historial):
            marca = " <--" if s.transiciono else ""
            L.append(f"{i:>4} {s.estado.value:<13} {s.u:>6.2f} {s.omega:>6.2f} "
                     f"{s.racha_ic:>6} {s.motivo.value}{marca}")
        return "\n".join(L)


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 70)
    print("DEMOSTRACION DE LA MAQUINA DE ESTADOS CON ENCLAVAMIENTO")
    print("=" * 70)
    print()
    print(f"Racha necesaria para confirmar: {CONFIG.bci.n_conf} ventanas "
          f"= {CONFIG.bci.t_confirmacion:.2f} s")
    print(f"Duracion tipica de una transicion de mirada: "
          f"{CONFIG.mock.dur_tr_min}-{CONFIG.mock.dur_tr_max} s "
          f"= {CONFIG.mock.dur_tr_min/CONFIG.bci.paso_tasm:.0f}-"
          f"{CONFIG.mock.dur_tr_max/CONFIG.bci.paso_tasm:.0f} ventanas")
    print()
    print("Por construccion, una transicion tipica NO alcanza a completar la")
    print("racha. Esa es la proteccion estructural del sistema.")
    print()

    fsm = CommandFSM(CONFIG)
    paso = CONFIG.bci.paso_tasm
    t = 0.0

    # --- Escenario 1: el usuario mira 'avanzar' (indice 2) el tiempo justo
    print("-" * 70)
    print("ESCENARIO 1: el usuario fija la mirada en AVANZAR (14 Hz)")
    print("-" * 70)
    for i in range(10):
        s = fsm.actualizar(EstadoTASM.IC, freq_idx=2, t=t)
        if s.transiciono:
            print(f"  Ventana {i+1}: TRANSICION -> {s.estado.value}, "
                  f"u={s.u:.2f} m/s")
        t += paso
    print(f"  Estado final: {fsm.estado.value}")
    print()

    # --- Escenario 2: transicion de mirada con comando espurio
    print("-" * 70)
    print("ESCENARIO 2: transicion de mirada de 0.75 s (6 ventanas)")
    print("             con comandos espurios de 'girar' intercalados")
    print("-" * 70)
    espurios = 0
    for i in range(6):
        # El clasificador se equivoca en la mitad de las ventanas
        if i % 2 == 0:
            s = fsm.actualizar(EstadoTASM.IC, freq_idx=0, t=t)  # espurio
            espurios += 1
        else:
            s = fsm.actualizar(EstadoTASM.TR, freq_idx=-1, t=t)
        t += paso
    print(f"  Comandos espurios recibidos: {espurios}")
    print(f"  Estado final: {fsm.estado.value} "
          f"(sigue avanzando, el espurio no se enclavo)")
    print(f"  Racha alcanzada: {fsm.racha_ic} de {CONFIG.bci.n_conf}")
    print()

    # --- Escenario 3: el usuario decide parar
    print("-" * 70)
    print("ESCENARIO 3: el usuario fija la mirada en PARAR (15.2 Hz)")
    print("-" * 70)
    for i in range(10):
        s = fsm.actualizar(EstadoTASM.IC, freq_idx=3, t=t)
        if s.transiciono:
            print(f"  Ventana {i+1}: TRANSICION -> {s.estado.value}, "
                  f"u={s.u:.2f} m/s")
        t += paso
    print(f"  Estado final: {fsm.estado.value}")
    print()

    print("=" * 70)
    print("Historial completo:")
    print("=" * 70)
    print(fsm.resumen_historial())

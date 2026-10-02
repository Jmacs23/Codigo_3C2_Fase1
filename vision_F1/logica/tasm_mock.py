"""
tasm_mock.py --- Generador sintetico de la salida de TASM
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
Genera la SALIDA de TASM, no la senal EEG. Es un nivel de abstraccion mas
arriba: produce directamente el mensaje que TASM publicaria.

    estado = IC, freq_idx = 2, p_max = 0.87, lambda_bci = 0.92

Sirve para que puedas desarrollar y probar TODO el sistema robotico sin
esperar a que el modulo TASM real este listo, y sin necesitar un sujeto con
electrodos puestos.

DOS USOS, UNO PROHIBIDO
-----------------------
  USO 1 --- Durante el desarrollo: OBLIGATORIO.
    Necesitas algo que publique estados mientras TASM no existe. Esto es
    andamiaje de ingenieria, no ciencia.

  USO 2 --- En los experimentos formales: PROHIBIDO.
    Alli la senal debe ser EEG real. Usar senal sintetica destruiria uno de
    los diferenciadores del trabajo frente a publicaciones previas que si
    emplearon EEG simulado, y un revisor lo detectaria.

COMO FUNCIONA
-------------
El mock mantiene dos cosas separadas:

  - El ESTADO REAL (ground truth), que sigue un guion programado de
    duraciones tomadas de la literatura: IC de 1.5-4.0 s, TR de 0.6-0.9 s,
    Idle de 2.0-6.0 s.

  - El ESTADO REPORTADO, que es el anterior contaminado con las tasas de
    error configurables en `config.mock`.

Tener ambos permite calcular metricas como N_FP_TR comparando lo que el
sistema hizo contra lo que deberia haber hecho. Eso es imposible con EEG
real sin un protocolo de etiquetado, y es lo que hace util al mock para el
hito de viabilidad temprana.

MODO BINARIO
------------
Con `modo="binary"` el mock colapsa TR e Idle en una sola clase de
no-control, que es lo que hacen los detectores del estado del arte. Sirve
como condicion de comparacion (baseline) en el experimento.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import random

from config import Config
from command_fsm import EstadoTASM


# ===========================================================================
# TIPOS
# ===========================================================================

@dataclass
class MensajeTASM:
    """
    Replica del mensaje ROS2 `TASMState`.

    Los campos coinciden exactamente con el contrato acordado con la Linea 1,
    de modo que sustituir el mock por TASM real no requiera tocar el codigo
    que lo consume.
    """
    estado: EstadoTASM
    freq_idx: int           # -1 si el estado no es IC
    p_max: float            # confianza del clasificador de comando [0,1]
    lambda_bci: float       # P(IC | observaciones), salida del HMM [0,1]
    valido: bool = True     # False si la ventana se rechazo por artefactos

    # --- Solo para el mock: no existe en el mensaje real ---
    estado_real: Optional[EstadoTASM] = None
    freq_real: int = -1

    @property
    def es_error(self) -> bool:
        """True si el estado reportado difiere del real."""
        if self.estado_real is None:
            return False
        return self.estado != self.estado_real


# ===========================================================================
# GENERADOR
# ===========================================================================

class TASMMock:
    """
    Generador sintetico de estados TASM.

    Uso:
        mock = TASMMock(CONFIG, objetivo=2)
        for _ in range(100):
            msg = mock.siguiente()
            # msg.estado, msg.freq_idx -> alimentan la FSM

    Parametros
    ----------
    config   : objeto Config. Todos los valores salen de config.mock.
    objetivo : indice de frecuencia que el usuario simulado quiere seleccionar.
               Durante los periodos IC, el mock reportara esta frecuencia
               (con probabilidad p_acierto_frecuencia).
    modo     : "tasm" para 3 clases, "binary" para colapsar TR e Idle.
    semilla  : si se da, sobreescribe la de config. Util para variar entre
               trials manteniendo reproducibilidad.
    """

    def __init__(self,
                 config: Config,
                 objetivo: int = 0,
                 modo: Optional[str] = None,
                 semilla: Optional[int] = None):
        self.cfg = config
        self.objetivo = objetivo
        self.modo = modo if modo is not None else config.experimento.modo_bci

        if self.modo not in config.experimento.MODOS_VALIDOS:
            raise ValueError(
                f"modo='{self.modo}' no valido. "
                f"Opciones: {config.experimento.MODOS_VALIDOS}"
            )

        s = semilla if semilla is not None else config.mock.semilla
        self._rng = random.Random(s)

        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Reinicia el guion. Llamar al empezar cada trial."""
        self._t = 0.0
        self._estado_real = EstadoTASM.IDLE
        self._t_fin_estado = self._duracion(EstadoTASM.IDLE)
        self._historial: List[MensajeTASM] = []

        # Estado del episodio actual. Ver la nota sobre modelo de error por
        # episodios en config.TASMMockConfig.
        self._episodio_es_fp = False
        self._episodio_freq = -1

    # -------------------------------------------------------------------
    def _duracion(self, estado: EstadoTASM) -> float:
        """Duracion aleatoria del estado, en el rango de config.mock."""
        m = self.cfg.mock
        if estado == EstadoTASM.IC:
            return self._rng.uniform(m.dur_ic_min, m.dur_ic_max)
        if estado == EstadoTASM.TR:
            return self._rng.uniform(m.dur_tr_min, m.dur_tr_max)
        return self._rng.uniform(m.dur_idle_min, m.dur_idle_max)

    # -------------------------------------------------------------------
    def _siguiente_estado_real(self) -> EstadoTASM:
        """
        Decide el siguiente estado del guion.

        La secuencia respeta la restriccion fisica que el HMM de TASM tambien
        impone: no se puede pasar de Idle a IC directamente, porque para
        empezar a mirar un estimulo hay que mover los ojos hacia el.

            Idle -> TR -> IC -> TR -> Idle -> ...
        """
        if self._estado_real == EstadoTASM.IDLE:
            return EstadoTASM.TR
        if self._estado_real == EstadoTASM.TR:
            # Tras una transicion se llega a un estimulo o se vuelve al reposo
            return EstadoTASM.IC if self._rng.random() < 0.75 else EstadoTASM.IDLE
        # Desde IC siempre se sale por una transicion
        return EstadoTASM.TR

    # -------------------------------------------------------------------
    def _iniciar_episodio(self) -> None:
        """
        Decide, al comenzar cada episodio, si producira falsos positivos.

        Esta es la pieza clave del modelo de error. En lugar de sortear un
        error en cada ventana de forma independiente, se sortea UNA VEZ por
        episodio si el detector va a confundirse, y en caso afirmativo se fija
        una unica frecuencia que se mantendra durante todo el transitorio.

        Fisicamente: durante una transicion del estimulo A al B, la senal es
        una mezcla decreciente de A y creciente de B. Un clasificador que se
        confunda reportara A o B de forma sostenida, no las cuatro
        frecuencias al azar.
        """
        m = self.cfg.mock
        r = self._rng

        if self._estado_real == EstadoTASM.TR:
            p = (m.p_episodio_fp_binary if self.modo == "binary"
                 else m.p_episodio_fp_tasm)
            self._episodio_es_fp = (r.random() < p)
            if self._episodio_es_fp:
                # La frecuencia reportada es la del estimulo que se abandona
                # o la del que se adquiere, no una cualquiera.
                candidatas = [self.objetivo]
                otras = [i for i in range(len(self.cfg.bci.frecuencias))
                         if i != self.objetivo]
                if otras:
                    candidatas.append(r.choice(otras))
                self._episodio_freq = r.choice(candidatas)
            else:
                self._episodio_freq = -1

        elif self._estado_real == EstadoTASM.IDLE:
            p = (m.p_error_idle_como_ic_binary if self.modo == "binary"
                 else m.p_error_idle_como_ic_tasm)
            self._episodio_es_fp = (r.random() < p)
            self._episodio_freq = (r.randrange(len(self.cfg.bci.frecuencias))
                                   if self._episodio_es_fp else -1)
        else:
            self._episodio_es_fp = False
            self._episodio_freq = -1

    # -------------------------------------------------------------------
    def _contaminar(self, real: EstadoTASM) -> Tuple[EstadoTASM, int, float]:
        """
        Aplica las tasas de error para producir el estado reportado.

        Devuelve (estado_reportado, freq_idx, lambda_bci).
        """
        m = self.cfg.mock
        r = self._rng

        # Las tasas de error dependen del MODO. Esta es la diferencia
        # esencial entre las dos condiciones experimentales:
        #
        #   - El detector BINARIO no tiene una clase para la transicion.
        #     Durante una TR ve SSVEP parcial y solo puede decidir entre
        #     "se parece a IC" o "no". Su tasa de falsos IC es alta.
        #
        #   - TASM tiene una clase dedicada a TR, y el HMM anade la
        #     restriccion Idle -> IC. Su tasa es sustancialmente menor.
        #
        # Si ambos modos tuvieran la misma tasa, el experimento no podria
        # medir nada: seria una diferencia de etiqueta sin diferencia de
        # comportamiento.
        # --- Estado IC real ---
        if real == EstadoTASM.IC:
            if r.random() < m.p_acierto_frecuencia:
                idx = self.objetivo
            else:
                otros = [i for i in range(len(self.cfg.bci.frecuencias))
                         if i != self.objetivo]
                idx = r.choice(otros)
            lam = r.uniform(0.70, 0.98)
            return (EstadoTASM.IC, idx, lam)

        # --- Estado TR real ---
        if real == EstadoTASM.TR:
            if self._episodio_es_fp and \
               r.random() < m.p_ventana_dentro_episodio:
                # Falso positivo sostenido: se mantiene la MISMA frecuencia
                # durante todo el episodio, que es lo que permite acumular
                # una racha y enclavar un comando espurio.
                return (EstadoTASM.IC, self._episodio_freq,
                        r.uniform(0.50, 0.75))

            if self.modo == "binary":
                # El detector binario colapsa TR e Idle en una sola clase
                return (EstadoTASM.IDLE, -1, r.uniform(0.05, 0.30))
            return (EstadoTASM.TR, -1, r.uniform(0.10, 0.40))

        # --- Estado Idle real ---
        if self._episodio_es_fp and r.random() < m.p_ventana_dentro_episodio:
            return (EstadoTASM.IC, self._episodio_freq, r.uniform(0.50, 0.70))

        return (EstadoTASM.IDLE, -1, r.uniform(0.02, 0.25))

    # -------------------------------------------------------------------
    def siguiente(self) -> MensajeTASM:
        """
        Avanza un paso de ventana y devuelve el mensaje correspondiente.

        Cada llamada avanza `config.bci.paso_tasm` segundos.
        """
        # Avance del guion
        if self._t >= self._t_fin_estado:
            self._estado_real = self._siguiente_estado_real()
            self._t_fin_estado = self._t + self._duracion(self._estado_real)
            self._iniciar_episodio()

        estado_rep, idx, lam = self._contaminar(self._estado_real)

        # p_max es la confianza del clasificador de comando, distinta de
        # lambda_bci: mide certeza sobre CUAL frecuencia, no sobre si hay
        # intencion de comandar.
        p_max = self._rng.uniform(0.60, 0.99) if estado_rep == EstadoTASM.IC \
            else self._rng.uniform(0.10, 0.50)

        msg = MensajeTASM(
            estado=estado_rep,
            freq_idx=idx,
            p_max=p_max,
            lambda_bci=lam,
            valido=True,
            estado_real=self._estado_real,
            freq_real=self.objetivo if self._estado_real == EstadoTASM.IC else -1,
        )

        self._historial.append(msg)
        self._t += self.cfg.bci.paso_tasm
        return msg

    # -------------------------------------------------------------------
    @property
    def tiempo(self) -> float:
        """Tiempo simulado transcurrido (s)."""
        return self._t

    @property
    def estado_real(self) -> EstadoTASM:
        """Estado verdadero actual (ground truth)."""
        return self._estado_real

    @property
    def historial(self) -> List[MensajeTASM]:
        return self._historial

    # -------------------------------------------------------------------
    def estadisticas(self) -> dict:
        """
        Resumen del guion generado. Util para verificar que las tasas de
        error configuradas se estan aplicando de verdad.
        """
        if not self._historial:
            return {}

        n = len(self._historial)
        reales = [m.estado_real for m in self._historial]
        errores = sum(1 for m in self._historial if m.es_error)

        fp_tr = sum(1 for m in self._historial
                    if m.estado_real == EstadoTASM.TR
                    and m.estado == EstadoTASM.IC)
        n_tr = sum(1 for e in reales if e == EstadoTASM.TR)

        fp_idle = sum(1 for m in self._historial
                      if m.estado_real == EstadoTASM.IDLE
                      and m.estado == EstadoTASM.IC)
        n_idle = sum(1 for e in reales if e == EstadoTASM.IDLE)

        return {
            "ventanas": n,
            "tiempo_s": n * self.cfg.bci.paso_tasm,
            "ventanas_IC": sum(1 for e in reales if e == EstadoTASM.IC),
            "ventanas_TR": n_tr,
            "ventanas_Idle": n_idle,
            "errores_totales": errores,
            "tasa_error": errores / n,
            "FP_en_TR": fp_tr,
            "tasa_FP_TR": fp_tr / n_tr if n_tr else 0.0,
            "FP_en_Idle": fp_idle,
            "tasa_FP_Idle": fp_idle / n_idle if n_idle else 0.0,
        }


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 70)
    print("DEMOSTRACION DEL MOCK DE TASM")
    print("=" * 70)
    print()
    print("El mock genera la SALIDA de TASM, no la senal EEG.")
    print("Mantiene separados el estado real (ground truth) y el reportado,")
    print("lo que permite medir cuantos comandos espurios se producen.")
    print()

    for modo in ("tasm", "binary"):
        print("-" * 70)
        print(f"MODO: {modo}")
        print("-" * 70)

        mock = TASMMock(CONFIG, objetivo=2, modo=modo, semilla=7)
        n_ventanas = int(60.0 / CONFIG.bci.paso_tasm)   # 60 s simulados
        for _ in range(n_ventanas):
            mock.siguiente()

        st = mock.estadisticas()
        print(f"  Simulados {st['tiempo_s']:.0f} s "
              f"({st['ventanas']} ventanas)")
        print(f"  Distribucion real : IC={st['ventanas_IC']}  "
              f"TR={st['ventanas_TR']}  Idle={st['ventanas_Idle']}")
        print(f"  Tasa de error     : {st['tasa_error']*100:.1f}%")
        print(f"  Falsos IC en TR   : {st['FP_en_TR']} "
              f"({st['tasa_FP_TR']*100:.1f}% de las ventanas TR)")
        print(f"  Falsos IC en Idle : {st['FP_en_Idle']} "
              f"({st['tasa_FP_Idle']*100:.1f}% de las ventanas Idle)")
        print()

    print("-" * 70)
    print("Primeras 20 ventanas en modo 'tasm' (real -> reportado):")
    print("-" * 70)
    mock = TASMMock(CONFIG, objetivo=2, modo="tasm", semilla=7)
    print(f"{'n':>4} {'t(s)':>6} {'real':>6} {'reportado':>10} "
          f"{'freq':>5} {'lambda':>7}  error")
    for i in range(20):
        m = mock.siguiente()
        marca = "  <-- ERROR" if m.es_error else ""
        print(f"{i:>4} {i*CONFIG.bci.paso_tasm:>6.3f} "
              f"{m.estado_real.value:>6} {m.estado.value:>10} "
              f"{m.freq_idx:>5} {m.lambda_bci:>7.3f}{marca}")

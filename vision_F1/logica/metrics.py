"""
metrics.py --- Metricas del sistema
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
Calcula las metricas del experimento a partir del registro de un trial.

LA METRICA PRIMARIA ES N_FP_TR
------------------------------
Cambios espurios de enclavamiento: cuantas veces la maquina de estados
cambio de comando durante una ventana en la que el usuario estaba realmente
en transicion de mirada.

Es la metrica primaria por una razon de diseno: bajo el paradigma de
enclavamiento, un cambio de estado de la FSM es un evento DISCRETO y
DIRECTAMENTE CONTABLE. No requiere elegir un umbral arbitrario sobre una
senal continua, a diferencia de metricas basadas en velocidad o desviacion.

POR QUE NO SE USA UNA METRICA DE SEGURIDAD COMO PRIMARIA
--------------------------------------------------------
Contar colisiones o violaciones de margen tiene dos problemas en este
sistema:

  - El override de emergencia puede absorber las consecuencias de un comando
    espurio antes de que llegue a colision. El error ocurrio, pero no deja
    rastro en la metrica.

  - Esta confundida con el progreso de la tarea: una condicion que avanza
    poco tiene pocas oportunidades de colisionar, y saldria artificialmente
    bien.

Ambas se reportan como metricas secundarias.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Sequence
import math

from config import Config
from command_fsm import EstadoTASM, EstadoFSM, SalidaFSM


# ===========================================================================
# REGISTRO DE UN CICLO
# ===========================================================================

@dataclass
class RegistroCiclo:
    """
    Instantanea de un ciclo del sistema.

    Se acumula una por cada ventana de TASM procesada. El conjunto de
    registros de un trial es lo que consume el calculador de metricas.
    """
    t: float
    estado_tasm_reportado: EstadoTASM
    estado_tasm_real: Optional[EstadoTASM]
    estado_fsm: EstadoFSM
    transiciono: bool
    por_comando: bool
    """True si la transicion la causo un comando BCI confirmado.
    False si la forzo una salvaguarda (topes, watchdog).

    Distinguirlas es esencial: una salvaguarda no es un comando espurio."""
    u: float
    omega: float
    lambda_bci: float
    freq_idx: int
    rho_min: float = float("inf")
    emergencia: bool = False
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


# ===========================================================================
# RESULTADO
# ===========================================================================

@dataclass
class MetricasTrial:
    """Metricas calculadas de un trial completo."""

    # --- Primaria ---
    n_fp_tr: int = 0
    """Cambios espurios de enclavamiento: transiciones de la FSM ocurridas
    durante ventanas de TR real. METRICA PRIMARIA."""

    # --- Diagnostico del detector ---
    fpr_tr: float = 0.0
    """Fraccion de ventanas TR reales reportadas como IC. Comparable con el
    benchmark de la Linea 1 (LDA solo = 0.141, LDA+HMM = 0.089)."""

    fpr_idle: float = 0.0
    """Fraccion de ventanas Idle reales reportadas como IC."""

    t_tr: float = 0.0
    """Tiempo acumulado en estado TR real (s). Diagnostica si un tiempo de
    tarea elevado se debe a transiciones genuinas o a mala calibracion."""

    # --- Tarea ---
    exito: bool = False
    tiempo_total: float = 0.0
    n_transiciones: int = 0
    """Transiciones legitimas de la FSM (cambios de comando efectivos)."""

    n_forzadas: int = 0
    """Transiciones forzadas por salvaguarda (topes o watchdog). No son
    comandos espurios; se reportan aparte para diagnostico."""

    # --- Seguridad (secundarias) ---
    n_colisiones: int = 0
    n_emergencias: int = 0
    rho_min_global: float = float("inf")

    # --- Movimiento ---
    distancia_recorrida: float = 0.0
    jerk_medio: float = 0.0

    # --- Contexto ---
    modo_bci: str = ""
    escenario: str = ""
    n_objetos: int = 0
    n_ventanas: int = 0

    def resumen(self) -> str:
        """Vuelca las metricas de forma legible."""
        L = []
        L.append(f"  Modo BCI            : {self.modo_bci}")
        if self.escenario:
            L.append(f"  Escenario           : {self.escenario}")
        L.append(f"  Exito               : {'SI' if self.exito else 'NO'}")
        L.append(f"  Tiempo total        : {self.tiempo_total:.1f} s")
        L.append("")
        L.append(f"  N_FP_TR (primaria)  : {self.n_fp_tr}")
        L.append(f"  FPR en TR           : {self.fpr_tr*100:.1f}%")
        L.append(f"  FPR en Idle         : {self.fpr_idle*100:.1f}%")
        L.append(f"  Tiempo en TR        : {self.t_tr:.1f} s")
        L.append("")
        L.append(f"  Transiciones utiles : {self.n_transiciones}")
        L.append(f"  Forzadas (salvag.)  : {self.n_forzadas}")
        L.append(f"  Colisiones          : {self.n_colisiones}")
        L.append(f"  Emergencias         : {self.n_emergencias}")
        L.append(f"  rho_min global      : {self.rho_min_global:.3f} m")
        L.append(f"  Distancia recorrida : {self.distancia_recorrida:.2f} m")
        return "\n".join(L)


# ===========================================================================
# CALCULADOR
# ===========================================================================

class CalculadorMetricas:
    """
    Calcula metricas a partir de una lista de RegistroCiclo.

    Uso:
        calc = CalculadorMetricas(CONFIG)
        m = calc.calcular(registros, exito=True, escenario="Escenario 1")
    """

    def __init__(self, config: Config):
        self.cfg = config

    # -------------------------------------------------------------------
    def calcular(self,
                 registros: Sequence[RegistroCiclo],
                 exito: bool = False,
                 escenario: str = "",
                 n_objetos: int = 0,
                 modo_bci: Optional[str] = None) -> MetricasTrial:
        """Calcula todas las metricas de un trial."""

        m = MetricasTrial(
            modo_bci=modo_bci or self.cfg.experimento.modo_bci,
            escenario=escenario,
            n_objetos=n_objetos,
            exito=exito,
            n_ventanas=len(registros),
        )

        if not registros:
            return m

        paso = self.cfg.bci.paso_tasm
        m.tiempo_total = registros[-1].t - registros[0].t + paso

        # --- METRICA PRIMARIA: cambios espurios de enclavamiento --------
        #
        # Una transicion de la FSM es espuria si cumple DOS condiciones:
        #   (a) fue causada por un comando BCI, no por una salvaguarda
        #   (b) ocurrio mientras el estado REAL del usuario era TR
        #
        # La condicion (a) es esencial. Las transiciones forzadas por el tope
        # de rotacion, el tope de traslacion o el watchdog NO son comandos
        # espurios: son el sistema protegiendose. Contarlas inflaria la
        # metrica con eventos que no tienen nada que ver con el BCI, y como
        # ocurren igual en ambas condiciones, enmascararian por completo el
        # efecto que se quiere medir.
        m.n_fp_tr = sum(
            1 for r in registros
            if r.transiciono
            and r.por_comando
            and r.estado_tasm_real == EstadoTASM.TR
        )

        # Transiciones legitimas: por comando, durante IC real
        m.n_transiciones = sum(
            1 for r in registros
            if r.transiciono
            and r.por_comando
            and r.estado_tasm_real == EstadoTASM.IC
        )

        # Transiciones forzadas por salvaguarda. Se reportan aparte: si son
        # muchas, indica que algo va mal en la operacion o en los topes.
        m.n_forzadas = sum(
            1 for r in registros if r.transiciono and not r.por_comando
        )

        # --- FPR por estado ---------------------------------------------
        ventanas_tr = [r for r in registros
                       if r.estado_tasm_real == EstadoTASM.TR]
        if ventanas_tr:
            fp = sum(1 for r in ventanas_tr
                     if r.estado_tasm_reportado == EstadoTASM.IC)
            m.fpr_tr = fp / len(ventanas_tr)
        m.t_tr = len(ventanas_tr) * paso

        ventanas_idle = [r for r in registros
                         if r.estado_tasm_real == EstadoTASM.IDLE]
        if ventanas_idle:
            fp = sum(1 for r in ventanas_idle
                     if r.estado_tasm_reportado == EstadoTASM.IC)
            m.fpr_idle = fp / len(ventanas_idle)

        # --- Seguridad ---------------------------------------------------
        m.rho_min_global = min(r.rho_min for r in registros)

        # Una colision es un evento, no una muestra: se cuentan los flancos
        # de entrada en la condicion, no las muestras que la cumplen.
        umbral_colision = 0.05
        en_colision = False
        for r in registros:
            if r.rho_min < umbral_colision and not en_colision:
                m.n_colisiones += 1
                en_colision = True
            elif r.rho_min >= umbral_colision:
                en_colision = False

        en_emergencia = False
        for r in registros:
            if r.emergencia and not en_emergencia:
                m.n_emergencias += 1
                en_emergencia = True
            elif not r.emergencia:
                en_emergencia = False

        # --- Movimiento --------------------------------------------------
        d = 0.0
        for a, b in zip(registros[:-1], registros[1:]):
            d += math.hypot(b.x - a.x, b.y - a.y)
        m.distancia_recorrida = d

        m.jerk_medio = self._jerk(registros, paso)

        return m

    # -------------------------------------------------------------------
    @staticmethod
    def _jerk(registros: Sequence[RegistroCiclo], paso: float) -> float:
        """
        Jerk medio: derivada tercera de la posicion, aproximada como derivada
        segunda de la velocidad de comando. Mide la suavidad del movimiento.
        """
        if len(registros) < 3:
            return 0.0
        us = [r.u for r in registros]
        total = 0.0
        n = 0
        for i in range(1, len(us) - 1):
            d2 = (us[i + 1] - 2 * us[i] + us[i - 1]) / (paso ** 2)
            total += abs(d2)
            n += 1
        return total / n if n else 0.0


# ===========================================================================
# COMPARACION ENTRE CONDICIONES
# ===========================================================================

@dataclass
class ComparacionPareada:
    """
    Resultado de comparar la condicion TASM contra el baseline binario sobre
    los mismos trials.

    Esta es la comparacion central del trabajo: misma base de codigo, misma
    calibracion, mismo sujeto, misma sesion. La unica diferencia es si el
    detector distingue TR de Idle.
    """
    n_pares: int
    media_tasm: float
    media_binary: float
    delta_absoluto: float
    """Reduccion absoluta. Se usa esta y no la relativa porque la relativa
    SATURA: si la condicion propuesta llega a cero eventos, el cociente vale
    1 y no puede crecer, lo que hace inverificable la hipotesis de
    escalamiento."""
    delta_relativo: float
    d_cohen: float

    def resumen(self) -> str:
        L = []
        L.append(f"  Pares comparados    : {self.n_pares}")
        L.append(f"  Media TASM          : {self.media_tasm:.2f}")
        L.append(f"  Media baseline      : {self.media_binary:.2f}")
        L.append(f"  Reduccion absoluta  : {self.delta_absoluto:+.2f}")
        L.append(f"  Reduccion relativa  : {self.delta_relativo*100:+.1f}%")
        L.append(f"  d de Cohen          : {self.d_cohen:.2f}")
        return "\n".join(L)


def comparar_pareado(valores_tasm: Sequence[float],
                     valores_binary: Sequence[float]) -> ComparacionPareada:
    """
    Compara dos series pareadas (mismo sujeto, misma condicion geometrica).

    Devuelve la reduccion absoluta, la relativa y el tamano de efecto.
    El test estadistico formal (t-test pareado) se hace fuera de este modulo,
    con scipy, en el script de analisis.
    """
    if len(valores_tasm) != len(valores_binary):
        raise ValueError(
            f"Las series deben tener la misma longitud: "
            f"{len(valores_tasm)} vs {len(valores_binary)}"
        )
    if not valores_tasm:
        raise ValueError("Series vacias.")

    n = len(valores_tasm)
    mt = sum(valores_tasm) / n
    mb = sum(valores_binary) / n

    diffs = [b - t for t, b in zip(valores_tasm, valores_binary)]
    md = sum(diffs) / n

    if n > 1:
        var = sum((d - md) ** 2 for d in diffs) / (n - 1)
        sd = math.sqrt(var)
        d_cohen = md / sd if sd > 1e-12 else 0.0
    else:
        d_cohen = 0.0

    rel = (mb - mt) / mb if abs(mb) > 1e-12 else 0.0

    return ComparacionPareada(
        n_pares=n, media_tasm=mt, media_binary=mb,
        delta_absoluto=md, delta_relativo=rel, d_cohen=d_cohen,
    )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG
    from tasm_mock import TASMMock
    from command_fsm import CommandFSM, MotivoNoTransicion

    print("=" * 70)
    print("DEMOSTRACION DEL CALCULO DE METRICAS")
    print("=" * 70)
    print()
    print("Se simulan 60 s en cada modo y se comparan los cambios espurios")
    print("de enclavamiento. Ambos usan la MISMA semilla, por lo que el")
    print("guion de estados reales es identico: la unica diferencia es como")
    print("los reporta el detector.")
    print()

    calc = CalculadorMetricas(CONFIG)
    resultados = {}

    for modo in ("binary", "tasm"):
        mock = TASMMock(CONFIG, objetivo=2, modo=modo, semilla=11)
        fsm = CommandFSM(CONFIG)
        registros = []
        t = 0.0

        for _ in range(int(60.0 / CONFIG.bci.paso_tasm)):
            msg = mock.siguiente()
            s = fsm.actualizar(msg.estado, msg.freq_idx, t, msg.valido)
            registros.append(RegistroCiclo(
                t=t,
                estado_tasm_reportado=msg.estado,
                estado_tasm_real=msg.estado_real,
                estado_fsm=s.estado,
                transiciono=s.transiciono,
                por_comando=(s.motivo == MotivoNoTransicion.TRANSICIONO),
                u=s.u, omega=s.omega,
                lambda_bci=msg.lambda_bci,
                freq_idx=msg.freq_idx,
            ))
            t += CONFIG.bci.paso_tasm

        m = calc.calcular(registros, exito=True, modo_bci=modo)
        resultados[modo] = m

        print("-" * 70)
        print(f"MODO: {modo}")
        print("-" * 70)
        print(m.resumen())
        print()

    print("=" * 70)
    print("COMPARACION")
    print("=" * 70)
    mt = resultados["tasm"]
    mb = resultados["binary"]
    print(f"  Cambios espurios con TASM     : {mt.n_fp_tr}")
    print(f"  Cambios espurios con baseline : {mb.n_fp_tr}")
    print(f"  Reduccion absoluta            : {mb.n_fp_tr - mt.n_fp_tr}")
    print()
    print("  NOTA: con un solo trial la diferencia puede ser pequena o nula.")
    print("  El experimento real promedia 5 trials por celda y 8 sujetos.")

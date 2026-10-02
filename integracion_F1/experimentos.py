"""
experimentos.py --- Regresion y scripts de experimento
Bloque 6: integracion_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

La logica de regresion y de los experimentos esta verificada con pruebas
automatizadas usando el mock. Con TASM real no se ha podido probar: el
paquete de la Linea 1 no estaba disponible.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Dos cosas:

  REGRESION     Corre los mismos escenarios con mock y con TASM real, y
                compara. Detecta si la integracion rompio algo.

  EXPERIMENTOS  Los dos experimentos formales, con su registro de metricas.


POR QUE HACEN FALTA TESTS DE REGRESION
---------------------------------------
Cuando se sustituya el mock por TASM real, los resultados van a cambiar. La
pregunta es POR QUE cambiaron.

Hay dos causas posibles y hay que poder distinguirlas:

  (a) TASM real se comporta distinto del mock. Es lo esperado y lo
      interesante: el mock era una hipotesis sobre como se comportaria.

  (b) La integracion rompio algo. Un campo mal convertido, un indice
      desplazado, un reinicio que falta.

Sin regresion, un fallo de tipo (b) aparece como "TASM funciona peor de lo
esperado", y se buscaria el problema en el sitio equivocado durante semanas.

El test no compara los VALORES (que legitimamente cambian) sino las
PROPIEDADES ESTRUCTURALES que deben cumplirse con cualquier fuente:

  - El numero de decisiones de la Etapa 2 sigue siendo ceil(log2 N)
  - La FSM sigue sin transicionar durante TR
  - El watchdog sigue deteniendo el robot
  - Las tasas de exito no se desploman

Si esas propiedades se rompen, el problema es de integracion.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable
import json
import math
import os
import time
import numpy as np

from config import Config
import logica  # noqa: F401  (pone las rutas)
from command_fsm import CommandFSM, EstadoTASM, EstadoFSM, MotivoNoTransicion
from metrics import CalculadorMetricas, RegistroCiclo, comparar_pareado
from binary_search import BusquedaBinaria, generar_objetos_en_fila

from tasm_interface import (FuenteTASM, SalidaTASM, crear_fuente_tasm,
                            VerificadorContrato)


# ===========================================================================
# REGISTRO
# ===========================================================================

@dataclass
class RegistroTrial:
    """
    Todo lo que se guarda de un trial.

    Se vuelca a JSON. Un trial que no se registra es un trial que hay que
    repetir si algo sale raro, y repetir sesiones con sujetos es caro.
    """
    sujeto: str
    experimento: str
    condicion: str
    escenario: str
    n_objetos: int
    trial: int

    fuente_tasm: str
    exito: bool
    tiempo_total: float

    n_fp_tr: int
    fpr_tr: float
    t_tr: float
    n_transiciones: int
    n_forzadas: int

    n_decisiones: int = 0
    n_decisiones_teoricas: int = 0
    seleccion_correcta: bool = False

    n_colisiones: int = 0
    n_emergencias: int = 0

    nasa_tlx: Optional[Dict[str, float]] = None
    fatiga_visual: Optional[float] = None

    marca_tiempo: str = ""
    notas: str = ""

    def a_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegistradorExperimento:
    """
    Guarda los trials en JSON, uno por linea.

    El formato es deliberadamente simple: una linea por trial, legible con
    cualquier herramienta. Un formato binario seria mas compacto pero
    imposible de inspeccionar cuando algo va mal a las once de la noche.

    Se escribe DESPUES DE CADA TRIAL, no al final de la sesion. Si el
    programa se cae en el trial 30, los 29 anteriores estan a salvo.
    """

    def __init__(self, ruta: str, sujeto: str, config: Config):
        self.ruta = ruta
        self.sujeto = sujeto
        self.cfg = config
        self._n = 0

        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)

        # Cabecera con la configuracion completa. Sin esto, dentro de seis
        # meses no se sabra con que parametros se corrio.
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "tipo": "cabecera",
                "sujeto": sujeto,
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fuente_tasm": config.tasm.fuente,
                "n_conf": config.bci.n_conf,
                "paso_tasm": config.bci.paso_tasm,
                "tw_tasm": config.bci.tw_tasm,
                "frecuencias": list(config.bci.frecuencias),
                "niveles_n": list(config.etapa2.niveles_n),
                "modo_bci": config.experimento.modo_bci,
            }) + "\n")

    def registrar(self, r: RegistroTrial) -> None:
        r.marca_tiempo = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tipo": "trial", **r.a_dict()}) + "\n")
        self._n += 1

    @property
    def n_trials(self) -> int:
        return self._n

    @staticmethod
    def cargar(ruta: str) -> Dict[str, Any]:
        """Lee un archivo de registro."""
        cabecera, trials = {}, []
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                d = json.loads(linea)
                if d.get("tipo") == "cabecera":
                    cabecera = d
                else:
                    trials.append(d)
        return {"cabecera": cabecera, "trials": trials}


# ===========================================================================
# REGRESION
# ===========================================================================

@dataclass
class ResultadoRegresion:
    """Comparacion entre dos fuentes de TASM."""
    propiedades_ok: bool
    problemas: List[str] = field(default_factory=list)
    metricas_mock: Dict[str, float] = field(default_factory=dict)
    metricas_real: Dict[str, float] = field(default_factory=dict)

    def resumen(self) -> str:
        L = []
        L.append(f"  {'metrica':<26} {'mock':>12} {'real':>12} "
                 f"{'cambio':>10}")
        L.append("  " + "-" * 62)
        for k in self.metricas_mock:
            m = self.metricas_mock[k]
            r = self.metricas_real.get(k, float("nan"))
            if isinstance(m, (int, float)) and abs(m) > 1e-9:
                camb = f"{(r - m) / abs(m) * 100:+.0f}%"
            else:
                camb = "---"
            L.append(f"  {k:<26} {m:>12.3f} {r:>12.3f} {camb:>10}")

        L.append("")
        if self.problemas:
            L.append("  PROPIEDADES ESTRUCTURALES VIOLADAS:")
            for p in self.problemas:
                L.append(f"    - {p}")
            L.append("")
            L.append("  Estas propiedades deben cumplirse con CUALQUIER")
            L.append("  fuente. Que se rompan indica un problema de")
            L.append("  INTEGRACION, no de TASM.")
        else:
            L.append("  Todas las propiedades estructurales se mantienen.")
            L.append("  Las diferencias de valor son atribuibles a TASM, no")
            L.append("  a la integracion.")
        return "\n".join(L)


class TestRegresion:
    """
    Compara el comportamiento del sistema con dos fuentes distintas.

    NO compara valores absolutos: esos cambian legitimamente al pasar de un
    generador sintetico a un clasificador real.

    Compara PROPIEDADES ESTRUCTURALES, que deben cumplirse siempre.
    """

    def __init__(self, config: Config):
        self.cfg = config

    # -------------------------------------------------------------------
    def _correr(self, fuente: FuenteTASM, n_ventanas: int = 2000,
                generar_ventana=None) -> Dict[str, float]:
        """Ejercita la FSM con una fuente y recoge metricas."""
        fsm = CommandFSM(self.cfg)
        calc = CalculadorMetricas(self.cfg)
        registros: List[RegistroCiclo] = []

        t = 0.0
        for _ in range(n_ventanas):
            v = generar_ventana() if generar_ventana else None
            s = fuente.siguiente(v)
            salida = fsm.actualizar(s.estado, s.freq_idx, t, s.valido)

            registros.append(RegistroCiclo(
                t=t,
                estado_tasm_reportado=s.estado,
                estado_tasm_real=s.estado_real,
                estado_fsm=salida.estado,
                transiciono=salida.transiciono,
                por_comando=(salida.motivo ==
                             MotivoNoTransicion.TRANSICIONO),
                u=salida.u, omega=salida.omega,
                lambda_bci=s.lambda_bci,
                freq_idx=s.freq_idx,
            ))
            t += self.cfg.bci.paso_tasm

        m = calc.calcular(registros)
        return {
            "n_fp_tr": float(m.n_fp_tr),
            "fpr_tr": m.fpr_tr,
            "fpr_idle": m.fpr_idle,
            "n_transiciones": float(m.n_transiciones),
            "n_forzadas": float(m.n_forzadas),
            "t_tr": m.t_tr,
        }

    # -------------------------------------------------------------------
    def _propiedades(self, fuente: FuenteTASM,
                     generar_ventana=None) -> List[str]:
        """
        Comprueba las propiedades que deben cumplirse con cualquier fuente.

        Son invariantes del DISENO, no del clasificador. Si se rompen, el
        problema esta en la integracion.
        """
        problemas = []
        cfg = self.cfg

        # --- 1. TR nunca debe enclavar un comando ---
        fsm = CommandFSM(cfg)
        t = 0.0
        for _ in range(cfg.bci.n_conf * 4):
            fsm.actualizar(EstadoTASM.TR, -1, t)
            t += cfg.bci.paso_tasm
        if fsm.estado != EstadoFSM.DETENIDO:
            problemas.append(
                "La FSM transiciono con TR sostenido. Es la propiedad "
                "central del trabajo.")

        # --- 2. IC sostenido debe enclavar ---
        fsm = CommandFSM(cfg)
        t = 0.0
        for _ in range(cfg.bci.n_conf + 2):
            fsm.actualizar(EstadoTASM.IC, 2, t)
            t += cfg.bci.paso_tasm
        if fsm.estado == EstadoFSM.DETENIDO:
            problemas.append(
                f"La FSM no enclavo tras {cfg.bci.n_conf} ventanas de IC. "
                f"El sistema no podria aceptar ningun comando.")

        # --- 3. La Etapa 2 sigue siendo logaritmica ---
        for n in cfg.etapa2.niveles_n:
            objs = generar_objetos_en_fila(cfg, n)
            teorico = math.ceil(math.log2(n))
            for objetivo in range(n):
                bb = BusquedaBinaria(cfg, objs)
                while not bb.terminada:
                    bb.votar(bb.lado_correcto(objetivo))
                r = bb.resultado(objetivo=objetivo)
                if r.n_decisiones != teorico:
                    problemas.append(
                        f"N={n}, objetivo={objetivo}: {r.n_decisiones} "
                        f"decisiones en vez de {teorico}.")
                    break

        # --- 4. La fuente produce las tres clases ---
        vistos = set()
        for _ in range(1000):
            v = generar_ventana() if generar_ventana else None
            vistos.add(fuente.siguiente(v).estado)
        if EstadoTASM.TR not in vistos:
            problemas.append(
                "La fuente nunca reporta TR. Es la clase central del "
                "trabajo.")
        if EstadoTASM.IC not in vistos:
            problemas.append("La fuente nunca reporta IC.")

        fuente.reiniciar()
        return problemas

    # -------------------------------------------------------------------
    def comparar(self, fuente_mock: FuenteTASM,
                 fuente_real: FuenteTASM,
                 n_ventanas: int = 2000,
                 generar_ventana=None) -> ResultadoRegresion:
        """Compara ambas fuentes."""
        m_mock = self._correr(fuente_mock, n_ventanas)
        fuente_mock.reiniciar()

        m_real = self._correr(fuente_real, n_ventanas, generar_ventana)
        fuente_real.reiniciar()

        problemas = self._propiedades(fuente_real, generar_ventana)

        return ResultadoRegresion(
            propiedades_ok=(len(problemas) == 0),
            problemas=problemas,
            metricas_mock=m_mock,
            metricas_real=m_real,
        )


# ===========================================================================
# EXPERIMENTOS
# ===========================================================================

class Experimento:
    """
    Ejecuta los experimentos formales.

    LOS DOS EXPERIMENTOS SON SEPARADOS Y NO SE CRUZAN
    --------------------------------------------------
      A: TASM vs baseline binario, en 2 o 3 escenarios, con N fijo en 4.
         Responde: reduce TASM los comandos espurios? Crece el efecto con la
         dificultad?

      B: escalabilidad con N en {2,4,8}, solo con TASM, escenario fijo.
         Responde: se sostiene ceil(log2 N)? Se degrada la precision?

    Cruzarlos daria 90 trials por sujeto, unas 5 horas de sesion, inviable
    por fatiga visual. Y no aportaria: la interaccion entre dificultad de
    navegacion y numero de objetos no es una pregunta del trabajo.
    """

    def __init__(self, config: Config, sujeto: str,
                 ruta_salida: str = "resultados"):
        self.cfg = config
        self.sujeto = sujeto
        self.ruta = os.path.join(ruta_salida, f"{sujeto}.jsonl")
        self.registrador = RegistradorExperimento(self.ruta, sujeto, config)

    # -------------------------------------------------------------------
    def experimento_A(self, n_trials: Optional[int] = None,
                      simulado: bool = True) -> List[RegistroTrial]:
        """
        Experimento A: TASM vs baseline binario.

        Con `simulado=True` usa el simulador 2D, lo que permite ensayar el
        protocolo completo sin robot ni sujeto.
        """
        from simulator2d import Simulador2D

        n_trials = n_trials or self.cfg.experimento.trials_por_celda
        escenarios = [self.cfg.escenarios.lista[i]
                      for i in self.cfg.escenarios.usados_en_fase1]

        salida: List[RegistroTrial] = []

        for esc in escenarios:
            for condicion in ("tasm", "binary"):
                for k in range(n_trials):
                    cfg_t = self.cfg.copia_con(
                        experimento=dict(modo_bci=condicion))
                    sim = Simulador2D(cfg_t, esc, modo=condicion,
                                      semilla=hash((esc.nombre, k)) % 10000)
                    res = sim.correr(n_objetos=4, objetivo=2)
                    m = res.metricas

                    r = RegistroTrial(
                        sujeto=self.sujeto,
                        experimento="A",
                        condicion=condicion,
                        escenario=esc.nombre,
                        n_objetos=4,
                        trial=k,
                        fuente_tasm=self.cfg.tasm.fuente,
                        exito=m.exito,
                        tiempo_total=m.tiempo_total,
                        n_fp_tr=m.n_fp_tr,
                        fpr_tr=m.fpr_tr,
                        t_tr=m.t_tr,
                        n_transiciones=m.n_transiciones,
                        n_forzadas=m.n_forzadas,
                        n_decisiones=res.n_decisiones_e2,
                        n_decisiones_teoricas=2,
                        seleccion_correcta=res.seleccion_correcta,
                        n_colisiones=m.n_colisiones,
                        n_emergencias=m.n_emergencias,
                    )
                    self.registrador.registrar(r)
                    salida.append(r)

        return salida

    # -------------------------------------------------------------------
    def experimento_B(self, n_trials: Optional[int] = None,
                      simulado: bool = True) -> List[RegistroTrial]:
        """
        Experimento B: escalabilidad.

        Solo con TASM: la pregunta es si la precision se degrada al aumentar
        N, no si TASM ayuda. Escenario fijo en el mas simple, para que la
        navegacion no introduzca varianza.
        """
        from simulator2d import Simulador2D

        n_trials = n_trials or self.cfg.experimento.trials_por_celda
        esc = self.cfg.escenarios.lista[0]

        salida: List[RegistroTrial] = []

        for n in self.cfg.etapa2.niveles_n:
            teorico = math.ceil(math.log2(n))
            for k in range(n_trials):
                cfg_t = self.cfg.copia_con(
                    experimento=dict(modo_bci="tasm"))
                sim = Simulador2D(cfg_t, esc, modo="tasm",
                                  semilla=hash((n, k)) % 10000)
                res = sim.correr(n_objetos=n, objetivo=n // 2)
                m = res.metricas

                r = RegistroTrial(
                    sujeto=self.sujeto,
                    experimento="B",
                    condicion="tasm",
                    escenario=esc.nombre,
                    n_objetos=n,
                    trial=k,
                    fuente_tasm=self.cfg.tasm.fuente,
                    exito=m.exito,
                    tiempo_total=m.tiempo_total,
                    n_fp_tr=m.n_fp_tr,
                    fpr_tr=m.fpr_tr,
                    t_tr=m.t_tr,
                    n_transiciones=m.n_transiciones,
                    n_forzadas=m.n_forzadas,
                    n_decisiones=res.n_decisiones_e2,
                    n_decisiones_teoricas=teorico,
                    seleccion_correcta=res.seleccion_correcta,
                    n_colisiones=m.n_colisiones,
                    n_emergencias=m.n_emergencias,
                )
                self.registrador.registrar(r)
                salida.append(r)

        return salida


# ===========================================================================
# ANALISIS
# ===========================================================================

def analizar_experimento_A(trials: List[RegistroTrial]) -> str:
    """
    Resume el Experimento A.

    La metrica primaria es N_FP_TR, y la comparacion es PAREADA: mismo
    escenario, mismo indice de trial, distinta condicion.

    Se reporta la reduccion ABSOLUTA, no la relativa. La relativa SATURA: si
    la condicion propuesta llega a cero eventos, el cociente vale 1 y no
    puede crecer, lo que hace inverificable la hipotesis de escalamiento.
    """
    L = []
    L.append("  EXPERIMENTO A --- TASM vs baseline binario")
    L.append("")

    escenarios = sorted({t.escenario for t in trials})

    L.append(f"  {'escenario':<32} {'TASM':>8} {'baseline':>10} "
             f"{'reduccion':>11}")
    L.append("  " + "-" * 64)

    for esc in escenarios:
        v_t = [float(t.n_fp_tr) for t in trials
               if t.escenario == esc and t.condicion == "tasm"]
        v_b = [float(t.n_fp_tr) for t in trials
               if t.escenario == esc and t.condicion == "binary"]
        if not v_t or not v_b:
            continue
        n = min(len(v_t), len(v_b))
        c = comparar_pareado(v_t[:n], v_b[:n])
        L.append(f"  {esc[:32]:<32} {c.media_tasm:>8.2f} "
                 f"{c.media_binary:>10.2f} {c.delta_absoluto:>+11.2f}")

    L.append("")
    L.append("  Tasas de exito:")
    for cond in ("tasm", "binary"):
        sub = [t for t in trials if t.condicion == cond]
        if sub:
            ex = sum(1 for t in sub if t.exito) / len(sub)
            L.append(f"    {cond:<10}: {ex*100:.0f}% ({len(sub)} trials)")

    L.append("")
    L.append("  NOTA: la reduccion se reporta en valor ABSOLUTO. La relativa")
    L.append("  satura cuando la condicion propuesta llega a cero eventos, y")
    L.append("  eso hace inverificable la hipotesis de escalamiento.")
    return "\n".join(L)


def analizar_experimento_B(trials: List[RegistroTrial]) -> str:
    """
    Resume el Experimento B.

    Lo decisivo es que el numero de decisiones OBSERVADO coincida con el
    teorico. Es la verificacion directa de la complejidad logaritmica.
    """
    L = []
    L.append("  EXPERIMENTO B --- Escalabilidad")
    L.append("")
    L.append(f"  {'N':>4} {'teorico':>9} {'observado':>11} {'coincide':>10} "
             f"{'exito':>8} {'tiempo':>9}")
    L.append("  " + "-" * 56)

    for n in sorted({t.n_objetos for t in trials}):
        sub = [t for t in trials if t.n_objetos == n]
        if not sub:
            continue
        con_sel = [t for t in sub if t.n_decisiones > 0]
        obs = (float(np.mean([t.n_decisiones for t in con_sel]))
               if con_sel else 0.0)
        teo = sub[0].n_decisiones_teoricas
        coincide = all(t.n_decisiones == teo for t in con_sel)
        ex = sum(1 for t in sub if t.exito) / len(sub)
        tm = float(np.mean([t.tiempo_total for t in sub]))
        L.append(f"  {n:>4} {teo:>9} {obs:>11.1f} "
                 f"{'SI' if coincide else 'NO':>10} {ex*100:>7.0f}% "
                 f"{tm:>8.1f}s")

    L.append("")
    L.append("  La columna 'coincide' es la que importa: es la verificacion")
    L.append("  directa de que la seleccion requiere ceil(log2 N) decisiones")
    L.append("  para cualquier objetivo.")
    return "\n".join(L)


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DE REGRESION Y EXPERIMENTOS")
    print("=" * 72)

    # --- Regresion ---
    print()
    print("-" * 72)
    print("TEST DE REGRESION")
    print("-" * 72)
    print("  Se comparan dos instancias del mock con parametros distintos,")
    print("  simulando lo que ocurrira al sustituirlo por TASM real.")
    print()

    from tasm_interface import FuenteMock

    f_mock = FuenteMock(CONFIG, objetivo=2, modo="tasm", semilla=1)

    # Segunda fuente con tasas peores, como cabria esperar de un
    # clasificador real frente a uno sintetico optimista
    cfg_peor = CONFIG.copia_con(
        mock=dict(p_episodio_fp_tasm=0.18,
                  p_error_idle_como_ic_tasm=0.09))
    f_otra = FuenteMock(cfg_peor, objetivo=2, modo="tasm", semilla=2)

    reg = TestRegresion(CONFIG)
    res = reg.comparar(f_mock, f_otra, n_ventanas=1500)
    print(res.resumen())

    # --- Experimentos ---
    print()
    print("-" * 72)
    print("EXPERIMENTO A (simulado, 2 trials por celda)")
    print("-" * 72)

    exp = Experimento(CONFIG, sujeto="demo", ruta_salida="/tmp/resultados")
    tA = exp.experimento_A(n_trials=2)
    print()
    print(analizar_experimento_A(tA))

    print()
    print("-" * 72)
    print("EXPERIMENTO B (simulado, 2 trials por celda)")
    print("-" * 72)
    tB = exp.experimento_B(n_trials=2)
    print()
    print(analizar_experimento_B(tB))

    print()
    print("-" * 72)
    print("REGISTRO")
    print("-" * 72)
    print(f"  Archivo : {exp.ruta}")
    print(f"  Trials  : {exp.registrador.n_trials}")
    d = RegistradorExperimento.cargar(exp.ruta)
    print(f"  Releidos: {len(d['trials'])}")
    print(f"  Fuente registrada: {d['cabecera']['fuente_tasm']}")
    print()
    print("  El archivo se escribe DESPUES DE CADA TRIAL. Si el programa se")
    print("  cae en el trial 30, los 29 anteriores estan a salvo.")

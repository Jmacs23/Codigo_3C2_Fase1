"""
run.py --- Script principal del Bloque 1
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

COMO USAR ESTE SCRIPT
---------------------
Es el punto de entrada unico del bloque. Cada subcomando corresponde a una
compuerta de verificacion: no pases a la siguiente hasta que la anterior
este en verde.

    python run.py config      Vuelca y valida la configuracion
    python run.py smoke       Prueba rapida de todos los modulos
    python run.py fsm         Compuerta 1: la FSM suprime durante TR
    python run.py fbcca       Compuerta 2: el clasificador recupera la frec.
    python run.py seleccion   Compuerta 3: log2(N) decisiones exactas
    python run.py trial       Un trial completo en el simulador
    python run.py viabilidad  HITO DE VIABILIDAD: TASM vs baseline
    python run.py barrido     Barrido de n_conf
    python run.py test        Bateria de tests

EL HITO DE VIABILIDAD
---------------------
`python run.py viabilidad` es el comando mas importante del bloque. Compara
la condicion TASM contra el baseline binario en el simulador, y responde la
pregunta que decide si el experimento con sujetos tiene sentido:

    ?La supresion de transiciones produce una reduccion medible de comandos
     espurios, o el mecanismo de confirmacion ya resuelve el problema por si
     solo?

Si la respuesta fuera que no hay diferencia, habria que revisar el diseno
ANTES de invertir meses en ROS2, hardware y sujetos. Ese es todo el proposito
de correrlo temprano.
"""

import sys
import math
import time
from typing import List, Optional

from config import CONFIG, Config


# ===========================================================================
# UTILIDADES DE PRESENTACION
# ===========================================================================

def titulo(txt: str) -> None:
    print()
    print("=" * 72)
    print(txt)
    print("=" * 72)


def seccion(txt: str) -> None:
    print()
    print("-" * 72)
    print(txt)
    print("-" * 72)


def veredicto(ok: bool, msg: str) -> bool:
    marca = "[ OK ]" if ok else "[FALLO]"
    print(f"  {marca} {msg}")
    return ok


# ===========================================================================
# COMANDOS
# ===========================================================================

def cmd_config() -> int:
    """Vuelca la configuracion y ejecuta la validacion."""
    titulo("CONFIGURACION")
    print(CONFIG.resumen())

    problemas = CONFIG.validar()
    seccion("VALIDACION")
    if problemas:
        for p in problemas:
            print(f"  - {p}")
        print()
        print("  Nota: los avisos de diseno no impiden ejecutar, pero")
        print("  conviene entender por que aparecen.")
    else:
        print("  Sin problemas.")
    return 0


def cmd_smoke() -> int:
    """Prueba rapida: importa y ejercita todos los modulos."""
    titulo("PRUEBA DE HUMO --- todos los modulos")

    ok = True

    seccion("1. Configuracion")
    ok &= veredicto(CONFIG.bci.fs > 0, f"fs = {CONFIG.bci.fs} Hz")
    ok &= veredicto(len(CONFIG.bci.frecuencias) == 4,
                    f"frecuencias = {CONFIG.bci.frecuencias}")
    ok &= veredicto(CONFIG.robot.diagonal < min(
        e.gap for e in CONFIG.escenarios.lista),
        f"el robot cabe girando en todos los escenarios "
        f"(diagonal {CONFIG.robot.diagonal:.3f} m)")

    seccion("2. FBCCA")
    try:
        from fbcca import FBCCA, generar_ssvep
        clf = FBCCA(CONFIG)
        x = generar_ssvep(CONFIG, 1, CONFIG.bci.tw_tasm, snr_db=0.0, semilla=1)
        r = clf.clasificar(x)
        ok &= veredicto(r.freq_idx == 1,
                        f"recupera la frecuencia inyectada "
                        f"({r.freq_hz} Hz, p_max={r.p_max:.3f})")
    except Exception as e:
        ok &= veredicto(False, f"excepcion: {e}")

    seccion("3. Maquina de estados")
    try:
        from command_fsm import CommandFSM, EstadoTASM, EstadoFSM
        fsm = CommandFSM(CONFIG)
        t = 0.0
        for _ in range(CONFIG.bci.n_conf + 1):
            s = fsm.actualizar(EstadoTASM.IC, 2, t)
            t += CONFIG.bci.paso_tasm
        ok &= veredicto(fsm.estado == EstadoFSM.AVANZANDO,
                        f"enclava tras {CONFIG.bci.n_conf} ventanas IC")
    except Exception as e:
        ok &= veredicto(False, f"excepcion: {e}")

    seccion("4. Mock de TASM")
    try:
        from tasm_mock import TASMMock
        mock = TASMMock(CONFIG, objetivo=2, semilla=1)
        for _ in range(200):
            mock.siguiente()
        st = mock.estadisticas()
        ok &= veredicto(st["ventanas_TR"] > 0,
                        f"genera los tres estados "
                        f"(IC={st['ventanas_IC']}, TR={st['ventanas_TR']}, "
                        f"Idle={st['ventanas_Idle']})")
    except Exception as e:
        ok &= veredicto(False, f"excepcion: {e}")

    seccion("5. Seleccion binaria")
    try:
        from binary_search import BusquedaBinaria, generar_objetos_en_fila
        objs = generar_objetos_en_fila(CONFIG, 8)
        bb = BusquedaBinaria(CONFIG, objs)
        while not bb.terminada:
            bb.votar(bb.lado_correcto(5))
        r = bb.resultado(objetivo=5)
        ok &= veredicto(r.acierto and r.coincide_teoria,
                        f"selecciona con {r.n_decisiones} decisiones "
                        f"(teoricas {r.n_decisiones_teoricas})")
    except Exception as e:
        ok &= veredicto(False, f"excepcion: {e}")

    seccion("6. Simulador")
    try:
        from simulator2d import Simulador2D
        sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0], semilla=1)
        res = sim.correr(n_objetos=4, objetivo=2)
        ok &= veredicto(res.llego_a_zona,
                        f"completa un trial "
                        f"({res.metricas.tiempo_total:.1f} s, "
                        f"{res.metricas.distancia_recorrida:.2f} m)")
    except Exception as e:
        ok &= veredicto(False, f"excepcion: {e}")

    seccion("RESULTADO")
    print(f"  {'TODO CORRECTO' if ok else 'HAY FALLOS'}")
    return 0 if ok else 1


def cmd_fsm() -> int:
    """Compuerta 1: la FSM no transiciona durante transiciones de mirada."""
    from command_fsm import CommandFSM, EstadoTASM, EstadoFSM

    titulo("COMPUERTA 1 --- Supresion durante transiciones de mirada")
    print()
    print("  Esta es la propiedad central del sistema: un comando detectado")
    print("  durante una transicion de mirada NO debe enclavarse.")
    print()

    cfg = CONFIG
    paso = cfg.bci.paso_tasm
    ok = True

    # --- Caso 1: fijacion sostenida SI enclava ---
    seccion("Caso 1: fijacion sostenida en 'avanzar'")
    fsm = CommandFSM(cfg)
    t = 0.0
    ventana_transicion = None
    for i in range(cfg.bci.n_conf + 3):
        s = fsm.actualizar(EstadoTASM.IC, 2, t)
        if s.transiciono and ventana_transicion is None:
            ventana_transicion = i + 1
        t += paso
    ok &= veredicto(fsm.estado == EstadoFSM.AVANZANDO,
                    f"enclava AVANZANDO en la ventana {ventana_transicion}")
    ok &= veredicto(ventana_transicion == cfg.bci.n_conf,
                    f"exactamente tras {cfg.bci.n_conf} ventanas "
                    f"({cfg.bci.t_confirmacion*1000:.0f} ms)")

    # --- Caso 2: transicion con espurios NO enclava ---
    seccion("Caso 2: transicion de mirada con comandos espurios")
    dur_tr = cfg.mock.dur_tr_max
    n_vent_tr = int(dur_tr / paso)
    print(f"  Duracion de TR simulada : {dur_tr*1000:.0f} ms "
          f"({n_vent_tr} ventanas)")
    print(f"  Racha necesaria         : {cfg.bci.n_conf} ventanas")
    print()

    estado_previo = fsm.estado
    espurios = 0
    for i in range(n_vent_tr):
        # Peor caso: TODAS las ventanas de la TR reportan IC espurio
        s = fsm.actualizar(EstadoTASM.IC, 0, t)
        espurios += 1
        t += paso

    cambio = (fsm.estado != estado_previo)
    ok &= veredicto(
        not cambio or n_vent_tr >= cfg.bci.n_conf,
        f"con {espurios} ventanas espurias consecutivas: "
        f"{'no cambio' if not cambio else 'CAMBIO'}")

    if n_vent_tr >= cfg.bci.n_conf:
        print()
        print(f"  ATENCION: una TR de duracion maxima ({n_vent_tr} ventanas)")
        print(f"  SUPERA la racha de confirmacion ({cfg.bci.n_conf}).")
        print(f"  Es el escenario en que TASM aporta: sin el, esas ventanas")
        print(f"  espurias enclavarian un comando.")

    # --- Caso 3: en movimiento solo se acepta parar ---
    seccion("Caso 3: transicion obligatoria por DETENIDO")
    fsm2 = CommandFSM(cfg)
    t = 0.0
    for _ in range(cfg.bci.n_conf):
        fsm2.actualizar(EstadoTASM.IC, 2, t); t += paso
    estado_avanzando = fsm2.estado
    for _ in range(cfg.bci.n_conf * 2):
        fsm2.actualizar(EstadoTASM.IC, 0, t); t += paso   # pide girar
    ok &= veredicto(fsm2.estado == estado_avanzando,
                    "en movimiento, un comando de giro se ignora")
    for _ in range(cfg.bci.n_conf):
        fsm2.actualizar(EstadoTASM.IC, 3, t); t += paso   # pide parar
    ok &= veredicto(fsm2.estado == EstadoFSM.DETENIDO,
                    "el comando de parar si se acepta")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    return 0 if ok else 1


def cmd_fbcca() -> int:
    """Compuerta 2: el clasificador recupera la frecuencia inyectada."""
    from fbcca import FBCCA, generar_ssvep

    titulo("COMPUERTA 2 --- Clasificador FBCCA")

    clf = FBCCA(CONFIG)
    b = CONFIG.bci
    n_rep = 10

    seccion(f"Precision por frecuencia ({n_rep} repeticiones, SNR 0 dB)")
    print(f"  {'frecuencia':>11} {'aciertos':>10} {'p_max medio':>13}")
    total_ok = 0
    total = 0
    for k, f in enumerate(b.frecuencias):
        ac = 0
        pm = 0.0
        for rep in range(n_rep):
            x = generar_ssvep(CONFIG, k, b.tw_tasm, snr_db=0.0,
                              semilla=k * 100 + rep)
            r = clf.clasificar(x)
            ac += (r.freq_idx == k)
            pm += r.p_max
        total_ok += ac
        total += n_rep
        print(f"  {f:>10.1f}Hz {ac:>6}/{n_rep:<3} {pm/n_rep:>13.3f}")

    acc = total_ok / total
    print()
    ok = veredicto(acc >= 0.80,
                   f"precision global {acc*100:.1f}% (criterio: >= 80%)")

    seccion("Efecto de la SNR")
    print(f"  {'SNR':>7} {'precision':>11}")
    for snr in (5.0, 0.0, -5.0, -10.0):
        ac = 0
        for k in range(len(b.frecuencias)):
            for rep in range(n_rep):
                x = generar_ssvep(CONFIG, k, b.tw_tasm, snr_db=snr,
                                  semilla=k * 50 + rep)
                ac += (clf.clasificar(x).freq_idx == k)
        print(f"  {snr:>5.0f}dB {ac/(len(b.frecuencias)*n_rep)*100:>10.1f}%")

    print()
    print("  Nota: la senal sintetica no reproduce la estructura espacial")
    print("  del EEG real. Estos numeros verifican que el pipeline funciona,")
    print("  no predicen el rendimiento con un sujeto.")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    return 0 if ok else 1


def cmd_seleccion() -> int:
    """Compuerta 3: la seleccion requiere exactamente log2(N) decisiones."""
    from binary_search import BusquedaBinaria, generar_objetos_en_fila

    titulo("COMPUERTA 3 --- Complejidad logaritmica de la seleccion")
    print()
    print("  Verifica la afirmacion central de la Etapa 2: con dos")
    print("  frecuencias, seleccionar entre N objetos requiere exactamente")
    print("  ceil(log2 N) decisiones, para CUALQUIER objetivo.")

    ok = True
    e2 = CONFIG.etapa2

    seccion("Decisiones por objetivo")
    print(f"  {'N':>4} {'teorico':>9} {'observado':>28} {'uniforme':>10}")
    for n in e2.niveles_n:
        objs = generar_objetos_en_fila(CONFIG, n)
        conteos = []
        aciertos = 0
        for obj in range(n):
            bb = BusquedaBinaria(CONFIG, objs)
            while not bb.terminada:
                bb.votar(bb.lado_correcto(obj))
            r = bb.resultado(objetivo=obj)
            conteos.append(r.n_decisiones)
            aciertos += r.acierto

        teorico = math.ceil(math.log2(n))
        uniforme = len(set(conteos)) == 1
        coincide = all(c == teorico for c in conteos)
        txt = str(conteos) if n <= 8 else f"{conteos[:4]}..."
        print(f"  {n:>4} {teorico:>9} {txt:>28} "
              f"{'SI' if uniforme else 'NO':>10}")
        ok &= (coincide and aciertos == n)

    print()
    veredicto(ok, "todas las configuraciones cumplen ceil(log2 N)")

    seccion("Viabilidad fisica de cada nivel")
    print(f"  {'N':>4} {'D_min':>8} {'px/obj':>9} {'cabe':>7} {'detectable':>12}")
    for n in e2.niveles_n:
        d = e2.distancia_minima(n)
        px = e2.pixeles_por_objeto(n)
        cabe = d <= e2.umbral_transicion
        det = px >= e2.px_minimo_deteccion
        ok &= (cabe and det)
        print(f"  {n:>4} {d:>7.2f}m {px:>8.0f} "
              f"{'SI' if cabe else 'NO':>7} {'SI' if det else 'NO':>12}")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    return 0 if ok else 1


def cmd_trial(escenario_idx: int = 0) -> int:
    """Ejecuta un trial completo y vuelca las metricas."""
    from simulator2d import Simulador2D

    esc = CONFIG.escenarios.lista[escenario_idx]
    titulo(f"TRIAL COMPLETO --- {esc.nombre}")

    print(f"  Sala         : {esc.sala[0]} x {esc.sala[1]} m")
    print(f"  Obstaculos   : {esc.n_obstaculos}")
    print(f"  Gap          : {esc.gap} m")
    print(f"  Margen libre : {esc.margen_libre(CONFIG.robot.ancho):.3f} m")

    for modo in ("binary", "tasm"):
        seccion(f"MODO: {modo}")
        sim = Simulador2D(CONFIG, esc, modo=modo, semilla=5)
        res = sim.correr(n_objetos=4, objetivo=2)
        print(res.metricas.resumen())
        print()
        print(f"  Llego a zona E2     : "
              f"{'SI' if res.llego_a_zona else 'NO'}")
        print(f"  Seleccion correcta  : "
              f"{'SI' if res.seleccion_correcta else 'NO'}")
        print(f"  Decisiones Etapa 2  : {res.n_decisiones_e2}")

    return 0


def cmd_viabilidad(n_semillas: int = 12) -> int:
    """
    HITO DE VIABILIDAD.

    Compara TASM contra el baseline binario. Es la verificacion que decide si
    tiene sentido invertir en el experimento con sujetos.
    """
    from simulator2d import Simulador2D
    from metrics import comparar_pareado

    titulo("HITO DE VIABILIDAD TEMPRANA")
    print()
    print("  Pregunta que responde:")
    print("    ?La supresion de transiciones produce una reduccion medible")
    print("     de comandos espurios, o el mecanismo de confirmacion ya")
    print("     resuelve el problema por si solo?")
    print()
    print(f"  Configuracion: n_conf = {CONFIG.bci.n_conf} ventanas "
          f"({CONFIG.bci.t_confirmacion*1000:.0f} ms)")
    print(f"  Semillas por condicion: {n_semillas}")

    t0 = time.time()

    for esc in CONFIG.escenarios.lista[:2]:
        seccion(esc.nombre)

        v_tasm: List[float] = []
        v_bin: List[float] = []
        exitos = {"tasm": 0, "binary": 0}
        fpr = {"tasm": 0.0, "binary": 0.0}

        for s in range(n_semillas):
            for modo, lista in (("tasm", v_tasm), ("binary", v_bin)):
                sim = Simulador2D(CONFIG, esc, modo=modo, semilla=s)
                r = sim.correr(n_objetos=4, objetivo=2)
                lista.append(float(r.metricas.n_fp_tr))
                exitos[modo] += int(r.metricas.exito)
                fpr[modo] += r.metricas.fpr_tr

        comp = comparar_pareado(v_tasm, v_bin)
        print(f"  {'':22} {'TASM':>10} {'baseline':>10}")
        print(f"  {'Comandos espurios':22} {sum(v_tasm):>10.0f} "
              f"{sum(v_bin):>10.0f}")
        print(f"  {'FPR en TR':22} {fpr['tasm']/n_semillas*100:>9.1f}% "
              f"{fpr['binary']/n_semillas*100:>9.1f}%")
        print(f"  {'Trials con exito':22} {exitos['tasm']:>10} "
              f"{exitos['binary']:>10}")
        print()
        print(comp.resumen())

    seccion("INTERPRETACION")
    print("  Criterio de la Bitacora TASM: reduccion de FPR > 30% frente al")
    print("  baseline binario para considerar que hay contribucion.")
    print()
    print("  Si la reduccion fuera nula o negativa, revisar antes de seguir:")
    print("    1. ?n_conf es demasiado grande? Prueba 'python run.py barrido'")
    print("    2. ?Los escenarios inducen suficientes transiciones?")
    print("    3. ?El override de emergencia esta absorbiendo el efecto?")
    print()
    print(f"  Tiempo de computo: {time.time()-t0:.1f} s")
    return 0


def cmd_barrido() -> int:
    """Barrido del parametro n_conf."""
    from simulator2d import Simulador2D

    titulo("BARRIDO DE n_conf")
    print()
    print("  n_conf es el numero de ventanas consecutivas en IC necesarias")
    print("  para enclavar un comando. Es el parametro de compromiso central:")
    print()
    print("    Muy pequeno -> entran comandos espurios")
    print("    Muy grande  -> ninguna transicion alcanza a completarlo, y el")
    print("                   baseline tampoco falla: no hay nada que medir")
    print()
    print("  Su valor definitivo debe fijarse con los datos de la Linea 1.")

    esc = CONFIG.escenarios.lista[0]
    n_sem = 20

    seccion(f"{esc.nombre}, {n_sem} semillas por condicion")
    print(f"  {'n_conf':>7} {'ms':>6} | {'TASM':>7} {'baseline':>9} "
          f"{'reduccion':>10} {'%':>7}")
    print("  " + "-" * 58)

    for nc in (4, 6, 8, 10, 12, 16):
        cfg = CONFIG.copia_con(bci=dict(n_conf=nc))
        tot_t = tot_b = 0
        for s in range(n_sem):
            for modo in ("tasm", "binary"):
                sim = Simulador2D(cfg, esc, modo=modo, semilla=s)
                r = sim.correr(n_objetos=4, objetivo=2)
                if modo == "tasm":
                    tot_t += r.metricas.n_fp_tr
                else:
                    tot_b += r.metricas.n_fp_tr
        red = tot_b - tot_t
        pct = red / tot_b * 100 if tot_b else 0.0
        marca = "  <-- actual" if nc == CONFIG.bci.n_conf else ""
        print(f"  {nc:>7} {nc*CONFIG.bci.paso_tasm*1000:>6.0f} | "
              f"{tot_t:>7} {tot_b:>9} {red:>10} {pct:>6.0f}%{marca}")

    print()
    print("  Criterio de seleccion: el valor debe dar reduccion clara SIN")
    print("  que los numeros absolutos sean tan pequenos que se pierda")
    print("  potencia estadistica en el experimento con sujetos.")
    return 0


def cmd_test() -> int:
    """Ejecuta la bateria de tests."""
    import subprocess
    titulo("BATERIA DE TESTS")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v",
                        "--tb=short", "-W", "ignore"],
                       capture_output=False)
    return r.returncode


# ===========================================================================
# PUNTO DE ENTRADA
# ===========================================================================

COMANDOS = {
    "config": cmd_config,
    "smoke": cmd_smoke,
    "fsm": cmd_fsm,
    "fbcca": cmd_fbcca,
    "seleccion": cmd_seleccion,
    "trial": cmd_trial,
    "viabilidad": cmd_viabilidad,
    "barrido": cmd_barrido,
    "test": cmd_test,
}


def ayuda() -> None:
    print(__doc__)
    print("Comandos disponibles:")
    for k in COMANDOS:
        print(f"  {k}")


def main() -> int:
    if len(sys.argv) < 2:
        ayuda()
        return 1

    cmd = sys.argv[1]
    if cmd not in COMANDOS:
        print(f"Comando desconocido: '{cmd}'")
        print()
        ayuda()
        return 1

    try:
        return COMANDOS[cmd]()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

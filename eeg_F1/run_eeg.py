"""
run_eeg.py --- Script principal del Bloque 3
Bloque 3: eeg_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado con el amplificador real.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

COMANDOS
--------
    python run_eeg.py config       Configuracion de adquisicion
    python run_eeg.py filtros      Respuesta de la cadena de filtrado
    python run_eeg.py causal       Por que filtfilt no sirve online
    python run_eeg.py fuente       Prueba de la fuente de senal
    python run_eeg.py buffer       Como funciona el ventaneo solapado
    python run_eeg.py pipeline     COMPUERTA: cadena completa
    python run_eeg.py latencia     Verificacion del presupuesto temporal
    python run_eeg.py impedancias  Comprobacion previa a una sesion
    python run_eeg.py test         Bateria de tests

ORDEN RECOMENDADO
-----------------
  1. config     Ver que parametros hay
  2. filtros    Comprobar que la cadena deja pasar los armonicos utiles
  3. causal     Entender por que el filtrado es causal con estado
  4. pipeline   COMPUERTA del bloque
  5. latencia   Confirmar que el computo cabe en el periodo

LA COMPUERTA DEL BLOQUE
-----------------------
`python run_eeg.py pipeline` verifica la cadena completa sobre senal
sintetica con respuesta conocida. Debe recuperar la frecuencia inyectada con
precision alta y con el computo dentro del periodo de ventana.

En el laboratorio hay que ejecutar ademas la comprobacion de impedancias
antes de cada sesion, y verificar con un sujeto real que los picos
espectrales aparecen (protocolo del Bloque 2).
"""

import sys
import time
import numpy as np

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
    titulo("CONFIGURACION DE ADQUISICION")
    b, a, l = CONFIG.bci, CONFIG.adquisicion, CONFIG.latencia

    print()
    print("  Senal")
    print(f"    Muestreo        : {b.fs} Hz")
    print(f"    Canales         : {b.n_canales}")
    print(f"    Orden           : {', '.join(b.canales)}")
    print()
    print("  Filtrado")
    print(f"    Notch           : {b.notch_hz} Hz, Q={a.q_notch}")
    semi = b.notch_hz / a.q_notch / 2
    print(f"    Banda suprimida : {b.notch_hz-semi:.2f} - "
          f"{b.notch_hz+semi:.2f} Hz")
    print(f"    Pasabanda       : {b.banda_hz[0]} - {b.banda_hz[1]} Hz, "
          f"orden {a.orden_pasabanda}")
    print(f"    Tipo            : "
          f"{'causal con estado' if a.usar_filtro_causal else 'NO CAUSAL'}")
    print()
    print("  Ventaneo")
    print(f"    Bloque          : {a.buffer_muestras} muestras "
          f"({a.buffer_muestras/b.fs*1000:.0f} ms)")
    print(f"    Ventana         : {b.muestras_ventana_tasm} muestras "
          f"({b.tw_tasm*1000:.0f} ms)")
    print(f"    Paso            : {b.muestras_paso_tasm} muestras "
          f"({b.paso_tasm*1000:.0f} ms)")
    solape = (1 - b.muestras_paso_tasm / b.muestras_ventana_tasm) * 100
    print(f"    Solapamiento    : {solape:.0f}%")
    print()
    print("  Artefactos")
    print(f"    Amplitud max    : {a.umbral_amplitud:.0f} uV")
    print(f"    Gradiente max   : "
          f"{a.umbral_gradiente(b.banda_hz[1], b.fs):.0f} uV "
          f"(derivado de la banda)")
    print()
    print("  Latencia")
    print(f"    Computo         : {l.procesamiento_ms:.0f} ms "
          f"(periodo {b.paso_tasm*1000:.0f} ms)")
    print(f"    Extremo a extremo: {l.extremo_a_extremo_ms:.0f} ms "
          f"(watchdog {b.watchdog_s*1000:.0f} ms)")

    probs = [p for p in CONFIG.validar()
             if any(k in p.lower() for k in
                    ("notch", "armonico", "buffer", "latencia", "filtro",
                     "fuente", "watchdog", "computo"))]
    seccion("VALIDACION")
    if probs:
        for p in probs:
            print(f"  - {p}")
    else:
        print("  Sin problemas en la configuracion de adquisicion.")
    return 0


def cmd_filtros() -> int:
    from filters import CadenaFiltrado

    titulo("RESPUESTA DE LA CADENA DE FILTRADO")
    b, a = CONFIG.bci, CONFIG.adquisicion
    cadena = CadenaFiltrado(CONFIG)

    seccion("Ganancia en las frecuencias de interes")
    print(f"  {'frecuencia':>12} {'ganancia':>10}  que es")

    puntos = [(2.0, "por debajo de la banda")]
    for f in b.frecuencias:
        puntos.append((f, "fundamental"))
        for o in (2, 3, 4):
            if f * o < b.fs / 2:
                puntos.append((f * o, f"armonico {o} de {f} Hz"))
    puntos.append((b.notch_hz, "RED ELECTRICA"))
    puntos.append((110.0, "por encima de la banda"))

    for f, desc in sorted(set(puntos)):
        if f >= b.fs / 2:
            continue
        g = cadena.ganancia_en(f)
        marca = ""
        if abs(f - b.notch_hz) < 0.1:
            marca = "  <-- debe suprimirse"
        elif abs(f - 60.8) < 0.1:
            marca = "  <-- CRITICO"
        print(f"  {f:>11.1f}Hz {g:>10.4f}  {desc}{marca}")

    seccion("EL PUNTO CRITICO")
    print("  El 4o armonico de 15.2 Hz cae en 60.8 Hz, a 0.8 Hz de la red.")
    print("  Por eso el factor Q del notch no puede ser el habitual de 30.")
    print()
    print(f"  {'Q':>5} {'banda suprimida':>22} {'ganancia 60.8 Hz':>18}")
    for q in (30, 45, 60, 90):
        cfg_q = CONFIG.copia_con(adquisicion=dict(q_notch=float(q)))
        c = CadenaFiltrado(cfg_q)
        anc = b.notch_hz / q
        marca = "  <-- actual" if q == int(a.q_notch) else ""
        print(f"  {q:>5} {b.notch_hz-anc/2:>10.2f}-{b.notch_hz+anc/2:<10.2f} "
              f"{c.ganancia_en(60.8):>18.4f}{marca}")

    seccion("VERIFICACION")
    probs = cadena.verificar()
    if probs:
        for p in probs:
            print(f"  - {p}")
        return 1
    print("  La cadena deja pasar todas las fundamentales y armonicos")
    print("  utiles, y suprime la red.")
    return 0


def cmd_causal() -> int:
    from filters import CadenaFiltrado
    from scipy.signal import lfilter, filtfilt

    titulo("POR QUE EL FILTRADO ES CAUSAL Y CON ESTADO")
    b, a = CONFIG.bci, CONFIG.adquisicion

    seccion("1. filtfilt NO es causal")
    print("  Se filtra un escalon: cero hasta la mitad, luego uno.")
    print("  Un filtro causal no puede reaccionar ANTES del escalon.")
    print()

    cadena = CadenaFiltrado(CONFIG)
    n = 200
    x = np.zeros(n)
    x[n // 2:] = 1.0
    bp, ap = cadena._coef_banda

    y_ff = filtfilt(bp, ap, x)
    y_lf = lfilter(bp, ap, x)

    print(f"  Maxima respuesta antes del escalon:")
    print(f"    filtfilt : {np.abs(y_ff[:n//2]).max():.6f}  "
          f"<-- reacciona antes de que ocurra")
    print(f"    lfilter  : {np.abs(y_lf[:n//2]).max():.6f}")
    print()
    print("  filtfilt usa muestras futuras. Offline es legitimo; online esas")
    print("  muestras no existen todavia. Usarlo daria un rendimiento en")
    print("  analisis que no se reproduce en tiempo real.")

    seccion("2. El estado debe persistir entre bloques")
    rng = np.random.default_rng(1)
    n_tot = int(3.0 * b.fs)
    t = np.arange(n_tot) / b.fs
    senal = (40 * np.sin(2 * np.pi * 14.0 * t) +
             25 * np.sin(2 * np.pi * b.notch_hz * t) +
             12 * rng.standard_normal(n_tot))
    senal = senal[:, None] * np.ones((1, b.n_canales))

    ref = CadenaFiltrado(CONFIG).aplicar(senal)

    nb = a.buffer_muestras
    c1 = CadenaFiltrado(CONFIG)
    con = np.concatenate(
        [c1.aplicar(senal[i:i + nb]) for i in range(0, n_tot, nb)])

    c2 = CadenaFiltrado(CONFIG)
    trozos = []
    for i in range(0, n_tot, nb):
        c2.reiniciar()
        trozos.append(c2.aplicar(senal[i:i + nb]))
    sin = np.concatenate(trozos)

    rango = float(np.abs(ref).max())
    e_con = float(np.abs(con - ref).max())
    e_sin = float(np.abs(sin - ref).max())

    print(f"  Rango de la senal filtrada : {rango:.1f} uV")
    print()
    print(f"  Error CON estado : {e_con:.6f} uV "
          f"({e_con/rango*100:.4f}%)")
    print(f"  Error SIN estado : {e_sin:.2f} uV "
          f"({e_sin/rango*100:.1f}%)")
    print()
    print(f"  Con estado, filtrar por bloques da EXACTAMENTE el mismo")
    print(f"  resultado que filtrar la senal entera.")
    print()
    print(f"  Sin estado, cada bloque arranca con un transitorio. A "
          f"{b.fs/nb:.0f} bloques")
    print(f"  por segundo, eso serian {b.fs/nb:.0f} transitorios por segundo")
    print(f"  contaminando la senal.")
    return 0


def cmd_fuente() -> int:
    from eeg_source import FuenteSinteticaEEG

    titulo("PRUEBA DE LA FUENTE DE SENAL")
    b = CONFIG.bci

    print()
    print(f"  Fuente configurada: {CONFIG.adquisicion.fuente}")
    print()

    seccion("Ritmo temporal")
    f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
    f.iniciar()
    bloques = []
    t0 = time.perf_counter()
    while len(bloques) < 30 and (time.perf_counter() - t0) < 4.0:
        bl = f.leer()
        if bl is not None:
            bloques.append(bl)
        else:
            time.sleep(0.001)
    dt = time.perf_counter() - t0
    total = sum(len(x) for x in bloques)
    print(f"  Bloques  : {len(bloques)}")
    print(f"  Muestras : {total}")
    print(f"  Tasa     : {total/dt:.1f} Hz (nominal {b.fs} Hz)")
    err = abs(total / dt - b.fs) / b.fs * 100
    veredicto(err < 10, f"desviacion de {err:.1f}% respecto a la nominal")
    f.detener()

    seccion("Contenido espectral (sin filtrar)")
    f2 = FuenteSinteticaEEG(CONFIG, freq_idx=2, snr_db=0.0, semilla=2)
    f2.iniciar()
    trozos = []
    while sum(len(x) for x in trozos) < int(4 * b.fs):
        bl = f2.leer()
        if bl is not None:
            trozos.append(bl)
        else:
            time.sleep(0.001)
    x = np.concatenate(trozos)[:, -1]
    f2.detener()

    esp = np.abs(np.fft.rfft(x - x.mean()))
    fr = np.fft.rfftfreq(len(x), d=1.0 / b.fs)
    print(f"  {'frecuencia':>12} {'relativa':>10}")
    for fv in list(b.frecuencias) + [b.notch_hz]:
        i = int(np.argmin(np.abs(fr - fv)))
        m = ""
        if abs(fv - b.frecuencias[2]) < 0.01:
            m = "  <-- atendida"
        elif abs(fv - b.notch_hz) < 0.01:
            m = "  <-- red (el notch la quitara)"
        print(f"  {fv:>11.1f}Hz {esp[i]/esp.max():>10.3f}{m}")
    return 0


def cmd_buffer() -> int:
    from acquisition import BufferCircular

    titulo("VENTANEO SOLAPADO")
    b, a = CONFIG.bci, CONFIG.adquisicion

    print()
    print("  Hay un desajuste que resolver:")
    print(f"    el amplificador entrega bloques de {a.buffer_muestras} "
          f"muestras")
    print(f"    la ventana de analisis es de {b.muestras_ventana_tasm}")
    print(f"    y avanza {b.muestras_paso_tasm} muestras cada vez")
    print()
    solape = (1 - b.muestras_paso_tasm / b.muestras_ventana_tasm) * 100
    print(f"  Cada ventana solapa un {solape:.0f}% con la anterior.")

    seccion("Simulacion")
    buf = BufferCircular(b.muestras_ventana_tasm, b.n_canales,
                         b.muestras_paso_tasm)
    print(f"  {'bloque':>8} {'muestras':>10} {'ventanas':>10} "
          f"{'acumuladas':>12}")
    acum = 0
    for i in range(1, 61):
        buf.escribir(np.random.randn(a.buffer_muestras, b.n_canales))
        nv = 0
        while buf.hay_ventana():
            buf.extraer_ventana()
            nv += 1
            acum += 1
        if i <= 3 or i in (16, 17, 18, 30, 45, 60):
            print(f"  {i:>8} {buf.muestras_escritas:>10} {nv:>10} "
                  f"{acum:>12}")

    print()
    esperado = (buf.muestras_escritas - b.muestras_ventana_tasm) \
        // b.muestras_paso_tasm + 1
    veredicto(abs(acum - esperado) <= 1,
              f"{acum} ventanas de {esperado} esperadas")
    return 0


def cmd_pipeline() -> int:
    """Compuerta del bloque."""
    from eeg_source import FuenteSinteticaEEG
    from acquisition import (PipelineAdquisicion, crear_procesador_fbcca)
    from collections import Counter

    titulo("COMPUERTA DEL BLOQUE 3 --- Cadena completa")
    b = CONFIG.bci
    ok = True

    print()
    print("  Se genera senal con respuesta SSVEP conocida, se pasa por toda")
    print("  la cadena, y se comprueba que la frecuencia se recupera.")

    for idx in range(len(b.frecuencias)):
        seccion(f"Frecuencia inyectada: {b.frecuencias[idx]} Hz "
                f"(indice {idx})")

        fuente = FuenteSinteticaEEG(CONFIG, freq_idx=idx, snr_db=-3.0,
                                    con_red=True, semilla=10 + idx)
        proc = crear_procesador_fbcca(CONFIG)
        pipe = PipelineAdquisicion(CONFIG, fuente, proc)

        detectadas = []
        pipe.correr(duracion=4.0,
                    callback=lambda m: detectadas.append(m.freq_idx))

        if not detectadas:
            ok &= veredicto(False, "no se genero ninguna ventana")
            continue

        cuenta = Counter(detectadas)
        acierto = cuenta.get(idx, 0) / len(detectadas)
        ok &= veredicto(acierto >= 0.80,
                        f"acierto {acierto*100:.1f}% "
                        f"({cuenta.get(idx,0)}/{len(detectadas)} ventanas)")

    seccion("Rendimiento")
    fuente = FuenteSinteticaEEG(CONFIG, freq_idx=2, snr_db=-3.0, semilla=99)
    proc = crear_procesador_fbcca(CONFIG)
    pipe = PipelineAdquisicion(CONFIG, fuente, proc)
    pipe.correr(duracion=5.0)
    print(pipe.stats.resumen(b.paso_tasm * 1000))

    ok &= veredicto(pipe.stats.computo_p95_ms < b.paso_tasm * 1000,
                    "el computo cabe en el periodo de ventana")
    ok &= veredicto(pipe.stats.tasa_ventanas > 1 / b.paso_tasm * 0.85,
                    f"tasa de ventanas {pipe.stats.tasa_ventanas:.1f} Hz")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    print()
    print("  RECORDATORIO: esto verifica la cadena con senal sintetica.")
    print("  En el laboratorio hay que comprobar ademas:")
    print("    - impedancias por debajo de 5 kOhm antes de cada sesion")
    print("    - picos espectrales con un sujeto real (protocolo Bloque 2)")
    print("    - orden de canales identico a config.bci.canales")
    return 0 if ok else 1


def cmd_latencia() -> int:
    from eeg_source import FuenteSinteticaEEG
    from acquisition import PipelineAdquisicion, crear_procesador_fbcca

    titulo("PRESUPUESTO DE LATENCIA")
    b, l = CONFIG.bci, CONFIG.latencia

    print()
    print("  Hay DOS latencias distintas y conviene no confundirlas.")

    seccion("1. Tiempo de COMPUTO por ventana")
    print("  Es lo unico que debe caber en el periodo de ventana. Si el")
    print("  procesamiento tardara mas que el intervalo entre ventanas, la")
    print("  cola creceria sin limite.")
    print()

    fuente = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
    proc = crear_procesador_fbcca(CONFIG)
    pipe = PipelineAdquisicion(CONFIG, fuente, proc)
    pipe.correr(duracion=8.0)

    print(f"  Medido (media)  : {pipe.stats.computo_medio_ms:.2f} ms")
    print(f"  Medido (p95)    : {pipe.stats.computo_p95_ms:.2f} ms")
    print(f"  Presupuestado   : {l.procesamiento_ms:.1f} ms")
    print(f"  Periodo         : {b.paso_tasm*1000:.1f} ms")
    print()
    margen = b.paso_tasm * 1000 - pipe.stats.computo_p95_ms
    veredicto(margen > 0, f"margen de {margen:.2f} ms sobre el periodo")

    seccion("2. Latencia EXTREMO A EXTREMO")
    print("  Retardo entre que ocurre algo en el cerebro y el robot")
    print("  reacciona. NO tiene que caber en el periodo: la cadena es un")
    print("  pipeline y las etapas se solapan.")
    print()
    print(f"    adquisicion   : {l.adquisicion_ms:>6.1f} ms")
    print(f"    filtrado      : {l.filtrado_ms:>6.1f} ms")
    print(f"    clasificacion : {l.clasificacion_ms:>6.1f} ms")
    print(f"    transporte    : {l.transporte_ms:>6.1f} ms")
    print(f"    ros2          : {l.ros2_ms:>6.1f} ms")
    print(f"    {'TOTAL':14}: {l.extremo_a_extremo_ms:>6.1f} ms")
    print()
    print(f"  Watchdog        : {b.watchdog_s*1000:.0f} ms")
    veredicto(l.margen_watchdog(b.watchdog_s) > 0,
              f"margen de {l.margen_watchdog(b.watchdog_s):.0f} ms "
              f"sobre el watchdog")

    seccion("3. Retardo total del comando")
    print("  Lo que el usuario percibe: desde que empieza a mirar el")
    print("  estimulo hasta que el robot se mueve.")
    print()
    total = b.t_confirmacion * 1000 + l.extremo_a_extremo_ms
    print(f"    racha de confirmacion : {b.t_confirmacion*1000:>6.0f} ms")
    print(f"    cadena                : {l.extremo_a_extremo_ms:>6.0f} ms")
    print(f"    {'TOTAL':22}: {total:>6.0f} ms")
    print()
    d = CONFIG.robot.u_max * total / 1000
    print(f"  A {CONFIG.robot.u_max} m/s, el robot recorre {d*100:.1f} cm")
    print(f"  durante ese retardo.")
    return 0


def cmd_impedancias() -> int:
    from eeg_source import crear_fuente_eeg, comprobar_impedancias

    titulo("COMPROBACION DE IMPEDANCIAS")
    print()
    print("  Ejecutar ANTES de cada sesion. Una impedancia alta degrada la")
    print("  relacion senal-ruido lo suficiente para comprometer la")
    print("  deteccion, y el remedio es simple: reaplicar gel.")
    print()
    print(f"  Umbral: {CONFIG.adquisicion.impedancia_maxima:.0f} ohm")

    fuente = crear_fuente_eeg(CONFIG)
    rep = comprobar_impedancias(fuente, CONFIG)

    seccion("RESULTADO")
    print(rep.resumen())

    if CONFIG.adquisicion.fuente == "sintetica":
        print()
        print("  NOTA: estos valores son simulados. Con el amplificador real")
        print("  hay que completar el metodo impedancias() de FuenteGUSBamp.")
    return 0 if rep.todos_correctos else 1


def cmd_test() -> int:
    import subprocess
    titulo("BATERIA DE TESTS")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v",
                        "--tb=short", "-W", "ignore"])
    return r.returncode


# ===========================================================================

COMANDOS = {
    "config": cmd_config,
    "filtros": cmd_filtros,
    "causal": cmd_causal,
    "fuente": cmd_fuente,
    "buffer": cmd_buffer,
    "pipeline": cmd_pipeline,
    "latencia": cmd_latencia,
    "impedancias": cmd_impedancias,
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

"""
run_estimulo.py --- Script principal del Bloque 2
Bloque 2: estimulo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

COMANDOS
--------
    python run_estimulo.py config       Configuracion del estimulo
    python run_estimulo.py realizable   Verifica que las frecuencias se
                                        pueden generar a 60 Hz
    python run_estimulo.py textura      Muestra la textura del estimulo
    python run_estimulo.py espectro     Comprueba el espectro generado
    python run_estimulo.py deriva       Demuestra el problema del reloj
    python run_estimulo.py validacion   Protocolo de validacion (simulado)
    python run_estimulo.py timing       Compuerta: temporizacion sin jitter
    python run_estimulo.py demo         Demostracion grafica (necesita
                                        PsychoPy y pantalla)
    python run_estimulo.py test         Bateria de tests

ORDEN RECOMENDADO
-----------------
  1. config       Entender que parametros hay
  2. realizable   Ver POR QUE se usa modulacion sinusoidal
  3. espectro     Verificar que la senal generada es correcta
  4. timing       Compuerta del bloque
  5. demo         Ya en la maquina del laboratorio, con pantalla

LA COMPUERTA DEL BLOQUE
-----------------------
`python run_estimulo.py timing` es la verificacion que hay que superar antes
de pasar al Bloque 3. Comprueba que la secuencia generada tiene los picos
espectrales exactamente donde deben, sin ensanchamiento.

En la maquina del laboratorio hay ademas que ejecutar `demo` y comprobar que
el porcentaje de frames perdidos se mantiene por debajo del 1%.
"""

import sys
import math
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
    titulo("CONFIGURACION DEL ESTIMULO")
    e, b = CONFIG.estimulo, CONFIG.bci

    print()
    print("  Pantalla")
    print(f"    Refresco          : {e.refresh_hz} Hz")
    print(f"    Resolucion        : {e.resolucion[0]} x {e.resolucion[1]}")
    print(f"    Distancia vision  : {e.distancia_vision} m")
    print()
    print("  Estimulo")
    print(f"    Tamano angular    : {e.tamano_angular} grados "
          f"= {e.lado_estimulo_px} px")
    print(f"    Rejilla           : {e.celdas_por_lado} x {e.celdas_por_lado}")
    print(f"    Densidad          : {e.densidad_pixeles*100:.0f}% "
          f"({e.n_celdas_activas} celdas encendidas)")
    print(f"    Distribucion      : "
          f"{'aleatoria' if e.distribucion_aleatoria else 'uniforme'}")
    print()
    print("  Frecuencias y fases")
    print(f"    {'indice':>7} {'f (Hz)':>8} {'fase':>10} {'posicion':>16}")
    nombres = ("giro izq", "giro der", "avanzar", "parar")
    for i, f in enumerate(b.frecuencias):
        px, py = e.posiciones_cruz[i]
        print(f"    {i:>7} {f:>8.1f} "
              f"{math.degrees(b.fases_jfpm[i]):>9.0f}o "
              f"{nombres[i]:>16}  ({px:+.2f}, {py:+.2f})")
    print()
    print("  Validacion")
    print(f"    Ventana FFT       : {e.ventana_fft} s "
          f"(resolucion {1/e.ventana_fft:.2f} Hz)")
    print(f"    Frames perdidos   : maximo {e.max_frames_perdidos*100:.0f}%")

    problemas = [p for p in CONFIG.validar()
                 if any(k in p.lower() for k in
                        ("estimulo", "densidad", "nyquist", "bin", "pixel"))]
    seccion("VALIDACION")
    if problemas:
        for p in problemas:
            print(f"  - {p}")
    else:
        print("  Sin problemas en la configuracion del estimulo.")
    return 0


def cmd_realizable() -> int:
    from stimulus import verificar_realizabilidad

    titulo("REALIZABILIDAD DE LAS FRECUENCIAS A 60 Hz")
    print()
    print("  Con onda cuadrada solo son realizables las frecuencias de la")
    print("  forma R/n con n PAR. Con modulacion sinusoidal muestreada,")
    print("  cualquier frecuencia por debajo de Nyquist.")

    v = verificar_realizabilidad(CONFIG)
    seccion(f"Refresco {v['refresh']} Hz, Nyquist {v['nyquist']} Hz")
    print(f"  {'f (Hz)':>8} {'R/f':>9} {'entero':>8} {'par':>6} "
          f"{'cuadrada':>10} {'sinusoidal':>12}")
    for x in v["frecuencias"]:
        c = x["R/f"]
        ent = abs(c - round(c)) < 1e-9
        par = ent and round(c) % 2 == 0
        print(f"  {x['frecuencia']:>8.1f} {c:>9.3f} "
              f"{'si' if ent else 'no':>8} {'si' if par else 'no':>6} "
              f"{'SI' if x['cuadrada_ok'] else 'NO':>10} "
              f"{'SI' if x['sinusoidal_ok'] else 'NO':>12}")

    seccion("CONCLUSION")
    n_ok = sum(1 for x in v["frecuencias"] if x["cuadrada_ok"])
    print(f"  Realizables con onda cuadrada : {n_ok} de "
          f"{len(v['frecuencias'])}")
    print(f"  Realizables con sinusoidal    : "
          f"{sum(1 for x in v['frecuencias'] if x['sinusoidal_ok'])} de "
          f"{len(v['frecuencias'])}")
    print()
    print("  Por eso el sistema usa modulacion sinusoidal muestreada.")
    return 0


def cmd_textura() -> int:
    from stimulus import TexturaEstimulo

    titulo("TEXTURA DEL ESTIMULO")
    e = CONFIG.estimulo

    tex = TexturaEstimulo(CONFIG)
    print()
    print(f"  Rejilla         : {tex.n} x {tex.n} = {tex.n**2} celdas")
    print(f"  Encendidas      : {int(tex.mascara.sum())}")
    print(f"  Densidad real   : {tex.densidad_real*100:.1f}% "
          f"(objetivo {e.densidad_pixeles*100:.0f}%)")
    print(f"  Lado en pantalla: {e.lado_estimulo_px} px")

    seccion("PATRON COMPLETO  (# encendida, . apagada)")
    for fila in tex.mascara:
        print("    " + "".join("#" if c else "." for c in fila))

    seccion("COMPARACION CON DISTRIBUCION UNIFORME")
    print("  Meng et al. (2023) encontraron que la distribucion ALEATORIA")
    print("  supera a la uniforme en precision y en fatiga. Se incluye la")
    print("  uniforme solo para poder reproducir su comparacion.")
    print()
    cfg_u = CONFIG.copia_con(estimulo=dict(distribucion_aleatoria=False))
    tex_u = TexturaEstimulo(cfg_u)
    print("  Uniforme (primeras 8 filas):")
    for fila in tex_u.mascara[:8]:
        print("    " + "".join("#" if c else "." for c in fila))
    return 0


def cmd_espectro() -> int:
    from stimulus import GeneradorLuminancia

    titulo("VERIFICACION ESPECTRAL DE LA SENAL GENERADA")
    print()
    print("  Se genera la secuencia ideal de luminancia y se comprueba que")
    print("  su espectro tiene el pico exactamente en la frecuencia nominal.")

    e, b = CONFIG.estimulo, CONFIG.bci
    gen = GeneradorLuminancia(CONFIG)
    n = int(e.ventana_fft * e.refresh_hz)
    sec = gen.secuencia(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / e.refresh_hz)

    seccion(f"Ventana de {e.ventana_fft} s ({n} frames)")
    print(f"  {'nominal':>9} {'pico':>9} {'error':>9} {'bin':>6} "
          f"{'bin teorico':>13}")

    ok = True
    for k, f in enumerate(b.frecuencias):
        x = sec[:, k] - sec[:, k].mean()
        esp = np.abs(np.fft.rfft(x))
        i = int(np.argmax(esp))
        err = abs(freqs[i] - f)
        ok &= (err < 1e-6)
        print(f"  {f:>8.1f}Hz {freqs[i]:>8.2f}Hz {err:>8.4f}Hz "
              f"{i:>6} {e.bin_fft(f):>13.0f}")

    print()
    veredicto(ok, "todas las frecuencias caen exactamente en su bin")

    seccion("PUREZA ESPECTRAL")
    print("  Energia fuera de la frecuencia nominal (deberia ser minima).")
    print()
    print(f"  {'frecuencia':>11} {'energia en pico':>17} {'resto':>12} "
          f"{'ratio':>10}")
    for k, f in enumerate(b.frecuencias):
        x = sec[:, k] - sec[:, k].mean()
        esp = np.abs(np.fft.rfft(x)) ** 2
        i = int(np.argmax(esp))
        e_pico = esp[i]
        e_resto = esp.sum() - e_pico
        print(f"  {f:>10.1f}Hz {e_pico:>17.1f} {e_resto:>12.4f} "
              f"{e_resto/e_pico:>10.2e}")
    print()
    print("  Un ratio muy bajo confirma que la senal es una sinusoide pura,")
    print("  sin los armonicos que introduciria una onda cuadrada.")
    return 0 if ok else 1


def cmd_deriva() -> int:
    titulo("POR QUE EL CONTADOR DE FRAMES Y NO EL RELOJ")
    print()
    print("  El error mas facil de cometer al implementar el estimulo es")
    print("  calcular la fase con el tiempo transcurrido en lugar del")
    print("  contador de frames.")
    print()
    print("  Se simula una implementacion que acumula el paso de tiempo con")
    print("  un error de solo 0.1 ms por frame.")

    e, b = CONFIG.estimulo, CONFIG.bci
    dt_nom = 1.0 / e.refresh_hz

    for err_ms in (0.01, 0.1, 0.5):
        seccion(f"Error de {err_ms} ms por frame")
        dt_real = dt_nom + err_ms / 1000.0
        print(f"  {'tiempo':>10} {'deriva de fase':>18} {'efecto'}")
        for minutos in (0.5, 1, 2, 5, 10):
            n = int(minutos * 60 * e.refresh_hz)
            deriva = 2 * math.pi * b.freq_parar * (n * dt_real - n * dt_nom)
            g = math.degrees(deriva) % 360
            if g < 30 or g > 330:
                efecto = "despreciable"
            elif 150 < g < 210:
                efecto = "SENAL INVERTIDA"
            else:
                efecto = "degradacion"
            print(f"  {minutos:>7} min {g:>17.1f}o  {efecto}")

    seccion("CONCLUSION")
    print("  Con contador de frames la deriva es EXACTAMENTE cero, porque la")
    print("  fase se calcula de un indice entero:")
    print()
    print("      angulo = 2*pi*f*(frame/refresh) + fase")
    print()
    print("  Si en el codigo de PsychoPy ves algo parecido a")
    print("  'fase = 2*pi*f*time.time()', esta mal.")
    return 0


def cmd_validacion() -> int:
    from validacion_fft import (ValidadorEspectral, ProtocoloValidacion,
                                simular_respuesta_eeg)

    titulo("PROTOCOLO DE VALIDACION ESPECTRAL")
    print()
    print("  Se ejecuta con senal EEG SIMULADA. En el laboratorio hay que")
    print("  sustituir la funcion de adquisicion por la lectura real del")
    print("  amplificador.")

    val = ValidadorEspectral(CONFIG)
    b = CONFIG.bci
    dur = CONFIG.estimulo.ventana_fft * 4

    seccion("PASO 1: barrido de frecuencia")
    ok = True
    for k, f in enumerate(b.frecuencias):
        s = simular_respuesta_eeg(CONFIG, k, dur, snr_db=-8.0, semilla=100 + k)
        r = val.analizar(s, b.fs)
        p = r.picos[k]
        bien = p.error_frecuencia < 0.2
        ok &= bien
        print(f"  {f:>5.1f}Hz -> {p.frecuencia_pico:>6.2f}Hz  "
              f"error {p.error_frecuencia:>5.3f}Hz  "
              f"SNR {p.snr_db:>5.1f}dB  {'OK' if bien else 'FALLO'}")

    seccion("PASO 3: deteccion de jitter")
    print(f"  {'jitter':>10} {'ancho':>12} {'SNR':>10}")
    for j in (0.0, 0.01, 0.03, 0.06):
        s = simular_respuesta_eeg(CONFIG, 2, dur, snr_db=-5.0,
                                  jitter_frames=j, semilla=7)
        p = val.analizar(s, b.fs).picos[2]
        print(f"  {j:>10.3f} {p.ancho_media_altura:>11.3f}Hz "
              f"{p.snr_db:>9.1f}dB")

    seccion("PASO 4: control de validez entre escenarios")

    def adquirir(escenario, condicion, duracion):
        idx = [x.nombre for x in CONFIG.escenarios.lista].index(escenario)
        penal = 1.2 * idx if condicion == "movimiento" else 0.0
        return simular_respuesta_eeg(CONFIG, 2, duracion,
                                     snr_db=-6.0 - penal, semilla=200 + idx)

    prot = ProtocoloValidacion(CONFIG)
    nombres = [x.nombre for x in CONFIG.escenarios.lista]
    print(prot.comparar_escenarios(prot.paso4_escenarios(adquirir, nombres)))

    seccion("RESULTADO")
    print(f"  {'PROTOCOLO SUPERADO' if ok else 'HAY FALLOS'}")
    return 0 if ok else 1


def cmd_timing() -> int:
    """Compuerta del bloque."""
    from stimulus import GeneradorLuminancia, TexturaEstimulo
    from interface import InterfazSimulada, EstadoInterfaz
    from video_source import crear_fuente

    titulo("COMPUERTA DEL BLOQUE 2 --- Temporizacion del estimulo")
    ok = True
    e, b = CONFIG.estimulo, CONFIG.bci

    seccion("1. Espectro exacto")
    gen = GeneradorLuminancia(CONFIG)
    n = int(e.ventana_fft * e.refresh_hz)
    sec = gen.secuencia(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / e.refresh_hz)
    for k, f in enumerate(b.frecuencias):
        x = sec[:, k] - sec[:, k].mean()
        i = int(np.argmax(np.abs(np.fft.rfft(x))))
        ok &= veredicto(abs(freqs[i] - f) < 1e-6,
                        f"{f} Hz cae en el bin {i} sin error")

    seccion("2. Fase estable a lo largo del trial")
    gen.reiniciar()
    largo = gen.secuencia(n * 6)
    from validacion_fft import ValidadorEspectral
    val = ValidadorEspectral(CONFIG)
    for k, f in enumerate(b.frecuencias):
        f1 = val._fase_en(largo[:n, k], e.refresh_hz, f)
        f6 = val._fase_en(largo[5 * n:, k], e.refresh_hz, f)
        d = abs(math.degrees(f6 - f1)) % 360
        d = min(d, 360 - d)
        ok &= veredicto(d < 1.0,
                        f"{f} Hz: deriva de {d:.3f} grados en 6 ventanas")

    seccion("3. Rango de luminancia")
    ok &= veredicto(sec.min() >= -1e-9 and sec.max() <= 1 + 1e-9,
                    f"luminancia dentro de [0,1] "
                    f"(min {sec.min():.4f}, max {sec.max():.4f})")

    seccion("4. Densidad de la textura")
    tex = TexturaEstimulo(CONFIG)
    err = abs(tex.densidad_real - e.densidad_pixeles)
    ok &= veredicto(err < 0.01,
                    f"densidad {tex.densidad_real*100:.1f}% "
                    f"(objetivo {e.densidad_pixeles*100:.0f}%)")

    seccion("5. Bucle de interfaz completo")
    fuente = crear_fuente(CONFIG, "sintetica")
    iface = InterfazSimulada(CONFIG, fuente)
    n_f = int(10.0 * e.refresh_hz)
    for i in range(n_f):
        iface.dibujar(EstadoInterfaz(etapa=1, estado_tasm="IC"))
    ok &= veredicto(len(iface.historial) == n_f,
                    f"{len(iface.historial)} frames en 10 s "
                    f"(esperados {n_f})")
    ok &= veredicto(iface.registro.fraccion_perdidos <= e.max_frames_perdidos,
                    f"frames perdidos "
                    f"{iface.registro.fraccion_perdidos*100:.2f}% "
                    f"(maximo {e.max_frames_perdidos*100:.0f}%)")
    fuente.cerrar()

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    print()
    print("  RECORDATORIO: esta compuerta verifica la LOGICA de generacion.")
    print("  En la maquina del laboratorio hay que ejecutar ademas:")
    print("     python run_estimulo.py demo")
    print("  y comprobar que los frames perdidos se mantienen bajo el 1%")
    print("  con el video real de fondo.")
    return 0 if ok else 1


def cmd_demo() -> int:
    """Demostracion grafica. Necesita PsychoPy y una pantalla."""
    try:
        from interface import InterfazSSVEP, EstadoInterfaz, _HAY_PSYCHOPY
        from video_source import crear_fuente
    except ImportError as ex:
        print(f"Error de importacion: {ex}")
        return 1

    if not _HAY_PSYCHOPY:
        titulo("DEMOSTRACION GRAFICA NO DISPONIBLE")
        print()
        print("  PsychoPy no esta instalado. Instala con:")
        print("     pip install psychopy")
        print()
        print("  Mientras tanto puedes verificar toda la logica temporal")
        print("  con:  python run_estimulo.py timing")
        return 1

    titulo("DEMOSTRACION GRAFICA")
    print()
    print("  Pulsa ESC para salir.")
    print("  Pulsa ESPACIO para alternar entre Etapa 1 y Etapa 2.")
    print()

    fuente = crear_fuente(CONFIG, "sintetica")
    etapa = 1
    estados = ("Idle", "TR", "IC")

    with InterfazSSVEP(CONFIG, fuente, pantalla_completa=False) as iface:
        i = 0
        while True:
            teclas = iface.teclas()
            if "escape" in teclas:
                break
            if "space" in teclas:
                etapa = 2 if etapa == 1 else 1

            est = EstadoInterfaz(
                etapa=etapa,
                estado_tasm=estados[(i // 90) % 3],
                activos=None if etapa == 1 else [0, 1],
                grupos_e2=([0, 1], [2, 3]) if etapa == 2 else None,
                bboxes=[(-300, 0, 150, 200), (-100, 0, 150, 200),
                        (100, 0, 150, 200), (300, 0, 150, 200)]
                if etapa == 2 else None,
                mensaje=f"Etapa {etapa} --- ESPACIO cambia, ESC sale",
            )
            iface.dibujar(est)
            i += 1

        print()
        print(iface.registro.resumen())

    fuente.cerrar()
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
    "realizable": cmd_realizable,
    "textura": cmd_textura,
    "espectro": cmd_espectro,
    "deriva": cmd_deriva,
    "validacion": cmd_validacion,
    "timing": cmd_timing,
    "demo": cmd_demo,
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

"""
tests/test_estimulo.py --- Bateria de tests del Bloque 2
Bloque 2: estimulo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo y que intentaste.
===========================================================================

QUE NO CUBREN ESTOS TESTS
-------------------------
El renderizado real con PsychoPy. Eso necesita una pantalla y solo puede
verificarse en la maquina del laboratorio con:

    python run_estimulo.py demo

Lo que si cubren es toda la logica que produce los numeros, que es donde
estan los errores dificiles de detectar a ojo: temporizacion, fase,
densidad, y el comportamiento de la interfaz frente a fallos del video.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CONFIG
from stimulus import (GeneradorLuminancia, TexturaEstimulo, RegistroFrames,
                      verificar_realizabilidad)
from validacion_fft import (ValidadorEspectral, simular_respuesta_eeg,
                            ProtocoloValidacion)
from video_source import FuenteSintetica, crear_fuente
from interface import InterfazSimulada, EstadoInterfaz


# ===========================================================================
# CONFIGURACION DEL ESTIMULO
# ===========================================================================

class TestConfigEstimulo:

    def test_una_posicion_por_frecuencia(self):
        assert len(CONFIG.estimulo.posiciones_cruz) == \
               len(CONFIG.bci.frecuencias)

    def test_frecuencias_bajo_nyquist(self):
        """Una frecuencia por encima de Nyquist de la pantalla no seria
        representable: aparecerian como aliasing a otra frecuencia."""
        nyq = CONFIG.estimulo.refresh_hz / 2
        for f in CONFIG.bci.frecuencias:
            assert f < nyq, f"{f} Hz supera Nyquist ({nyq} Hz)"

    def test_frecuencias_en_bin_entero(self):
        """Con la ventana de validacion, cada frecuencia debe caer en su bin.

        Si no, el error de fase medido seria un artefacto de la ventana y no
        una propiedad del renderizado.
        """
        for f in CONFIG.bci.frecuencias:
            b = CONFIG.estimulo.bin_fft(f)
            assert abs(b - round(b)) < 1e-9, \
                f"{f} Hz cae en el bin {b:.2f}, no entero"

    def test_densidad_no_bajo_el_umbral(self):
        """Por debajo del 60% la precision se desploma segun Meng 2023."""
        assert CONFIG.estimulo.densidad_pixeles >= 0.60

    def test_estimulo_mayor_que_su_rejilla(self):
        e = CONFIG.estimulo
        assert e.lado_estimulo_px >= e.celdas_por_lado, \
            "Cada celda mediria menos de un pixel"

    def test_onda_cuadrada_no_es_viable(self):
        """Documenta POR QUE se usa modulacion sinusoidal.

        Si este test empezara a pasar, es que alguien cambio las frecuencias
        a valores divisores de 60, lo que romperia la transferencia desde el
        dataset Benchmark.
        """
        v = verificar_realizabilidad(CONFIG)
        assert not v["cuadrada_viable"]
        assert v["sinusoidal_viable"]


# ===========================================================================
# GENERADOR DE LUMINANCIA
# ===========================================================================

class TestGenerador:

    def test_luminancia_en_rango(self):
        gen = GeneradorLuminancia(CONFIG)
        sec = gen.secuencia(1000)
        assert sec.min() >= -1e-9
        assert sec.max() <= 1 + 1e-9

    def test_una_luminancia_por_estimulo(self):
        gen = GeneradorLuminancia(CONFIG)
        lum = gen.siguiente_frame()
        assert len(lum) == len(CONFIG.bci.frecuencias)

    def test_contador_avanza(self):
        gen = GeneradorLuminancia(CONFIG)
        assert gen.frame == 0
        gen.siguiente_frame()
        assert gen.frame == 1

    def test_reiniciar_pone_el_contador_a_cero(self):
        gen = GeneradorLuminancia(CONFIG)
        for _ in range(50):
            gen.siguiente_frame()
        gen.reiniciar()
        assert gen.frame == 0

    def test_espectro_exacto(self):
        """PROPIEDAD CENTRAL: el pico debe caer exactamente en la nominal."""
        e = CONFIG.estimulo
        gen = GeneradorLuminancia(CONFIG)
        n = int(e.ventana_fft * e.refresh_hz)
        sec = gen.secuencia(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / e.refresh_hz)

        for k, f in enumerate(CONFIG.bci.frecuencias):
            x = sec[:, k] - sec[:, k].mean()
            i = int(np.argmax(np.abs(np.fft.rfft(x))))
            assert abs(freqs[i] - f) < 1e-6, \
                f"{f} Hz: pico en {freqs[i]:.3f} Hz"

    def test_fase_no_deriva(self):
        """El contador de frames garantiza fase exacta indefinidamente.

        Si este test fallara, alguien cambio el calculo de fase para usar
        tiempo acumulado.
        """
        e = CONFIG.estimulo
        gen = GeneradorLuminancia(CONFIG)
        n = int(e.ventana_fft * e.refresh_hz)
        largo = gen.secuencia(n * 10)
        val = ValidadorEspectral(CONFIG)

        for k, f in enumerate(CONFIG.bci.frecuencias):
            f1 = val._fase_en(largo[:n, k], e.refresh_hz, f)
            f10 = val._fase_en(largo[9 * n:, k], e.refresh_hz, f)
            d = abs(math.degrees(f10 - f1)) % 360
            d = min(d, 360 - d)
            assert d < 1.0, f"{f} Hz derivo {d:.2f} grados"

    def test_activos_apaga_los_demas(self):
        gen = GeneradorLuminancia(CONFIG)
        sec = gen.secuencia(100, activos=[0, 1])
        assert np.allclose(sec[:, 2], 0.0)
        assert np.allclose(sec[:, 3], 0.0)
        assert sec[:, 0].max() > 0.5

    def test_secuencia_no_avanza_el_contador(self):
        gen = GeneradorLuminancia(CONFIG)
        gen.secuencia(500)
        assert gen.frame == 0

    def test_reproducible(self):
        a = GeneradorLuminancia(CONFIG).secuencia(200)
        b = GeneradorLuminancia(CONFIG).secuencia(200)
        assert np.allclose(a, b)

    def test_fases_jfpm_distintas(self):
        """Con cuatro objetivos, el paso pi/2 da fases distintas.

        La Linea 2, con cinco objetivos, necesita paso 2pi/5 porque con pi/2
        se tendria phi_0 == phi_4.
        """
        fases = CONFIG.bci.fases_jfpm
        assert len(set(round(f, 6) for f in fases)) == len(fases)


# ===========================================================================
# TEXTURA
# ===========================================================================

class TestTextura:

    def test_densidad_correcta(self):
        tex = TexturaEstimulo(CONFIG)
        assert abs(tex.densidad_real - CONFIG.estimulo.densidad_pixeles) < 0.01

    def test_numero_exacto_de_celdas(self):
        """Se sortean posiciones concretas, no se aplica umbral a ruido.

        Con umbral el numero seria aproximado y la densidad real variaria.
        """
        tex = TexturaEstimulo(CONFIG)
        assert int(tex.mascara.sum()) == CONFIG.estimulo.n_celdas_activas

    def test_render_tiene_el_tamano_pedido(self):
        tex = TexturaEstimulo(CONFIG)
        img = tex.render(1.0, lado_px=128)
        assert img.shape == (128, 128)

    def test_render_escala_con_la_luminancia(self):
        tex = TexturaEstimulo(CONFIG)
        a = tex.render(1.0)
        b = tex.render(0.5)
        assert abs(a.max() - 1.0) < 1e-9
        assert abs(b.max() - 0.5) < 1e-9

    def test_celdas_apagadas_quedan_a_cero(self):
        """Las celdas apagadas se renderizan transparentes, lo que deja ver
        el video de fondo a traves del estimulo."""
        tex = TexturaEstimulo(CONFIG)
        img = tex.render(1.0)
        assert img.min() == 0.0

    def test_misma_semilla_misma_textura(self):
        a = TexturaEstimulo(CONFIG, semilla=1)
        b = TexturaEstimulo(CONFIG, semilla=1)
        assert np.array_equal(a.mascara, b.mascara)

    def test_semillas_distintas_texturas_distintas(self):
        a = TexturaEstimulo(CONFIG, semilla=1)
        b = TexturaEstimulo(CONFIG, semilla=2)
        assert not np.array_equal(a.mascara, b.mascara)

    def test_uniforme_es_diferente_de_aleatoria(self):
        cfg_u = CONFIG.copia_con(estimulo=dict(distribucion_aleatoria=False))
        a = TexturaEstimulo(CONFIG)
        u = TexturaEstimulo(cfg_u)
        assert not np.array_equal(a.mascara, u.mascara)


# ===========================================================================
# REGISTRO DE FRAMES
# ===========================================================================

class TestRegistroFrames:

    def test_sin_perdidas_con_intervalos_nominales(self):
        r = RegistroFrames(refresh_hz=60.0)
        for i in range(100):
            r.registrar(i / 60.0)
        assert r.frames_perdidos == 0

    def test_detecta_perdidas(self):
        r = RegistroFrames(refresh_hz=60.0)
        t = 0.0
        for i in range(50):
            r.registrar(t)
            # Cada 10 frames se pierde uno: el intervalo se duplica
            t += (2.0 / 60.0) if i % 10 == 0 else (1.0 / 60.0)
        assert r.frames_perdidos > 0

    def test_registro_vacio_no_rompe(self):
        r = RegistroFrames(refresh_hz=60.0)
        assert r.n_frames == 0
        assert r.frames_perdidos == 0
        assert r.fraccion_perdidos == 0.0


# ===========================================================================
# VALIDACION ESPECTRAL
# ===========================================================================

class TestValidacion:

    def test_recupera_la_frecuencia(self):
        val = ValidadorEspectral(CONFIG)
        dur = CONFIG.estimulo.ventana_fft * 3
        for k, f in enumerate(CONFIG.bci.frecuencias):
            s = simular_respuesta_eeg(CONFIG, k, dur, snr_db=-6.0,
                                      semilla=k)
            p = val.analizar(s, CONFIG.bci.fs).picos[k]
            assert p.error_frecuencia < 0.3, \
                f"{f} Hz: pico en {p.frecuencia_pico:.2f} Hz"

    def test_snr_mayor_con_mejor_senal(self):
        val = ValidadorEspectral(CONFIG)
        dur = CONFIG.estimulo.ventana_fft * 3
        snrs = []
        for db in (-15.0, -5.0, 5.0):
            s = simular_respuesta_eeg(CONFIG, 2, dur, snr_db=db, semilla=1)
            snrs.append(val.analizar(s, CONFIG.bci.fs).picos[2].snr_db)
        assert snrs[0] < snrs[1] < snrs[2]

    def test_jitter_degrada_la_snr(self):
        """El jitter de renderizado debe ser detectable."""
        val = ValidadorEspectral(CONFIG)
        dur = CONFIG.estimulo.ventana_fft * 3
        sin_j = val.analizar(
            simular_respuesta_eeg(CONFIG, 2, dur, snr_db=-3.0,
                                  jitter_frames=0.0, semilla=3),
            CONFIG.bci.fs).picos[2]
        con_j = val.analizar(
            simular_respuesta_eeg(CONFIG, 2, dur, snr_db=-3.0,
                                  jitter_frames=0.06, semilla=3),
            CONFIG.bci.fs).picos[2]
        assert con_j.snr_db < sin_j.snr_db

    def test_ruido_puro_da_snr_baja(self):
        val = ValidadorEspectral(CONFIG)
        s = simular_respuesta_eeg(CONFIG, -1,
                                  CONFIG.estimulo.ventana_fft * 3,
                                  semilla=99)
        r = val.analizar(s, CONFIG.bci.fs)
        assert r.snr_minima < 10.0

    def test_acepta_senal_multicanal(self):
        val = ValidadorEspectral(CONFIG)
        dur = CONFIG.estimulo.ventana_fft * 2
        multi = np.stack([
            simular_respuesta_eeg(CONFIG, 1, dur, snr_db=-6.0, semilla=i)
            for i in range(CONFIG.bci.n_canales)
        ], axis=1)
        r = val.analizar(multi, CONFIG.bci.fs)
        assert len(r.picos) == len(CONFIG.bci.frecuencias)


# ===========================================================================
# VIDEO
# ===========================================================================

class TestVideo:

    def test_fuente_sintetica_entrega_frames(self):
        f = FuenteSintetica(CONFIG, ancho=320, alto=240)
        frame = f.leer()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        f.cerrar()

    def test_frames_cambian_en_el_tiempo(self):
        """La escena debe tener movimiento: si no, no serviria para verificar
        el efecto de un fondo dinamico sobre el estimulo."""
        import time
        f = FuenteSintetica(CONFIG, ancho=320, alto=240, velocidad=5.0)
        a = f.leer()
        time.sleep(0.15)
        b = f.leer()
        assert not np.array_equal(a, b)
        f.cerrar()

    def test_fabrica_rechaza_tipo_desconocido(self):
        with pytest.raises(ValueError):
            crear_fuente(CONFIG, "inventado")

    def test_context_manager(self):
        with crear_fuente(CONFIG, "sintetica") as f:
            assert f.leer() is not None


# ===========================================================================
# INTERFAZ
# ===========================================================================

class TestInterfaz:

    def test_dibuja_el_numero_de_frames_pedido(self):
        iface = InterfazSimulada(CONFIG)
        for _ in range(100):
            iface.dibujar(EstadoInterfaz())
        assert len(iface.historial) == 100

    def test_etapa2_apaga_los_estimulos_no_usados(self):
        iface = InterfazSimulada(CONFIG)
        for _ in range(20):
            iface.dibujar(EstadoInterfaz(etapa=2, activos=[0, 1]))
        for h in iface.historial:
            assert h["luminancias"][2] == 0.0
            assert h["luminancias"][3] == 0.0

    def test_registra_el_estado_tasm(self):
        iface = InterfazSimulada(CONFIG)
        for est in ("IC", "TR", "Idle"):
            iface.dibujar(EstadoInterfaz(estado_tasm=est))
        assert [h["estado_tasm"] for h in iface.historial] == \
               ["IC", "TR", "Idle"]

    def test_funciona_sin_fuente_de_video(self):
        """El estimulo no debe depender de que haya video.

        Si el video fallara, el estimulo debe seguir generandose: son
        procesos independientes por diseno.
        """
        iface = InterfazSimulada(CONFIG, fuente_video=None)
        for _ in range(50):
            iface.dibujar(EstadoInterfaz())
        assert len(iface.historial) == 50

    def test_reiniciar_limpia_el_historial(self):
        iface = InterfazSimulada(CONFIG)
        for _ in range(30):
            iface.dibujar(EstadoInterfaz())
        iface.reiniciar()
        assert len(iface.historial) == 0
        assert iface.gen.frame == 0

    def test_la_luminancia_sigue_al_generador(self):
        """La interfaz no debe alterar los valores que produce el generador."""
        iface = InterfazSimulada(CONFIG)
        ref = GeneradorLuminancia(CONFIG).secuencia(30)
        for _ in range(30):
            iface.dibujar(EstadoInterfaz())
        for i, h in enumerate(iface.historial):
            assert np.allclose(h["luminancias"], ref[i])


# ===========================================================================
# INTEGRACION
# ===========================================================================

class TestIntegracion:

    def test_sesion_de_treinta_segundos(self):
        fuente = crear_fuente(CONFIG, "sintetica")
        iface = InterfazSimulada(CONFIG, fuente)
        n = int(30.0 * CONFIG.estimulo.refresh_hz)
        for i in range(n):
            iface.dibujar(EstadoInterfaz(
                etapa=1 if i < n // 2 else 2,
                estado_tasm="IC",
                activos=None if i < n // 2 else [0, 1],
            ))
        assert len(iface.historial) == n
        assert iface.registro.fraccion_perdidos <= \
               CONFIG.estimulo.max_frames_perdidos
        fuente.cerrar()

    def test_cambio_de_etapa_no_reinicia_la_fase(self):
        """La fase debe ser continua a traves del cambio de etapa.

        Un salto de fase al cambiar de etapa introduciria un transitorio que
        el clasificador leeria como senal.
        """
        iface = InterfazSimulada(CONFIG)
        for i in range(60):
            iface.dibujar(EstadoInterfaz(
                etapa=1 if i < 30 else 2,
                activos=None if i < 30 else [0, 1],
            ))
        ref = GeneradorLuminancia(CONFIG).secuencia(60)
        # El estimulo 0 sigue activo en ambas etapas: su fase debe ser
        # continua
        for i, h in enumerate(iface.historial):
            assert abs(h["luminancias"][0] - ref[i, 0]) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

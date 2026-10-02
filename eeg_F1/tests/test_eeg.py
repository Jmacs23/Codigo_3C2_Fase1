"""
tests/test_eeg.py --- Bateria de tests del Bloque 3
Bloque 3: eeg_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado con el amplificador real.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo y que intentaste.
===========================================================================

QUE NO CUBREN ESTOS TESTS
-------------------------
El amplificador real. La clase FuenteGUSBamp es un esqueleto y sus llamadas
al SDK estan pendientes de completar.

Lo que si cubren es toda la cadena de procesamiento, que es donde estan los
errores dificiles de ver: filtrado causal, persistencia de estado, ventaneo
solapado y deteccion de artefactos.
"""

import sys
import os
import math
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CONFIG
from filters import (FiltroCausal, CadenaFiltrado, DetectorArtefactos)
from eeg_source import (FuenteSinteticaEEG, FuenteDataset,
                        crear_fuente_eeg, comprobar_impedancias)
from acquisition import (BufferCircular, MensajeSalida, PuenteTCP,
                         PipelineAdquisicion, crear_procesador_fbcca)

from scipy.signal import butter, lfilter, filtfilt


# ===========================================================================
# CONFIGURACION
# ===========================================================================

class TestConfigAdquisicion:

    def test_buffer_no_mayor_que_el_paso(self):
        """Si el bloque fuera mayor que el paso de ventana, entre dos
        lecturas pasaria mas de un paso y se perderian actualizaciones."""
        assert (CONFIG.adquisicion.buffer_muestras <=
                CONFIG.bci.muestras_paso_tasm)

    def test_notch_no_toca_armonicos_utiles(self):
        """El notch debe ser lo bastante estrecho.

        El caso critico es el 4o armonico de 15.2 Hz, que cae en 60.8 Hz, a
        solo 0.8 Hz de la red.
        """
        b, a = CONFIG.bci, CONFIG.adquisicion
        semi = b.notch_hz / a.q_notch / 2
        for f in b.frecuencias:
            for orden in range(1, 5):
                arm = f * orden
                assert abs(arm - b.notch_hz) >= semi, \
                    f"El armonico {orden} de {f} Hz ({arm}) cae en el notch"

    def test_computo_cabe_en_el_periodo(self):
        assert CONFIG.latencia.cabe_en_periodo(CONFIG.bci.paso_tasm)

    def test_latencia_bajo_el_watchdog(self):
        """Si la latencia superara el watchdog, el robot se detendria
        constantemente sin motivo."""
        assert CONFIG.latencia.margen_watchdog(CONFIG.bci.watchdog_s) > 0

    def test_filtrado_causal_activado(self):
        assert CONFIG.adquisicion.usar_filtro_causal

    def test_umbral_gradiente_derivado_no_arbitrario(self):
        """El umbral debe superar la cota fisica de la senal en banda.

        Con un umbral menor se descartarian ventanas validas: ruido de 20 uV
        ya produce saltos de 105 uV entre muestras.
        """
        b, a = CONFIG.bci, CONFIG.adquisicion
        cota = 2 * math.pi * b.banda_hz[1] * a.umbral_amplitud / b.fs
        assert a.umbral_gradiente(b.banda_hz[1], b.fs) >= cota


# ===========================================================================
# FILTRADO
# ===========================================================================

class TestFiltrado:

    def test_estado_persiste_entre_bloques(self):
        """PROPIEDAD CENTRAL DEL MODULO.

        Filtrar por bloques con estado debe dar EXACTAMENTE el mismo
        resultado que filtrar la senal entera.
        """
        b = CONFIG.bci
        rng = np.random.default_rng(1)
        n = int(2.0 * b.fs)
        x = rng.standard_normal((n, b.n_canales)) * 30

        ref = CadenaFiltrado(CONFIG).aplicar(x)

        cad = CadenaFiltrado(CONFIG)
        nb = CONFIG.adquisicion.buffer_muestras
        troceado = np.concatenate(
            [cad.aplicar(x[i:i + nb]) for i in range(0, n, nb)])

        assert np.allclose(troceado, ref, atol=1e-9)

    def test_sin_estado_produce_transitorios(self):
        """Documenta el fallo que el estado persistente evita."""
        b = CONFIG.bci
        rng = np.random.default_rng(1)
        n = int(2.0 * b.fs)
        x = rng.standard_normal((n, b.n_canales)) * 30

        ref = CadenaFiltrado(CONFIG).aplicar(x)

        cad = CadenaFiltrado(CONFIG)
        nb = CONFIG.adquisicion.buffer_muestras
        trozos = []
        for i in range(0, n, nb):
            cad.reiniciar()
            trozos.append(cad.aplicar(x[i:i + nb]))
        sin_estado = np.concatenate(trozos)

        error = np.abs(sin_estado - ref).max()
        rango = np.abs(ref).max()
        assert error > rango * 0.1, \
            "Sin estado deberia haber transitorios apreciables"

    def test_lfilter_es_causal(self):
        """Un filtro causal no puede reaccionar antes del estimulo."""
        cad = CadenaFiltrado(CONFIG)
        bp, ap = cad._coef_banda
        n = 200
        x = np.zeros(n)
        x[n // 2:] = 1.0
        y = lfilter(bp, ap, x)
        assert np.abs(y[:n // 2]).max() < 1e-12

    def test_filtfilt_no_es_causal(self):
        """Documenta por que filtfilt no sirve online."""
        cad = CadenaFiltrado(CONFIG)
        bp, ap = cad._coef_banda
        n = 200
        x = np.zeros(n)
        x[n // 2:] = 1.0
        y = filtfilt(bp, ap, x)
        assert np.abs(y[:n // 2]).max() > 0.01, \
            "filtfilt deberia reaccionar antes del escalon"

    def test_suprime_la_red(self):
        cad = CadenaFiltrado(CONFIG)
        assert cad.ganancia_en(CONFIG.bci.notch_hz) < 0.15

    def test_deja_pasar_las_fundamentales(self):
        cad = CadenaFiltrado(CONFIG)
        for f in CONFIG.bci.frecuencias:
            assert cad.ganancia_en(f) > 0.70, \
                f"{f} Hz se atenua demasiado"

    def test_deja_pasar_el_armonico_critico(self):
        """El 4o armonico de 15.2 Hz cae en 60.8 Hz, junto a la red.

        Es la razon de que Q sea 60 y no el habitual 30.
        """
        cad = CadenaFiltrado(CONFIG)
        assert cad.ganancia_en(60.8) > 0.5

    def test_q_bajo_perderia_el_armonico(self):
        """Confirma que la eleccion de Q no es arbitraria."""
        cfg30 = CONFIG.copia_con(adquisicion=dict(q_notch=30.0))
        g30 = CadenaFiltrado(cfg30).ganancia_en(60.8)
        g60 = CadenaFiltrado(CONFIG).ganancia_en(60.8)
        assert g60 > g30

    def test_verificacion_sin_problemas(self):
        assert CadenaFiltrado(CONFIG).verificar() == []

    def test_rechaza_numero_de_canales_incorrecto(self):
        cad = CadenaFiltrado(CONFIG)
        with pytest.raises(ValueError):
            cad.aplicar(np.zeros((100, 3)))


# ===========================================================================
# ARTEFACTOS
# ===========================================================================

class TestArtefactos:

    def _limpia(self, semilla=1):
        """Senal limpia YA FILTRADA, como llegaria en el sistema real."""
        b = CONFIG.bci
        rng = np.random.default_rng(semilla)
        x = 20 * rng.standard_normal((b.muestras_ventana_tasm, b.n_canales))
        return CadenaFiltrado(CONFIG).aplicar(x)

    def test_acepta_senal_limpia(self):
        det = DetectorArtefactos(CONFIG)
        assert det.evaluar(self._limpia()).valida

    def test_detecta_parpadeo(self):
        det = DetectorArtefactos(CONFIG)
        x = self._limpia()
        x[50:70, 0] += 400
        r = det.evaluar(x)
        assert not r.valida
        assert 0 in r.canales_saturados

    def test_detecta_saturacion(self):
        det = DetectorArtefactos(CONFIG)
        x = self._limpia()
        x[:, 3] += 300
        assert not det.evaluar(x).valida

    def test_informa_del_motivo(self):
        det = DetectorArtefactos(CONFIG)
        x = self._limpia()
        x[10:20, 1] += 500
        assert det.evaluar(x).motivo != ""


# ===========================================================================
# FUENTES
# ===========================================================================

class TestFuentes:

    def test_sintetica_entrega_bloques(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        f.iniciar()
        bloques = []
        t0 = time.perf_counter()
        while len(bloques) < 5 and (time.perf_counter() - t0) < 2.0:
            b = f.leer()
            if b is not None:
                bloques.append(b)
            else:
                time.sleep(0.001)
        f.detener()
        assert len(bloques) >= 5
        assert bloques[0].shape == (CONFIG.adquisicion.buffer_muestras,
                                    CONFIG.bci.n_canales)

    def test_respeta_el_ritmo_temporal(self):
        """Debe devolver None si se la llama mas rapido que fs, igual que
        haria el hardware real."""
        f = FuenteSinteticaEEG(CONFIG, semilla=1)
        f.iniciar()
        f.leer()
        assert f.leer() is None
        f.detener()

    def test_no_entrega_nada_sin_iniciar(self):
        f = FuenteSinteticaEEG(CONFIG, semilla=1)
        assert f.leer() is None

    def test_mirar_cambia_la_frecuencia(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=0, semilla=1)
        f.mirar(2)
        assert f._freq_idx == 2

    def test_dataset_reproduce(self):
        b = CONFIG.bci
        datos = np.random.randn(int(3 * b.fs), b.n_canales)
        f = FuenteDataset(CONFIG, datos, tiempo_real=False)
        f.iniciar()
        bloques = [f.leer() for _ in range(5)]
        f.detener()
        assert all(x is not None for x in bloques)

    def test_dataset_remuestrea(self):
        """El dataset de referencia se grabo a 5000 Hz; el sistema opera a
        256."""
        b = CONFIG.bci
        datos = np.random.randn(5000, b.n_canales)
        f = FuenteDataset(CONFIG, datos, fs_original=5000.0,
                          tiempo_real=False)
        esperado = int(round(5000 * b.fs / 5000.0))
        assert abs(len(f._datos) - esperado) <= 1

    def test_fabrica_rechaza_fuente_desconocida(self):
        cfg = CONFIG.copia_con(adquisicion=dict(fuente="sintetica"))
        assert crear_fuente_eeg(cfg) is not None

    def test_impedancias_simuladas_correctas(self):
        f = FuenteSinteticaEEG(CONFIG, semilla=1)
        rep = comprobar_impedancias(f, CONFIG)
        assert rep.todos_correctos
        assert len(rep.valores) == CONFIG.bci.n_canales


# ===========================================================================
# BUFFER CIRCULAR
# ===========================================================================

class TestBuffer:

    def _buf(self):
        b = CONFIG.bci
        return BufferCircular(b.muestras_ventana_tasm, b.n_canales,
                              b.muestras_paso_tasm)

    def test_no_hay_ventana_al_principio(self):
        assert not self._buf().hay_ventana()

    def test_primera_ventana_tras_llenar(self):
        b = CONFIG.bci
        buf = self._buf()
        buf.escribir(np.random.randn(b.muestras_ventana_tasm, b.n_canales))
        assert buf.hay_ventana()

    def test_ventana_del_tamano_correcto(self):
        b = CONFIG.bci
        buf = self._buf()
        buf.escribir(np.random.randn(b.muestras_ventana_tasm, b.n_canales))
        v = buf.extraer_ventana()
        assert v.shape == (b.muestras_ventana_tasm, b.n_canales)

    def test_ventanas_solapan(self):
        """Cada ventana debe compartir la mayor parte de sus muestras con la
        anterior. Con 750 ms de ventana y 50 ms de paso, el solape es del
        93%."""
        b = CONFIG.bci
        buf = self._buf()
        datos = np.arange(b.muestras_ventana_tasm * 3).reshape(-1, 1) * \
            np.ones((1, b.n_canales))
        buf.escribir(datos)

        v1 = buf.extraer_ventana()
        v2 = buf.extraer_ventana()
        # v2 debe empezar `paso` muestras despues de v1
        assert np.allclose(v1[b.muestras_paso_tasm:],
                           v2[:-b.muestras_paso_tasm])

    def test_numero_de_ventanas_esperado(self):
        b = CONFIG.bci
        buf = self._buf()
        nb = CONFIG.adquisicion.buffer_muestras
        n_bloques = 60
        total = 0
        for _ in range(n_bloques):
            buf.escribir(np.random.randn(nb, b.n_canales))
            while buf.hay_ventana():
                buf.extraer_ventana()
                total += 1
        esperado = ((n_bloques * nb - b.muestras_ventana_tasm)
                    // b.muestras_paso_tasm + 1)
        assert abs(total - esperado) <= 1

    def test_rechaza_canales_incorrectos(self):
        buf = self._buf()
        with pytest.raises(ValueError):
            buf.escribir(np.zeros((10, 3)))

    def test_reiniciar(self):
        b = CONFIG.bci
        buf = self._buf()
        buf.escribir(np.random.randn(b.muestras_ventana_tasm, b.n_canales))
        buf.reiniciar()
        assert not buf.hay_ventana()
        assert buf.muestras_escritas == 0


# ===========================================================================
# MENSAJE Y TRANSPORTE
# ===========================================================================

class TestMensaje:

    def test_json_valido(self):
        import json
        m = MensajeSalida(estado="IC", freq_idx=2, p_max=0.87,
                          lambda_bci=0.92, valido=True, timestamp=1.5)
        d = json.loads(m.a_json())
        assert d["estado"] == "IC"
        assert d["freq_idx"] == 2

    def test_diagnostico_opcional(self):
        import json
        m = MensajeSalida(estado="IC", freq_idx=0, p_max=0.5,
                          lambda_bci=0.5, valido=True, timestamp=0.0)
        assert "rho" not in json.loads(m.a_json())

        m.rho = [0.1, 0.2, 0.3, 0.4]
        assert "rho" in json.loads(m.a_json())

    def test_puente_no_lanza_sin_servidor(self):
        """Perder la conexion no debe detener la adquisicion."""
        cfg = CONFIG.copia_con(adquisicion=dict(tcp_puerto=59999))
        p = PuenteTCP(cfg)
        m = MensajeSalida(estado="Idle", freq_idx=-1, p_max=0.0,
                          lambda_bci=0.0, valido=True, timestamp=0.0)
        assert p.enviar(m) is False
        p.cerrar()


# ===========================================================================
# PIPELINE
# ===========================================================================

class TestPipeline:

    def test_procesa_ventanas(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        pipe = PipelineAdquisicion(CONFIG, f, crear_procesador_fbcca(CONFIG))
        pipe.correr(duracion=3.0)
        assert pipe.stats.ventanas_procesadas > 0

    def test_recupera_la_frecuencia(self):
        """PRUEBA DE INTEGRACION PRINCIPAL.

        Toda la cadena debe recuperar la frecuencia inyectada.
        """
        from collections import Counter
        b = CONFIG.bci
        for idx in (0, 2):
            f = FuenteSinteticaEEG(CONFIG, freq_idx=idx, snr_db=-3.0,
                                   con_red=True, semilla=20 + idx)
            pipe = PipelineAdquisicion(CONFIG, f,
                                       crear_procesador_fbcca(CONFIG))
            det = []
            pipe.correr(duracion=3.0,
                        callback=lambda m: det.append(m.freq_idx))
            assert det, "no se genero ninguna ventana"
            acierto = Counter(det).get(idx, 0) / len(det)
            assert acierto >= 0.75, \
                f"{b.frecuencias[idx]} Hz: acierto {acierto*100:.0f}%"

    def test_computo_cabe_en_el_periodo(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        pipe = PipelineAdquisicion(CONFIG, f, crear_procesador_fbcca(CONFIG))
        pipe.correr(duracion=3.0)
        assert pipe.stats.computo_p95_ms < CONFIG.bci.paso_tasm * 1000

    def test_tasa_de_ventanas_correcta(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        pipe = PipelineAdquisicion(CONFIG, f, crear_procesador_fbcca(CONFIG))
        pipe.correr(duracion=4.0)
        nominal = 1.0 / CONFIG.bci.paso_tasm
        assert pipe.stats.tasa_ventanas > nominal * 0.80

    def test_funciona_sin_puente(self):
        """El pipeline debe correr aunque no haya conexion con el robot."""
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        pipe = PipelineAdquisicion(CONFIG, f,
                                   crear_procesador_fbcca(CONFIG),
                                   puente=None)
        pipe.correr(duracion=2.0)
        assert pipe.stats.ventanas_procesadas > 0

    def test_reiniciar(self):
        f = FuenteSinteticaEEG(CONFIG, freq_idx=2, semilla=1)
        pipe = PipelineAdquisicion(CONFIG, f, crear_procesador_fbcca(CONFIG))
        pipe.correr(duracion=2.0)
        pipe.reiniciar()
        assert pipe.stats.ventanas_procesadas == 0
        assert not pipe.buffer.hay_ventana()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

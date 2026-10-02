"""
tests/test_integracion.py --- Bateria de tests del Bloque 6
Bloque 6: integracion_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Estos tests cubren la interfaz, la fabrica, la verificacion de contrato, la
regresion y los experimentos, usando el mock.

NO cubren el adaptador a TASM real, que es un esqueleto pendiente de
completar.

Si un test falla, reportalo indicando QUE TEST, el comando exacto, el
traceback completo y que intentaste.
===========================================================================

EL TEST MAS IMPORTANTE DE ESTE BLOQUE
--------------------------------------
TestVerificadorContrato::test_detecta_* comprueba que el verificador
encuentra los desajustes tipicos.

Importa porque esos desajustes NO dan error en ejecucion: producen un
sistema que corre, da numeros plausibles, y ninguno significa lo que uno
cree. Si el verificador dejara de detectarlos, el bloque perderia su razon
de ser.
"""

import sys
import os
import math
import json
import tempfile
import numpy as np
import pytest

_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _raiz)

from config import CONFIG
import logica  # noqa: F401
from command_fsm import EstadoTASM, EstadoFSM, CommandFSM

from tasm_interface import (SalidaTASM, FuenteTASM, FuenteMock,
                            crear_fuente_tasm, VerificadorContrato)
from experimentos import (RegistroTrial, RegistradorExperimento,
                          TestRegresion, Experimento,
                          analizar_experimento_A, analizar_experimento_B)


# ===========================================================================
# UTILIDADES
# ===========================================================================

class FuenteFija(FuenteTASM):
    """Fuente que devuelve siempre lo mismo. Para probar el verificador."""

    def __init__(self, salida_fn):
        self._fn = salida_fn
        self.i = 0

    def siguiente(self, ventana=None):
        self.i += 1
        return self._fn(self.i)

    def reiniciar(self):
        self.i = 0

    @property
    def nombre(self):
        return "fija"


def salida_valida(i=0):
    return SalidaTASM(EstadoTASM.IC, 2, 0.9, 0.9, True)


# ===========================================================================
# CONFIGURACION
# ===========================================================================

class TestConfigTASM:

    def test_fuente_valida(self):
        assert CONFIG.tasm.fuente in CONFIG.tasm.FUENTES_VALIDAS

    def test_verificacion_activada(self):
        """NO desactivar: un desajuste de contrato produce fallos
        silenciosos."""
        assert CONFIG.tasm.verificar_contrato

    def test_transiciones_inducidas_activadas(self):
        """Sin segmentos de TR etiquetados no se puede entrenar la clase
        central del trabajo."""
        assert CONFIG.tasm.inducir_transiciones

    def test_propiedades_de_fuente(self):
        cfg_m = CONFIG.copia_con(tasm=dict(fuente="mock"))
        assert cfg_m.tasm.usa_mock
        assert not cfg_m.tasm.usa_tasm_real

        cfg_r = CONFIG.copia_con(tasm=dict(fuente="real"))
        assert not cfg_r.tasm.usa_mock
        assert cfg_r.tasm.usa_tasm_real

    def test_avisa_de_mock(self):
        """La validacion debe avisar de que mock no vale para experimentos formales."""
        probs = CONFIG.validar()
        assert any("mock" in p.lower() for p in probs)


# ===========================================================================
# FABRICA
# ===========================================================================

class TestFabrica:

    def test_crea_mock(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        assert isinstance(f, FuenteMock)

    def test_rechaza_fuente_desconocida(self):
        cfg = CONFIG.copia_con(tasm=dict(fuente="mock"))
        object.__setattr__(cfg.tasm, "fuente", "inventada")
        with pytest.raises(ValueError):
            crear_fuente_tasm(cfg)

    def test_real_lanza_no_implementado(self):
        """El esqueleto debe fallar de forma explicita, no en silencio."""
        from tasm_interface import FuenteTASMReal
        cfg = CONFIG.copia_con(tasm=dict(fuente="real"))
        with pytest.raises(NotImplementedError):
            FuenteTASMReal(cfg)


# ===========================================================================
# CONTRATO
# ===========================================================================

class TestContrato:

    def test_mock_respeta_el_contrato(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        r = VerificadorContrato(CONFIG).verificar(f, n_muestras=500)
        assert r.correcto, f"problemas: {r.problemas}"

    def test_mock_produce_las_tres_clases(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        r = VerificadorContrato(CONFIG).verificar(f, n_muestras=800)
        assert r.estadisticas["IC"] > 0
        assert r.estadisticas["TR"] > 0
        assert r.estadisticas["Idle"] > 0

    def test_mock_tiene_ground_truth(self):
        """Es lo que permite calcular N_FP_TR."""
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        assert f.tiene_ground_truth

    def test_estado_es_enumerado(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        assert isinstance(f.siguiente().estado, EstadoTASM)

    def test_freq_idx_menos_uno_si_no_es_ic(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=3)
        for _ in range(500):
            s = f.siguiente()
            if s.estado != EstadoTASM.IC:
                assert s.freq_idx == -1


# ===========================================================================
# VERIFICADOR
# ===========================================================================

class TestVerificadorContrato:
    """
    LOS TESTS MAS IMPORTANTES DEL BLOQUE.

    Cada uno simula un desajuste que NO daria error en ejecucion pero
    produciria un sistema que miente.
    """

    def _ver(self):
        return VerificadorContrato(CONFIG)

    def test_detecta_indice_base_1(self):
        """Si TASM usa base 1, el indice apunta a la frecuencia equivocada."""
        n = len(CONFIG.bci.frecuencias)
        f = FuenteFija(lambda i: SalidaTASM(EstadoTASM.IC, n, 0.9, 0.9, True))
        assert not self._ver().verificar(f, n_muestras=50).correcto

    def test_detecta_lambda_en_porcentaje(self):
        f = FuenteFija(
            lambda i: SalidaTASM(EstadoTASM.IC, 2, 0.9, 95.0, True))
        assert not self._ver().verificar(f, n_muestras=50).correcto

    def test_detecta_p_max_fuera_de_rango(self):
        f = FuenteFija(
            lambda i: SalidaTASM(EstadoTASM.IC, 2, 1.5, 0.9, True))
        assert not self._ver().verificar(f, n_muestras=50).correcto

    def test_detecta_ausencia_de_tr(self):
        """Si el modelo nunca reporta TR, la clase central del trabajo no
        existe."""
        f = FuenteFija(salida_valida)
        r = self._ver().verificar(f, n_muestras=200)
        assert not r.correcto
        assert any("TR" in p for p in r.problemas)

    def test_detecta_ausencia_de_ic(self):
        f = FuenteFija(
            lambda i: SalidaTASM(EstadoTASM.TR, -1, 0.3, 0.2, True))
        r = self._ver().verificar(f, n_muestras=200)
        assert not r.correcto

    def test_detecta_rho_de_longitud_incorrecta(self):
        f = FuenteFija(
            lambda i: SalidaTASM(EstadoTASM.IC, 2, 0.9, 0.9, True,
                                 rho=np.zeros(2)))
        assert not self._ver().verificar(f, n_muestras=50).correcto

    def test_detecta_valido_no_booleano(self):
        f = FuenteFija(
            lambda i: SalidaTASM(EstadoTASM.IC, 2, 0.9, 0.9, 1))
        assert not self._ver().verificar(f, n_muestras=50).correcto

    def test_detecta_excepcion(self):
        def rompe(i):
            raise RuntimeError("fallo del adaptador")
        f = FuenteFija(rompe)
        r = self._ver().verificar(f, n_muestras=50)
        assert not r.correcto
        assert any("Excepcion" in p for p in r.problemas)

    def test_avisa_de_lambda_constante(self):
        """En la Fase 2 lambda_bci se usa como ganancia continua;
        un valor casi constante la haria inutil."""
        def gen(i):
            return SalidaTASM(
                EstadoTASM.IC if i % 3 else EstadoTASM.TR,
                2 if i % 3 else -1, 0.9, 0.500, True)
        r = self._ver().verificar(FuenteFija(gen), n_muestras=200)
        assert any("lambda" in a.lower() for a in r.avisos)

    def test_acepta_fuente_correcta(self):
        def gen(i):
            m = i % 10
            if m < 6:
                return SalidaTASM(EstadoTASM.IC, i % 4, 0.8, 0.9, True)
            if m < 8:
                return SalidaTASM(EstadoTASM.TR, -1, 0.3, 0.3, True)
            return SalidaTASM(EstadoTASM.IDLE, -1, 0.2, 0.1, True)
        r = self._ver().verificar(FuenteFija(gen), n_muestras=300)
        assert r.correcto, f"problemas: {r.problemas}"


# ===========================================================================
# REGISTRO
# ===========================================================================

class TestRegistro:

    def _registro(self, **kw):
        base = dict(
            sujeto="S01", experimento="A", condicion="tasm",
            escenario="Escenario 1", n_objetos=4, trial=0,
            fuente_tasm="mock", exito=True, tiempo_total=45.0,
            n_fp_tr=2, fpr_tr=0.09, t_tr=3.5, n_transiciones=8,
            n_forzadas=0)
        base.update(kw)
        return RegistroTrial(**base)

    def test_escribe_y_relee(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "s.jsonl")
            reg = RegistradorExperimento(ruta, "S01", CONFIG)
            for i in range(5):
                reg.registrar(self._registro(trial=i))
            datos = RegistradorExperimento.cargar(ruta)
            assert len(datos["trials"]) == 5
            assert datos["cabecera"]["sujeto"] == "S01"

    def test_cabecera_guarda_la_configuracion(self):
        """Sin esto, dentro de seis meses no se sabra con que parametros se
        corrio."""
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "s.jsonl")
            RegistradorExperimento(ruta, "S01", CONFIG)
            c = RegistradorExperimento.cargar(ruta)["cabecera"]
            assert c["n_conf"] == CONFIG.bci.n_conf
            assert c["fuente_tasm"] == CONFIG.tasm.fuente
            assert c["frecuencias"] == list(CONFIG.bci.frecuencias)

    def test_escribe_incrementalmente(self):
        """Si el programa se cae en el trial 30, los 29 anteriores deben
        estar a salvo."""
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "s.jsonl")
            reg = RegistradorExperimento(ruta, "S01", CONFIG)
            reg.registrar(self._registro(trial=0))
            assert len(RegistradorExperimento.cargar(ruta)["trials"]) == 1
            reg.registrar(self._registro(trial=1))
            assert len(RegistradorExperimento.cargar(ruta)["trials"]) == 2

    def test_marca_de_tiempo(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "s.jsonl")
            reg = RegistradorExperimento(ruta, "S01", CONFIG)
            r = self._registro()
            reg.registrar(r)
            assert r.marca_tiempo != ""


# ===========================================================================
# REGRESION
# ===========================================================================

class TestRegresionSistema:

    def test_propiedades_con_mock(self):
        f = FuenteMock(CONFIG, objetivo=2, modo="tasm", semilla=1)
        problemas = TestRegresion(CONFIG)._propiedades(f)
        assert problemas == [], f"propiedades violadas: {problemas}"

    def test_detecta_fuente_sin_tr(self):
        """Una fuente que nunca reporta TR debe hacer fallar la regresion."""
        f = FuenteFija(salida_valida)
        problemas = TestRegresion(CONFIG)._propiedades(f)
        assert any("TR" in p for p in problemas)

    def test_comparacion_produce_metricas(self):
        a = FuenteMock(CONFIG, objetivo=2, modo="tasm", semilla=1)
        b = FuenteMock(CONFIG, objetivo=2, modo="binary", semilla=2)
        r = TestRegresion(CONFIG).comparar(a, b, n_ventanas=500)
        assert "fpr_tr" in r.metricas_mock
        assert "fpr_tr" in r.metricas_real

    def test_binario_tiene_peor_fpr(self):
        """Es la diferencia que el experimento mide."""
        a = FuenteMock(CONFIG, objetivo=2, modo="tasm", semilla=5)
        b = FuenteMock(CONFIG, objetivo=2, modo="binary", semilla=5)
        r = TestRegresion(CONFIG).comparar(a, b, n_ventanas=2000)
        assert r.metricas_real["fpr_tr"] > r.metricas_mock["fpr_tr"]


# ===========================================================================
# EXPERIMENTOS
# ===========================================================================

class TestExperimentos:

    def test_experimento_A_corre(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T01", ruta_salida=d)
            trials = exp.experimento_A(n_trials=1)
            assert len(trials) > 0
            assert {t.condicion for t in trials} == {"tasm", "binary"}

    def test_experimento_B_corre(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T02", ruta_salida=d)
            trials = exp.experimento_B(n_trials=1)
            assert len(trials) == len(CONFIG.etapa2.niveles_n)
            assert {t.condicion for t in trials} == {"tasm"}

    def test_experimento_B_registra_las_decisiones_teoricas(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T03", ruta_salida=d)
            for t in exp.experimento_B(n_trials=1):
                assert t.n_decisiones_teoricas == \
                    math.ceil(math.log2(t.n_objetos))

    def test_decisiones_coinciden_con_la_teoria(self):
        """VERIFICACION CENTRAL DE LA ETAPA 2.

        Cuando hay seleccion, el numero de decisiones debe ser exactamente
        ceil(log2 N).
        """
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T04", ruta_salida=d)
            for t in exp.experimento_B(n_trials=2):
                if t.n_decisiones > 0:
                    assert t.n_decisiones == t.n_decisiones_teoricas, \
                        f"N={t.n_objetos}: {t.n_decisiones} != " \
                        f"{t.n_decisiones_teoricas}"

    def test_los_experimentos_no_se_cruzan(self):
        """A varia escenario con N fijo; B varia N con escenario fijo.

        Cruzarlos daria 90 trials por sujeto, inviable por fatiga.
        """
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T05", ruta_salida=d)
            tA = exp.experimento_A(n_trials=1)
            assert len({t.n_objetos for t in tA}) == 1

            exp2 = Experimento(CONFIG, "T06", ruta_salida=d)
            tB = exp2.experimento_B(n_trials=1)
            assert len({t.escenario for t in tB}) == 1

    def test_analisis_no_rompe(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Experimento(CONFIG, "T07", ruta_salida=d)
            assert isinstance(
                analizar_experimento_A(exp.experimento_A(n_trials=1)), str)
            exp2 = Experimento(CONFIG, "T08", ruta_salida=d)
            assert isinstance(
                analizar_experimento_B(exp2.experimento_B(n_trials=1)), str)


# ===========================================================================
# INTEGRACION
# ===========================================================================

class TestIntegracionCompleta:

    def test_fuente_alimenta_la_fsm(self):
        f = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        fsm = CommandFSM(CONFIG)
        t = 0.0
        transiciones = 0
        for _ in range(2000):
            s = f.siguiente()
            if fsm.actualizar(s.estado, s.freq_idx, t, s.valido).transiciono:
                transiciones += 1
            t += CONFIG.bci.paso_tasm
        assert transiciones > 0, "la FSM nunca transiciono"

    def test_cambiar_de_fuente_no_rompe_nada(self):
        """LA PROPIEDAD QUE JUSTIFICA TODO EL BLOQUE.

        El resto del sistema debe funcionar igual con cualquier fuente.
        """
        for modo in ("tasm", "binary"):
            f = FuenteMock(CONFIG, objetivo=2, modo=modo, semilla=1)
            fsm = CommandFSM(CONFIG)
            t = 0.0
            for _ in range(500):
                s = f.siguiente()
                fsm.actualizar(s.estado, s.freq_idx, t, s.valido)
                t += CONFIG.bci.paso_tasm
            assert fsm.estado in EstadoFSM

    def test_misma_semilla_misma_secuencia(self):
        """REPRODUCIBILIDAD DEL EXPERIMENTO.

        Dos ejecuciones del programa con la misma semilla deben dar la misma
        secuencia. Es lo que permite reproducir un resultado.
        """
        a = [FuenteMock(CONFIG, objetivo=2, semilla=7).siguiente().estado
             for _ in range(1)]
        f1 = FuenteMock(CONFIG, objetivo=2, semilla=7)
        f2 = FuenteMock(CONFIG, objetivo=2, semilla=7)
        s1 = [f1.siguiente().estado for _ in range(200)]
        s2 = [f2.siguiente().estado for _ in range(200)]
        assert s1 == s2

    def test_reiniciar_no_repite_la_secuencia(self):
        """COMPORTAMIENTO DELIBERADO, no un fallo.

        reiniciar() se llama ENTRE TRIALS y reinicia el guion de estados,
        pero NO el generador aleatorio.

        Es lo correcto: dos trials consecutivos del mismo sujeto no deben
        tener exactamente la misma secuencia de estados, igual que una
        persona no repite el mismo patron de miradas.

        La reproducibilidad del experimento la garantiza la semilla al
        construir, no el reinicio.
        """
        f = FuenteMock(CONFIG, objetivo=2, semilla=7)
        a = [f.siguiente().estado for _ in range(200)]
        f.reiniciar()
        b = [f.siguiente().estado for _ in range(200)]
        assert a != b, "reiniciar deberia dar una secuencia distinta"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

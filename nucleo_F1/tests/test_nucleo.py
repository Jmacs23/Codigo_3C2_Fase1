"""
tests/test_nucleo.py --- Bateria de tests del Bloque 1
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

COMO USAR ESTOS TESTS
---------------------
    python run.py test
    o bien
    python -m pytest tests/ -v

CUANDO UN TEST FALLA
--------------------
No lo silencies ni cambies el criterio para que pase. Cada test comprueba
una propiedad que el sistema DEBE cumplir. Si falla, o bien el codigo tiene
un error, o bien la configuracion es incoherente.

Al reportar un fallo, indica QUE TEST falla y pega la salida completa de
pytest. Eso convierte una conversacion difusa en un dato.
"""

import sys
import os
import math
import pytest

# Permite importar los modulos del bloque estando en tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CONFIG, Config
from command_fsm import (CommandFSM, EstadoTASM, EstadoFSM,
                         MotivoNoTransicion)
from tasm_mock import TASMMock
from fbcca import FBCCA, generar_ssvep
from binary_search import (BusquedaBinaria, LadoSeleccion,
                           generar_objetos_en_fila, Objeto)
from metrics import (CalculadorMetricas, RegistroCiclo, comparar_pareado)


# ===========================================================================
# CONFIGURACION
# ===========================================================================

class TestConfig:
    """La configuracion es la fuente unica de verdad y debe ser coherente."""

    def test_validacion_sin_errores_criticos(self):
        """La configuracion por defecto no debe tener errores criticos.

        Los avisos de diseno (que empiezan por 'AVISO') se toleran: senalan
        compromisos, no errores.
        """
        problemas = CONFIG.validar()
        criticos = [p for p in problemas if not p.startswith("AVISO")]
        assert criticos == [], f"Errores criticos: {criticos}"

    def test_es_inmutable(self):
        """No debe poder modificarse en caliente.

        Si esto fallara, un modulo podria cambiar un parametro global sin que
        nadie se entere, y el sistema dejaria de ser reproducible.
        """
        with pytest.raises(Exception):
            CONFIG.bci.fs = 512.0

    def test_copia_con_no_altera_el_original(self):
        n_original = CONFIG.bci.n_conf
        cfg2 = CONFIG.copia_con(bci=dict(n_conf=99))
        assert cfg2.bci.n_conf == 99
        assert CONFIG.bci.n_conf == n_original

    def test_copia_con_rechaza_parametros_inexistentes(self):
        with pytest.raises(ValueError):
            CONFIG.copia_con(bci=dict(no_existe=1))

    def test_frecuencias_en_grilla_benchmark(self):
        """Todas las frecuencias deben estar en la grilla 8.0-15.8 paso 0.2.

        Fuera de esa grilla, los componentes preentrenados con el dataset
        Benchmark no transfieren.
        """
        grilla = {round(8.0 + 0.2 * i, 1) for i in range(40)}
        for f in CONFIG.bci.frecuencias:
            assert round(f, 1) in grilla, f"{f} Hz fuera de la grilla"

    def test_sin_armonico_sobre_la_red(self):
        """Ningun armonico de orden <=4 debe caer sobre el notch."""
        for f in CONFIG.bci.frecuencias:
            for orden in range(1, 5):
                assert abs(f * orden - CONFIG.bci.notch_hz) > 0.15, \
                    f"El armonico {orden} de {f} Hz cae en el notch"

    def test_banda_cubre_cuarto_armonico(self):
        f_max = max(CONFIG.bci.frecuencias)
        assert f_max * 4 <= CONFIG.bci.banda_hz[1], \
            "La banda de paso no cubre el 4o armonico de la frecuencia mayor"

    def test_robot_cabe_girando_en_todos_los_escenarios(self):
        """El robot debe poder reorientarse en sitio en cualquier escenario.

        Es lo que justifica no incluir un comando de retroceso.
        """
        for esc in CONFIG.escenarios.lista:
            assert esc.gap > CONFIG.robot.diagonal, \
                f"'{esc.nombre}': gap {esc.gap} < diagonal " \
                f"{CONFIG.robot.diagonal:.3f}"

    def test_escenarios_navegables(self):
        """Tras descontar robot y margenes de seguridad debe quedar holgura."""
        for esc in CONFIG.escenarios.lista:
            m = esc.margen_tras_seguridad(CONFIG.robot.ancho,
                                          CONFIG.robot.rho_safe)
            assert m > 0, f"'{esc.nombre}' no es navegable (margen {m:.3f} m)"

    def test_niveles_n_son_potencias_de_dos(self):
        for n in CONFIG.etapa2.niveles_n:
            assert (n & (n - 1)) == 0, \
                f"N={n} no es potencia de dos: el arbol quedaria desbalanceado"

    def test_niveles_n_caben_en_el_umbral(self):
        e2 = CONFIG.etapa2
        for n in e2.niveles_n:
            d = e2.distancia_minima(n)
            assert d <= e2.umbral_transicion, \
                f"N={n} requiere {d:.2f} m > umbral {e2.umbral_transicion} m"

    def test_objetos_detectables(self):
        e2 = CONFIG.etapa2
        for n in e2.niveles_n:
            px = e2.pixeles_por_objeto(n)
            assert px >= e2.px_minimo_deteccion, \
                f"N={n}: {px:.0f} px < minimo {e2.px_minimo_deteccion}"


# ===========================================================================
# MAQUINA DE ESTADOS
# ===========================================================================

class TestFSM:
    """La FSM es donde vive el mecanismo central del trabajo."""

    def _avanzar(self, fsm, cfg, estado, idx, n, t0=0.0):
        """Aplica n ventanas y devuelve el tiempo final."""
        t = t0
        for _ in range(n):
            fsm.actualizar(estado, idx, t)
            t += cfg.bci.paso_tasm
        return t

    def test_arranca_detenido(self):
        fsm = CommandFSM(CONFIG)
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_enclava_tras_la_racha_completa(self):
        fsm = CommandFSM(CONFIG)
        self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf)
        assert fsm.estado == EstadoFSM.AVANZANDO

    def test_no_enclava_antes_de_la_racha(self):
        """Con una ventana menos NO debe haber transicionado.

        Este test verifica el limite exacto, que es donde suelen estar los
        errores de contador.
        """
        fsm = CommandFSM(CONFIG)
        self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf - 1)
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_tr_no_transiciona(self):
        """PROPIEDAD CENTRAL: durante TR la FSM nunca cambia de estado."""
        fsm = CommandFSM(CONFIG)
        self._avanzar(fsm, CONFIG, EstadoTASM.TR, -1, CONFIG.bci.n_conf * 3)
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_idle_no_transiciona(self):
        fsm = CommandFSM(CONFIG)
        self._avanzar(fsm, CONFIG, EstadoTASM.IDLE, -1, CONFIG.bci.n_conf * 3)
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_idle_mantiene_el_comando_enclavado(self):
        """Durante Idle el robot MANTIENE el comando, no se detiene.

        Bajo el paradigma de enclavamiento el usuario esta legitimamente en
        Idle la mayor parte del trial. Si el robot se detuviera, el sistema
        seria inutilizable.
        """
        fsm = CommandFSM(CONFIG)
        t = self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf)
        assert fsm.estado == EstadoFSM.AVANZANDO

        # Idle prolongado
        for _ in range(100):
            s = fsm.actualizar(EstadoTASM.IDLE, -1, t)
            t += CONFIG.bci.paso_tasm
            assert s.u == CONFIG.robot.u_max, \
                "El robot dejo de avanzar durante Idle"
        assert fsm.estado == EstadoFSM.AVANZANDO

    def test_racha_se_reinicia_si_cambia_la_frecuencia(self):
        """Alternar entre dos frecuencias no debe acumular racha.

        Es lo que impide que un usuario que salta entre estimulos sin fijar
        la mirada enclave un comando por acumulacion.
        """
        fsm = CommandFSM(CONFIG)
        t = 0.0
        for i in range(CONFIG.bci.n_conf * 4):
            fsm.actualizar(EstadoTASM.IC, i % 2, t)
            t += CONFIG.bci.paso_tasm
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_en_movimiento_solo_se_acepta_parar(self):
        fsm = CommandFSM(CONFIG)
        t = self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf)

        # Pide girar: debe ignorarse
        t = self._avanzar(fsm, CONFIG, EstadoTASM.IC, 0,
                          CONFIG.bci.n_conf * 2, t)
        assert fsm.estado == EstadoFSM.AVANZANDO

        # Pide parar: debe aceptarse
        self._avanzar(fsm, CONFIG, EstadoTASM.IC, 3, CONFIG.bci.n_conf, t)
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_velocidades_dentro_de_limites(self):
        fsm = CommandFSM(CONFIG)
        t = 0.0
        for idx in range(4):
            fsm.reiniciar()
            for _ in range(CONFIG.bci.n_conf + 2):
                s = fsm.actualizar(EstadoTASM.IC, idx, t)
                t += CONFIG.bci.paso_tasm
                assert abs(s.u) <= CONFIG.robot.u_max + 1e-9
                assert abs(s.omega) <= CONFIG.robot.omega_max + 1e-9

    def test_tope_de_traslacion(self):
        """Tras el tope de tiempo, el robot debe detenerse solo."""
        fsm = CommandFSM(CONFIG)
        t = self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf)
        assert fsm.estado == EstadoFSM.AVANZANDO

        limite = CONFIG.fsm.tope_traslacion_s + 1.0
        n = int(limite / CONFIG.bci.paso_tasm)
        for _ in range(n):
            fsm.actualizar(EstadoTASM.IDLE, -1, t)
            t += CONFIG.bci.paso_tasm
        assert fsm.estado == EstadoFSM.DETENIDO

    def test_reiniciar_deja_el_estado_inicial(self):
        fsm = CommandFSM(CONFIG)
        self._avanzar(fsm, CONFIG, EstadoTASM.IC, 2, CONFIG.bci.n_conf)
        fsm.reiniciar()
        assert fsm.estado == EstadoFSM.DETENIDO
        assert fsm.racha_ic == 0


# ===========================================================================
# MOCK
# ===========================================================================

class TestMock:

    def test_genera_los_tres_estados(self):
        mock = TASMMock(CONFIG, objetivo=2, semilla=1)
        for _ in range(2000):
            mock.siguiente()
        st = mock.estadisticas()
        assert st["ventanas_IC"] > 0
        assert st["ventanas_TR"] > 0
        assert st["ventanas_Idle"] > 0

    def test_respeta_la_restriccion_idle_a_ic(self):
        """El guion nunca debe pasar de Idle a IC sin TR en medio.

        Es la misma restriccion que el HMM de TASM impone, y tiene base
        fisica: para empezar a mirar un estimulo hay que mover los ojos.
        """
        mock = TASMMock(CONFIG, objetivo=2, semilla=3)
        previo = None
        for _ in range(3000):
            m = mock.siguiente()
            if previo == EstadoTASM.IDLE:
                assert m.estado_real != EstadoTASM.IC, \
                    "Transicion directa Idle -> IC en el guion"
            previo = m.estado_real

    def test_binario_nunca_reporta_tr(self):
        """En modo binario, TR debe colapsarse con Idle."""
        mock = TASMMock(CONFIG, objetivo=2, modo="binary", semilla=1)
        for _ in range(2000):
            m = mock.siguiente()
            assert m.estado != EstadoTASM.TR

    def test_tasm_tiene_menor_fpr_que_binario(self):
        """PROPIEDAD QUE EL EXPERIMENTO MIDE.

        TASM debe producir menos falsos IC durante TR que el baseline. Si
        esto fallara, no habria nada que medir.
        """
        res = {}
        for modo in ("tasm", "binary"):
            fp = tr = 0
            for s in range(10):
                mock = TASMMock(CONFIG, objetivo=2, modo=modo, semilla=s)
                for _ in range(2000):
                    m = mock.siguiente()
                    if m.estado_real == EstadoTASM.TR:
                        tr += 1
                        if m.estado == EstadoTASM.IC:
                            fp += 1
            res[modo] = fp / tr if tr else 0.0
        assert res["tasm"] < res["binary"], \
            f"FPR TASM ({res['tasm']:.3f}) no es menor que " \
            f"binario ({res['binary']:.3f})"

    def test_reproducible_con_la_misma_semilla(self):
        a = [m.estado for m in
             (TASMMock(CONFIG, 2, semilla=7).siguiente() for _ in range(100))]
        b = [m.estado for m in
             (TASMMock(CONFIG, 2, semilla=7).siguiente() for _ in range(100))]
        assert a == b


# ===========================================================================
# FBCCA
# ===========================================================================

class TestFBCCA:

    def test_recupera_la_frecuencia_inyectada(self):
        clf = FBCCA(CONFIG)
        for k in range(len(CONFIG.bci.frecuencias)):
            x = generar_ssvep(CONFIG, k, CONFIG.bci.tw_tasm,
                              snr_db=0.0, semilla=k)
            assert clf.clasificar(x).freq_idx == k

    def test_rho_tiene_la_dimension_correcta(self):
        clf = FBCCA(CONFIG)
        x = generar_ssvep(CONFIG, 0, CONFIG.bci.tw_tasm, semilla=1)
        r = clf.clasificar(x)
        assert len(r.rho) == len(CONFIG.bci.frecuencias)

    def test_p_max_en_rango(self):
        clf = FBCCA(CONFIG)
        for k in range(len(CONFIG.bci.frecuencias)):
            x = generar_ssvep(CONFIG, k, CONFIG.bci.tw_tasm, semilla=k)
            assert 0.0 <= clf.clasificar(x).p_max <= 1.0

    def test_gradiente_muestra_patron_cruzado(self):
        """Durante una transicion, una correlacion baja y otra sube.

        Es la firma que TASM detecta.
        """
        clf = FBCCA(CONFIG)
        x0 = generar_ssvep(CONFIG, 0, CONFIG.bci.tw_tasm, semilla=1)
        x2 = generar_ssvep(CONFIG, 2, CONFIG.bci.tw_tasm, semilla=2)
        g = clf.gradiente(clf.clasificar(x2).rho, clf.clasificar(x0).rho)
        assert g.min() < 0 < g.max(), "No hay patron cruzado en el gradiente"

    def test_ruido_puro_no_produce_correlacion_dominante(self):
        """Con ruido puro ninguna frecuencia debe destacar mucho."""
        clf = FBCCA(CONFIG)
        x = generar_ssvep(CONFIG, -1, CONFIG.bci.tw_tasm, semilla=99)
        r = clf.clasificar(x)
        assert r.p_max < 0.75, \
            f"p_max={r.p_max:.3f} demasiado alto para ruido puro"


# ===========================================================================
# SELECCION
# ===========================================================================

class TestSeleccion:

    def test_decisiones_exactamente_logaritmicas(self):
        """AFIRMACION CENTRAL DE LA ETAPA 2."""
        for n in CONFIG.etapa2.niveles_n:
            objs = generar_objetos_en_fila(CONFIG, n)
            teorico = math.ceil(math.log2(n))
            for obj in range(n):
                bb = BusquedaBinaria(CONFIG, objs)
                while not bb.terminada:
                    bb.votar(bb.lado_correcto(obj))
                r = bb.resultado(objetivo=obj)
                assert r.n_decisiones == teorico, \
                    f"N={n}, objetivo={obj}: {r.n_decisiones} != {teorico}"
                assert r.acierto

    def test_todos_los_objetivos_alcanzables(self):
        for n in CONFIG.etapa2.niveles_n:
            objs = generar_objetos_en_fila(CONFIG, n)
            for obj in range(n):
                bb = BusquedaBinaria(CONFIG, objs)
                while not bb.terminada:
                    bb.votar(bb.lado_correcto(obj))
                assert bb.resultado(objetivo=obj).objeto_elegido == obj

    def test_ordena_por_posicion_horizontal(self):
        desordenados = [
            Objeto(id_aruco=0, x_center=500.0),
            Objeto(id_aruco=1, x_center=100.0),
            Objeto(id_aruco=2, x_center=300.0),
            Objeto(id_aruco=3, x_center=700.0),
        ]
        bb = BusquedaBinaria(CONFIG, desordenados)
        xs = [o.x_center for o in bb.objetos]
        assert xs == sorted(xs)

    def test_rechaza_menos_de_dos_objetos(self):
        with pytest.raises(ValueError):
            BusquedaBinaria(CONFIG, [Objeto(id_aruco=0, x_center=0.0)])

    def test_timeout_no_avanza_la_busqueda(self):
        """Un timeout repite la iteracion, no avanza con un voto dudoso."""
        objs = generar_objetos_en_fila(CONFIG, 4)
        bb = BusquedaBinaria(CONFIG, objs)
        cand_antes = bb.n_candidatos
        bb.registrar_timeout(duracion=8.0)
        assert bb.n_candidatos == cand_antes

    def test_n_no_potencia_de_dos_da_decisiones_variables(self):
        """Documenta por que el experimento usa potencias de dos."""
        objs = generar_objetos_en_fila(CONFIG, 6)
        conteos = []
        for obj in range(6):
            bb = BusquedaBinaria(CONFIG, objs)
            while not bb.terminada:
                bb.votar(bb.lado_correcto(obj))
            conteos.append(bb.resultado(objetivo=obj).n_decisiones)
        assert len(set(conteos)) > 1, \
            "Con N=6 se esperaba numero de decisiones variable"
        assert max(conteos) <= math.ceil(math.log2(6))


# ===========================================================================
# METRICAS
# ===========================================================================

class TestMetricas:

    def _registro(self, real, transiciono=False, por_comando=True, t=0.0):
        return RegistroCiclo(
            t=t, estado_tasm_reportado=EstadoTASM.IC,
            estado_tasm_real=real, estado_fsm=EstadoFSM.DETENIDO,
            transiciono=transiciono, por_comando=por_comando,
            u=0.0, omega=0.0, lambda_bci=0.5, freq_idx=0,
        )

    def test_cuenta_transiciones_durante_tr(self):
        calc = CalculadorMetricas(CONFIG)
        regs = [
            self._registro(EstadoTASM.IC, t=0.0),
            self._registro(EstadoTASM.TR, transiciono=True, t=0.05),
            self._registro(EstadoTASM.IDLE, t=0.10),
        ]
        assert calc.calcular(regs).n_fp_tr == 1

    def test_no_cuenta_transiciones_forzadas(self):
        """Una transicion por salvaguarda NO es un comando espurio.

        Este test protege contra un error que enmascaraba por completo el
        efecto medido: las salvaguardas se disparan igual en ambas
        condiciones experimentales.
        """
        calc = CalculadorMetricas(CONFIG)
        regs = [
            self._registro(EstadoTASM.TR, transiciono=True,
                           por_comando=False, t=0.0),
        ]
        assert calc.calcular(regs).n_fp_tr == 0
        assert calc.calcular(regs).n_forzadas == 1

    def test_fpr_tr(self):
        calc = CalculadorMetricas(CONFIG)
        regs = []
        for i in range(10):
            r = self._registro(EstadoTASM.TR, t=i * 0.05)
            r.estado_tasm_reportado = (EstadoTASM.IC if i < 3
                                       else EstadoTASM.TR)
            regs.append(r)
        assert abs(calc.calcular(regs).fpr_tr - 0.3) < 1e-9

    def test_registros_vacios_no_rompen(self):
        calc = CalculadorMetricas(CONFIG)
        m = calc.calcular([])
        assert m.n_fp_tr == 0

    def test_comparacion_pareada(self):
        c = comparar_pareado([1.0, 2.0, 1.0], [3.0, 4.0, 5.0])
        assert c.n_pares == 3
        assert c.delta_absoluto > 0     # el baseline es peor
        assert c.media_tasm < c.media_binary

    def test_comparacion_rechaza_series_desiguales(self):
        with pytest.raises(ValueError):
            comparar_pareado([1.0, 2.0], [1.0])


# ===========================================================================
# INTEGRACION
# ===========================================================================

class TestIntegracion:

    def test_trial_completo_termina(self):
        from simulator2d import Simulador2D
        sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0], semilla=1)
        res = sim.correr(n_objetos=4, objetivo=2)
        assert res.metricas.tiempo_total > 0

    def test_trial_llega_a_la_zona(self):
        """El usuario simulado debe ser capaz de completar la navegacion.

        Si esto falla, revisa el usuario simulado o la dinamica de planta:
        probablemente el robot no esta alcanzando la velocidad comandada.
        """
        from simulator2d import Simulador2D
        exitos = 0
        for s in range(5):
            sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0], semilla=s)
            exitos += int(sim.correr(n_objetos=4, objetivo=2).llego_a_zona)
        assert exitos >= 3, f"Solo {exitos}/5 trials llegaron a la zona"

    def test_seleccion_usa_decisiones_teoricas(self):
        from simulator2d import Simulador2D
        for n in (2, 4):
            sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0], semilla=1)
            res = sim.correr(n_objetos=n, objetivo=0)
            if res.llego_a_zona and res.n_decisiones_e2 > 0:
                assert res.n_decisiones_e2 == math.ceil(math.log2(n))

    def test_sin_colisiones_en_escenario_facil(self):
        from simulator2d import Simulador2D
        for s in range(5):
            sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0], semilla=s)
            res = sim.correr(n_objetos=4, objetivo=2)
            assert res.metricas.n_colisiones == 0, \
                f"Colision en el escenario mas facil (semilla {s})"

    def test_ambos_modos_corren(self):
        from simulator2d import Simulador2D
        for modo in ("tasm", "binary"):
            sim = Simulador2D(CONFIG, CONFIG.escenarios.lista[0],
                              modo=modo, semilla=1)
            assert sim.correr(n_objetos=4, objetivo=2) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

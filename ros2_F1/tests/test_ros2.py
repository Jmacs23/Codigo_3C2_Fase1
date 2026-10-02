"""
tests/test_ros2.py --- Bateria de tests del Bloque 4
Bloque 4: ros2_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Estos tests cubren la LOGICA de decision, que es donde estan los errores
dificiles. NO cubren los nodos ROS2, porque ROS2 no estaba disponible en el
entorno donde se escribieron.

Si un test falla, reportalo indicando QUE TEST, el comando exacto, el
traceback completo y que intentaste.
===========================================================================

QUE HAY QUE PROBAR EN EL ROBOT, QUE ESTOS TESTS NO CUBREN
----------------------------------------------------------
  - Que los topics se publican con los nombres y tipos correctos
  - Que la frecuencia real de cada nodo es la esperada
  - Que la latencia entre las dos maquinas cabe en el presupuesto
  - Que el orden de arranque funciona (safety_node primero)
  - Que el signo del campo potencial es correcto sobre el robot real
"""

import sys
import os
import math
import pytest

_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _raiz)
sys.path.insert(0, os.path.join(_raiz, "logica"))

from config import CONFIG
from logica.nodos_logica import (
    LogicaSeguridad, LogicaNavegacion, LogicaControl, LogicaMision,
    ReceptorTASM, MensajeTASMRecibido, ModoOperacion, Etapa, Velocidad,
)


N_RAYOS = CONFIG.robot.lidar_n_rayos


def escaneo(rho_min=5.0, indice=0):
    """Escaneo LiDAR con un obstaculo en un rayo concreto."""
    d = [5.0] * N_RAYOS
    d[indice] = rho_min
    return d


# ===========================================================================
# CONFIGURACION ROS2
# ===========================================================================

class TestConfigROS2:

    def test_frecuencias_coherentes(self):
        probs = CONFIG.ros2.verificar_coherencia(
            CONFIG.bci.paso_tasm, CONFIG.robot.ts_control)
        assert probs == [], f"Incoherencias: {probs}"

    def test_bci_coincide_con_tasa_de_tasm(self):
        """Si no coincidieran se perderian mensajes o se procesarian
        repetidos."""
        tasa = 1.0 / CONFIG.bci.paso_tasm
        assert abs(CONFIG.ros2.hz_bci - tasa) < 0.5

    def test_seguridad_no_va_por_detras_del_control(self):
        """La seguridad nunca debe correr mas lento que el control."""
        assert CONFIG.ros2.hz_seguridad >= CONFIG.ros2.hz_control

    def test_cola_corta(self):
        """En control en tiempo real un mensaje viejo es peor que ninguno."""
        assert CONFIG.ros2.qos_profundidad <= 10


# ===========================================================================
# SEGURIDAD
# ===========================================================================

class TestSeguridad:

    def test_arranca_sin_emergencia(self):
        assert not LogicaSeguridad(CONFIG).en_emergencia

    def test_entra_bajo_el_umbral(self):
        s = LogicaSeguridad(CONFIG)
        e = s.evaluar(escaneo(CONFIG.robot.rho_safe - 0.01))
        assert e.emergencia

    def test_no_entra_sobre_el_umbral(self):
        s = LogicaSeguridad(CONFIG)
        e = s.evaluar(escaneo(CONFIG.robot.rho_safe + 0.10))
        assert not e.emergencia

    def test_histeresis_no_sale_de_inmediato(self):
        """PROPIEDAD CENTRAL.

        Sin histeresis, un obstaculo en el umbral haria oscilar el sistema
        varias veces por segundo.
        """
        s = LogicaSeguridad(CONFIG)
        s.evaluar(escaneo(0.10))
        assert s.en_emergencia
        e = s.evaluar(escaneo(5.0))
        assert e.emergencia, "salio en el primer ciclo"

    def test_sale_tras_los_ciclos_de_histeresis(self):
        s = LogicaSeguridad(CONFIG)
        s.evaluar(escaneo(0.10))
        for _ in range(CONFIG.robot.n_hist):
            e = s.evaluar(escaneo(5.0))
        assert not e.emergencia

    def test_no_sale_si_vuelve_a_acercarse(self):
        """El contador de ciclos seguros debe reiniciarse."""
        s = LogicaSeguridad(CONFIG)
        s.evaluar(escaneo(0.10))
        for _ in range(CONFIG.robot.n_hist - 1):
            s.evaluar(escaneo(5.0))
        s.evaluar(escaneo(0.10))            # vuelve a acercarse
        e = s.evaluar(escaneo(5.0))
        assert e.emergencia

    def test_sin_datos_asume_emergencia(self):
        """Opcion conservadora: mejor una emergencia espuria que una
        colision."""
        assert LogicaSeguridad(CONFIG).evaluar([]).emergencia

    def test_escape_retrocede(self):
        s = LogicaSeguridad(CONFIG)
        e = s.evaluar(escaneo(0.10))
        v = s.comando_escape(e)
        assert v.lineal < 0

    def test_ignora_lecturas_cero(self):
        """Un cero del LiDAR suele significar lectura invalida, no un
        obstaculo pegado al sensor."""
        s = LogicaSeguridad(CONFIG)
        d = [0.0] * N_RAYOS
        e = s.evaluar(d)
        assert not e.emergencia

    def test_no_depende_del_estado_cognitivo(self):
        """La clase no debe tener forma alguna de conocer el estado de TASM.

        Acoplarlas introduciria un modo de fallo: un error del detector
        podria desactivar la proteccion.
        """
        s = LogicaSeguridad(CONFIG)
        firma = s.evaluar.__code__.co_varnames
        assert not any("tasm" in v.lower() or "cognit" in v.lower()
                       for v in firma)


# ===========================================================================
# NAVEGACION
# ===========================================================================

class TestNavegacion:

    def test_no_toca_el_comando_lejos_de_obstaculos(self):
        nav = LogicaNavegacion(CONFIG)
        cmd = Velocidad(CONFIG.robot.u_max, 0.0)
        out, _ = nav.asistir(cmd, escaneo(5.0), 0.0)
        assert abs(out.lineal - cmd.lineal) < 1e-9

    def test_frena_al_acercarse(self):
        nav = LogicaNavegacion(CONFIG)
        cmd = Velocidad(CONFIG.robot.u_max, 0.0)
        lejos, _ = nav.asistir(cmd, escaneo(0.35), 0.0)
        cerca, _ = nav.asistir(cmd, escaneo(0.20), 0.1)
        assert cerca.lineal < lejos.lineal

    def test_signo_del_gradiente(self):
        """VERIFICACION CRITICA.

        Si el signo estuviera invertido, el robot se PEGARIA a las paredes
        en lugar de evitarlas. Es un error facil de cometer al derivar el
        gradiente del potencial.
        """
        nav = LogicaNavegacion(CONFIG)
        cmd = Velocidad(CONFIG.robot.u_max, 0.0)

        # Obstaculo a la izquierda (rayo a 90 grados) -> debe girar a la
        # derecha, es decir omega negativo
        out, _ = nav.asistir(cmd, escaneo(0.25, N_RAYOS // 4), 0.0)
        assert out.angular < 0

        nav.reiniciar()
        # Obstaculo a la derecha -> debe girar a la izquierda
        out, _ = nav.asistir(cmd, escaneo(0.25, 3 * N_RAYOS // 4), 0.0)
        assert out.angular > 0

    def test_respeta_los_limites(self):
        nav = LogicaNavegacion(CONFIG)
        cmd = Velocidad(10.0, 10.0)      # muy por encima de los limites
        out, _ = nav.asistir(cmd, escaneo(0.25), 0.0)
        assert abs(out.lineal) <= CONFIG.robot.u_max + 1e-9
        assert abs(out.angular) <= CONFIG.robot.omega_max + 1e-9

    def test_sin_lidar_no_modifica(self):
        nav = LogicaNavegacion(CONFIG)
        cmd = Velocidad(0.1, 0.2)
        out, _ = nav.asistir(cmd, [], 0.0)
        assert out.lineal == cmd.lineal


# ===========================================================================
# CONTROL
# ===========================================================================

class TestControl:

    def test_sin_error_no_hay_accion_proporcional(self):
        c = LogicaControl(CONFIG)
        v = Velocidad(0.1, 0.0)
        u = c.actualizar(v, v)
        # El termino integral puede aportar algo, pero poco
        assert abs(u.lineal) < 0.02

    def test_error_positivo_accion_positiva(self):
        c = LogicaControl(CONFIG)
        u = c.actualizar(Velocidad(0.15, 0.0), Velocidad(0.0, 0.0))
        assert u.lineal > 0

    def test_respeta_los_limites(self):
        c = LogicaControl(CONFIG)
        for _ in range(50):
            u = c.actualizar(Velocidad(10.0, 10.0), Velocidad(0.0, 0.0))
        assert abs(u.lineal) <= CONFIG.robot.u_max + 1e-9
        assert abs(u.angular) <= CONFIG.robot.omega_max + 1e-9

    def test_antiwindup_acota_el_integrador(self):
        """Sin antiwindup, el integrador crece sin limite mientras el error
        persiste, y al resolverse produce un sobreimpulso brusco."""
        c = LogicaControl(CONFIG)
        for _ in range(500):
            c.actualizar(Velocidad(10.0, 0.0), Velocidad(0.0, 0.0))
        i_u, _ = c.integrales
        assert abs(i_u) <= CONFIG.control.antiwindup + 1e-9

    def test_reiniciar_limpia_el_integrador(self):
        c = LogicaControl(CONFIG)
        for _ in range(20):
            c.actualizar(Velocidad(0.15, 0.0), Velocidad(0.0, 0.0))
        c.reiniciar()
        assert c.integrales == (0.0, 0.0)


# ===========================================================================
# RECEPTOR
# ===========================================================================

class TestReceptor:

    VALIDO = ('{"estado":"IC","freq_idx":2,"p_max":0.9,"lambda_bci":0.95,'
              '"valido":true,"timestamp":1.0,"secuencia":1}')

    def test_acepta_mensaje_valido(self):
        r = ReceptorTASM(CONFIG)
        assert r.procesar_linea(self.VALIDO, t=1.0) is not None

    def test_descarta_json_malformado(self):
        r = ReceptorTASM(CONFIG)
        assert r.procesar_linea("{roto", t=1.0) is None

    def test_descarta_estado_desconocido(self):
        r = ReceptorTASM(CONFIG)
        m = ('{"estado":"XX","freq_idx":0,"p_max":0.5,"lambda_bci":0.5,'
             '"valido":true,"timestamp":1.0,"secuencia":1}')
        assert r.procesar_linea(m, t=1.0) is None

    def test_descarta_indice_fuera_de_rango(self):
        r = ReceptorTASM(CONFIG)
        m = ('{"estado":"IC","freq_idx":99,"p_max":0.5,"lambda_bci":0.5,'
             '"valido":true,"timestamp":1.0,"secuencia":1}')
        assert r.procesar_linea(m, t=1.0) is None

    def test_descarta_campos_faltantes(self):
        r = ReceptorTASM(CONFIG)
        assert r.procesar_linea('{"estado":"IC"}', t=1.0) is None

    def test_descarta_linea_vacia(self):
        r = ReceptorTASM(CONFIG)
        assert r.procesar_linea("   \n  ", t=1.0) is None

    def test_un_mensaje_corrupto_no_rompe_el_siguiente(self):
        """PROPIEDAD IMPORTANTE: el nodo debe sobrevivir a datos malos."""
        r = ReceptorTASM(CONFIG)
        r.procesar_linea("{basura", t=1.0)
        assert r.procesar_linea(self.VALIDO, t=1.1) is not None

    def test_watchdog_expirado_al_principio(self):
        assert ReceptorTASM(CONFIG).watchdog_expirado(t=0.0)

    def test_watchdog_no_expirado_tras_mensaje(self):
        r = ReceptorTASM(CONFIG)
        r.procesar_linea(self.VALIDO, t=10.0)
        assert not r.watchdog_expirado(t=10.1)

    def test_watchdog_expira_por_silencio(self):
        r = ReceptorTASM(CONFIG)
        r.procesar_linea(self.VALIDO, t=10.0)
        assert r.watchdog_expirado(t=10.0 + CONFIG.bci.watchdog_s + 0.1)

    def test_cuenta_descartados(self):
        r = ReceptorTASM(CONFIG)
        r.procesar_linea(self.VALIDO, t=1.0)
        r.procesar_linea("{roto", t=1.1)
        st = r.estadisticas
        assert st["recibidos"] == 1
        assert st["descartados"] == 1


# ===========================================================================
# MISION
# ===========================================================================

class TestMision:

    def _msg(self, estado="IC", idx=2):
        return MensajeTASMRecibido(estado, idx, 0.9, 0.95, True, 0.0, 1)

    def _enclavar(self, mis, libre, t0=0.0):
        """Aplica la racha completa y devuelve el tiempo final."""
        t = t0
        for _ in range(CONFIG.bci.n_conf):
            d = mis.ciclo(self._msg(), libre, t)
            t += CONFIG.bci.paso_tasm
        return t, d

    def test_enclava_tras_la_racha(self):
        mis = LogicaMision(CONFIG)
        _, d = self._enclavar(mis, escaneo(5.0))
        assert d.velocidad.lineal > 0

    def test_no_enclava_antes_de_la_racha(self):
        mis = LogicaMision(CONFIG)
        t = 0.0
        for _ in range(CONFIG.bci.n_conf - 1):
            d = mis.ciclo(self._msg(), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.velocidad.es_cero()

    def test_emergencia_tiene_prioridad(self):
        """PROPIEDAD CENTRAL DEL ARBITRAJE.

        Debe imponerse incluso sobre un comando con confianza maxima.
        """
        mis = LogicaMision(CONFIG)
        t, _ = self._enclavar(mis, escaneo(5.0))
        d = mis.ciclo(self._msg(), escaneo(0.10), t)
        assert d.modo == ModoOperacion.EMERGENCIA
        assert d.velocidad.lineal < 0

    def test_watchdog_detiene(self):
        mis = LogicaMision(CONFIG)
        d = mis.ciclo(None, escaneo(5.0), 0.0, watchdog_expirado=True)
        assert d.modo == ModoOperacion.DETENIDO_SEGURO
        assert d.velocidad.es_cero()

    def test_watchdog_no_mantiene_el_comando(self):
        """Si no sabemos si el usuario sigue conectado, seguir moviendose no
        es aceptable."""
        mis = LogicaMision(CONFIG)
        t, _ = self._enclavar(mis, escaneo(5.0))
        d = mis.ciclo(None, escaneo(5.0), t, watchdog_expirado=True)
        assert d.velocidad.es_cero()

    def test_tr_no_enclava(self):
        """La propiedad que define el trabajo."""
        mis = LogicaMision(CONFIG)
        t = 0.0
        for _ in range(CONFIG.bci.n_conf * 4):
            d = mis.ciclo(self._msg("TR", -1), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.velocidad.es_cero()

    def test_idle_no_enclava(self):
        mis = LogicaMision(CONFIG)
        t = 0.0
        for _ in range(CONFIG.bci.n_conf * 4):
            d = mis.ciclo(self._msg("Idle", -1), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.velocidad.es_cero()

    def test_idle_mantiene_el_comando_enclavado(self):
        """Bajo el paradigma de enclavamiento el usuario esta legitimamente
        en Idle la mayor parte del trial."""
        mis = LogicaMision(CONFIG)
        t, _ = self._enclavar(mis, escaneo(5.0))
        for _ in range(50):
            d = mis.ciclo(self._msg("Idle", -1), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.velocidad.lineal > 0

    def test_etapa2_robot_estatico(self):
        mis = LogicaMision(CONFIG)
        mis.cambiar_etapa(Etapa.SELECCION)
        t = 0.0
        for _ in range(CONFIG.bci.n_conf + 5):
            d = mis.ciclo(self._msg(), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.velocidad.es_cero()
        assert d.etapa == Etapa.SELECCION

    def test_cambio_de_etapa_reinicia_la_fsm(self):
        """Al cambiar de etapa las frecuencias activas cambian; el estado de
        comando anterior deja de tener sentido."""
        mis = LogicaMision(CONFIG)
        self._enclavar(mis, escaneo(5.0))
        mis.cambiar_etapa(Etapa.SELECCION)
        assert mis.fsm.racha_ic == 0

    def test_recupera_tras_la_emergencia(self):
        mis = LogicaMision(CONFIG)
        t, _ = self._enclavar(mis, escaneo(5.0))
        mis.ciclo(self._msg(), escaneo(0.10), t)
        t += CONFIG.bci.paso_tasm
        for _ in range(CONFIG.robot.n_hist + 1):
            d = mis.ciclo(self._msg(), escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm
        assert d.modo == ModoOperacion.BCI_MANUAL


# ===========================================================================
# INTEGRACION
# ===========================================================================

class TestIntegracion:

    def test_cadena_completa(self):
        """Receptor -> mision -> control, encadenados como lo harian los
        nodos."""
        rec = ReceptorTASM(CONFIG)
        mis = LogicaMision(CONFIG)
        ctl = LogicaControl(CONFIG)

        medida = Velocidad(0.0, 0.0)
        t = 0.0
        for i in range(CONFIG.bci.n_conf + 10):
            linea = (f'{{"estado":"IC","freq_idx":2,"p_max":0.9,'
                     f'"lambda_bci":0.9,"valido":true,"timestamp":{t},'
                     f'"secuencia":{i}}}')
            msg = rec.procesar_linea(linea, t=t)
            d = mis.ciclo(msg, escaneo(5.0), t)
            u = ctl.actualizar(d.velocidad, medida)
            medida = Velocidad(
                medida.lineal + (u.lineal - medida.lineal) * 0.25,
                medida.angular + (u.angular - medida.angular) * 0.25,
            )
            t += CONFIG.bci.paso_tasm

        assert medida.lineal > 0, "el robot deberia estar avanzando"

    def test_emergencia_de_extremo_a_extremo(self):
        rec = ReceptorTASM(CONFIG)
        mis = LogicaMision(CONFIG)
        ctl = LogicaControl(CONFIG)

        t = 0.0
        for i in range(CONFIG.bci.n_conf + 2):
            linea = (f'{{"estado":"IC","freq_idx":2,"p_max":0.9,'
                     f'"lambda_bci":0.9,"valido":true,"timestamp":{t},'
                     f'"secuencia":{i}}}')
            msg = rec.procesar_linea(linea, t=t)
            mis.ciclo(msg, escaneo(5.0), t)
            t += CONFIG.bci.paso_tasm

        d = mis.ciclo(rec.ultimo, escaneo(0.08), t)
        u = ctl.actualizar(d.velocidad, Velocidad(0.15, 0.0))
        assert d.modo == ModoOperacion.EMERGENCIA
        assert u.lineal < 0.15, "el control deberia estar frenando"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

"""
tests/test_vision.py --- Bateria de tests del Bloque 5
Bloque 5: vision_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Estos tests cubren:
  - Deteccion ArUco y estimacion de pose (VERIFICADO por ejecucion real)
  - Fusion, transicion y aproximacion (logica verificada)

NO cubren:
  - YOLO, que es un esqueleto sin modelo cargado
  - El comportamiento con la camara real: distorsion de lente, iluminacion
    del laboratorio y desenfoque de movimiento afectan de formas que una
    imagen sintetica no reproduce

Si un test falla, reportalo indicando QUE TEST, el comando exacto, el
traceback completo y que intentaste.
===========================================================================
"""

import sys
import os
import math
import numpy as np
import pytest

_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _raiz)
sys.path.insert(0, os.path.join(_raiz, "logica"))

from config import CONFIG
from aruco_detector import (DetectorArUco, MarcadorDetectado,
                            generar_marcador, generar_hoja_marcadores,
                            renderizar_escena, _HAY_ARUCO)
from vision_fusion import (DetectorYOLO, DeteccionYOLO, FusionVision,
                           ObjetoSeleccionable, LogicaTransicion,
                           ControlAproximacion)

pytestmark = pytest.mark.skipif(
    not _HAY_ARUCO, reason="OpenCV sin soporte aruco")

N_RAYOS = CONFIG.robot.lidar_n_rayos


def escena_fila(n, distancia, sep=0.25):
    """Genera n marcadores en fila a una distancia dada."""
    x0 = -(n - 1) * sep / 2
    return [(i, x0 + i * sep, 0.0, distancia) for i in range(n)]


def lidar_libre():
    return [5.0] * N_RAYOS


def lidar_con_obstaculo(dist=0.60, indice=0):
    d = [5.0] * N_RAYOS
    d[indice] = dist
    return d


# ===========================================================================
# CONFIGURACION
# ===========================================================================

class TestConfigVision:

    def test_resolucion_coherente_con_etapa2(self):
        assert CONFIG.vision.ancho_px == CONFIG.etapa2.ancho_imagen_px

    def test_fov_coherente_con_etapa2(self):
        assert abs(CONFIG.vision.fov_horizontal -
                   CONFIG.etapa2.fov_horizontal) < 0.1

    def test_umbral_cubre_el_mayor_n(self):
        """El umbral de transicion debe superar la distancia que exige el
        mayor N del experimento, o la transicion nunca se disparara."""
        n_max = max(CONFIG.etapa2.niveles_n)
        d = CONFIG.etapa2.distancia_minima(n_max)
        assert CONFIG.vision.umbral_distancia_transicion >= d

    def test_marcador_legible_en_el_umbral(self):
        v = CONFIG.vision
        px = v.px_marcador_a(v.umbral_distancia_transicion)
        assert px >= v.px_minimo_marcador

    def test_alcance_supera_el_umbral(self):
        v = CONFIG.vision
        assert v.distancia_maxima_deteccion() >= \
            v.umbral_distancia_transicion


# ===========================================================================
# DETECCION ArUco
# ===========================================================================

class TestArUco:

    def test_genera_marcador(self):
        img = generar_marcador(CONFIG, 0, lado_px=200, borde=20)
        assert img.shape == (240, 240)

    def test_genera_hoja(self):
        hoja = generar_hoja_marcadores(CONFIG, [0, 1, 2, 3], lado_px=100)
        assert hoja.ndim == 2
        assert hoja.size > 0

    def test_detecta_marcador_unico(self):
        det = DetectorArUco(CONFIG)
        img = renderizar_escena(CONFIG, [(0, 0.0, 0.0, 1.0)])
        res = det.detectar(img)
        assert len(res) == 1
        assert res[0].id == 0

    def test_detecta_varios(self):
        det = DetectorArUco(CONFIG)
        for n in (2, 4, 8):
            d = CONFIG.etapa2.distancia_minima(n)
            img = renderizar_escena(CONFIG, escena_fila(n, d))
            res = det.filtrar_fiables(det.detectar(img))
            assert len(res) == n, f"N={n}: detectados {len(res)}"

    def test_ordena_por_posicion_horizontal(self):
        """PROPIEDAD NECESARIA PARA LA BUSQUEDA BINARIA.

        El grupo izquierdo debe estar realmente a la izquierda.
        """
        det = DetectorArUco(CONFIG)
        img = renderizar_escena(CONFIG, escena_fila(4, 1.0))
        res = det.detectar(img)
        xs = [m.centro_px[0] for m in res]
        assert xs == sorted(xs)

    def test_pose_aproximadamente_correcta(self):
        det = DetectorArUco(CONFIG)
        for d_real in (0.5, 1.0, 1.5):
            img = renderizar_escena(CONFIG, [(0, 0.0, 0.0, d_real)])
            res = det.detectar(img)
            assert res, f"no detectado a {d_real} m"
            err = abs(res[0].profundidad - d_real)
            # Tolerancia del 10%: con intrinsecos nominales y renderizado
            # sintetico no cabe esperar mas precision
            assert err < d_real * 0.10, \
                f"a {d_real} m el error fue {err*1000:.0f} mm"

    def test_angulo_horizontal_tiene_el_signo_correcto(self):
        det = DetectorArUco(CONFIG)

        img = renderizar_escena(CONFIG, [(0, 0.30, 0.0, 1.0)])
        res = det.detectar(img)
        assert res and res[0].angulo_horizontal > 0, \
            "marcador a la derecha deberia dar angulo positivo"

        img = renderizar_escena(CONFIG, [(0, -0.30, 0.0, 1.0)])
        res = det.detectar(img)
        assert res and res[0].angulo_horizontal < 0

    def test_imagen_vacia_no_rompe(self):
        det = DetectorArUco(CONFIG)
        assert det.detectar(np.zeros((100, 100), dtype=np.uint8)) == []

    def test_imagen_none_no_rompe(self):
        det = DetectorArUco(CONFIG)
        assert det.detectar(None) == []

    def test_filtra_marcadores_pequenos(self):
        """Un marcador de pocos pixeles puede decodificarse y aun asi dar
        una pose imprecisa."""
        det = DetectorArUco(CONFIG)
        img = renderizar_escena(CONFIG, [(0, 0.0, 0.0, 3.5)])
        todos = det.detectar(img)
        fiables = det.filtrar_fiables(todos)
        assert len(fiables) <= len(todos)

    def test_robusto_al_ruido(self):
        det = DetectorArUco(CONFIG)
        img = renderizar_escena(CONFIG, escena_fila(4, 1.0),
                                ruido=0.10, semilla=1)
        assert len(det.detectar(img)) >= 3

    def test_ids_distintos(self):
        det = DetectorArUco(CONFIG)
        img = renderizar_escena(CONFIG, escena_fila(4, 1.0))
        ids = [m.id for m in det.detectar(img)]
        assert len(set(ids)) == len(ids)


# ===========================================================================
# FUSION
# ===========================================================================

class TestFusion:

    def _marcadores(self, n=4, d=1.2):
        det = DetectorArUco(CONFIG)
        return det.detectar(renderizar_escena(CONFIG, escena_fila(n, d)))

    def _recuadros(self, marcadores, factor=2.2):
        salida = []
        for m in marcadores:
            cx, cy = m.centro_px
            w = m.lado_px * factor
            h = m.lado_px * factor * 1.4
            salida.append(DeteccionYOLO(
                clase="objeto", confianza=0.9,
                bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        return salida

    def test_sin_yolo_funciona(self):
        """El sistema debe operar aunque YOLO no este disponible: la
        identidad y la pose las da el marcador."""
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, [])
        assert len(objs) == len(ms)
        assert all(not o.tiene_yolo for o in objs)

    def test_asocia_recuadros(self):
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, self._recuadros(ms))
        assert all(o.tiene_yolo for o in objs)

    def test_descarta_recuadro_sin_marcador(self):
        """DISCRIMINACION OBJETO/OBSTACULO.

        Un recuadro sin marcador asociado no es seleccionable.
        """
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        extra = DeteccionYOLO("obstaculo", 0.95, (10, 10, 60, 200))
        objs = f.fusionar(ms, self._recuadros(ms) + [extra])
        assert len(objs) == len(ms)

    def test_no_reutiliza_un_recuadro(self):
        """Cada recuadro se asocia a un solo marcador."""
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, self._recuadros(ms))
        usados = [id(o.deteccion) for o in objs if o.tiene_yolo]
        assert len(usados) == len(set(usados))

    def test_ordenados(self):
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, [])
        xs = [o.centro_px[0] for o in objs]
        assert xs == sorted(xs)

    def test_bbox_usa_yolo_si_existe(self):
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        recs = self._recuadros(ms)
        objs = f.fusionar(ms, recs)
        for o in objs:
            if o.tiene_yolo:
                assert o.bbox_interfaz() == o.deteccion.bbox

    def test_bbox_estimado_sin_yolo(self):
        """Sin YOLO se estima un recuadro ampliando el marcador. Es peor
        visualmente pero funcional."""
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, [])
        for o in objs:
            b = o.bbox_interfaz()
            assert b[2] > b[0] and b[3] > b[1]

    def test_id_viene_del_marcador(self):
        f = FusionVision(CONFIG)
        ms = self._marcadores()
        objs = f.fusionar(ms, self._recuadros(ms))
        for o in objs:
            assert o.id == o.marcador.id


# ===========================================================================
# YOLO
# ===========================================================================

class TestYOLO:

    def test_esqueleto_no_rompe(self):
        """Sin modelo cargado debe devolver lista vacia, no lanzar."""
        y = DetectorYOLO(CONFIG)
        assert y.detectar(np.zeros((100, 100, 3), dtype=np.uint8)) == []

    def test_informa_de_su_disponibilidad(self):
        assert isinstance(DetectorYOLO(CONFIG).disponible, bool)

    def test_deteccion_calcula_geometria(self):
        d = DeteccionYOLO("obj", 0.9, (100, 200, 300, 500))
        assert d.centro == (200.0, 350.0)
        assert d.ancho == 200.0
        assert d.alto == 300.0
        assert abs(d.diagonal - math.hypot(200, 300)) < 1e-9


# ===========================================================================
# TRANSICION
# ===========================================================================

class TestTransicion:

    def _objetos(self, n=4, d=1.2):
        det = DetectorArUco(CONFIG)
        f = FusionVision(CONFIG)
        return f.fusionar(
            det.detectar(renderizar_escena(CONFIG, escena_fila(n, d))), [])

    def _con_permanencia(self, log, objs, lidar, t=5.0):
        """Aplica el tiempo de permanencia completo."""
        log.evaluar(objs, lidar, True, 0.0)
        return log.evaluar(objs, lidar, True, t)

    def test_procede_con_las_tres(self):
        log = LogicaTransicion(CONFIG)
        est = self._con_permanencia(log, self._objetos(), lidar_libre())
        assert est.procede

    def test_no_procede_lejos(self):
        log = LogicaTransicion(CONFIG)
        est = self._con_permanencia(log, self._objetos(d=3.0), lidar_libre())
        assert not est.procede

    def test_no_procede_sin_objetos(self):
        log = LogicaTransicion(CONFIG)
        est = self._con_permanencia(log, [], lidar_libre())
        assert not est.procede

    def test_no_procede_con_un_objeto(self):
        """Hacen falta al menos dos para que una decision binaria tenga
        sentido."""
        log = LogicaTransicion(CONFIG)
        est = self._con_permanencia(log, self._objetos(n=1), lidar_libre())
        assert not est.procede

    def test_no_procede_con_obstaculo(self):
        log = LogicaTransicion(CONFIG)
        est = self._con_permanencia(log, self._objetos(),
                                    lidar_con_obstaculo(0.60))
        assert not est.procede

    def test_no_procede_en_movimiento(self):
        log = LogicaTransicion(CONFIG)
        est = log.evaluar(self._objetos(), lidar_libre(), False, 0.0)
        assert not est.procede

    def test_no_procede_sin_permanencia(self):
        """LA CONDICION QUE EVITA EL FALSO POSITIVO MAS PROBABLE.

        Sin ella, una parada momentanea para evaluar el entorno dispararia
        la transicion.
        """
        log = LogicaTransicion(CONFIG)
        log.evaluar(self._objetos(), lidar_libre(), True, 0.0)
        est = log.evaluar(self._objetos(), lidar_libre(), True, 0.5)
        assert not est.procede

    def test_contador_se_reinicia_al_moverse(self):
        log = LogicaTransicion(CONFIG)
        objs, lid = self._objetos(), lidar_libre()
        log.evaluar(objs, lid, True, 0.0)
        log.evaluar(objs, lid, True, 1.5)
        log.evaluar(objs, lid, False, 2.0)      # se mueve
        est = log.evaluar(objs, lid, True, 2.5)
        assert not est.procede

    def test_obstaculo_lateral_no_impide(self):
        """Solo importa el sector frontal: un obstaculo a los lados no
        impide llegar."""
        log = LogicaTransicion(CONFIG)
        lat = lidar_con_obstaculo(0.60, indice=N_RAYOS // 4)
        est = self._con_permanencia(log, self._objetos(), lat)
        assert est.procede

    def test_obstaculo_mas_lejos_que_objetos_no_impide(self):
        log = LogicaTransicion(CONFIG)
        lid = lidar_con_obstaculo(3.0)
        est = self._con_permanencia(log, self._objetos(d=1.2), lid)
        assert est.procede

    def test_informa_del_motivo(self):
        log = LogicaTransicion(CONFIG)
        est = log.evaluar(self._objetos(d=3.0), lidar_libre(), True, 5.0)
        assert est.motivo != ""

    def test_reiniciar(self):
        log = LogicaTransicion(CONFIG)
        objs, lid = self._objetos(), lidar_libre()
        log.evaluar(objs, lid, True, 0.0)
        log.reiniciar()
        est = log.evaluar(objs, lid, True, 5.0)
        assert not est.procede, "tras reiniciar deberia empezar de cero"


# ===========================================================================
# APROXIMACION
# ===========================================================================

class TestAproximacion:

    def test_completada_en_el_objetivo(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(CONFIG.vision.distancia_objetivo, 0.0)
        assert r.completada
        assert r.lineal == 0.0 and r.angular == 0.0

    def test_avanza_si_esta_lejos(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(1.0, 0.0)
        assert not r.completada
        assert r.lineal > 0

    def test_retrocede_si_esta_demasiado_cerca(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(0.10, 0.0)
        assert r.lineal < 0

    def test_alinea_antes_de_avanzar(self):
        """Avanzar muy desalineado alejaria al robot de la trayectoria
        util."""
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(1.0, math.radians(30))
        assert abs(r.lineal) < 1e-9
        assert abs(r.angular) > 0

    def test_signo_del_giro(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(1.0, math.radians(20))
        assert r.angular > 0, "objetivo a la derecha -> girar a la derecha"
        r = c.actualizar(1.0, math.radians(-20))
        assert r.angular < 0

    def test_respeta_la_velocidad_maxima(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(10.0, 0.0)
        assert abs(r.lineal) <= CONFIG.vision.velocidad_aproximacion + 1e-9

    def test_respeta_la_velocidad_angular_maxima(self):
        c = ControlAproximacion(CONFIG)
        r = c.actualizar(1.0, math.radians(170))
        assert abs(r.angular) <= CONFIG.robot.omega_max + 1e-9

    def test_converge(self):
        """PRUEBA DE INTEGRACION: el lazo debe cerrar."""
        c = ControlAproximacion(CONFIG)
        d, a = 1.20, math.radians(20)
        dt = CONFIG.robot.ts_control
        for _ in range(600):
            r = c.actualizar(d, a)
            if r.completada:
                break
            d -= r.lineal * dt
            a -= r.angular * dt
        assert r.completada, f"no convergio: d={d:.3f}, a={math.degrees(a):.1f}"

    def test_converge_desde_varias_condiciones(self):
        dt = CONFIG.robot.ts_control
        for d0, a0 in ((1.5, 30.0), (0.8, -25.0), (0.35, 5.0), (2.0, 0.0)):
            c = ControlAproximacion(CONFIG)
            d, a = d0, math.radians(a0)
            ok = False
            for _ in range(800):
                r = c.actualizar(d, a)
                if r.completada:
                    ok = True
                    break
                d -= r.lineal * dt
                a -= r.angular * dt
            assert ok, f"no convergio desde {d0} m, {a0} grados"

    def test_fijar_objetivo(self):
        det = DetectorArUco(CONFIG)
        f = FusionVision(CONFIG)
        objs = f.fusionar(
            det.detectar(renderizar_escena(CONFIG, escena_fila(4, 1.2))), [])
        c = ControlAproximacion(CONFIG)
        assert c.objetivo is None
        c.fijar_objetivo(objs[0])
        assert c.objetivo is objs[0]


# ===========================================================================
# INTEGRACION
# ===========================================================================

class TestIntegracion:

    def test_cadena_deteccion_a_seleccion(self):
        """Deteccion -> fusion -> busqueda binaria."""
        from binary_search import BusquedaBinaria, Objeto

        det = DetectorArUco(CONFIG)
        f = FusionVision(CONFIG)

        for n in (2, 4, 8):
            d = CONFIG.etapa2.distancia_minima(n)
            objs = f.fusionar(
                det.filtrar_fiables(
                    det.detectar(renderizar_escena(CONFIG, escena_fila(n, d)))),
                [])
            assert len(objs) == n

            lista = [Objeto(id_aruco=o.id, x_center=o.centro_px[0],
                            distancia=o.distancia) for o in objs]
            bb = BusquedaBinaria(CONFIG, lista)
            objetivo = n // 2
            while not bb.terminada:
                bb.votar(bb.lado_correcto(objetivo))
            r = bb.resultado(objetivo=objetivo)
            assert r.acierto
            assert r.n_decisiones == math.ceil(math.log2(n))

    def test_flujo_completo(self):
        """Navegacion -> transicion -> seleccion -> aproximacion."""
        from binary_search import BusquedaBinaria, Objeto

        det = DetectorArUco(CONFIG)
        f = FusionVision(CONFIG)
        log = LogicaTransicion(CONFIG)
        apr = ControlAproximacion(CONFIG)

        # Lejos: no procede
        objs = f.fusionar(
            det.detectar(renderizar_escena(CONFIG, escena_fila(4, 3.0))), [])
        assert not log.evaluar(objs, lidar_libre(), True, 5.0).procede

        # Cerca y parado: procede
        log.reiniciar()
        objs = f.fusionar(
            det.detectar(renderizar_escena(CONFIG, escena_fila(4, 1.2))), [])
        log.evaluar(objs, lidar_libre(), True, 0.0)
        assert log.evaluar(objs, lidar_libre(), True, 5.0).procede

        # Seleccion
        lista = [Objeto(id_aruco=o.id, x_center=o.centro_px[0],
                        distancia=o.distancia) for o in objs]
        bb = BusquedaBinaria(CONFIG, lista)
        while not bb.terminada:
            bb.votar(bb.lado_correcto(2))
        assert bb.resultado(objetivo=2).acierto

        # Aproximacion
        elegido = objs[2]
        apr.fijar_objetivo(elegido)
        d, a = elegido.distancia, elegido.angulo
        dt = CONFIG.robot.ts_control
        for _ in range(600):
            r = apr.actualizar(d, a)
            if r.completada:
                break
            d -= r.lineal * dt
            a -= r.angular * dt
        assert r.completada


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

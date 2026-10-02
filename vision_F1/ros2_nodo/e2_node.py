#!/usr/bin/env python3
"""
e2_node.py --- Nodo de vision y seleccion de objeto (Etapa 2)
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: codigo funcional pero NO verificado sobre ROS2. La logica de vision,
transicion y aproximacion esta en vision_fusion.py y SI tiene pruebas.

Copiar este archivo a:
    ros2_ws/src/turtlebot3_bci_3c2/turtlebot3_bci_3c2/
===========================================================================

QUE HACE ESTE NODO
------------------
Cuatro cosas, en este orden:

  1. Detecta objetos en la imagen de la camara (ArUco + YOLO).
  2. Decide cuando pasar de navegacion a seleccion.
  3. Ejecuta la busqueda binaria, compuertada por TASM.
  4. Aproxima el robot al objeto elegido.

Es el nodo mas complejo del sistema porque coordina vision, BCI y control.
Por eso toda su logica esta fuera, en modulos verificados: aqui solo queda
la secuencia.


LA COMPUERTA DE TASM TAMBIEN APLICA AQUI
-----------------------------------------
Cada voto de la busqueda binaria exige estado IC confirmado durante n_conf
ventanas, igual que en la Etapa 1.

Durante TR e Idle la iteracion espera sin registrar voto. Si se agota el
timeout, la iteracion SE REPITE en lugar de avanzar con un voto dudoso.

Esa decision importa: un error aqui es irrecuperable. Si el sistema elige el
grupo equivocado, el objeto correcto desaparece de la lista y ya no se puede
alcanzar. Es preferible repetir la iteracion.
"""

import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8, String
from turtlebot3_bci_3c2.msg import TASMState, BciCommand

from turtlebot3_bci_3c2.logica_bridge import cargar_config
from aruco_detector import DetectorArUco
from vision_fusion import (DetectorYOLO, FusionVision, LogicaTransicion,
                           ControlAproximacion)
from binary_search import BusquedaBinaria, LadoSeleccion, Objeto

try:
    import cv2
    _HAY_CV2 = True
except ImportError:
    _HAY_CV2 = False


NOMBRE_ESTADO = {0: "Idle", 1: "TR", 2: "IC"}


class E2Node(Node):
    """Vision y seleccion de objeto."""

    def __init__(self):
        super().__init__("e2_node")

        if not _HAY_CV2:
            self.get_logger().error("OpenCV no disponible")
            raise RuntimeError("OpenCV requerido")

        self.cfg = cargar_config()

        self.det_aruco = DetectorArUco(self.cfg)
        self.det_yolo = DetectorYOLO(self.cfg)
        self.fusion = FusionVision(self.cfg)
        self.transicion = LogicaTransicion(self.cfg)
        self.aproximacion = ControlAproximacion(self.cfg)

        self.declare_parameter("hz", 10.0)
        self.declare_parameter("n_objetos", 4)
        hz = self.get_parameter("hz").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, "/camera/image_raw",
                                 self._cb_imagen, qos)
        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos)
        self.create_subscription(TASMState, "/tasm/state", self._cb_tasm, qos)
        self.create_subscription(BciCommand, "/bci/command",
                                 self._cb_comando, qos)

        self.pub_etapa = self.create_publisher(Int8, "/bci/etapa", 10)
        self.pub_objetos = self.create_publisher(String, "/bci/objetos", 10)
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", qos)

        # --- Estado ---
        self._imagen: Optional[np.ndarray] = None
        self._distancias: List[float] = []
        self._objetos = []
        self._etapa = 1
        self._robot_detenido = True
        self._busqueda: Optional[BusquedaBinaria] = None
        self._racha_ic = 0
        self._freq_racha = -1
        self._t_iteracion = 0.0
        self._aproximando = False
        self._t0 = time.monotonic()

        self.timer = self.create_timer(1.0 / hz, self._ciclo)

        self.get_logger().info(
            f"e2_node a {hz} Hz. "
            f"Transicion: <{self.cfg.vision.umbral_distancia_transicion} m, "
            f">={self.cfg.vision.min_objetos_transicion} objetos, "
            f">={self.cfg.vision.permanencia_detenido} s parado")

    # -------------------------------------------------------------------
    def _cb_imagen(self, msg: Image) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "bgr8"):
                img = arr.reshape(msg.height, msg.width, 3)
                self._imagen = cv2.cvtColor(
                    img, cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8"
                    else cv2.COLOR_BGR2GRAY)
            elif msg.encoding == "mono8":
                self._imagen = arr.reshape(msg.height, msg.width)
        except ValueError:
            self.get_logger().warn("Frame malformado",
                                   throttle_duration_sec=5.0)

    def _cb_scan(self, msg: LaserScan) -> None:
        rmax = self.cfg.robot.lidar_rango_max
        self._distancias = [
            rmax if (r != r or r == float("inf") or r <= 0.0)
            else min(float(r), rmax) for r in msg.ranges]

    def _cb_tasm(self, msg: TASMState) -> None:
        """Acumula la racha de confirmacion para los votos de la Etapa 2."""
        estado = NOMBRE_ESTADO.get(msg.state, "Idle")
        if estado == "IC" and msg.is_valid and msg.freq_idx >= 0:
            if self._freq_racha == msg.freq_idx:
                self._racha_ic += 1
            else:
                self._freq_racha = msg.freq_idx
                self._racha_ic = 1
        else:
            self._racha_ic = 0
            self._freq_racha = -1

    def _cb_comando(self, msg: BciCommand) -> None:
        self._robot_detenido = (abs(msg.u_ref) < 1e-3 and
                                abs(msg.omega_ref) < 1e-3)

    # -------------------------------------------------------------------
    def _detectar(self) -> None:
        """Detecta objetos en el frame actual."""
        if self._imagen is None:
            return
        marcadores = self.det_aruco.filtrar_fiables(
            self.det_aruco.detectar(self._imagen))
        detecciones = self.det_yolo.detectar(self._imagen)
        self._objetos = self.fusion.fusionar(marcadores, detecciones)

    # -------------------------------------------------------------------
    def _publicar_objetos(self) -> None:
        """Publica los objetos y su agrupacion, para la interfaz."""
        import json
        izq, der = ([], [])
        if self._busqueda is not None and not self._busqueda.terminada:
            izq, der = self._busqueda.grupos_actuales()

        d = {
            "objetos": [
                {"id": o.id,
                 "bbox": list(o.bbox_interfaz()),
                 "distancia": round(o.distancia, 3)}
                for o in self._objetos
            ],
            "grupo_izq": izq,
            "grupo_der": der,
        }
        m = String()
        m.data = json.dumps(d)
        self.pub_objetos.publish(m)

    # -------------------------------------------------------------------
    def _ciclo(self) -> None:
        t = time.monotonic() - self._t0
        self._detectar()

        if self._etapa == 1:
            self._ciclo_navegacion(t)
        else:
            self._ciclo_seleccion(t)

        self._publicar_objetos()

    # -------------------------------------------------------------------
    def _ciclo_navegacion(self, t: float) -> None:
        """Vigila si procede pasar a la fase de seleccion."""
        est = self.transicion.evaluar(
            self._objetos, self._distancias, self._robot_detenido, t)

        if est.procede:
            self.get_logger().info(
                f"Transicion a Etapa 2: {est.n_objetos} objetos a "
                f"{est.distancia_media:.2f} m")
            self._etapa = 2
            self._iniciar_busqueda()
            m = Int8()
            m.data = 2
            self.pub_etapa.publish(m)

    # -------------------------------------------------------------------
    def _iniciar_busqueda(self) -> None:
        """Arranca la busqueda binaria sobre los objetos detectados."""
        objs = [
            Objeto(id_aruco=o.id, x_center=o.centro_px[0],
                   y_center=o.centro_px[1], distancia=o.distancia)
            for o in self._objetos
        ]
        try:
            self._busqueda = BusquedaBinaria(self.cfg, objs)
            self._t_iteracion = time.monotonic()
            self.get_logger().info(
                f"Busqueda iniciada: {len(objs)} objetos, "
                f"{self._busqueda.decisiones_teoricas} decisiones")
        except ValueError as ex:
            self.get_logger().error(f"No se pudo iniciar: {ex}")
            self._volver_a_navegacion()

    # -------------------------------------------------------------------
    def _ciclo_seleccion(self, t: float) -> None:
        """Ejecuta la busqueda binaria y la aproximacion."""
        if self._aproximando:
            self._ciclo_aproximacion()
            return

        if self._busqueda is None:
            self._volver_a_navegacion()
            return

        if self._busqueda.terminada:
            self._finalizar_seleccion()
            return

        # --- Voto por racha confirmada ---
        if self._racha_ic >= self.cfg.bci.n_conf:
            # Indice 0 = izquierda, 1 = derecha (mismas frecuencias que en
            # la Etapa 1, lo que mantiene coherencia semantica)
            lado = (LadoSeleccion.IZQUIERDA if self._freq_racha == 0
                    else LadoSeleccion.DERECHA)
            dur = time.monotonic() - self._t_iteracion
            self._busqueda.votar(lado, duracion=dur)
            self._racha_ic = 0
            self._freq_racha = -1
            self._t_iteracion = time.monotonic()
            self.get_logger().info(f"Voto: {lado.value}")
            return

        # --- Timeout ---
        if (time.monotonic() - self._t_iteracion) > \
                self.cfg.etapa2.t_max_iteracion:
            self.get_logger().warn("Timeout: se repite la iteracion")
            self._busqueda.registrar_timeout(
                duracion=self.cfg.etapa2.t_max_iteracion)
            self._t_iteracion = time.monotonic()

    # -------------------------------------------------------------------
    def _finalizar_seleccion(self) -> None:
        """Pasa a la fase de aproximacion."""
        r = self._busqueda.resultado()
        if r.objeto_elegido is None:
            self._volver_a_navegacion()
            return

        elegido = self._busqueda.objetos[r.objeto_elegido]
        obj = next((o for o in self._objetos
                    if o.id == elegido.id_aruco), None)

        self.get_logger().info(
            f"Objeto seleccionado: id {elegido.id_aruco} "
            f"en {r.n_decisiones} decisiones "
            f"(teoricas: {r.n_decisiones_teoricas})")

        if obj is not None:
            self.aproximacion.fijar_objetivo(obj)
            self._aproximando = True
        else:
            self.get_logger().warn(
                "El objeto elegido ya no se detecta; se cancela la "
                "aproximacion")
            self._volver_a_navegacion()

    # -------------------------------------------------------------------
    def _ciclo_aproximacion(self) -> None:
        """Aproxima el robot al objeto elegido."""
        objetivo = self.aproximacion.objetivo
        if objetivo is None:
            self._volver_a_navegacion()
            return

        # Se busca el objetivo en la deteccion actual para refinar la
        # medida. Si ya no se ve, se usa la ultima conocida.
        actual = next((o for o in self._objetos if o.id == objetivo.id), None)
        if actual is not None:
            d, a = actual.distancia, actual.angulo
        else:
            d, a = objetivo.distancia, objetivo.angulo

        c = self.aproximacion.actualizar(d, a)

        t = Twist()
        t.linear.x = float(c.lineal)
        t.angular.z = float(c.angular)
        self.pub_vel.publish(t)

        if c.completada:
            self.get_logger().info(
                f"Aproximacion completada. Error: "
                f"{c.error_distancia*100:.1f} cm")
            self.pub_vel.publish(Twist())
            self._aproximando = False

    # -------------------------------------------------------------------
    def _volver_a_navegacion(self) -> None:
        """Vuelve a la Etapa 1. La transicion es reversible."""
        self._etapa = 1
        self._busqueda = None
        self._aproximando = False
        self.transicion.reiniciar()
        m = Int8()
        m.data = 1
        self.pub_etapa.publish(m)
        self.get_logger().info("Vuelta a Etapa 1")


def main(args=None):
    rclpy.init(args=args)
    try:
        nodo = E2Node()
    except RuntimeError:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.pub_vel.publish(Twist())
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
e2_node.py --- Nodo de la Etapa 2 (seleccion de objeto)
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO SOBRE ESTE CODIGO

Codigo funcional pero NO verificado sobre ROS2: no estaba disponible en el
entorno donde se escribio.

La logica que envuelve (deteccion, fusion, transicion, aproximacion,
particion binaria) SI esta verificada con pruebas automatizadas.

Si algo falla, comprueba primero si el problema esta en la logica o en la
plomeria: `python run_vision.py etapa2` ejercita la logica sin ROS2.
===========================================================================

QUE HACE ESTE NODO
------------------
Gestiona todo lo que ocurre delante de la camara:

  1. Detecta marcadores y objetos en cada frame
  2. Decide cuando pasar de navegacion a seleccion
  3. Conduce la particion binaria: agrupa objetos, recibe votos, subdivide
  4. Una vez elegido el objeto, conduce la aproximacion

Publica en /bci/etapa el numero de etapa, que mission_node consume para
saber si debe mover el robot o mantenerlo estatico.

Publica en /bci/objetos la lista de objetos y su agrupacion, que
interface_node reenvia a Windows para que la interfaz sepa que recuadros
dibujar parpadeando.

EL ROBOT ESTA ESTATICO DURANTE LA SELECCION
--------------------------------------------
Es una decision de diseno importante. La camara no se mueve entre
iteraciones: la imagen es la misma y lo unico que cambia es que objetos
parpadean.

Eso tiene dos consecuencias: el numero de decisiones es exactamente
ceil(log2 N) independientemente de la geometria, y no hay degradacion de la
senal SSVEP por vibracion de camara o cambio de perspectiva entre
decisiones.

El desplazamiento se reserva para la fase de aproximacion, ya elegido el
objeto.
"""

import json
import time
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8, String
from turtlebot3_bci_3c2.msg import TASMState, BciCommand

from turtlebot3_bci_3c2.vision_bridge import (
    cargar_config, DetectorArUco, DetectorYOLO, FusionVision,
    DetectorTransicion, ControlAproximacion, EstadoAproximacion,
    BusquedaBinaria, LadoSeleccion, Objeto)

try:
    import cv2
    _HAY_CV2 = True
except ImportError:
    _HAY_CV2 = False


class E2Node(Node):
    """Nodo de vision y seleccion de objeto."""

    def __init__(self):
        super().__init__("e2_node")

        self.cfg = cargar_config()

        self.declare_parameter("hz", 10.0)
        self.declare_parameter("n_objetos", 4)
        hz = self.get_parameter("hz").value
        self.n_objetos = self.get_parameter("n_objetos").value

        # --- Modulos de vision ---
        self.det_aruco = DetectorArUco(self.cfg)
        self.det_yolo = DetectorYOLO(self.cfg)
        self.fusion = FusionVision(self.cfg)
        self.transicion = DetectorTransicion(self.cfg)
        self.aproximacion = ControlAproximacion(self.cfg)

        if not self.det_yolo.disponible:
            self.get_logger().warn(
                "YOLO no disponible. Funcionando en modo solo-ArUco: la "
                "seleccion funciona, pero los recuadros seran menos "
                "ajustados al objeto.")

        # --- Estado ---
        self._etapa = 1
        self._objetos: List = []
        self._busqueda: Optional[BusquedaBinaria] = None
        self._objetivo_final: Optional[int] = None
        self._robot_detenido = True
        self._distancias: List[float] = []
        self._ultimo_frame: Optional[np.ndarray] = None
        self._t0 = time.monotonic()
        self._t_iter: Optional[float] = None

        # --- Comunicacion ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, "/camera/image_raw",
                                 self._cb_imagen, qos)
        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos)
        self.create_subscription(TASMState, "/tasm/state", self._cb_tasm, qos)
        self.create_subscription(BciCommand, "/bci/command", self._cb_cmd, qos)

        self.pub_etapa = self.create_publisher(Int8, "/bci/etapa", 10)
        self.pub_objetos = self.create_publisher(String, "/bci/objetos", 10)
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", qos)

        self.timer = self.create_timer(1.0 / hz, self._ciclo)

        self.get_logger().info(
            f"e2_node a {hz} Hz, N={self.n_objetos} objetos, "
            f"{self.cfg.etapa2.decisiones_necesarias(self.n_objetos)} "
            f"decisiones esperadas")

    # -------------------------------------------------------------------
    def _cb_imagen(self, msg: Image) -> None:
        if not _HAY_CV2:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "bgr8":
                self._ultimo_frame = arr.reshape(msg.height, msg.width, 3)
            elif msg.encoding == "rgb8":
                img = arr.reshape(msg.height, msg.width, 3)
                self._ultimo_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                self._ultimo_frame = arr.reshape(msg.height, msg.width)
        except ValueError:
            self.get_logger().warn("Frame malformado",
                                   throttle_duration_sec=5.0)

    def _cb_scan(self, msg: LaserScan) -> None:
        rmax = self.cfg.robot.lidar_rango_max
        self._distancias = [
            rmax if (r != r or r == float("inf") or r <= 0.0)
            else min(float(r), rmax) for r in msg.ranges]

    def _cb_cmd(self, msg: BciCommand) -> None:
        # estado_fsm == 0 es DETENIDO
        self._robot_detenido = (msg.estado_fsm == 0)

    def _cb_tasm(self, msg: TASMState) -> None:
        """
        Recibe el voto de la busqueda binaria.

        Solo cuenta si el estado es IC (codigo 2) y la ventana es valida.
        Durante TR e Idle NO se registra voto: la iteracion espera.

        Esa es la compuerta cognitiva aplicada a la Etapa 2. Un voto durante
        una transicion de mirada seria irrecuperable: el objeto correcto
        saldria de la lista y no habria forma de volver.
        """
        if self._etapa != 2 or self._busqueda is None:
            return
        if msg.state != 2 or not msg.is_valid:
            return
        if self._busqueda.terminada:
            return

        # Los indices 0 y 1 son giro izquierda y derecha, que en la Etapa 2
        # significan grupo izquierdo y derecho. Es coherente: el usuario
        # aprende una asociacion y le vale en ambas etapas.
        if msg.freq_idx == 0:
            lado = LadoSeleccion.IZQUIERDA
        elif msg.freq_idx == 1:
            lado = LadoSeleccion.DERECHA
        else:
            return

        dur = (time.monotonic() - self._t_iter) if self._t_iter else 0.0
        self._busqueda.votar(lado, duracion=dur)
        self._t_iter = time.monotonic()

        self.get_logger().info(
            f"Voto: {lado.value}, quedan {self._busqueda.n_candidatos} "
            f"objetos")

        if self._busqueda.terminada:
            r = self._busqueda.resultado()
            self._objetivo_final = r.objeto_elegido
            self.aproximacion.reiniciar()
            self.get_logger().info(
                f"Objeto seleccionado: indice {r.objeto_elegido} "
                f"en {r.n_decisiones} decisiones "
                f"(teoricas {r.n_decisiones_teoricas})")

    # -------------------------------------------------------------------
    def _detectar(self) -> None:
        """Procesa el ultimo frame."""
        if self._ultimo_frame is None:
            return
        marcadores = self.det_aruco.detectar(self._ultimo_frame)
        detecciones = self.det_yolo.detectar(self._ultimo_frame)
        self._objetos = self.fusion.fusionar(marcadores, detecciones)

    # -------------------------------------------------------------------
    def _publicar_objetos(self) -> None:
        """Envia la lista de objetos y su agrupacion a la interfaz."""
        izq, der = ([], [])
        if self._busqueda is not None and not self._busqueda.terminada:
            izq, der = self._busqueda.grupos_actuales()

        d = {
            "objetos": [
                {"id": o.id_aruco,
                 "bbox": [round(x, 1) for x in o.bbox],
                 "d": round(o.distancia, 3),
                 "etiqueta": o.etiqueta}
                for o in self._objetos
            ],
            "grupo_izq": list(izq),
            "grupo_der": list(der),
            "objetivo": self._objetivo_final,
        }
        m = String()
        m.data = json.dumps(d)
        self.pub_objetos.publish(m)

    # -------------------------------------------------------------------
    def _ciclo(self) -> None:
        self._detectar()
        t = time.monotonic() - self._t0

        # --- ETAPA 1: comprobar si procede transicionar ---
        if self._etapa == 1:
            ev = self.transicion.evaluar(
                self._objetos, self._distancias, self._robot_detenido, t)

            if ev.procede:
                self._etapa = 2
                objs = [Objeto(id_aruco=o.id_aruco, x_center=o.centro_x)
                        for o in self._objetos]
                if len(objs) >= 2:
                    self._busqueda = BusquedaBinaria(self.cfg, objs)
                    self._t_iter = time.monotonic()
                    self.get_logger().info(
                        f"ETAPA 2 con {len(objs)} objetos. "
                        f"{self._busqueda.decisiones_teoricas} decisiones "
                        f"esperadas")
                else:
                    self._etapa = 1

        # --- ETAPA 2: seleccion o aproximacion ---
        elif self._objetivo_final is not None:
            objetivo = (self._objetos[self._objetivo_final]
                        if self._objetivo_final < len(self._objetos)
                        else None)
            cmd = self.aproximacion.ciclo(objetivo, t)

            tw = Twist()
            tw.linear.x = float(cmd.u)
            tw.angular.z = float(cmd.omega)
            self.pub_vel.publish(tw)

            if cmd.terminado:
                self.get_logger().info(
                    f"Aproximacion terminada: {cmd.estado.value} "
                    f"--- {cmd.motivo}")
                self.pub_vel.publish(Twist())
                self._objetivo_final = None

        # --- Publicar estado ---
        e = Int8()
        e.data = self._etapa
        self.pub_etapa.publish(e)
        self._publicar_objetos()

    # -------------------------------------------------------------------
    def volver_a_etapa1(self) -> None:
        """
        Vuelve a navegacion.

        La transicion es REVERSIBLE: si el sistema cambio de etapa cuando el
        usuario no lo queria, el comando de parar lo devuelve a navegacion.
        """
        self._etapa = 1
        self._busqueda = None
        self._objetivo_final = None
        self.transicion.reiniciar()
        self.aproximacion.reiniciar()


def main(args=None):
    rclpy.init(args=args)
    nodo = E2Node()
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

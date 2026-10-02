#!/usr/bin/env python3
"""
safety_node.py --- Supervision de seguridad independiente
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: codigo funcional pero NO verificado sobre ROS2. La logica de
seguridad esta en logica/nodos_logica.py y si tiene pruebas.
===========================================================================

QUE HACE
--------
Vigila el LiDAR de forma INDEPENDIENTE del resto del sistema y publica en
/cmd_vel un comando de escape si detecta un obstaculo demasiado cerca.

POR QUE UN NODO SEPARADO SI mission_node YA EVALUA LA SEGURIDAD
----------------------------------------------------------------
Redundancia deliberada. mission_node evalua la seguridad como parte de su
arbitraje, pero si mission_node se colgara, dejara de publicar o tuviera un
error de logica, el robot quedaria sin proteccion.

Este nodo es simple a proposito: lee el LiDAR, compara con un umbral, y
publica. Cuanto menos codigo tenga, menos posibilidades de fallar.

ESTE NODO ES DELIBERADAMENTE AJENO A TASM
------------------------------------------
No se suscribe a /tasm/state y no debe hacerlo nunca.

El override tiene que activarse igual si el usuario esta en control
intencional, en transicion de mirada o en reposo: un obstaculo a 15 cm es
igual de peligroso en los tres casos.

Ademas, acoplarlo a TASM introduciria un modo de fallo nuevo: un error del
detector podria desactivar la proteccion.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from turtlebot3_bci_3c2.logica_bridge import cargar_config, LogicaSeguridad


class SafetyNode(Node):
    """Supervision de seguridad."""

    def __init__(self):
        super().__init__("safety_node")

        self.cfg = cargar_config()
        self.logica = LogicaSeguridad(self.cfg)

        self.declare_parameter("hz", self.cfg.ros2.hz_seguridad)
        hz = self.get_parameter("hz").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos)

        # El comando de emergencia va por transporte FIABLE. Es la excepcion
        # a la regla general de best-effort: un comando de parada perdido es
        # mucho peor que uno con 20 ms de retraso.
        qos_fiable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", qos_fiable)
        self.pub_est = self.create_publisher(Bool, "/bci/emergencia", 10)

        self._distancias = []
        self.timer = self.create_timer(1.0 / hz, self._ciclo)

        self.get_logger().info(
            f"safety_node a {hz} Hz. "
            f"Umbral: {self.cfg.robot.rho_safe} m, "
            f"salida: {self.cfg.robot.rho_safe + self.cfg.robot.rho_hist} m "
            f"durante {self.cfg.robot.n_hist} ciclos")

    # -------------------------------------------------------------------
    def _cb_scan(self, msg: LaserScan) -> None:
        rango_max = self.cfg.robot.lidar_rango_max
        self._distancias = [
            rango_max if (r != r or r == float("inf") or r <= 0.0)
            else min(float(r), rango_max)
            for r in msg.ranges
        ]

    # -------------------------------------------------------------------
    def _ciclo(self) -> None:
        if not self._distancias:
            return

        est = self.logica.evaluar(self._distancias)

        b = Bool()
        b.data = bool(est.emergencia)
        self.pub_est.publish(b)

        if est.emergencia:
            v = self.logica.comando_escape(est)
            t = Twist()
            t.linear.x = float(v.lineal)
            t.angular.z = float(v.angular)
            self.pub_vel.publish(t)

            self.get_logger().warn(
                f"EMERGENCIA: obstaculo a {est.rho_min:.3f} m",
                throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    nodo = SafetyNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

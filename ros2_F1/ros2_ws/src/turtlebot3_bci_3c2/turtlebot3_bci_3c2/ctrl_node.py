#!/usr/bin/env python3
"""
ctrl_node.py --- Lazo de control de velocidad
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: codigo funcional pero NO verificado sobre ROS2. La logica del PID
esta en logica/nodos_logica.py y si tiene pruebas.
===========================================================================

QUE HACE
--------
Sigue la velocidad de referencia que publica mission_node, usando un PID
discreto con antiwindup, y publica en /cmd_vel.

DONDE ENTRA lambda_bci (FASE 2)
-----------------------------------------
En la Fase 1 el controlador es un PID simple: la referencia
entra tal cual.

En la Fase 2, la ganancia de seguimiento se pondera por
lambda_bci, la probabilidad continua de que el usuario este en control
intencional. Cuando esa confianza baja, el controlador prioriza trayectoria
suave sobre seguir agresivamente un comando incierto.

El campo lambda_bci YA VIAJA en el mensaje TASMState y se registra, aunque
en esta fase no se use. Asi el paso a la Fase 2 no requiere tocar
la cadena de comunicacion.

El bloque marcado abajo es el punto exacto donde se conecta.
"""

import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from turtlebot3_bci_3c2.msg import BciCommand, TASMState

from turtlebot3_bci_3c2.logica_bridge import (
    cargar_config, LogicaControl, Velocidad)


class CtrlNode(Node):
    """Lazo de control de velocidad."""

    def __init__(self):
        super().__init__("ctrl_node")

        self.cfg = cargar_config()
        self.pid = LogicaControl(self.cfg)

        self.declare_parameter("hz", self.cfg.ros2.hz_control)
        self.declare_parameter("usar_lambda_bci", False)
        hz = self.get_parameter("hz").value
        self.usar_lambda = self.get_parameter("usar_lambda_bci").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(BciCommand, "/bci/command", self._cb_cmd, qos)
        self.create_subscription(Odometry, "/odom", self._cb_odom, qos)
        self.create_subscription(TASMState, "/tasm/state", self._cb_tasm, qos)

        self.pub = self.create_publisher(Twist, "/cmd_vel", qos)

        self._ref = Velocidad(0.0, 0.0)
        self._medida = Velocidad(0.0, 0.0)
        self._lambda = 1.0
        self._t_cmd: Optional[float] = None

        self.timer = self.create_timer(1.0 / hz, self._ciclo)

        self.get_logger().info(
            f"ctrl_node a {hz} Hz. "
            f"Ganancia variable: {'activa' if self.usar_lambda else 'inactiva'}")

    # -------------------------------------------------------------------
    def _cb_cmd(self, msg: BciCommand) -> None:
        self._ref = Velocidad(float(msg.u_ref), float(msg.omega_ref))
        self._t_cmd = time.monotonic()

    def _cb_odom(self, msg: Odometry) -> None:
        self._medida = Velocidad(
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

    def _cb_tasm(self, msg: TASMState) -> None:
        self._lambda = float(msg.lambda_bci)

    # -------------------------------------------------------------------
    def _ciclo(self) -> None:
        # Si mission_node deja de publicar, parar. No mantener la ultima
        # referencia: podria ser de hace varios segundos.
        if self._t_cmd is None or \
                (time.monotonic() - self._t_cmd) > self.cfg.bci.watchdog_s:
            self.pub.publish(Twist())
            self.pid.reiniciar()
            return

        ref = self._ref

        # ===== INICIO BLOQUE FASE 2 --- NO ACTIVAR EN FASE 1 =====
        #
        # Ganancia de seguimiento ponderada por la confianza cognitiva.
        #
        # La referencia efectiva mezcla el comando nuevo con el anterior en
        # proporcion a lambda_bci. Bajo el paradigma de enclavamiento el
        # punto fijo de esa mezcla es el comando ya enclavado, que es
        # legitimo: durante TR la FSM no transiciona, de modo que el comando
        # nuevo coincide con el previo y la mezcla es inocua.
        #
        # La supresion de comandos espurios la hace la compuerta de la FSM,
        # no esta ecuacion. Esta solo suaviza el salto de velocidad.
        #
        if self.usar_lambda:
            lam = max(0.0, min(1.0, self._lambda))
            ref = Velocidad(
                lam * ref.lineal + (1 - lam) * self._medida.lineal,
                lam * ref.angular + (1 - lam) * self._medida.angular,
            )
        # ===== FIN BLOQUE FASE 2 =====

        u = self.pid.actualizar(ref, self._medida)

        t = Twist()
        t.linear.x = float(u.lineal)
        t.angular.z = float(u.angular)
        self.pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    nodo = CtrlNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.pub.publish(Twist())   # detener el robot al salir
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

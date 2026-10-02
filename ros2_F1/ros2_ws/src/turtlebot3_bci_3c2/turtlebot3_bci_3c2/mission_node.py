#!/usr/bin/env python3
"""
mission_node.py --- Arbitraje de mision
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional, pero NO se ha podido verificar
por ejecucion: ROS2 no estaba disponible en el entorno donde se escribio.

La logica de decision esta en logica/nodos_logica.py y SI esta verificada
con pruebas automatizadas. Este nodo se limita a envolverla.

Si algo falla, comprueba primero si el problema esta en la plomeria ROS2 o
en la decision: `python run_ros2.py mision` ejercita la logica sin ROS2.
===========================================================================

QUE HACE ESTE NODO
------------------
Es el que decide. Recibe el estado cognitivo, el escaneo LiDAR y la etapa
actual, y publica el comando de velocidad de referencia.

Contiene la maquina de estados con enclavamiento, que es el mecanismo
central del trabajo: un comando solo se engancha si TASM confirma control
intencional durante n_conf ventanas consecutivas.


EL ORDEN DE PRIORIDADES NO ES NEGOCIABLE
-----------------------------------------
  1. EMERGENCIA   Se impone sobre todo, incluso sobre un comando con
                  confianza maxima. Un obstaculo a 15 cm es peligroso
                  independientemente de lo que el usuario quiera.

  2. WATCHDOG     Sin mensajes de TASM, parada segura. No se mantiene el
                  comando enclavado: si no sabemos si el usuario sigue ahi,
                  seguir moviendose no es aceptable.

  3. ETAPA 2      Robot estatico durante la seleccion.

  4. ETAPA 1      Comando del usuario, compuertado por TASM y asistido por
                  el campo potencial.

Si la emergencia no fuera lo primero, un error de TASM podria impedir que se
activara la proteccion.
"""

import time
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from turtlebot3_bci_3c2.msg import TASMState, BciCommand

from turtlebot3_bci_3c2.logica_bridge import (
    cargar_config, LogicaMision, MensajeTASMRecibido, ModoOperacion, Etapa)


NOMBRE_ESTADO = {0: "Idle", 1: "TR", 2: "IC"}
CODIGO_MODO = {
    ModoOperacion.BCI_MANUAL: 0,
    ModoOperacion.EMERGENCIA: 1,
    ModoOperacion.DETENIDO_SEGURO: 2,
}
CODIGO_FSM = {
    "DETENIDO": 0, "AVANZANDO": 1,
    "ROTANDO_IZQ": 2, "ROTANDO_DER": 3,
}


class MissionNode(Node):
    """Arbitro de mision."""

    def __init__(self):
        super().__init__("mission_node")

        self.cfg = cargar_config()
        self.logica = LogicaMision(self.cfg)

        self.declare_parameter("hz", self.cfg.ros2.hz_control)
        hz = self.get_parameter("hz").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Suscripciones ---
        self.create_subscription(TASMState, "/tasm/state", self._cb_tasm, qos)
        self.create_subscription(LaserScan, "/scan", self._cb_scan, qos)
        self.create_subscription(Int8, "/bci/etapa", self._cb_etapa, 10)

        # --- Publicador ---
        self.pub = self.create_publisher(BciCommand, "/bci/command", qos)

        # --- Estado ---
        self._msg_tasm: Optional[MensajeTASMRecibido] = None
        self._t_tasm: Optional[float] = None
        self._distancias: List[float] = []
        self._t0 = time.monotonic()

        self.timer = self.create_timer(1.0 / hz, self._ciclo)

        self.get_logger().info(
            f"mission_node a {hz} Hz. "
            f"Racha de confirmacion: {self.cfg.bci.n_conf} ventanas "
            f"({self.cfg.bci.t_confirmacion*1000:.0f} ms)")

    # -------------------------------------------------------------------
    def _cb_tasm(self, msg: TASMState) -> None:
        self._msg_tasm = MensajeTASMRecibido(
            estado=NOMBRE_ESTADO.get(msg.state, "Idle"),
            freq_idx=int(msg.freq_idx),
            p_max=float(msg.p_max),
            lambda_bci=float(msg.lambda_bci),
            valido=bool(msg.is_valid),
            timestamp=time.monotonic(),
        )
        self._t_tasm = time.monotonic()

    # -------------------------------------------------------------------
    def _cb_scan(self, msg: LaserScan) -> None:
        # Los infinitos y NaN del LiDAR indican "nada detectado en ese rayo",
        # que a efectos de seguridad equivale al rango maximo.
        rango_max = self.cfg.robot.lidar_rango_max
        self._distancias = [
            rango_max if (r != r or r == float("inf") or r <= 0.0)
            else min(float(r), rango_max)
            for r in msg.ranges
        ]

    # -------------------------------------------------------------------
    def _cb_etapa(self, msg: Int8) -> None:
        nueva = Etapa.SELECCION if msg.data == 2 else Etapa.NAVEGACION
        if nueva != self.logica.etapa:
            self.get_logger().info(f"Cambio a etapa {nueva.value}")
            self.logica.cambiar_etapa(nueva)

    # -------------------------------------------------------------------
    def _watchdog_expirado(self) -> bool:
        if self._t_tasm is None:
            return True
        return (time.monotonic() - self._t_tasm) > self.cfg.bci.watchdog_s

    # -------------------------------------------------------------------
    def _ciclo(self) -> None:
        """Un ciclo de decision."""
        t = time.monotonic() - self._t0
        expirado = self._watchdog_expirado()

        d = self.logica.ciclo(
            mensaje=None if expirado else self._msg_tasm,
            distancias=self._distancias,
            t=t,
            watchdog_expirado=expirado,
        )

        m = BciCommand()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.modo = CODIGO_MODO[d.modo]
        m.etapa = d.etapa.value
        m.u_ref = float(d.velocidad.lineal)
        m.omega_ref = float(d.velocidad.angular)
        m.estado_fsm = CODIGO_FSM.get(self.logica.fsm.estado.value, 0)
        m.transiciono = bool(d.transiciono_fsm)
        m.rho_min = float(min(self._distancias)) if self._distancias else 0.0
        m.motivo = d.motivo

        self.pub.publish(m)

        # Avisar de los cambios de modo, que son eventos poco frecuentes y
        # relevantes para entender un trial despues.
        if d.modo == ModoOperacion.EMERGENCIA:
            self.get_logger().warn(
                f"EMERGENCIA: {d.motivo}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    nodo = MissionNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

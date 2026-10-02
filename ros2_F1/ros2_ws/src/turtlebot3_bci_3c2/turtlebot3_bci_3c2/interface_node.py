#!/usr/bin/env python3
"""
interface_node.py --- Envio de estado a la interfaz de Windows
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: codigo funcional pero NO verificado sobre ROS2.
===========================================================================

QUE HACE
--------
Envia a la maquina Windows la informacion que la interfaz necesita para
dibujar: estado cognitivo, etapa actual, si hay emergencia, y los objetos
detectados en la Etapa 2.

Es el camino inverso de bci_node: alli la informacion va de Windows a Linux;
aqui vuelve.

POR QUE LA INTERFAZ NECESITA SABER EL ESTADO COGNITIVO
-------------------------------------------------------
En un sistema sincrono el robot siempre responde, asi que el usuario nunca
se pregunta por que no pasa nada.

En este sistema el robot puede legitimamente NO aceptar un comando durante
una transicion de mirada. Sin realimentacion, el usuario interpretaria ese
silencio como una averia y probablemente insistiria, empeorando la
situacion.

El borde de color de la interfaz resuelve eso: verde cuando el sistema esta
aceptando, ambar durante una transicion, gris en reposo.

Es relevante para la medida de carga cognitiva del experimento.
"""

import json
import socket
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, Int8, String
from turtlebot3_bci_3c2.msg import TASMState, BciCommand


NOMBRE_ESTADO = {0: "Idle", 1: "TR", 2: "IC"}


class InterfaceNode(Node):
    """Puente ROS2 -> interfaz de Windows."""

    def __init__(self):
        super().__init__("interface_node")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("puerto", 5557)
        self.declare_parameter("hz", 30.0)

        self.host = self.get_parameter("host").value
        self.puerto = self.get_parameter("puerto").value
        hz = self.get_parameter("hz").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(TASMState, "/tasm/state", self._cb_tasm, qos)
        self.create_subscription(BciCommand, "/bci/command", self._cb_cmd, qos)
        self.create_subscription(Bool, "/bci/emergencia", self._cb_emer, 10)
        self.create_subscription(String, "/bci/objetos", self._cb_obj, 10)

        self._estado = "Idle"
        self._etapa = 1
        self._emergencia = False
        self._objetos = "[]"
        self._grupos = "[[],[]]"

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

        self._conectar()
        self.timer = self.create_timer(1.0 / hz, self._enviar)

        self.get_logger().info(
            f"interface_node -> {self.host}:{self.puerto} a {hz} Hz")

    # -------------------------------------------------------------------
    def _conectar(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.host, self.puerto))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(0.02)
            with self._lock:
                self._sock = s
            return True
        except OSError:
            with self._lock:
                self._sock = None
            return False

    # -------------------------------------------------------------------
    def _cb_tasm(self, msg: TASMState) -> None:
        self._estado = NOMBRE_ESTADO.get(msg.state, "Idle")

    def _cb_cmd(self, msg: BciCommand) -> None:
        self._etapa = int(msg.etapa)

    def _cb_emer(self, msg: Bool) -> None:
        self._emergencia = bool(msg.data)

    def _cb_obj(self, msg: String) -> None:
        # El nodo de vision (Bloque 5) publica aqui los objetos detectados y
        # su agrupacion en la busqueda binaria.
        self._objetos = msg.data

    # -------------------------------------------------------------------
    def _enviar(self) -> None:
        with self._lock:
            sock = self._sock
        if sock is None:
            self._conectar()
            return

        d = {
            "estado_tasm": self._estado,
            "etapa": self._etapa,
            "emergencia": self._emergencia,
            "objetos": self._objetos,
        }

        try:
            sock.sendall((json.dumps(d) + "\n").encode("utf-8"))
        except socket.timeout:
            pass    # se descarta: el siguiente sera mas fresco
        except OSError:
            with self._lock:
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None


def main(args=None):
    rclpy.init(args=args)
    nodo = InterfaceNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

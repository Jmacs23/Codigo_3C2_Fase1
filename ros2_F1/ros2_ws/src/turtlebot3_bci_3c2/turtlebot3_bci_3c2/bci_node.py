#!/usr/bin/env python3
"""
bci_node.py --- Puente entre la maquina Windows y ROS2
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional, pero NO se ha podido verificar
por ejecucion: ROS2 no estaba disponible en el entorno donde se escribio.

La logica que SI esta verificada es la de logica/nodos_logica.py, que este
nodo se limita a envolver. Los errores probables aqui son de plomeria ROS2
(nombres de topics, tipos de mensaje, QoS), no de decision.

Si algo falla:
  1. Comprueba primero con `ros2 topic echo /tasm/state` que el mensaje sale.
  2. Verifica que el emisor de Windows esta corriendo y conectado.
  3. Reportalo indicando el traceback completo y la version de ROS2.
===========================================================================

QUE HACE ESTE NODO
------------------
Escucha en un socket TCP los mensajes que envia la maquina Windows, los
traduce a mensajes ROS2, y los publica en /tasm/state.

Es el unico punto por el que la informacion cognitiva entra al sistema
robotico. Todo lo demas del lado Linux consume ese topic.


POR QUE UN HILO SEPARADO PARA EL SOCKET
----------------------------------------
La lectura del socket es bloqueante por naturaleza. Si se hiciera en el
callback del temporizador, un retraso de red congelaria el nodo entero.

El hilo lector escribe en una variable protegida por un lock, y el
temporizador publica lo que haya. Si no ha llegado nada nuevo, no publica: es
preferible que el consumidor note el silencio y active su watchdog a que
reciba un mensaje repetido que parezca fresco.
"""

import json
import socket
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from turtlebot3_bci_3c2.msg import TASMState


# Traduccion de nombre de estado a codigo del mensaje.
# El orden importa y debe coincidir con el del contrato.
CODIGO_ESTADO = {"Idle": 0, "TR": 1, "IC": 2}


class BciNode(Node):
    """Puente TCP -> ROS2 para el estado cognitivo."""

    def __init__(self):
        super().__init__("bci_node")

        # --- Parametros ---
        self.declare_parameter("tcp_puerto", 5556)
        self.declare_parameter("tcp_host", "0.0.0.0")
        self.declare_parameter("hz", 20.0)
        self.declare_parameter("watchdog_s", 0.4)

        self.puerto = self.get_parameter("tcp_puerto").value
        self.host = self.get_parameter("tcp_host").value
        hz = self.get_parameter("hz").value
        self.watchdog_s = self.get_parameter("watchdog_s").value

        # --- Publicador ---
        # Best-effort y cola corta: en control en tiempo real un mensaje
        # viejo es peor que ningun mensaje.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(TASMState, "/tasm/state", qos)

        # --- Estado compartido entre el hilo y el temporizador ---
        self._lock = threading.Lock()
        self._ultimo: Optional[dict] = None
        self._t_ultimo = 0.0
        self._nuevo = False

        self._recibidos = 0
        self._descartados = 0

        # --- Hilo lector ---
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle_socket, daemon=True)
        self._hilo.start()

        # --- Temporizador de publicacion ---
        self.timer = self.create_timer(1.0 / hz, self._publicar)

        self.get_logger().info(
            f"bci_node escuchando en {self.host}:{self.puerto} a {hz} Hz")

    # -------------------------------------------------------------------
    def _bucle_socket(self) -> None:
        """
        Hilo lector. Acepta conexiones y procesa lineas JSON.

        Si la conexion se cae, vuelve a esperar. No se rinde: la maquina
        Windows puede reiniciarse a mitad de sesion y debe poder reconectar.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.puerto))
        srv.listen(1)
        srv.settimeout(1.0)

        while not self._parar.is_set():
            try:
                con, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self.get_logger().info(f"Conexion desde {addr}")
            con.settimeout(0.5)
            buffer = ""

            while not self._parar.is_set():
                try:
                    datos = con.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if not datos:
                    break

                buffer += datos.decode("utf-8", errors="ignore")

                # Los mensajes van separados por salto de linea. Puede llegar
                # mas de uno en el mismo paquete, o uno partido entre dos.
                while "\n" in buffer:
                    linea, buffer = buffer.split("\n", 1)
                    self._procesar(linea)

            con.close()
            self.get_logger().warn("Conexion perdida; esperando reconexion")

        srv.close()

    # -------------------------------------------------------------------
    def _procesar(self, linea: str) -> None:
        """Interpreta una linea JSON y la guarda."""
        linea = linea.strip()
        if not linea:
            return

        try:
            d = json.loads(linea)
        except (json.JSONDecodeError, ValueError):
            self._descartados += 1
            return

        if d.get("estado") not in CODIGO_ESTADO:
            self._descartados += 1
            return

        with self._lock:
            self._ultimo = d
            self._t_ultimo = time.monotonic()
            self._nuevo = True
        self._recibidos += 1

    # -------------------------------------------------------------------
    def _publicar(self) -> None:
        """
        Publica el ultimo mensaje recibido, si es nuevo.

        NO republica mensajes antiguos. Si la conexion se cae, el topic
        simplemente deja de recibir, y el watchdog de mission_node se
        encarga. Republicar daria la falsa impresion de que el usuario sigue
        conectado.
        """
        with self._lock:
            if not self._nuevo or self._ultimo is None:
                return
            d = dict(self._ultimo)
            self._nuevo = False

        m = TASMState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "bci"

        m.state = CODIGO_ESTADO[d["estado"]]
        m.freq_idx = int(d.get("freq_idx", -1))
        m.p_max = float(d.get("p_max", 0.0))
        m.lambda_bci = float(d.get("lambda_bci", 0.0))
        m.is_valid = bool(d.get("valido", True))
        m.rho = [float(x) for x in d.get("rho", [])]
        m.rho_grad = [float(x) for x in d.get("rho_grad", [])]

        self.pub.publish(m)

    # -------------------------------------------------------------------
    def destroy_node(self):
        self._parar.set()
        if self._hilo.is_alive():
            self._hilo.join(timeout=2.0)
        self.get_logger().info(
            f"Recibidos: {self._recibidos}, descartados: {self._descartados}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    nodo = BciNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

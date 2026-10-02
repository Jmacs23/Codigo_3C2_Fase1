#!/usr/bin/env python3
"""
video_publisher.py --- Envio del video de la camara a la maquina Windows
Paquete: turtlebot3_bci_3c2

===========================================================================
AVISO: codigo funcional pero NO verificado sobre ROS2.
===========================================================================

QUE HACE
--------
Toma los frames de la camara del robot, los comprime a JPEG, y los envia por
TCP a la maquina Windows, donde PsychoPy los usa como fondo de la interfaz.

Es el lado emisor; el receptor esta en el Bloque 2 (video_source.FuenteTCP).

PROTOCOLO
---------
    [4 bytes: longitud, big-endian] [datos JPEG]

El prefijo de longitud es necesario porque TCP es un flujo continuo sin
fronteras de mensaje. Sin el no habria forma de saber donde acaba un frame.

POR QUE JPEG Y NO DATOS CRUDOS
-------------------------------
Un frame de 1280x960 en RGB ocupa 3.5 MB. A 30 fps serian 110 MB/s, que
saturaria una red de gigabit y añadiria latencia.

Comprimido al 70% baja a unos 80 kB por frame, es decir 2.4 MB/s.

La perdida de calidad afecta a lo que el usuario VE, no a la deteccion de
objetos: esa ocurre en Linux, sobre el frame original antes de comprimir.

SI LA RED SE SATURA, SE DESCARTAN FRAMES
-----------------------------------------
El envio es no bloqueante. Si el socket no admite mas datos, el frame se
descarta en lugar de acumularse en una cola.

Es la decision correcta: un frame de video atrasado no sirve de nada, y
encolarlos aumentaria la latencia progresivamente hasta hacer la interfaz
inusable. Mejor perder frames y mantener la imagen fresca.

Ademas, el estimulo SSVEP NO depende de este video: se genera con el
contador de frames de la pantalla de Windows. Una perdida de frames degrada
la experiencia visual pero no la senal.
"""

import socket
import struct
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

try:
    import cv2
    _HAY_CV2 = True
except ImportError:
    _HAY_CV2 = False


class VideoPublisher(Node):
    """Emisor de video hacia la maquina Windows."""

    def __init__(self):
        super().__init__("video_publisher")

        if not _HAY_CV2:
            self.get_logger().error(
                "OpenCV no esta instalado. Instala con: "
                "pip install opencv-python")
            raise RuntimeError("OpenCV requerido")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("puerto", 5555)
        self.declare_parameter("calidad", 70)
        self.declare_parameter("topic", "/camera/image_raw")

        self.host = self.get_parameter("host").value
        self.puerto = self.get_parameter("puerto").value
        self.calidad = int(self.get_parameter("calidad").value)
        topic = self.get_parameter("topic").value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, topic, self._cb_imagen, qos)

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._enviados = 0
        self._descartados = 0

        self._conectar()
        self.get_logger().info(
            f"video_publisher: {topic} -> {self.host}:{self.puerto} "
            f"(JPEG {self.calidad}%)")

    # -------------------------------------------------------------------
    def _conectar(self) -> bool:
        """Intenta conectar con el receptor de Windows."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.puerto))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # No bloqueante para el envio: si el buffer esta lleno,
            # preferimos descartar el frame a esperar.
            s.settimeout(0.02)
            with self._lock:
                self._sock = s
            self.get_logger().info("Conectado con la interfaz de Windows")
            return True
        except OSError:
            with self._lock:
                self._sock = None
            return False

    # -------------------------------------------------------------------
    def _cb_imagen(self, msg: Image) -> None:
        """Comprime y envia un frame."""
        with self._lock:
            sock = self._sock

        if sock is None:
            # Reintentar cada cierto numero de frames, no en cada uno
            self._descartados += 1
            if self._descartados % 60 == 0:
                self._conectar()
            return

        # --- Convertir el mensaje ROS a array ---
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "bgr8"):
                img = arr.reshape(msg.height, msg.width, 3)
                if msg.encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                img = cv2.cvtColor(arr.reshape(msg.height, msg.width),
                                   cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(
                    f"Codificacion no soportada: {msg.encoding}",
                    throttle_duration_sec=5.0)
                return
        except ValueError:
            self.get_logger().warn("Frame malformado",
                                   throttle_duration_sec=5.0)
            return

        # --- Comprimir ---
        ok, jpg = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.calidad])
        if not ok:
            return

        datos = jpg.tobytes()

        # --- Enviar ---
        try:
            sock.sendall(struct.pack(">I", len(datos)) + datos)
            self._enviados += 1
        except socket.timeout:
            # El buffer esta lleno: se descarta el frame. Un frame de video
            # atrasado no sirve, y encolarlo aumentaria la latencia.
            self._descartados += 1
        except OSError:
            self.get_logger().warn("Conexion perdida con la interfaz")
            with self._lock:
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None

    # -------------------------------------------------------------------
    def destroy_node(self):
        total = self._enviados + self._descartados
        self.get_logger().info(
            f"Frames enviados: {self._enviados}, "
            f"descartados: {self._descartados} "
            f"({self._descartados/total*100:.1f}%)" if total else "")
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        nodo = VideoPublisher()
    except RuntimeError:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

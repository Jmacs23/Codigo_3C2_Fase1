"""
video_source.py --- Fuente de video para el fondo de la interfaz
Bloque 2: estimulo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Entrega frames de video para usarlos como fondo de la interfaz, sobre el que
se superponen los estimulos parpadeantes.

EL PROBLEMA DE LAS DOS MAQUINAS
-------------------------------
La camara del robot publica en ROS2, que corre en Linux. PsychoPy y el
amplificador g.tec corren en Windows, porque el SDK del amplificador solo
existe para esa plataforma.

Hay que transportar el video de una maquina a la otra. Este modulo abstrae
ese transporte detras de una interfaz comun, de modo que el resto del codigo
no dependa de como llegue el video.

TRES FUENTES INTERCAMBIABLES
----------------------------
  FuenteSintetica  Genera una escena artificial. Para desarrollar sin robot.
  FuenteArchivo    Reproduce un video grabado. Para pruebas reproducibles.
  FuenteTCP        Recibe frames por red desde el nodo ROS2. Produccion.

Todas exponen el mismo metodo `leer()`. Cambiar de una a otra es cambiar una
linea en la configuracion, no reescribir la interfaz.

UNA ADVERTENCIA SOBRE LATENCIA
------------------------------
El video de fondo puede llegar con retardo o perder frames sin que eso afecte
al estimulo, PORQUE SON PROCESOS INDEPENDIENTES. El estimulo se calcula del
contador de frames de la pantalla, no del video.

Es una propiedad deliberada del diseno: si el estimulo dependiera de la
llegada del video, un tiron de red degradaria la senal SSVEP. Al
desacoplarlos, un tiron de red solo degrada la experiencia visual del
usuario, que es mucho menos grave.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import socket
import struct
import time
import numpy as np

from config import Config


# ===========================================================================
# INTERFAZ COMUN
# ===========================================================================

class FuenteVideo(ABC):
    """
    Interfaz comun de todas las fuentes de video.

    Un frame es un array (alto, ancho, 3) con valores en [0, 255] y tipo
    uint8, en orden RGB.
    """

    @abstractmethod
    def leer(self) -> Optional[np.ndarray]:
        """Devuelve el siguiente frame, o None si no hay ninguno disponible.

        Debe ser NO BLOQUEANTE: si no hay frame nuevo, devolver None en lugar
        de esperar. Bloquear aqui detendria el bucle de renderizado del
        estimulo y provocaria perdida de frames de pantalla."""
        ...

    @abstractmethod
    def cerrar(self) -> None:
        """Libera los recursos."""
        ...

    @property
    @abstractmethod
    def resolucion(self) -> Tuple[int, int]:
        """Resolucion (ancho, alto) en pixeles."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()


# ===========================================================================
# FUENTE SINTETICA
# ===========================================================================

class FuenteSintetica(FuenteVideo):
    """
    Genera una escena artificial en movimiento.

    Sirve para desarrollar la interfaz sin robot ni camara. La escena imita
    la estructura visual de un corredor con obstaculos: gradiente de fondo,
    lineas de fuga y rectangulos que se desplazan.

    No pretende parecerse a la escena real. Su proposito es tener contenido
    con textura y movimiento, que es lo que puede afectar a la percepcion del
    estimulo superpuesto.
    """

    def __init__(self, config: Config,
                 ancho: int = 640, alto: int = 480,
                 velocidad: float = 1.0):
        self.cfg = config
        self._ancho = ancho
        self._alto = alto
        self.velocidad = velocidad
        self._t0 = time.time()

        # Fondo estatico: gradiente vertical mas lineas de fuga
        self._fondo = self._construir_fondo()

    # -------------------------------------------------------------------
    def _construir_fondo(self) -> np.ndarray:
        img = np.zeros((self._alto, self._ancho, 3), dtype=np.float32)

        # Gradiente vertical: suelo mas claro, techo mas oscuro
        grad = np.linspace(0.15, 0.45, self._alto)[:, None]
        img += grad[:, :, None] * np.array([0.9, 0.9, 1.0])

        # Lineas de fuga hacia el punto central
        cx, cy = self._ancho // 2, int(self._alto * 0.55)
        for k in range(-4, 5):
            x0 = cx + k * self._ancho // 10
            for y in range(cy, self._alto):
                fr = (y - cy) / max(1, self._alto - cy)
                x = int(cx + (x0 - cx) * fr)
                if 0 <= x < self._ancho:
                    img[y, x] = np.minimum(img[y, x] + 0.18, 1.0)

        return img

    # -------------------------------------------------------------------
    def leer(self) -> Optional[np.ndarray]:
        t = (time.time() - self._t0) * self.velocidad
        img = self._fondo.copy()

        # Tres rectangulos que se desplazan, simulando obstaculos
        for i in range(3):
            fase = t * 0.35 + i * 2.1
            x = int((math.sin(fase) * 0.35 + 0.5) * self._ancho)
            y = int(self._alto * (0.45 + 0.12 * math.sin(fase * 0.7)))
            w = int(self._ancho * 0.09)
            h = int(self._alto * 0.16)
            x0, x1 = max(0, x - w // 2), min(self._ancho, x + w // 2)
            y0, y1 = max(0, y - h // 2), min(self._alto, y + h // 2)
            tono = 0.35 + 0.1 * i
            img[y0:y1, x0:x1] = tono

        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def cerrar(self) -> None:
        pass

    @property
    def resolucion(self) -> Tuple[int, int]:
        return (self._ancho, self._alto)


# ===========================================================================
# FUENTE DESDE ARCHIVO
# ===========================================================================

class FuenteArchivo(FuenteVideo):
    """
    Reproduce un video grabado.

    Util para pruebas reproducibles: el mismo video da exactamente la misma
    secuencia visual en cada ejecucion, lo que permite comparar condiciones
    sin que la escena introduzca variabilidad.

    Requiere OpenCV.
    """

    def __init__(self, ruta: str, en_bucle: bool = True):
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "FuenteArchivo requiere OpenCV. Instala con: "
                "pip install opencv-python"
            )
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(ruta)
        if not self._cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {ruta}")

        self.en_bucle = en_bucle
        self._ancho = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._alto = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def leer(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            if self.en_bucle:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        # OpenCV entrega BGR; el resto del sistema trabaja en RGB
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def cerrar(self) -> None:
        self._cap.release()

    @property
    def resolucion(self) -> Tuple[int, int]:
        return (self._ancho, self._alto)


# ===========================================================================
# FUENTE POR RED
# ===========================================================================

class FuenteTCP(FuenteVideo):
    """
    Recibe frames por TCP desde el nodo ROS2 que corre en Linux.

    PROTOCOLO
    ---------
    Cada frame se envia como:

        [4 bytes: longitud en big-endian] [datos JPEG]

    El prefijo de longitud es necesario porque TCP es un flujo continuo sin
    fronteras de mensaje: sin el, no habria forma de saber donde acaba un
    frame y empieza el siguiente.

    Se usa JPEG y no datos crudos porque un frame de 1280x960 en RGB ocupa
    3.5 MB, y a 30 fps serian 110 MB/s. Comprimido baja a unos 2 MB/s.

    NO BLOQUEANTE
    -------------
    El socket se configura con timeout corto. Si no hay frame nuevo, `leer`
    devuelve None inmediatamente en lugar de esperar. Esto es esencial: el
    bucle de renderizado del estimulo no puede detenerse a esperar video.
    """

    def __init__(self, host: str = "0.0.0.0", puerto: int = 5555,
                 timeout: float = 0.005):
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "FuenteTCP requiere OpenCV para decodificar JPEG. "
                "Instala con: pip install opencv-python"
            )
        self._cv2 = cv2

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, puerto))
        self._srv.listen(1)
        self._srv.settimeout(30.0)

        print(f"[FuenteTCP] Esperando conexion en {host}:{puerto} ...")
        self._con, addr = self._srv.accept()
        self._con.settimeout(timeout)
        print(f"[FuenteTCP] Conectado desde {addr}")

        self._buffer = b""
        self._ancho = 0
        self._alto = 0

    # -------------------------------------------------------------------
    def _recibir(self, n: int) -> Optional[bytes]:
        """Acumula bytes hasta tener n, o devuelve None si no llegan."""
        while len(self._buffer) < n:
            try:
                trozo = self._con.recv(65536)
            except socket.timeout:
                return None
            except OSError:
                return None
            if not trozo:
                return None
            self._buffer += trozo
        datos, self._buffer = self._buffer[:n], self._buffer[n:]
        return datos

    # -------------------------------------------------------------------
    def leer(self) -> Optional[np.ndarray]:
        cab = self._recibir(4)
        if cab is None:
            return None

        (longitud,) = struct.unpack(">I", cab)
        if longitud <= 0 or longitud > 50_000_000:
            # Longitud absurda: el flujo se desincronizo. Se limpia el buffer
            # para intentar recuperar en el siguiente frame.
            self._buffer = b""
            return None

        datos = self._recibir(longitud)
        if datos is None:
            return None

        arr = np.frombuffer(datos, dtype=np.uint8)
        frame = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if frame is None:
            return None

        self._alto, self._ancho = frame.shape[:2]
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def cerrar(self) -> None:
        try:
            self._con.close()
        except OSError:
            pass
        try:
            self._srv.close()
        except OSError:
            pass

    @property
    def resolucion(self) -> Tuple[int, int]:
        return (self._ancho, self._alto)


# ===========================================================================
# FABRICA
# ===========================================================================

def crear_fuente(config: Config, tipo: str = "sintetica",
                 **kwargs) -> FuenteVideo:
    """
    Crea la fuente de video indicada.

    Tipos: "sintetica", "archivo", "tcp".
    """
    if tipo == "sintetica":
        return FuenteSintetica(config, **kwargs)
    if tipo == "archivo":
        return FuenteArchivo(**kwargs)
    if tipo == "tcp":
        return FuenteTCP(**kwargs)
    raise ValueError(
        f"Tipo de fuente desconocido: '{tipo}'. "
        f"Opciones: sintetica, archivo, tcp"
    )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DE LA FUENTE DE VIDEO")
    print("=" * 72)
    print()

    fuente = crear_fuente(CONFIG, "sintetica", ancho=640, alto=480)
    print(f"  Tipo       : sintetica")
    print(f"  Resolucion : {fuente.resolucion}")
    print()

    print("-" * 72)
    print("LECTURA DE FRAMES")
    print("-" * 72)
    t0 = time.time()
    n = 0
    for i in range(60):
        f = fuente.leer()
        if f is not None:
            n += 1
            if i < 3:
                print(f"  Frame {i}: shape={f.shape}, dtype={f.dtype}, "
                      f"rango=[{f.min()}, {f.max()}]")
    dt = time.time() - t0
    print(f"  ...")
    print(f"  {n} frames leidos en {dt*1000:.1f} ms "
          f"({n/dt:.0f} fps equivalentes)")
    print()

    print("-" * 72)
    print("NOTA SOBRE LATENCIA")
    print("-" * 72)
    print("  El video de fondo y el estimulo son procesos INDEPENDIENTES.")
    print("  El estimulo se calcula del contador de frames de la pantalla,")
    print("  no de la llegada del video.")
    print()
    print("  Consecuencia: un tiron de red degrada la experiencia visual")
    print("  del usuario, pero NO la senal SSVEP. Si estuvieran acoplados,")
    print("  cada perdida de paquete corromperia el estimulo.")

    fuente.cerrar()

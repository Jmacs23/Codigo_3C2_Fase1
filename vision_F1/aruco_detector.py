"""
aruco_detector.py --- Deteccion de marcadores fiduciales y pose 3D
Bloque 5: vision_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este modulo SI se ha podido verificar por ejecucion: OpenCV con soporte
ArUco estaba disponible. Las pruebas generan marcadores sinteticos, los
detectan y comprueban la pose.

Lo que NO se ha verificado es el comportamiento con la camara real. La
distorsion de lente, la iluminacion del laboratorio y el desenfoque de
movimiento afectan a la deteccion de formas que una imagen sintetica no
reproduce.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Detecta marcadores ArUco en una imagen y calcula su posicion 3D respecto a
la camara.

Aporta tres cosas que YOLO por si solo no da:

  IDENTIDAD UNIVOCA   Cada marcador tiene un numero. Dos objetos identicos
                      no se confunden nunca.

  POSE 3D             Distancia y orientacion sin camara de profundidad. Es
                      lo que permite la condicion de transicion de etapa y
                      la aproximacion final.

  DISCRIMINACION      Los obstaculos no llevan marcador. Si tiene ArUco es
                      seleccionable; si no, es obstaculo. Sin ambiguedad y
                      sin entrenar nada.


DE DONDE SALE LA DISTANCIA
---------------------------
De la geometria de proyeccion. Si el marcador mide L metros de lado y en la
imagen ocupa p pixeles, la distancia es aproximadamente

    Z = f * L / p

con f la distancia focal en pixeles. OpenCV resuelve el problema completo
(PnP), que tiene en cuenta la perspectiva y devuelve tambien la orientacion.


DOS COSAS QUE HAY QUE MEDIR BIEN
---------------------------------
1. El lado del marcador IMPRESO, no el del archivo. Las impresoras escalan.
   Un error del 5% aqui da un error del 5% en TODAS las distancias.

2. Los intrinsecos de la camara. Los valores por defecto de config son
   nominales, derivados del campo visual. Sirven para desarrollar, pero la
   distorsion de lente de las camaras pequenas es apreciable y hay que
   calibrar con un patron de ajedrez antes de confiar en las distancias.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np

try:
    import cv2
    _HAY_CV2 = True
    _HAY_ARUCO = hasattr(cv2, "aruco")
except ImportError:
    _HAY_CV2 = False
    _HAY_ARUCO = False
    cv2 = None

from config import Config


# ===========================================================================
# RESULTADO
# ===========================================================================

@dataclass
class MarcadorDetectado:
    """Un marcador ArUco localizado en la imagen."""

    id: int
    """Identificador del marcador. Es la identidad del objeto."""

    esquinas: np.ndarray
    """Las cuatro esquinas en pixeles, forma (4, 2)."""

    centro_px: Tuple[float, float]

    posicion: Tuple[float, float, float]
    """
    Posicion en el marco de la camara (m): (x, y, z).
      x positivo a la derecha
      y positivo hacia abajo
      z positivo hacia adelante (profundidad)
    """

    rotacion: Tuple[float, float, float]
    """Vector de Rodrigues de la orientacion (rad)."""

    lado_px: float
    """Lado medio en pixeles. Sirve para juzgar si la pose es fiable."""

    @property
    def distancia(self) -> float:
        """Distancia euclidea al marcador (m)."""
        return float(np.linalg.norm(self.posicion))

    @property
    def profundidad(self) -> float:
        """
        Componente Z, la distancia frontal (m).

        Para la transicion de etapa importa mas que la euclidea: lo que
        cuenta es cuanto falta por avanzar, no la distancia en linea recta.
        """
        return float(self.posicion[2])

    @property
    def angulo_horizontal(self) -> float:
        """
        Angulo horizontal respecto al eje optico (rad).

        Positivo si el marcador esta a la derecha. Es lo que el controlador
        de aproximacion usa para centrarse.
        """
        x, _, z = self.posicion
        return math.atan2(x, z) if abs(z) > 1e-6 else 0.0

    def es_fiable(self, px_minimo: int) -> bool:
        """True si el marcador es lo bastante grande para confiar en su pose."""
        return self.lado_px >= px_minimo


# ===========================================================================
# DETECTOR
# ===========================================================================

class DetectorArUco:
    """
    Detector de marcadores con estimacion de pose.

    Uso:
        det = DetectorArUco(CONFIG)
        for m in det.detectar(imagen):
            print(m.id, m.profundidad, m.angulo_horizontal)
    """

    def __init__(self, config: Config):
        if not _HAY_CV2:
            raise ImportError(
                "DetectorArUco requiere OpenCV. "
                "Instala con: pip install opencv-python")
        if not _HAY_ARUCO:
            raise ImportError(
                "Esta version de OpenCV no incluye el modulo aruco. "
                "Instala opencv-contrib-python.")

        self.cfg = config
        v = config.vision

        self._dic = cv2.aruco.getPredefinedDictionary(v.diccionario_cv2())

        # La API de ArUco cambio entre OpenCV 4.6 y 4.7. Se soportan ambas
        # porque los laboratorios rara vez tienen la ultima version.
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._dic, self._params)
            self._api_nueva = True
        else:
            self._params = cv2.aruco.DetectorParameters_create()
            self._detector = None
            self._api_nueva = False

        self._K = v.matriz_camara()
        self._dist = v.distorsion()

        # Puntos del marcador en su propio marco, centrado en el origen. El
        # orden coincide con el de las esquinas que devuelve el detector:
        # superior-izquierda, superior-derecha, inferior-derecha,
        # inferior-izquierda.
        L = v.lado_marcador / 2.0
        self._puntos_objeto = np.array([
            [-L,  L, 0.0],
            [ L,  L, 0.0],
            [ L, -L, 0.0],
            [-L, -L, 0.0],
        ], dtype=np.float64)

    # -------------------------------------------------------------------
    def detectar(self, imagen: np.ndarray) -> List[MarcadorDetectado]:
        """
        Detecta todos los marcadores de una imagen.

        Devuelve la lista ORDENADA por posicion horizontal. Ese orden es el
        que la busqueda binaria necesita: el grupo izquierdo debe estar
        realmente a la izquierda de la pantalla.
        """
        if imagen is None or imagen.size == 0:
            return []

        gris = (cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                if imagen.ndim == 3 else imagen)

        if self._api_nueva:
            esquinas, ids, _ = self._detector.detectMarkers(gris)
        else:
            esquinas, ids, _ = cv2.aruco.detectMarkers(
                gris, self._dic, parameters=self._params)

        if ids is None or len(ids) == 0:
            return []

        salida: List[MarcadorDetectado] = []
        for esq, idm in zip(esquinas, ids.flatten()):
            m = self._construir(esq.reshape(4, 2), int(idm))
            if m is not None:
                salida.append(m)

        salida.sort(key=lambda m: m.centro_px[0])
        return salida

    # -------------------------------------------------------------------
    def _construir(self, esquinas: np.ndarray,
                   idm: int) -> Optional[MarcadorDetectado]:
        """Calcula la pose de un marcador a partir de sus esquinas."""
        pts = esquinas.astype(np.float64)

        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        lados = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
                 for i in range(4)]
        lado_px = float(np.mean(lados))

        # IPPE_SQUARE esta especializado en cuadrados planos y es mas estable
        # que el metodo generico para este caso concreto.
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self._puntos_objeto, pts, self._K, self._dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self._puntos_objeto, pts, self._K, self._dist)
            except cv2.error:
                return None

        if not ok:
            return None

        t = tvec.flatten()
        r = rvec.flatten()

        return MarcadorDetectado(
            id=idm,
            esquinas=pts,
            centro_px=(cx, cy),
            posicion=(float(t[0]), float(t[1]), float(t[2])),
            rotacion=(float(r[0]), float(r[1]), float(r[2])),
            lado_px=lado_px,
        )

    # -------------------------------------------------------------------
    def filtrar_fiables(self, marcadores: List[MarcadorDetectado]
                        ) -> List[MarcadorDetectado]:
        """
        Descarta los marcadores demasiado pequenos.

        Un marcador de pocos pixeles puede decodificarse bien y aun asi dar
        una pose imprecisa: el error de un pixel en las esquinas se amplifica
        al estimar la distancia.
        """
        px_min = self.cfg.vision.px_minimo_marcador
        return [m for m in marcadores if m.es_fiable(px_min)]


# ===========================================================================
# GENERACION DE MARCADORES
# ===========================================================================

def generar_marcador(config: Config, idm: int,
                     lado_px: int = 400,
                     borde: int = 40) -> np.ndarray:
    """
    Genera la imagen de un marcador para imprimir.

    IMPORTANTE AL IMPRIMIR
    ----------------------
    Comprueba que la impresora NO escale: elige "tamano real" o "100%", no
    "ajustar a pagina".

    Despues MIDE con una regla el marcador impreso y verifica que coincide
    con config.vision.lado_marcador. Un error del 5% aqui se traduce en un
    error del 5% en todas las distancias que el sistema calcule.

    El borde blanco no es decorativo: el detector lo necesita para encontrar
    el contorno del marcador.
    """
    if not _HAY_ARUCO:
        raise ImportError("Se requiere OpenCV con soporte aruco.")

    dic = cv2.aruco.getPredefinedDictionary(config.vision.diccionario_cv2())

    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dic, idm, lado_px)
    else:
        img = cv2.aruco.drawMarker(dic, idm, lado_px)

    if borde > 0:
        img = cv2.copyMakeBorder(img, borde, borde, borde, borde,
                                 cv2.BORDER_CONSTANT, value=255)
    return img


def generar_hoja_marcadores(config: Config, ids: List[int],
                            lado_px: int = 400) -> np.ndarray:
    """Genera una hoja con varios marcadores, listos para recortar."""
    imgs = [generar_marcador(config, i, lado_px) for i in ids]
    if not imgs:
        return np.zeros((10, 10), dtype=np.uint8)

    cols = min(3, len(imgs))
    filas = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape

    hoja = np.full((filas * h, cols * w), 255, dtype=np.uint8)
    for k, img in enumerate(imgs):
        f, c = divmod(k, cols)
        hoja[f * h:(f + 1) * h, c * w:(c + 1) * w] = img
    return hoja


# ===========================================================================
# ESCENA SINTETICA
# ===========================================================================

def renderizar_escena(config: Config,
                      marcadores: List[Tuple[int, float, float, float]],
                      ruido: float = 0.0,
                      semilla: Optional[int] = None) -> np.ndarray:
    """
    Genera una imagen sintetica con marcadores colocados en 3D.

    Cada entrada de `marcadores` es (id, x, y, z) en metros, en el marco de
    la camara.

    Proyecta segun el modelo pinhole. NO modela distorsion de lente ni
    desenfoque de movimiento: sirve para verificar que la cadena de
    deteccion funciona, no para predecir el rendimiento con la camara real.
    """
    if not _HAY_CV2:
        raise ImportError("Se requiere OpenCV.")

    v = config.vision
    img = np.full((v.alto_px, v.ancho_px), 200, dtype=np.uint8)

    for idm, X, Y, Z in marcadores:
        if Z <= 0.05:
            continue

        u = v.fx * X / Z + v.cx
        w = v.fy * Y / Z + v.cy
        lado = v.fx * v.lado_marcador / Z

        if lado < 8:
            continue

        n = int(round(lado))
        borde = max(2, n // 5)
        try:
            patron = generar_marcador(config, idm, lado_px=n, borde=borde)
        except Exception:
            continue

        ph, pw = patron.shape
        x0 = int(round(u - pw / 2))
        y0 = int(round(w - ph / 2))

        xs0, ys0 = max(0, x0), max(0, y0)
        xs1 = min(v.ancho_px, x0 + pw)
        ys1 = min(v.alto_px, y0 + ph)
        if xs1 <= xs0 or ys1 <= ys0:
            continue

        img[ys0:ys1, xs0:xs1] = patron[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]

    if ruido > 0:
        rng = np.random.default_rng(semilla)
        r = rng.normal(0, ruido * 255, img.shape)
        img = np.clip(img.astype(float) + r, 0, 255).astype(np.uint8)

    return img


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DEL DETECTOR ArUco")
    print("=" * 72)

    if not _HAY_ARUCO:
        print()
        print("  OpenCV sin soporte aruco. Instala opencv-contrib-python.")
        raise SystemExit(1)

    v = CONFIG.vision

    print()
    print(f"  Diccionario      : {v.diccionario_aruco}")
    print(f"  Lado del marcador: {v.lado_marcador*1000:.0f} mm")
    print(f"  Camara           : {v.ancho_px}x{v.alto_px}, fx={v.fx:.0f} px")
    print(f"  Alcance teorico  : {v.distancia_maxima_deteccion():.2f} m "
          f"(con {v.px_minimo_marcador} px minimos)")

    if not v.calibracion_verificada:
        print()
        print("  AVISO: intrinsecos NOMINALES. Las distancias tendran error")
        print("  sistematico hasta calibrar la camara.")

    # --- Tamano aparente ---
    print()
    print("-" * 72)
    print("TAMANO APARENTE DEL MARCADOR")
    print("-" * 72)
    print(f"  {'distancia':>11} {'lado (px)':>12}  decodificable")
    for d in (0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5):
        px = v.px_marcador_a(d)
        marca = ("  <-- umbral de transicion"
                 if abs(d - v.umbral_distancia_transicion) < 0.01 else "")
        print(f"  {d:>10.1f}m {px:>12.0f}  "
              f"{'si' if px >= v.px_minimo_marcador else 'NO':>13}{marca}")
    print()
    print("  La fisica de la deteccion actua como filtro de distancia: mas")
    print("  alla del alcance el marcador simplemente no se detecta, antes")
    print("  incluso de aplicar el umbral de transicion.")

    # --- Deteccion ---
    print()
    print("-" * 72)
    print("DETECCION SOBRE ESCENA SINTETICA")
    print("-" * 72)

    det = DetectorArUco(CONFIG)
    escena = [
        (0, -0.375, 0.0, 0.79),
        (1, -0.125, 0.0, 0.79),
        (2,  0.125, 0.0, 0.79),
        (3,  0.375, 0.0, 0.79),
    ]
    print("  Escena: 4 marcadores en fila a 0.79 m, separados 0.25 m")
    print()

    img = renderizar_escena(CONFIG, escena)
    detectados = det.detectar(img)

    print(f"  Detectados: {len(detectados)} de {len(escena)}")
    print()
    print(f"  {'id':>4} {'x real':>9} {'x medida':>10} {'z real':>9} "
          f"{'z medida':>10} {'error':>10} {'lado px':>9}")
    for m in detectados:
        real = next((e for e in escena if e[0] == m.id), None)
        if real is None:
            continue
        _, xr, _, zr = real
        err = math.hypot(m.posicion[0] - xr, m.posicion[2] - zr)
        print(f"  {m.id:>4} {xr:>8.3f}m {m.posicion[0]:>9.3f}m "
              f"{zr:>8.3f}m {m.profundidad:>9.3f}m {err*1000:>9.1f}mm "
              f"{m.lado_px:>9.1f}")

    # --- Ordenamiento ---
    print()
    print("-" * 72)
    print("ORDENAMIENTO POR POSICION HORIZONTAL")
    print("-" * 72)
    print("  El detector devuelve los marcadores ordenados de izquierda a")
    print("  derecha. Es lo que la busqueda binaria necesita: el grupo")
    print("  izquierdo debe estar realmente a la izquierda en la pantalla.")
    print()
    print(f"  Orden por id : {[m.id for m in detectados]}")
    print(f"  Centro x (px): {[round(m.centro_px[0]) for m in detectados]}")

    # --- Alcance medido ---
    print()
    print("-" * 72)
    print("ALCANCE MEDIDO")
    print("-" * 72)
    print(f"  {'distancia':>11} {'detectado':>11} {'z medida':>11} "
          f"{'error':>10}")
    for d in (0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0):
        det_d = det.detectar(renderizar_escena(CONFIG, [(0, 0.0, 0.0, d)]))
        if det_d:
            z = det_d[0].profundidad
            print(f"  {d:>10.1f}m {'si':>11} {z:>10.3f}m "
                  f"{abs(z-d)*1000:>9.1f}mm")
        else:
            print(f"  {d:>10.1f}m {'no':>11} {'---':>11} {'---':>10}")

    # --- Ruido ---
    print()
    print("-" * 72)
    print("ROBUSTEZ AL RUIDO")
    print("-" * 72)
    print(f"  {'ruido':>8} {'detectados':>12} {'error medio':>14}")
    for r in (0.0, 0.02, 0.05, 0.10, 0.20):
        d_r = det.detectar(
            renderizar_escena(CONFIG, escena, ruido=r, semilla=1))
        if d_r:
            errs = []
            for m in d_r:
                real = next((e for e in escena if e[0] == m.id), None)
                if real:
                    errs.append(abs(m.profundidad - real[3]))
            err = np.mean(errs) * 1000 if errs else float("nan")
            print(f"  {r:>8.2f} {len(d_r):>9}/{len(escena)} {err:>13.1f}mm")
        else:
            print(f"  {r:>8.2f} {0:>9}/{len(escena)} {'---':>14}")

"""
vision_fusion.py --- Fusion YOLO+ArUco, transicion de etapa y aproximacion
Bloque 5: vision_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

La logica de fusion, transicion y aproximacion esta verificada con pruebas
automatizadas. El detector YOLO NO: ultralytics no estaba disponible en el
entorno donde se escribio, y su clase es un esqueleto con las llamadas
marcadas para completar.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Tres cosas que dependen de la vision:

  FUSION       Combina lo que ve YOLO (recuadros de objetos reales) con lo
               que ve ArUco (identidad y pose 3D).

  TRANSICION   Decide cuando pasar de navegacion a seleccion.

  APROXIMACION Lleva el robot hasta el objeto elegido, una vez seleccionado.


POR QUE DOS DETECTORES
----------------------
No es redundancia: resuelven problemas distintos.

YOLO detecta objetos reales y produce los recuadros que el usuario ve
parpadear. Eso conserva la validez ecologica: la persona ve munecos, no
cuadrados en blanco y negro.

ArUco aporta la identidad univoca y la pose 3D. Sin el haria falta camara de
profundidad para saber a que distancia esta cada objeto, y dos objetos
identicos serian indistinguibles para YOLO.

Y hay un tercer papel, quiza el mas importante para el experimento: los
obstaculos NO llevan marcador. Eso discrimina objetos de obstaculos sin
ambiguedad y sin entrenar ninguna clase.


LAS TRES CONDICIONES DE TRANSICION
-----------------------------------
Pasar a la fase de seleccion exige que se cumplan las tres a la vez:

  1. DISTANCIA        Los objetos a menos del umbral.
  2. SIN OBSTACULOS   Nada intermedio en el sector frontal.
  3. PERMANENCIA      El robot lleva un tiempo minimo detenido.

La tercera es la que evita el falso positivo mas probable: el usuario para a
mitad de camino para evaluar el entorno, y el sistema lo interpreta como que
ha llegado. Con la exigencia de permanencia, una parada momentanea no basta.

La transicion es AUTOMATICA, sin confirmacion adicional: el usuario ya
expreso su intencion al detenerse frente a los objetos, y el cambio de
interfaz es autoevidente. Pero es REVERSIBLE: el comando de parar en la
etapa 2 devuelve el sistema a navegacion.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
import math
import numpy as np

from config import Config
from aruco_detector import MarcadorDetectado


# ===========================================================================
# YOLO
# ===========================================================================

@dataclass
class DeteccionYOLO:
    """Un objeto detectado por YOLO."""
    clase: str
    confianza: float
    bbox: Tuple[float, float, float, float]
    """(x_min, y_min, x_max, y_max) en pixeles."""

    @property
    def centro(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def ancho(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def alto(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def diagonal(self) -> float:
        return math.hypot(self.ancho, self.alto)


class DetectorYOLO:
    """
    Detector de objetos con YOLOv8.

    ESQUELETO PENDIENTE DE COMPLETAR
    --------------------------------
    Las llamadas a ultralytics estan marcadas con  # >>> COMPLETAR

    QUE HAY QUE HACER
      1. Instalar: pip install ultralytics
      2. Descomentar el import y la carga del modelo en __init__
      3. Descomentar el cuerpo de detectar()

    QUE NO CAMBIAR
      El contrato: detectar() devuelve una lista de DeteccionYOLO con bbox en
      pixeles de la imagen original. Si el modelo trabaja a otra resolucion,
      la conversion va DENTRO de esta clase.

    SOBRE EL ENTRENAMIENTO
      El modelo preentrenado (yolov8n.pt) reconoce las 80 clases de COCO, que
      incluyen algunas utiles (bottle, cup, teddy bear). Para objetos que no
      estan en esas clases hay dos opciones:

        a) Afinar el modelo con imagenes propias. Da generalidad pero exige
           etiquetar unas cien imagenes.
        b) Aceptar que YOLO no reconozca la clase y usar solo el recuadro
           generico. Como la identidad la da el ArUco, esto es suficiente
           para el experimento.

      Para la Fase 1 basta la opcion (b): el trabajo no versa
      sobre vision por computadora, y controlar esa variable es preferible a
      introducir la variabilidad de un detector afinado a medias.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._modelo = None

        # >>> COMPLETAR: cargar el modelo
        #
        #     from ultralytics import YOLO
        #     self._modelo = YOLO(config.vision.modelo_yolo)
        #
        # Si se afina con datos propios, apuntar aqui al .pt resultante.

    @property
    def disponible(self) -> bool:
        return self._modelo is not None

    def detectar(self, imagen: np.ndarray) -> List[DeteccionYOLO]:
        """
        Detecta objetos en la imagen.

        Mientras el modelo no este cargado devuelve lista vacia, lo que hace
        que la fusion caiga en modo "solo ArUco". Eso permite trabajar sin
        YOLO durante el desarrollo.
        """
        if self._modelo is None:
            return []

        # >>> COMPLETAR: inferencia
        #
        #     v = self.cfg.vision
        #     res = self._modelo.predict(
        #         imagen, conf=v.umbral_confianza, iou=v.umbral_iou,
        #         verbose=False)
        #
        #     salida = []
        #     for r in res:
        #         for caja in r.boxes:
        #             x0, y0, x1, y1 = caja.xyxy[0].tolist()
        #             salida.append(DeteccionYOLO(
        #                 clase=r.names[int(caja.cls[0])],
        #                 confianza=float(caja.conf[0]),
        #                 bbox=(x0, y0, x1, y1),
        #             ))
        #     return salida
        #
        return []


# ===========================================================================
# FUSION
# ===========================================================================

@dataclass
class ObjetoSeleccionable:
    """
    Un objeto candidato a ser seleccionado.

    Combina la informacion de ambos detectores. El campo `id` viene siempre
    del marcador: es la identidad que no admite ambiguedad.
    """
    id: int
    marcador: MarcadorDetectado
    deteccion: Optional[DeteccionYOLO] = None

    @property
    def tiene_yolo(self) -> bool:
        return self.deteccion is not None

    @property
    def centro_px(self) -> Tuple[float, float]:
        """Centro para ordenar de izquierda a derecha."""
        return self.marcador.centro_px

    @property
    def distancia(self) -> float:
        return self.marcador.profundidad

    @property
    def angulo(self) -> float:
        return self.marcador.angulo_horizontal

    def bbox_interfaz(self, margen: float = 1.6
                      ) -> Tuple[float, float, float, float]:
        """
        Recuadro que la interfaz debe dibujar parpadeando.

        Si YOLO detecto el objeto, se usa SU recuadro: el usuario ve
        encuadrado el objeto completo, no el cuadradito de papel.

        Si no, se estima uno ampliando el marcador. Es peor visualmente pero
        funcional: el usuario sigue viendo donde esta el objeto.
        """
        if self.deteccion is not None:
            return self.deteccion.bbox

        cx, cy = self.marcador.centro_px
        semi = self.marcador.lado_px * margen / 2
        return (cx - semi, cy - semi, cx + semi, cy + semi)


class FusionVision:
    """
    Combina las detecciones de ArUco y YOLO.

    La asociacion se hace por proximidad: un recuadro de YOLO pertenece al
    marcador cuyo centro cae mas cerca, siempre que la distancia no supere
    una fraccion de la diagonal del recuadro.

    Ese margen es generoso a proposito. El marcador no esta en el centro del
    objeto sino en su frente, y la perspectiva lo desplaza respecto al
    centroide del recuadro.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def fusionar(self, marcadores: List[MarcadorDetectado],
                 detecciones: List[DeteccionYOLO]
                 ) -> List[ObjetoSeleccionable]:
        """
        Asocia cada marcador con un recuadro de YOLO, si lo hay.

        Devuelve la lista ordenada de izquierda a derecha, que es el orden
        que la busqueda binaria necesita.

        NOTA: solo se devuelven objetos CON marcador. Un recuadro de YOLO sin
        marcador asociado se descarta: es un obstaculo o un objeto del
        entorno, no un candidato seleccionable. Esa es la discriminacion.
        """
        v = self.cfg.vision
        libres = list(detecciones)
        salida: List[ObjetoSeleccionable] = []

        for m in marcadores:
            mejor = None
            mejor_d = float("inf")
            mx, my = m.centro_px

            for d in libres:
                dx, dy = d.centro
                dist = math.hypot(mx - dx, my - dy)
                if dist < mejor_d and dist <= d.diagonal * \
                        v.max_distancia_asociacion:
                    mejor, mejor_d = d, dist

            if mejor is not None:
                libres.remove(mejor)

            salida.append(ObjetoSeleccionable(
                id=m.id, marcador=m, deteccion=mejor))

        salida.sort(key=lambda o: o.centro_px[0])
        return salida


# ===========================================================================
# TRANSICION DE ETAPA
# ===========================================================================

@dataclass
class EstadoTransicion:
    """Evaluacion de las condiciones de transicion."""
    procede: bool
    n_objetos: int
    distancia_media: float
    obstaculo_intermedio: bool
    tiempo_detenido: float
    motivo: str = ""

    def resumen(self) -> str:
        L = []
        L.append(f"  Objetos detectados : {self.n_objetos}")
        L.append(f"  Distancia media    : {self.distancia_media:.2f} m")
        L.append(f"  Obstaculo delante  : "
                 f"{'SI' if self.obstaculo_intermedio else 'no'}")
        L.append(f"  Tiempo detenido    : {self.tiempo_detenido:.1f} s")
        L.append(f"  Transicion         : "
                 f"{'PROCEDE' if self.procede else 'no procede'}")
        if self.motivo:
            L.append(f"  Motivo             : {self.motivo}")
        return "\n".join(L)


class LogicaTransicion:
    """
    Decide cuando pasar de navegacion a seleccion.

    Uso:
        log = LogicaTransicion(CONFIG)
        est = log.evaluar(objetos, distancias_lidar, robot_detenido, t)
        if est.procede:
            cambiar_a_etapa_2()
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        self._t_detenido_desde: Optional[float] = None

    # -------------------------------------------------------------------
    def _obstaculo_frontal(self, distancias: List[float],
                           d_objetos: float) -> bool:
        """
        Comprueba si hay algo entre el robot y los objetos.

        Solo mira el sector frontal: un obstaculo a los lados no impide
        llegar, y exigir el pasillo entero despejado seria innecesariamente
        restrictivo.
        """
        if not distancias:
            return False

        v = self.cfg.vision
        n = len(distancias)
        # Sector centrado en el frente (indice 0 del escaneo)
        semi = int(n * (v.sector_frontal / 2.0) / 360.0)

        indices = list(range(0, semi + 1)) + list(range(n - semi, n))
        umbral = d_objetos - v.margen_obstaculo_frontal

        for i in indices:
            d = distancias[i]
            if 1e-6 < d < umbral:
                return True
        return False

    # -------------------------------------------------------------------
    def evaluar(self, objetos: List[ObjetoSeleccionable],
                distancias_lidar: List[float],
                robot_detenido: bool,
                t: float) -> EstadoTransicion:
        """Evalua las tres condiciones."""
        v = self.cfg.vision

        # Contador de permanencia
        if robot_detenido:
            if self._t_detenido_desde is None:
                self._t_detenido_desde = t
            t_det = t - self._t_detenido_desde
        else:
            self._t_detenido_desde = None
            t_det = 0.0

        n = len(objetos)
        d_media = (float(np.mean([o.distancia for o in objetos]))
                   if objetos else float("inf"))

        # --- Condicion 1: objetos suficientes y cerca ---
        if n < v.min_objetos_transicion:
            return EstadoTransicion(
                False, n, d_media, False, t_det,
                f"hacen falta al menos {v.min_objetos_transicion} objetos")

        if d_media > v.umbral_distancia_transicion:
            return EstadoTransicion(
                False, n, d_media, False, t_det,
                f"objetos a {d_media:.2f} m, umbral "
                f"{v.umbral_distancia_transicion} m")

        # --- Condicion 2: sin obstaculos intermedios ---
        obst = self._obstaculo_frontal(distancias_lidar, d_media)
        if obst:
            return EstadoTransicion(
                False, n, d_media, True, t_det,
                "hay un obstaculo entre el robot y los objetos")

        # --- Condicion 3: permanencia ---
        if not robot_detenido:
            return EstadoTransicion(
                False, n, d_media, False, t_det, "el robot esta en movimiento")

        if t_det < v.permanencia_detenido:
            return EstadoTransicion(
                False, n, d_media, False, t_det,
                f"lleva {t_det:.1f} s detenido, hacen falta "
                f"{v.permanencia_detenido} s")

        return EstadoTransicion(True, n, d_media, False, t_det,
                                "las tres condiciones se cumplen")


# ===========================================================================
# APROXIMACION
# ===========================================================================

@dataclass
class ComandoAproximacion:
    """Comando de velocidad durante la aproximacion final."""
    lineal: float
    angular: float
    completada: bool
    error_distancia: float
    error_angulo: float


class ControlAproximacion:
    """
    Lleva el robot hasta el objeto seleccionado.

    Es un control proporcional sobre dos errores: la distancia que falta y el
    angulo respecto al objetivo.

    POR QUE PROPORCIONAL Y NO PID
    ------------------------------
    La maniobra es corta y termina en reposo. Un termino integral acumularia
    error durante la aproximacion y produciria sobreimpulso justo al final,
    que es lo peor posible cuando el robot se esta acercando a un objeto.

    LA REALIMENTACION ES DE ODOMETRIA, NO VISUAL
    ---------------------------------------------
    La pose del objetivo se mide UNA VEZ, al seleccionarlo, y se navega hacia
    ese punto.

    Refinar continuamente con realimentacion visual (visual servoing)
    compensaria la deriva odometrica y daria mejor precision final. Se
    identifica como extension natural, no se implementa aqui: el trabajo
    versa sobre arbitracion cognitiva, y anadirlo mezclaria dos
    contribuciones distintas.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._objetivo: Optional[ObjetoSeleccionable] = None

    # -------------------------------------------------------------------
    def fijar_objetivo(self, objeto: ObjetoSeleccionable) -> None:
        """Fija el objeto al que aproximarse."""
        self._objetivo = objeto

    # -------------------------------------------------------------------
    @property
    def objetivo(self) -> Optional[ObjetoSeleccionable]:
        return self._objetivo

    # -------------------------------------------------------------------
    def actualizar(self, distancia_actual: float,
                   angulo_actual: float) -> ComandoAproximacion:
        """
        Calcula el comando de aproximacion.

        Parametros
        ----------
        distancia_actual : distancia frontal al objetivo (m)
        angulo_actual    : angulo horizontal al objetivo (rad)
        """
        v = self.cfg.vision

        e_d = distancia_actual - v.distancia_objetivo
        e_a = angulo_actual

        completada = (abs(e_d) <= v.tolerancia_posicion and
                      abs(e_a) <= v.tolerancia_angulo)

        if completada:
            return ComandoAproximacion(0.0, 0.0, True, e_d, e_a)

        # Primero alinear, luego avanzar. Avanzar mientras se esta muy
        # desalineado alejaria el robot de la trayectoria util.
        if abs(e_a) > v.tolerancia_angulo:
            w = v.kp_aproximacion_angular * e_a
            w = max(-self.cfg.robot.omega_max,
                    min(self.cfg.robot.omega_max, w))
            return ComandoAproximacion(0.0, w, False, e_d, e_a)

        u = v.kp_aproximacion_lineal * e_d
        u = max(-v.velocidad_aproximacion,
                min(v.velocidad_aproximacion, u))
        w = v.kp_aproximacion_angular * e_a * 0.5

        return ComandoAproximacion(u, w, False, e_d, e_a)


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG
    from aruco_detector import DetectorArUco, renderizar_escena

    print("=" * 72)
    print("DEMOSTRACION DE FUSION, TRANSICION Y APROXIMACION")
    print("=" * 72)

    v = CONFIG.vision
    det = DetectorArUco(CONFIG)
    fusion = FusionVision(CONFIG)

    # --- Fusion ---
    print()
    print("-" * 72)
    print("FUSION YOLO + ArUco")
    print("-" * 72)

    escena = [(i, -0.375 + i * 0.25, 0.0, 1.20) for i in range(4)]
    img = renderizar_escena(CONFIG, escena)
    marcadores = det.detectar(img)

    yolo = DetectorYOLO(CONFIG)
    print(f"  YOLO disponible: {'si' if yolo.disponible else 'NO (esqueleto)'}")
    detecciones = yolo.detectar(img)

    objetos = fusion.fusionar(marcadores, detecciones)
    print(f"  Marcadores      : {len(marcadores)}")
    print(f"  Recuadros YOLO  : {len(detecciones)}")
    print(f"  Objetos fusion  : {len(objetos)}")
    print()
    print(f"  {'id':>4} {'x (px)':>9} {'distancia':>11} {'angulo':>9} "
          f"{'con YOLO':>10}")
    for o in objetos:
        print(f"  {o.id:>4} {o.centro_px[0]:>9.0f} {o.distancia:>10.2f}m "
              f"{math.degrees(o.angulo):>8.1f}o "
              f"{'si' if o.tiene_yolo else 'no':>10}")
    print()
    print("  Sin YOLO el sistema sigue funcionando: la identidad y la pose")
    print("  las da el marcador. Lo que se pierde es el recuadro que encuadra")
    print("  el objeto completo, y se sustituye por uno estimado.")

    # --- Transicion ---
    print()
    print("-" * 72)
    print("LAS TRES CONDICIONES DE TRANSICION")
    print("-" * 72)
    print(f"  1. Objetos a menos de {v.umbral_distancia_transicion} m")
    print(f"  2. Sin obstaculos en el sector frontal de "
          f"{v.sector_frontal}o")
    print(f"  3. Robot detenido durante {v.permanencia_detenido} s")

    n_rayos = CONFIG.robot.lidar_n_rayos
    libre = [5.0] * n_rayos
    con_obst = [5.0] * n_rayos
    con_obst[0] = 0.60

    log = LogicaTransicion(CONFIG)

    casos = [
        ("objetos lejos (3.0 m)", 3.00, libre, True, 5.0),
        ("objetos cerca, en movimiento", 1.20, libre, False, 0.0),
        ("objetos cerca, recien parado", 1.20, libre, True, 0.5),
        ("objetos cerca, obstaculo delante", 1.20, con_obst, True, 5.0),
        ("las tres condiciones", 1.20, libre, True, 5.0),
    ]

    print()
    print(f"  {'situacion':>34} {'transicion':>12}  motivo")
    for desc, dist, lidar, parado, t_det in casos:
        esc = [(i, -0.375 + i * 0.25, 0.0, dist) for i in range(4)]
        objs = fusion.fusionar(
            det.detectar(renderizar_escena(CONFIG, esc)), [])

        log.reiniciar()
        if parado and t_det > 0:
            log.evaluar(objs, lidar, True, 0.0)
            est = log.evaluar(objs, lidar, True, t_det)
        else:
            est = log.evaluar(objs, lidar, parado, 0.0)

        print(f"  {desc:>34} {'SI' if est.procede else 'no':>12}  "
              f"{est.motivo}")

    # --- Aproximacion ---
    print()
    print("-" * 72)
    print("APROXIMACION AL OBJETO SELECCIONADO")
    print("-" * 72)
    print(f"  Distancia objetivo : {v.distancia_objetivo} m")
    print(f"  Velocidad maxima   : {v.velocidad_aproximacion} m/s")
    print()
    print("  Simulacion: el robot arranca a 1.20 m y 15 grados desviado.")
    print()

    ctrl = ControlAproximacion(CONFIG)
    if objetos:
        ctrl.fijar_objetivo(objetos[2])

    d, a = 1.20, math.radians(15)
    dt = CONFIG.robot.ts_control
    print(f"  {'t (s)':>7} {'distancia':>11} {'angulo':>9} {'u':>8} "
          f"{'omega':>8}  fase")

    for i in range(120):
        c = ctrl.actualizar(d, a)
        if i % 10 == 0 or c.completada:
            fase = ("completada" if c.completada
                    else "alineando" if abs(c.lineal) < 1e-6 else "avanzando")
            print(f"  {i*dt:>7.2f} {d:>10.3f}m "
                  f"{math.degrees(a):>8.1f}o {c.lineal:>8.3f} "
                  f"{c.angular:>8.3f}  {fase}")
        if c.completada:
            break
        # Integracion simple del movimiento
        d -= c.lineal * dt
        a -= c.angular * dt

    print()
    print("  El controlador alinea primero y avanza despues. Avanzar mientras")
    print("  se esta muy desalineado alejaria al robot de la trayectoria util.")

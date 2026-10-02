"""
interface.py --- Interfaz visual con PsychoPy
Bloque 2: estimulo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional, pero a diferencia del resto
del bloque NO se ha podido verificar por ejecucion: PsychoPy necesita una
pantalla real y no estaba disponible en el entorno donde se escribio.

La logica que SI esta verificada es la de stimulus.py, que produce los
numeros. Este modulo solo los dibuja.

Es por tanto el archivo del bloque con mayor probabilidad de necesitar
ajustes. Si algo falla:
  1. Ejecuta primero `python run_estimulo.py demo` para aislar el problema.
  2. Comprueba la version de PsychoPy: la API de `ElementArrayStim` cambio
     entre versiones mayores.
  3. Reportalo indicando la version de PsychoPy y el traceback completo.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Dibuja la pantalla que ve el usuario:

  - Video de la camara del robot como fondo, a pantalla completa
  - Cuatro estimulos parpadeantes superpuestos, en cruz (Etapa 1)
  - Recuadros parpadeantes sobre los objetos detectados (Etapa 2)
  - Borde de color que indica el estado cognitivo detectado por TASM

LAS DOS ETAPAS
--------------
ETAPA 1 (navegacion): cuatro estimulos en cruz, uno por comando. El usuario
mira brevemente uno, el comando se enclava, y deja de mirar los estimulos
para observar el video mientras el robot se mueve.

ETAPA 2 (seleccion): los cuatro estimulos desaparecen. Los objetos
detectados se agrupan en dos mitades que parpadean a dos frecuencias. El
usuario elige un lado y el grupo se subdivide.

El cambio entre etapas es visualmente evidente por si mismo, lo que hace
innecesario un aviso explicito.

REGLAS DE TEMPORIZACION QUE NO DEBEN VIOLARSE
---------------------------------------------
  1. La fase se calcula del CONTADOR DE FRAMES, nunca del reloj.
     Un error de 0.1 ms por frame produce 265 grados de deriva en medio
     minuto.

  2. Sincronizacion vertical activada y tasa fijada a 60 Hz.
     Sin ello la tasa varia con la carga de renderizado del video.

  3. Se registra el instante de cada frame presentado.
     Permite detectar frames perdidos a posteriori en lugar de suponer que
     no los hubo.

  4. El estimulo se actualiza aunque no haya frame de video nuevo.
     Si se esperara al video, un tiron de red corromperia la senal SSVEP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import time
import numpy as np

from config import Config
from stimulus import GeneradorLuminancia, TexturaEstimulo, RegistroFrames


# ===========================================================================
# DISPONIBILIDAD DE PSYCHOPY
# ===========================================================================

try:
    from psychopy import visual, core, event
    _HAY_PSYCHOPY = True
except ImportError:
    _HAY_PSYCHOPY = False
    visual = None
    core = None
    event = None


# ===========================================================================
# ESTADO DE LA INTERFAZ
# ===========================================================================

@dataclass
class EstadoInterfaz:
    """
    Que debe mostrarse en el frame actual.

    Es el mensaje que la interfaz recibe del resto del sistema. Se mantiene
    deliberadamente simple: la interfaz no decide nada, solo dibuja lo que le
    dicen.
    """
    etapa: int = 1
    """1 = navegacion, 2 = seleccion de objeto."""

    estado_tasm: str = "Idle"
    """"IC", "TR" o "Idle". Determina el color del borde."""

    activos: Optional[List[int]] = None
    """Indices de los estimulos que deben parpadear. None = todos.
    En Etapa 2 se usa para encender solo los dos grupos."""

    grupos_e2: Optional[Tuple[List[int], List[int]]] = None
    """(indices_izquierda, indices_derecha) de los objetos en juego."""

    bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    """Recuadros de los objetos detectados, en pixeles: (x, y, ancho, alto).
    Coordenadas del centro."""

    emergencia: bool = False
    """Si esta activo el override de seguridad. Se muestra un aviso rojo."""

    mensaje: str = ""
    """Texto opcional a mostrar (instrucciones, avisos)."""


# ===========================================================================
# INTERFAZ
# ===========================================================================

class InterfazSSVEP:
    """
    Interfaz visual completa.

    Uso:
        with InterfazSSVEP(CONFIG, fuente_video) as iface:
            while corriendo:
                estado = EstadoInterfaz(etapa=1, estado_tasm="IC")
                iface.dibujar(estado)

    El bucle debe llamar a `dibujar` una vez por frame de pantalla, a 60 Hz.
    Es responsabilidad del llamante mantener ese ritmo; PsychoPy lo hace
    automaticamente si vsync esta activo.
    """

    def __init__(self, config: Config,
                 fuente_video=None,
                 pantalla_completa: bool = True,
                 monitor_idx: int = 0):
        if not _HAY_PSYCHOPY:
            raise ImportError(
                "PsychoPy no esta instalado. Instala con:\n"
                "  pip install psychopy\n"
                "Para desarrollar sin pantalla, usa las clases de "
                "stimulus.py directamente."
            )

        self.cfg = config
        self.fuente = fuente_video
        e = config.estimulo

        # --- Ventana ---
        self.win = visual.Window(
            size=e.resolucion,
            fullscr=pantalla_completa,
            screen=monitor_idx,
            color=e.color_fondo,
            colorSpace="rgb1",
            units="pix",
            waitBlanking=True,      # sincronizacion vertical
            allowGUI=False,
        )

        # Fijar la tasa de refresco esperada. Si el monitor real no es de
        # 60 Hz, TODAS las frecuencias del sistema estarian mal.
        self._verificar_refresco()

        # --- Generadores ---
        self.gen = GeneradorLuminancia(config)
        self.textura = TexturaEstimulo(config)
        self.registro = RegistroFrames(refresh_hz=e.refresh_hz)

        # --- Elementos graficos ---
        self._crear_elementos()

        self._t_inicio = time.perf_counter()

    # -------------------------------------------------------------------
    def _verificar_refresco(self) -> None:
        """
        Mide la tasa de refresco real y avisa si no coincide con la esperada.

        Es una comprobacion barata que evita un error catastrofico y
        silencioso: si el monitor va a 144 Hz y el codigo asume 60, todas las
        frecuencias generadas serian 2.4 veces mas altas de lo previsto.
        """
        medido = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1)

        esperado = self.cfg.estimulo.refresh_hz
        if medido is None:
            print("[AVISO] No se pudo medir la tasa de refresco. "
                  "Verifica manualmente que el monitor va a "
                  f"{esperado} Hz.")
            return

        if abs(medido - esperado) > 2.0:
            print(f"[ADVERTENCIA GRAVE] Tasa medida: {medido:.1f} Hz, "
                  f"esperada: {esperado} Hz.")
            print("  Las frecuencias generadas NO seran las nominales.")
            print("  Corrige config.estimulo.refresh_hz o la configuracion "
                  "del monitor antes de continuar.")
        else:
            print(f"[OK] Tasa de refresco: {medido:.1f} Hz")

    # -------------------------------------------------------------------
    def _crear_elementos(self) -> None:
        """Crea los objetos graficos una sola vez, al inicio.

        Crearlos en cada frame seria costoso y provocaria perdida de frames.
        """
        e = self.cfg.estimulo
        W, H = e.resolucion
        lado = e.lado_estimulo_px

        # --- Fondo de video ---
        self.img_video = visual.ImageStim(
            self.win, image=None, units="pix", size=(W, H), pos=(0, 0),
        )

        # --- Estimulos de la Etapa 1 ---
        # Cada estimulo es una imagen cuya textura se recalcula por frame con
        # la luminancia correspondiente.
        self.estimulos = []
        for i, (fx, fy) in enumerate(e.posiciones_cruz):
            self.estimulos.append(visual.ImageStim(
                self.win,
                image=np.zeros((lado, lado)),
                units="pix",
                size=(lado, lado),
                pos=(fx * W / 2, fy * H / 2),
                colorSpace="rgb1",
            ))

        # --- Borde indicador de estado TASM ---
        g = e.grosor_indicador_px
        self.borde = visual.Rect(
            self.win, width=W - g, height=H - g,
            lineWidth=g, lineColor=e.color_idle, fillColor=None,
            units="pix", colorSpace="rgb1", opacity=e.opacidad_indicador,
        )

        # --- Recuadros de la Etapa 2 ---
        # Se crean con antelacion para el maximo N previsto, y se muestran
        # solo los necesarios.
        max_n = max(self.cfg.etapa2.niveles_n)
        self.bboxes = [
            visual.Rect(self.win, width=100, height=100,
                        lineWidth=e.grosor_bbox_px, lineColor=(1, 1, 1),
                        fillColor=None, units="pix", colorSpace="rgb1")
            for _ in range(max_n)
        ]

        # --- Aviso de emergencia ---
        self.aviso_emergencia = visual.Rect(
            self.win, width=W, height=int(H * 0.08),
            pos=(0, H / 2 - int(H * 0.04)),
            fillColor=(0.8, 0.1, 0.1), lineColor=None,
            units="pix", colorSpace="rgb1", opacity=0.75,
        )
        self.texto_emergencia = visual.TextStim(
            self.win, text="EVASION AUTOMATICA ACTIVA",
            pos=(0, H / 2 - int(H * 0.04)), height=int(H * 0.035),
            color=(1, 1, 1), colorSpace="rgb1", units="pix",
        )

        # --- Mensaje generico ---
        self.texto = visual.TextStim(
            self.win, text="", pos=(0, -H / 2 + int(H * 0.06)),
            height=int(H * 0.03), color=(0.9, 0.9, 0.9),
            colorSpace="rgb1", units="pix",
        )

    # -------------------------------------------------------------------
    def _color_borde(self, estado_tasm: str) -> Tuple[float, float, float]:
        e = self.cfg.estimulo
        if estado_tasm == "IC":
            return e.color_ic
        if estado_tasm == "TR":
            return e.color_tr
        return e.color_idle

    # -------------------------------------------------------------------
    def _actualizar_video(self) -> None:
        """
        Actualiza el fondo si hay frame nuevo.

        Si no lo hay, se mantiene el anterior. El estimulo se dibuja igual:
        NO se espera al video.
        """
        if self.fuente is None:
            return
        frame = self.fuente.leer()
        if frame is not None:
            # PsychoPy espera valores en [-1, 1] con colorSpace por defecto
            self.img_video.image = (frame.astype(np.float32) / 127.5) - 1.0

    # -------------------------------------------------------------------
    def dibujar(self, estado: EstadoInterfaz) -> None:
        """
        Dibuja un frame completo y lo presenta.

        Debe llamarse una vez por frame de pantalla. La llamada a `flip()`
        bloquea hasta el siguiente refresco si vsync esta activo, lo que
        mantiene el ritmo automaticamente.
        """
        e = self.cfg.estimulo

        # --- Fondo ---
        self._actualizar_video()
        if self.fuente is not None:
            self.img_video.draw()

        # --- Luminancias del frame actual ---
        # AQUI SE USA EL CONTADOR DE FRAMES, no el reloj.
        lums = self.gen.siguiente_frame(activos=estado.activos)

        # --- Estimulos ---
        if estado.etapa == 1:
            for i, est in enumerate(self.estimulos):
                if lums[i] > 0.0:
                    est.image = self.textura.render(lums[i])
                    est.draw()
        else:
            self._dibujar_etapa2(estado, lums)

        # --- Indicador de estado ---
        if e.mostrar_indicador_tasm:
            self.borde.lineColor = self._color_borde(estado.estado_tasm)
            self.borde.draw()

        # --- Emergencia ---
        if estado.emergencia:
            self.aviso_emergencia.draw()
            self.texto_emergencia.draw()

        # --- Mensaje ---
        if estado.mensaje:
            self.texto.text = estado.mensaje
            self.texto.draw()

        # --- Presentar ---
        self.win.flip()
        self.registro.registrar(time.perf_counter() - self._t_inicio)

    # -------------------------------------------------------------------
    def _dibujar_etapa2(self, estado: EstadoInterfaz,
                        lums: np.ndarray) -> None:
        """
        Dibuja los recuadros de la fase de seleccion.

        Los objetos del grupo izquierdo parpadean a la frecuencia del indice
        0 y los del derecho a la del indice 1. Son las mismas frecuencias que
        en la Etapa 1 para giro izquierda y derecha, lo que mantiene
        coherencia semantica: el usuario aprende una asociacion y le vale en
        ambas etapas.

        IMPORTANTE: el recuadro encuadra el OBJETO completo, no el marcador
        ArUco. El marcador es infraestructura invisible para el usuario.
        """
        if estado.bboxes is None or estado.grupos_e2 is None:
            return

        izq, der = estado.grupos_e2

        for idx, (cx, cy, w, h) in enumerate(estado.bboxes):
            if idx in izq:
                lum = lums[0]
            elif idx in der:
                lum = lums[1]
            else:
                continue    # objeto descartado: no se dibuja

            if idx >= len(self.bboxes):
                continue

            r = self.bboxes[idx]
            r.width = w
            r.height = h
            r.pos = (cx, cy)
            r.lineColor = (float(lum), float(lum), float(lum))
            r.draw()

    # -------------------------------------------------------------------
    def teclas(self) -> List[str]:
        """Teclas pulsadas desde la ultima llamada."""
        return event.getKeys()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Reinicia contadores. Llamar al empezar cada trial."""
        self.gen.reiniciar()
        self.registro = RegistroFrames(
            refresh_hz=self.cfg.estimulo.refresh_hz)
        self._t_inicio = time.perf_counter()

    # -------------------------------------------------------------------
    def cerrar(self) -> None:
        self.win.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()


# ===========================================================================
# SIMULADOR DE INTERFAZ (sin PsychoPy)
# ===========================================================================

class InterfazSimulada:
    """
    Sustituto de la interfaz que no dibuja nada.

    Permite ejercitar toda la logica temporal ---generacion de luminancias,
    conteo de frames, cambio de etapa--- sin necesidad de pantalla ni de
    PsychoPy. Se usa en las pruebas automatizadas.

    Expone la misma interfaz que InterfazSSVEP, de modo que el codigo que la
    consume no distingue entre una y otra.
    """

    def __init__(self, config: Config, fuente_video=None):
        self.cfg = config
        self.fuente = fuente_video
        self.gen = GeneradorLuminancia(config)
        self.textura = TexturaEstimulo(config)
        self.registro = RegistroFrames(
            refresh_hz=config.estimulo.refresh_hz)

        self.historial: List[Dict] = []
        self._t_inicio = time.perf_counter()

    def dibujar(self, estado: EstadoInterfaz) -> None:
        lums = self.gen.siguiente_frame(activos=estado.activos)
        if self.fuente is not None:
            self.fuente.leer()

        self.historial.append({
            "frame": self.gen.frame - 1,
            "etapa": estado.etapa,
            "estado_tasm": estado.estado_tasm,
            "luminancias": lums.copy(),
        })
        # Se simula el tiempo desde el contador de frames, no del reloj real
        self.registro.registrar(
            (self.gen.frame - 1) / self.cfg.estimulo.refresh_hz)

    def teclas(self) -> List[str]:
        return []

    def reiniciar(self) -> None:
        self.gen.reiniciar()
        self.registro = RegistroFrames(
            refresh_hz=self.cfg.estimulo.refresh_hz)
        self.historial.clear()

    def cerrar(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG
    from video_source import crear_fuente

    print("=" * 72)
    print("DEMOSTRACION DE LA INTERFAZ")
    print("=" * 72)
    print()

    if _HAY_PSYCHOPY:
        print("  PsychoPy detectado. Para la demostracion grafica ejecuta:")
        print("     python run_estimulo.py demo")
    else:
        print("  PsychoPy NO esta instalado en este entorno.")
        print("  Se usara InterfazSimulada, que ejercita la misma logica")
        print("  temporal sin dibujar.")

    print()
    print("-" * 72)
    print("SIMULACION DE 5 SEGUNDOS")
    print("-" * 72)

    fuente = crear_fuente(CONFIG, "sintetica")
    iface = InterfazSimulada(CONFIG, fuente)

    n = int(5.0 * CONFIG.estimulo.refresh_hz)
    for i in range(n):
        # Alterna de etapa a mitad de la simulacion
        etapa = 1 if i < n // 2 else 2
        est = EstadoInterfaz(
            etapa=etapa,
            estado_tasm="IC" if (i // 30) % 2 == 0 else "Idle",
            activos=None if etapa == 1 else [0, 1],
            grupos_e2=([0, 1], [2, 3]) if etapa == 2 else None,
            bboxes=[(0, 0, 100, 100)] * 4 if etapa == 2 else None,
        )
        iface.dibujar(est)

    print(f"  Frames dibujados : {len(iface.historial)}")
    print(f"  Frames esperados : {n}")
    print()
    print(iface.registro.resumen())

    print()
    print("-" * 72)
    print("LUMINANCIAS EN ETAPA 2 (solo dos grupos activos)")
    print("-" * 72)
    print("  En Etapa 2 solo parpadean los estimulos 0 y 1; los otros dos")
    print("  quedan a cero.")
    print()
    print(f"  {'frame':>7} " +
          " ".join(f"{f:>8.1f}Hz" for f in CONFIG.bci.frecuencias))
    for h in iface.historial[n // 2: n // 2 + 6]:
        print(f"  {h['frame']:>7} " +
              " ".join(f"{x:>10.4f}" for x in h["luminancias"]))

    fuente.cerrar()

"""
stimulus.py --- Generacion del estimulo SSVEP
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
Genera la luminancia de cada estimulo en cada frame, y la textura de celdas
encendidas. NO dibuja nada: solo produce numeros. El dibujado es
responsabilidad de interface.py.

Esa separacion es deliberada. Toda la logica temporal ---que es donde estan
los errores dificiles de detectar--- queda verificable con numpy, sin
necesidad de abrir una ventana grafica ni tener PsychoPy instalado.

EL PROBLEMA QUE RESUELVE: LA ONDA CUADRADA NO SIRVE A 60 Hz
------------------------------------------------------------
Con encendido/apagado por frame, solo se pueden generar frecuencias de la
forma R/n con n entero, y para ciclo de trabajo del 50% hace falta ademas
que n sea PAR.

    8.0 Hz  ->  60/8.0  = 7.50   no entero
    12.0 Hz ->  60/12.0 = 5.00   entero pero IMPAR
    14.0 Hz ->  60/14.0 = 4.29   no entero
    15.2 Hz ->  60/15.2 = 3.95   no entero

Ninguna de las cuatro frecuencias del sistema es realizable asi.

LA SOLUCION: MODULACION SINUSOIDAL MUESTREADA
---------------------------------------------
En lugar de conmutar entre encendido y apagado, se modula la LUMINANCIA de
forma continua:

    s_k(i) = 0.5 * [1 + sin(2*pi*f_k*i/R + phi_k)]

donde i es el indice de frame, R la tasa de refresco y phi_k la fase JFPM.

Con esto CUALQUIER frecuencia por debajo de Nyquist es realizable, porque la
informacion ya no esta en el instante de conmutacion sino en la envolvente
de luminancia. Es el mismo metodo con que se grabo el dataset Benchmark, lo
que ademas garantiza que los componentes preentrenados con el transfieran.

EL ERROR MAS FACIL DE COMETER
------------------------------
Calcular la fase con el tiempo transcurrido (time.time(), deltaTime) en
lugar del CONTADOR DE FRAMES.

El tiempo acumulado arrastra error de redondeo y produce deriva de fase
progresiva: al principio del trial el estimulo esta en fase, y varios
minutos despues ha derivado lo suficiente para degradar la clasificacion.
Con contador de frames la fase es exacta por construccion.

Este modulo solo expone la version con contador. Si en la implementacion de
PsychoPy ves algo parecido a `fase = 2*pi*f*time.time()`, esta mal.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import math
import numpy as np

from config import Config


# ===========================================================================
# SECUENCIA TEMPORAL DE LUMINANCIA
# ===========================================================================

class GeneradorLuminancia:
    """
    Produce el valor de luminancia de cada estimulo en cada frame.

    Uso:
        gen = GeneradorLuminancia(CONFIG)
        for _ in range(n_frames):
            lums = gen.siguiente_frame()   # una luminancia por estimulo
            # ... dibujar con esas luminancias ...

    El contador de frames es interno y solo avanza al llamar
    siguiente_frame(). Eso hace que la fase sea reproducible exactamente:
    dos ejecuciones con el mismo numero de llamadas dan la misma secuencia.
    """

    def __init__(self, config: Config):
        self.cfg = config
        b, e = config.bci, config.estimulo

        self.frecuencias = np.asarray(b.frecuencias, dtype=float)
        self.fases = np.asarray(b.fases_jfpm, dtype=float)
        self.refresh = e.refresh_hz
        self.n_estimulos = len(self.frecuencias)

        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Pone el contador de frames a cero. Llamar al empezar cada trial."""
        self._frame = 0

    # -------------------------------------------------------------------
    @property
    def frame(self) -> int:
        return self._frame

    @property
    def tiempo(self) -> float:
        """Tiempo transcurrido segun el contador de frames (s).

        Se deriva del contador, NO del reloj del sistema. Si difiere del
        tiempo real, es que se han perdido frames."""
        return self._frame / self.refresh

    # -------------------------------------------------------------------
    def luminancia_en(self, frame: int,
                      activos: Optional[List[int]] = None) -> np.ndarray:
        """
        Luminancia de cada estimulo en un frame dado, sin avanzar el contador.

        Parametros
        ----------
        frame   : indice de frame (entero)
        activos : indices de los estimulos que deben parpadear. Los que no
                  estan en la lista devuelven 0 (apagados). Se usa en la
                  Etapa 2, donde solo dos grupos parpadean.

        Devuelve
        --------
        Array de longitud n_estimulos con valores en [0, 1].
        """
        # AQUI ESTA LA ECUACION CENTRAL DEL MODULO.
        # Notese que usa `frame`, un entero, y no un tiempo en segundos.
        ang = 2 * np.pi * self.frecuencias * (frame / self.refresh) + self.fases
        lum = 0.5 * (1.0 + np.sin(ang))

        if activos is not None:
            mascara = np.zeros(self.n_estimulos, dtype=bool)
            for i in activos:
                if 0 <= i < self.n_estimulos:
                    mascara[i] = True
            lum = np.where(mascara, lum, 0.0)

        return lum

    # -------------------------------------------------------------------
    def siguiente_frame(self,
                        activos: Optional[List[int]] = None) -> np.ndarray:
        """Devuelve la luminancia del frame actual y avanza el contador."""
        lum = self.luminancia_en(self._frame, activos)
        self._frame += 1
        return lum

    # -------------------------------------------------------------------
    def secuencia(self, n_frames: int,
                  activos: Optional[List[int]] = None) -> np.ndarray:
        """
        Genera la secuencia completa de un tramo, sin avanzar el contador.

        Devuelve un array (n_frames, n_estimulos). Se usa para la validacion
        espectral y para las pruebas.
        """
        frames = np.arange(n_frames)
        ang = (2 * np.pi * self.frecuencias[None, :] *
               (frames[:, None] / self.refresh) + self.fases[None, :])
        lum = 0.5 * (1.0 + np.sin(ang))

        if activos is not None:
            mascara = np.zeros(self.n_estimulos, dtype=bool)
            for i in activos:
                if 0 <= i < self.n_estimulos:
                    mascara[i] = True
            lum = lum * mascara[None, :]

        return lum


# ===========================================================================
# TEXTURA DE CELDAS
# ===========================================================================

class TexturaEstimulo:
    """
    Patron de celdas encendidas dentro del cuadrado del estimulo.

    Segun Meng et al. (2023), la forma optima es un CUADRADO con
    distribucion ALEATORIA de pixeles al 60% de densidad. El cuadrado supera
    al damero en unos 30 puntos de precision, diferencia que atribuyen a la
    cancelacion de fase entre celdas vecinas desfasadas pi en el damero: sus
    respuestas se restan en corteza.

    La textura se genera UNA VEZ y se mantiene fija durante todo el
    experimento. Lo que varia frame a frame es la luminancia global, no que
    celdas estan encendidas. Regenerarla en cada frame introduciria ruido
    espacial que compite con la senal.
    """

    def __init__(self, config: Config, semilla: Optional[int] = None):
        self.cfg = config
        e = config.estimulo

        self.n = e.celdas_por_lado
        self.n_activas = e.n_celdas_activas

        rng = np.random.default_rng(
            semilla if semilla is not None else 20260101)

        if e.distribucion_aleatoria:
            self.mascara = self._mascara_aleatoria(rng)
        else:
            self.mascara = self._mascara_uniforme()

    # -------------------------------------------------------------------
    def _mascara_aleatoria(self, rng) -> np.ndarray:
        """
        Selecciona n_activas celdas al azar, sin repeticion.

        Se sortean posiciones concretas en lugar de aplicar un umbral a ruido
        uniforme, porque asi el numero de celdas encendidas es EXACTAMENTE el
        pedido. Con umbral seria aproximado, y la densidad real variaria entre
        estimulos.
        """
        total = self.n * self.n
        idx = rng.choice(total, size=self.n_activas, replace=False)
        m = np.zeros(total, dtype=bool)
        m[idx] = True
        return m.reshape(self.n, self.n)

    # -------------------------------------------------------------------
    def _mascara_uniforme(self) -> np.ndarray:
        """
        Distribucion uniforme (rejilla regular).

        Se incluye solo para poder reproducir la comparacion de Meng et al.
        No es la opcion adoptada: su estudio muestra que la aleatoria es
        mejor tanto en precision como en fatiga.
        """
        total = self.n * self.n
        m = np.zeros(total, dtype=bool)
        if self.n_activas > 0:
            paso = total / self.n_activas
            idx = np.round(np.arange(self.n_activas) * paso).astype(int)
            idx = np.clip(idx, 0, total - 1)
            m[idx] = True
        return m.reshape(self.n, self.n)

    # -------------------------------------------------------------------
    @property
    def densidad_real(self) -> float:
        """Fraccion de celdas efectivamente encendidas."""
        return float(self.mascara.sum()) / self.mascara.size

    # -------------------------------------------------------------------
    def render(self, luminancia: float,
               lado_px: Optional[int] = None) -> np.ndarray:
        """
        Genera la imagen del estimulo para una luminancia dada.

        Devuelve un array (lado_px, lado_px) con valores en [0, 1]. Las
        celdas apagadas quedan a 0, que en la interfaz se renderiza
        transparente para que se vea el video de fondo.
        """
        lado = lado_px if lado_px is not None \
            else self.cfg.estimulo.lado_estimulo_px

        img = self.mascara.astype(float) * float(luminancia)

        # Escalado por repeticion de bloques (nearest neighbour). Interpolar
        # suavizaria los bordes entre celdas y reduciria el contraste, que es
        # lo que determina la amplitud de la respuesta SSVEP.
        rep = max(1, lado // self.n)
        img = np.kron(img, np.ones((rep, rep)))

        # Ajuste fino al tamano exacto pedido
        if img.shape[0] != lado:
            recorte = min(img.shape[0], lado)
            salida = np.zeros((lado, lado))
            salida[:recorte, :recorte] = img[:recorte, :recorte]
            img = salida

        return img


# ===========================================================================
# CONTROL DE FRAMES
# ===========================================================================

@dataclass
class RegistroFrames:
    """
    Registro de los tiempos de presentacion de cada frame.

    Sirve para detectar frames perdidos a posteriori. No basta con suponer
    que no se perdio ninguno: la carga de renderizado del video de fondo
    varia con la escena, y un escenario visualmente denso puede provocar
    perdidas que un corredor vacio no provoca.

    Si eso ocurriera de forma desigual entre escenarios, seria un confusor
    de la hipotesis de escalamiento: parte del efecto observado se deberia a
    la calidad del estimulo y no a la arbitracion cognitiva.
    """
    tiempos: List[float] = field(default_factory=list)
    refresh_hz: float = 60.0

    def registrar(self, t: float) -> None:
        self.tiempos.append(t)

    @property
    def n_frames(self) -> int:
        return len(self.tiempos)

    @property
    def intervalos(self) -> np.ndarray:
        if len(self.tiempos) < 2:
            return np.array([])
        return np.diff(np.asarray(self.tiempos))

    @property
    def frames_perdidos(self) -> int:
        """
        Numero estimado de frames perdidos.

        Un intervalo de duracion cercana a k veces el periodo nominal indica
        que se perdieron k-1 frames. Se usa un umbral de 1.5 periodos para
        no contar como perdida la jitter normal del sistema.
        """
        iv = self.intervalos
        if iv.size == 0:
            return 0
        periodo = 1.0 / self.refresh_hz
        return int(np.sum(np.round(iv / periodo) - 1)[()]) if iv.size else 0

    @property
    def fraccion_perdidos(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return self.frames_perdidos / self.n_frames

    def resumen(self) -> str:
        iv = self.intervalos
        L = []
        L.append(f"  Frames presentados : {self.n_frames}")
        if iv.size:
            L.append(f"  Intervalo medio    : {iv.mean()*1000:.2f} ms "
                     f"(nominal {1000/self.refresh_hz:.2f} ms)")
            L.append(f"  Desviacion tipica  : {iv.std()*1000:.3f} ms")
            L.append(f"  Intervalo maximo   : {iv.max()*1000:.2f} ms")
        L.append(f"  Frames perdidos    : {self.frames_perdidos} "
                 f"({self.fraccion_perdidos*100:.2f}%)")
        return "\n".join(L)


# ===========================================================================
# VERIFICACION DE REALIZABILIDAD
# ===========================================================================

def verificar_realizabilidad(config: Config) -> Dict[str, object]:
    """
    Comprueba si las frecuencias son realizables con onda cuadrada y con
    modulacion sinusoidal.

    Es la verificacion que motiva la eleccion del metodo de generacion, y
    conviene poder reproducirla en cualquier momento.
    """
    b, e = config.bci, config.estimulo
    R = e.refresh_hz

    filas = []
    for i, f in enumerate(b.frecuencias):
        cociente = R / f
        entero = abs(cociente - round(cociente)) < 1e-9
        par = entero and (round(cociente) % 2 == 0)
        filas.append({
            "indice": i,
            "frecuencia": f,
            "R/f": cociente,
            "cuadrada_ok": par,
            "sinusoidal_ok": f < R / 2,
            "bin_fft": e.bin_fft(f),
        })

    return {
        "refresh": R,
        "nyquist": R / 2,
        "frecuencias": filas,
        "cuadrada_viable": all(x["cuadrada_ok"] for x in filas),
        "sinusoidal_viable": all(x["sinusoidal_ok"] for x in filas),
    }


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DEL GENERADOR DE ESTIMULO")
    print("=" * 72)

    b, e = CONFIG.bci, CONFIG.estimulo

    # --- Realizabilidad ---
    print()
    print("-" * 72)
    print("REALIZABILIDAD DE LAS FRECUENCIAS")
    print("-" * 72)
    v = verificar_realizabilidad(CONFIG)
    print(f"  Refresco: {v['refresh']} Hz, Nyquist: {v['nyquist']} Hz")
    print()
    print(f"  {'f (Hz)':>8} {'R/f':>8} {'cuadrada':>10} {'sinusoidal':>12} "
          f"{'bin FFT':>9}")
    for x in v["frecuencias"]:
        print(f"  {x['frecuencia']:>8.1f} {x['R/f']:>8.3f} "
              f"{'SI' if x['cuadrada_ok'] else 'NO':>10} "
              f"{'SI' if x['sinusoidal_ok'] else 'NO':>12} "
              f"{x['bin_fft']:>9.0f}")
    print()
    print(f"  Onda cuadrada viable   : "
          f"{'SI' if v['cuadrada_viable'] else 'NO'}")
    print(f"  Sinusoidal viable      : "
          f"{'SI' if v['sinusoidal_viable'] else 'NO'}")
    print()
    print("  Por eso se usa modulacion sinusoidal muestreada.")

    # --- Textura ---
    print()
    print("-" * 72)
    print("TEXTURA DEL ESTIMULO")
    print("-" * 72)
    tex = TexturaEstimulo(CONFIG)
    print(f"  Rejilla        : {tex.n} x {tex.n} = {tex.n*tex.n} celdas")
    print(f"  Celdas activas : {int(tex.mascara.sum())} "
          f"(densidad real {tex.densidad_real*100:.1f}%)")
    print(f"  Lado en pantalla: {e.lado_estimulo_px} px "
          f"({e.tamano_angular} grados a {e.distancia_vision} m)")
    print()
    print("  Vista de las primeras 12x12 celdas (# encendida, . apagada):")
    for fila in tex.mascara[:12, :12]:
        print("    " + "".join("#" if c else "." for c in fila))

    # --- Secuencia temporal ---
    print()
    print("-" * 72)
    print("SECUENCIA DE LUMINANCIA (primeros 12 frames)")
    print("-" * 72)
    gen = GeneradorLuminancia(CONFIG)
    print(f"  {'frame':>6} {'t (ms)':>8} " +
          " ".join(f"{f:>7.1f}Hz" for f in b.frecuencias))
    for i in range(12):
        lum = gen.siguiente_frame()
        print(f"  {i:>6} {i/e.refresh_hz*1000:>8.2f} " +
              " ".join(f"{x:>9.4f}" for x in lum))

    # --- Verificacion espectral ---
    print()
    print("-" * 72)
    print("VERIFICACION ESPECTRAL DE LA SECUENCIA GENERADA")
    print("-" * 72)
    print("  Se genera la senal ideal y se comprueba que su espectro tiene el")
    print("  pico exactamente en la frecuencia nominal.")
    print()
    gen.reiniciar()
    n = int(e.ventana_fft * e.refresh_hz)
    sec = gen.secuencia(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / e.refresh_hz)

    print(f"  {'nominal':>9} {'pico':>9} {'error':>9} {'bin':>6}")
    for k, f in enumerate(b.frecuencias):
        x = sec[:, k] - sec[:, k].mean()
        esp = np.abs(np.fft.rfft(x))
        pico = freqs[int(np.argmax(esp))]
        print(f"  {f:>8.1f}Hz {pico:>8.2f}Hz {abs(pico-f):>8.3f}Hz "
              f"{int(np.argmax(esp)):>6}")

    # --- Efecto de usar tiempo en vez de contador ---
    print()
    print("-" * 72)
    print("POR QUE EL CONTADOR DE FRAMES Y NO EL TIEMPO")
    print("-" * 72)
    print("  Se simula una implementacion que acumula deltaTime con un error")
    print("  de solo 0.1 ms por frame, y se mide la deriva de fase.")
    print()
    f_test = b.frecuencias[3]
    dt_nominal = 1.0 / e.refresh_hz
    dt_real = dt_nominal + 0.0001

    for minutos in (0.5, 1, 2, 5):
        n_fr = int(minutos * 60 * e.refresh_hz)
        t_correcto = n_fr * dt_nominal
        t_acumulado = n_fr * dt_real
        deriva_rad = 2 * np.pi * f_test * (t_acumulado - t_correcto)
        deriva_grados = math.degrees(deriva_rad) % 360
        print(f"  Tras {minutos:>4} min: deriva de fase = "
              f"{deriva_grados:>6.1f} grados")
    print()
    print("  Con contador de frames la deriva es exactamente cero, porque la")
    print("  fase se calcula del indice entero y no de un tiempo acumulado.")

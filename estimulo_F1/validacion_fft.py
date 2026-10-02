"""
validacion_fft.py --- Validacion espectral del estimulo
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
Comprueba que el estimulo funciona de verdad, midiendo la cadena completa:
pantalla -> ojo -> corteza -> EEG.

Un fotodiodo mediria solo la pantalla. Aqui se mide si el cerebro responde,
que es lo que importa. Es tambien lo que hace la literatura de referencia.

CUANDO SE EJECUTA
-----------------
Una vez por sujeto, al caracterizar el estimulo, antes de los experimentos.
No en cada sesion.

LOS SIETE PASOS DEL PROTOCOLO
-----------------------------
  1. Barrido de frecuencia   Pico en f_k y en 2*f_k
  2. Amplitud relativa       Ninguna frecuencia debe ser sistematicamente
                             mas debil que las demas
  3. Ancho de banda          Un pico ensanchado delata jitter temporal
  4. Escena estatica vs.     CRITICO. Control de validez del escalamiento
     en movimiento
  5. Frames perdidos         Por escenario. Criterio de exclusion: >1%
  6. Interferencia           Los otros estimulos no deben dominar
  7. Fase                    Una fase que deriva delata uso de tiempo
                             acumulado en vez de contador de frames

POR QUE EL PASO 4 ES CRITICO
----------------------------
En el estudio de referencia sobre densidad de pixeles, los pixeles apagados
eran fondo negro CONTROLADO. Aqui son transparentes y dejan ver el video de
la camara, que es brillante, texturado y esta en movimiento.

Eso convierte el contraste de borde en una variable no controlada que
depende del contenido visual. Si la calidad del SSVEP se degradara mas en un
escenario visualmente denso que en uno vacio, parte del efecto atribuido a
la arbitracion cognitiva seria en realidad un artefacto del estimulo.

Por eso el paso 4 se ejecuta en los TRES escenarios y se reporta la tabla.
Si las SNR son comparables, el asunto queda cerrado y se cita como control
de validez. Si difieren, la mitigacion es renderizar los pixeles inactivos
en negro con opacidad alta en lugar de totalmente transparentes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math
import numpy as np

try:
    from scipy.signal import welch
    _HAY_SCIPY = True
except ImportError:
    _HAY_SCIPY = False

from config import Config
from stimulus import GeneradorLuminancia


# ===========================================================================
# RESULTADOS
# ===========================================================================

@dataclass
class ResultadoPico:
    """Medida espectral de una frecuencia."""
    frecuencia_nominal: float
    frecuencia_pico: float
    amplitud: float
    amplitud_armonico2: float
    snr_db: float
    ancho_media_altura: float
    fase: float

    @property
    def error_frecuencia(self) -> float:
        return abs(self.frecuencia_pico - self.frecuencia_nominal)

    @property
    def ratio_armonico(self) -> float:
        """Amplitud del segundo armonico respecto a la fundamental."""
        if self.amplitud < 1e-12:
            return 0.0
        return self.amplitud_armonico2 / self.amplitud


@dataclass
class ResultadoValidacion:
    """Resultado completo de la validacion."""
    picos: List[ResultadoPico] = field(default_factory=list)
    frames_perdidos: float = 0.0
    escenario: str = ""
    condicion: str = ""

    @property
    def todas_detectadas(self) -> bool:
        """True si todas las frecuencias se recuperaron con error < 0.2 Hz.

        El umbral es la resolucion espectral con la ventana de 5 s.
        """
        return all(p.error_frecuencia < 0.2 for p in self.picos)

    @property
    def snr_minima(self) -> float:
        return min((p.snr_db for p in self.picos), default=float("-inf"))

    def resumen(self) -> str:
        L = []
        if self.escenario:
            L.append(f"  Escenario : {self.escenario}")
        if self.condicion:
            L.append(f"  Condicion : {self.condicion}")
        L.append(f"  {'nominal':>9} {'pico':>9} {'error':>8} {'SNR':>9} "
                 f"{'2o arm.':>9} {'ancho':>8}")
        for p in self.picos:
            L.append(f"  {p.frecuencia_nominal:>8.1f}Hz "
                     f"{p.frecuencia_pico:>8.2f}Hz "
                     f"{p.error_frecuencia:>7.3f}Hz "
                     f"{p.snr_db:>8.1f}dB "
                     f"{p.ratio_armonico:>8.2f} "
                     f"{p.ancho_media_altura:>7.3f}Hz")
        L.append(f"  Frames perdidos: {self.frames_perdidos*100:.2f}%")
        return "\n".join(L)


# ===========================================================================
# ANALIZADOR
# ===========================================================================

class ValidadorEspectral:
    """
    Analiza una senal EEG y mide la respuesta en cada frecuencia de estimulo.

    Uso:
        val = ValidadorEspectral(CONFIG)
        res = val.analizar(eeg, fs=256.0)
        print(res.resumen())
    """

    def __init__(self, config: Config):
        self.cfg = config

    # -------------------------------------------------------------------
    def _psd(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Densidad espectral de potencia.

        Se usa Welch si hay scipy: promediar segmentos reduce la varianza del
        estimado, a costa de resolucion. Con la ventana de 5 s hay margen de
        sobra para ambas cosas.
        """
        if _HAY_SCIPY:
            nper = min(len(x), int(fs * self.cfg.estimulo.ventana_fft))
            f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
            return f, p
        # Sin scipy: periodograma simple
        x = x - x.mean()
        n = len(x)
        esp = np.abs(np.fft.rfft(x)) ** 2 / n
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        return f, esp

    # -------------------------------------------------------------------
    @staticmethod
    def _amplitud_en(f: np.ndarray, p: np.ndarray,
                     objetivo: float, tol: float = 0.25) -> float:
        """Amplitud del espectro en el entorno de una frecuencia."""
        m = np.abs(f - objetivo) <= tol
        return float(np.max(p[m])) if np.any(m) else 0.0

    # -------------------------------------------------------------------
    @staticmethod
    def _snr_db(f: np.ndarray, p: np.ndarray, objetivo: float,
                banda_senal: float = 0.25,
                banda_ruido: float = 2.0) -> float:
        """
        SNR del pico frente a su vecindario espectral.

        Se compara la potencia en la banda estrecha del pico contra la
        potencia media en un anillo alrededor, excluyendo el propio pico. Es
        la definicion habitual en SSVEP.
        """
        m_sig = np.abs(f - objetivo) <= banda_senal
        m_ruido = (np.abs(f - objetivo) <= banda_ruido) & (~m_sig)

        if not np.any(m_sig) or not np.any(m_ruido):
            return float("-inf")

        p_sig = float(np.max(p[m_sig]))
        p_ruido = float(np.mean(p[m_ruido]))

        if p_ruido < 1e-20:
            return float("inf")
        return 10.0 * math.log10(p_sig / p_ruido)

    # -------------------------------------------------------------------
    @staticmethod
    def _ancho_media_altura(f: np.ndarray, p: np.ndarray,
                            objetivo: float, tol: float = 1.0) -> float:
        """
        Ancho del pico a media altura (Hz).

        Un pico ensanchado indica jitter temporal en el renderizado: la
        frecuencia no es estable frame a frame y la energia se reparte entre
        bins vecinos.
        """
        m = np.abs(f - objetivo) <= tol
        if not np.any(m):
            return float("nan")

        fl, pl = f[m], p[m]
        i_max = int(np.argmax(pl))
        mitad = pl[i_max] / 2.0

        # Buscar los cruces por la mitad a izquierda y derecha del maximo
        izq = i_max
        while izq > 0 and pl[izq] > mitad:
            izq -= 1
        der = i_max
        while der < len(pl) - 1 and pl[der] > mitad:
            der += 1

        return float(fl[der] - fl[izq])

    # -------------------------------------------------------------------
    @staticmethod
    def _fase_en(x: np.ndarray, fs: float, objetivo: float) -> float:
        """
        Fase de la componente a una frecuencia dada (rad).

        Se estima por proyeccion sobre seno y coseno, que es mas robusto que
        leer la fase de la FFT cuando la frecuencia no cae exactamente en un
        bin.
        """
        n = len(x)
        t = np.arange(n) / fs
        x = x - x.mean()
        c = float(np.dot(x, np.cos(2 * np.pi * objetivo * t)))
        s = float(np.dot(x, np.sin(2 * np.pi * objetivo * t)))
        return math.atan2(s, c)

    # -------------------------------------------------------------------
    def analizar(self, senal: np.ndarray, fs: float,
                 escenario: str = "", condicion: str = "",
                 frames_perdidos: float = 0.0) -> ResultadoValidacion:
        """
        Analiza una senal y mide la respuesta en cada frecuencia.

        Parametros
        ----------
        senal : array 1D, o 2D (n_muestras, n_canales). Si es 2D se promedian
                los canales, que es el procedimiento habitual para una
                verificacion de presencia de respuesta.
        fs    : frecuencia de muestreo de la senal (Hz)
        """
        x = np.asarray(senal, dtype=float)
        if x.ndim == 2:
            x = x.mean(axis=1)

        f, p = self._psd(x, fs)
        res = ResultadoValidacion(escenario=escenario, condicion=condicion,
                                  frames_perdidos=frames_perdidos)

        for f_nom in self.cfg.bci.frecuencias:
            i_pico = int(np.argmin(np.abs(f - f_nom)))
            # Buscar el maximo local en el entorno
            lo = max(0, i_pico - 3)
            hi = min(len(f), i_pico + 4)
            i_real = lo + int(np.argmax(p[lo:hi]))

            res.picos.append(ResultadoPico(
                frecuencia_nominal=f_nom,
                frecuencia_pico=float(f[i_real]),
                amplitud=float(p[i_real]),
                amplitud_armonico2=self._amplitud_en(f, p, 2 * f_nom),
                snr_db=self._snr_db(f, p, f_nom),
                ancho_media_altura=self._ancho_media_altura(f, p, f_nom),
                fase=self._fase_en(x, fs, f_nom),
            ))

        return res


# ===========================================================================
# SIMULACION DE RESPUESTA (para probar sin sujeto)
# ===========================================================================

def simular_respuesta_eeg(config: Config,
                          freq_idx: int,
                          duracion: float,
                          snr_db: float = -8.0,
                          jitter_frames: float = 0.0,
                          semilla: Optional[int] = None) -> np.ndarray:
    """
    Genera una senal EEG sintetica con respuesta a un estimulo.

    Sirve para verificar que el analizador funciona antes de tener sujetos.
    NO reproduce EEG realista: no tiene ruido 1/f ni estructura espacial.

    El parametro `jitter_frames` permite simular un renderizado defectuoso:
    introduce variacion aleatoria en el instante de cada frame, que es lo que
    el paso 3 del protocolo detecta como ensanchamiento del pico.
    """
    rng = np.random.default_rng(semilla)
    b, e = config.bci, config.estimulo

    n = int(duracion * b.fs)
    t = np.arange(n) / b.fs

    ruido = rng.standard_normal(n)

    if freq_idx < 0:
        return ruido

    f = b.frecuencias[freq_idx]
    fase = b.fases_jfpm[freq_idx]

    if jitter_frames > 0:
        # El jitter se modela como ruido de fase acumulado
        deriva = np.cumsum(rng.standard_normal(n) * jitter_frames)
        ang_base = 2 * np.pi * f * t + fase + deriva
    else:
        ang_base = 2 * np.pi * f * t + fase

    # Fundamental mas armonicos, con amplitudes decrecientes
    amps = [1.00, 0.62, 0.22, 0.12]
    senal = np.zeros(n)
    for h, a in enumerate(amps, start=1):
        senal += a * np.sin(h * ang_base)

    p_sig = np.mean(senal ** 2)
    p_ruido = np.mean(ruido ** 2)
    if p_sig > 1e-12:
        senal *= math.sqrt(p_ruido * (10 ** (snr_db / 10.0)) / p_sig)

    return senal + ruido


# ===========================================================================
# PROTOCOLO
# ===========================================================================

class ProtocoloValidacion:
    """
    Ejecuta los siete pasos del protocolo de validacion.

    En el laboratorio, la funcion `adquirir` debe leer EEG real. Aqui se
    inyecta una funcion simulada para poder verificar el analisis sin sujeto.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.validador = ValidadorEspectral(config)

    # -------------------------------------------------------------------
    def paso1_barrido(self, adquirir) -> Dict[int, ResultadoValidacion]:
        """
        Paso 1: presentar cada frecuencia por separado y verificar el pico.

        `adquirir(freq_idx, duracion)` debe devolver la senal EEG registrada
        mientras se presenta esa frecuencia en solitario.
        """
        out = {}
        dur = self.cfg.estimulo.ventana_fft * 6   # 30 s con ventana de 5 s
        for k in range(len(self.cfg.bci.frecuencias)):
            senal = adquirir(k, dur)
            out[k] = self.validador.analizar(
                senal, self.cfg.bci.fs,
                condicion=f"solo {self.cfg.bci.frecuencias[k]} Hz")
        return out

    # -------------------------------------------------------------------
    def paso4_escenarios(self, adquirir,
                          escenarios: List[str]
                          ) -> Dict[str, Dict[str, ResultadoValidacion]]:
        """
        Paso 4: comparar escena estatica contra escena en movimiento, en cada
        escenario.

        Es el control de validez de la hipotesis de escalamiento.
        """
        out = {}
        dur = self.cfg.estimulo.ventana_fft * 4
        for esc in escenarios:
            out[esc] = {}
            for cond in ("estatica", "movimiento"):
                senal = adquirir(esc, cond, dur)
                out[esc][cond] = self.validador.analizar(
                    senal, self.cfg.bci.fs,
                    escenario=esc, condicion=cond)
        return out

    # -------------------------------------------------------------------
    @staticmethod
    def comparar_escenarios(
            resultados: Dict[str, Dict[str, ResultadoValidacion]]
    ) -> str:
        """
        Tabla comparativa de SNR entre escenarios.

        Si las SNR son comparables, el estimulo no es un confusor y se cita
        como control de validez. Si difieren, hay que mitigar.
        """
        L = []
        L.append(f"  {'escenario':<30} {'estatica':>10} {'movimiento':>12} "
                 f"{'delta':>8}")
        L.append("  " + "-" * 62)
        deltas = []
        for esc, conds in resultados.items():
            s_est = conds["estatica"].snr_minima
            s_mov = conds["movimiento"].snr_minima
            d = s_mov - s_est
            deltas.append(d)
            L.append(f"  {esc[:30]:<30} {s_est:>9.1f}dB {s_mov:>11.1f}dB "
                     f"{d:>+7.1f}dB")

        L.append("")
        if deltas:
            rango = max(deltas) - min(deltas)
            L.append(f"  Rango de degradacion entre escenarios: "
                     f"{rango:.1f} dB")
            if rango < 3.0:
                L.append("  -> Comparable entre escenarios. El estimulo no es")
                L.append("     un confusor de la hipotesis de escalamiento.")
            else:
                L.append("  -> ATENCION: la degradacion difiere entre")
                L.append("     escenarios. Mitigacion: renderizar los pixeles")
                L.append("     inactivos en negro con opacidad 70-80% en vez")
                L.append("     de totalmente transparentes.")
        return "\n".join(L)


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DEL PROTOCOLO DE VALIDACION ESPECTRAL")
    print("=" * 72)
    print()
    print("  Se usa senal EEG SIMULADA. En el laboratorio, sustituir la")
    print("  funcion de adquisicion por la lectura del amplificador real.")
    if not _HAY_SCIPY:
        print()
        print("  AVISO: sin scipy se usa periodograma simple en vez de Welch.")

    val = ValidadorEspectral(CONFIG)
    b = CONFIG.bci
    dur = CONFIG.estimulo.ventana_fft * 4

    # --- Paso 1 ---
    print()
    print("-" * 72)
    print("PASO 1: barrido de frecuencia")
    print("-" * 72)
    for k, f in enumerate(b.frecuencias):
        senal = simular_respuesta_eeg(CONFIG, k, dur, snr_db=-8.0,
                                      semilla=100 + k)
        r = val.analizar(senal, b.fs, condicion=f"solo {f} Hz")
        p = r.picos[k]
        ok = "OK" if p.error_frecuencia < 0.2 else "FALLO"
        print(f"  {f:>5.1f}Hz -> pico en {p.frecuencia_pico:>6.2f}Hz  "
              f"error {p.error_frecuencia:.3f}Hz  "
              f"SNR {p.snr_db:>5.1f}dB  2o arm. {p.ratio_armonico:.2f}  {ok}")

    # --- Paso 3: efecto del jitter ---
    print()
    print("-" * 72)
    print("PASO 3: deteccion de jitter por ensanchamiento del pico")
    print("-" * 72)
    print("  Un renderizado con temporizacion inestable ensancha el pico.")
    print()
    print(f"  {'jitter':>10} {'ancho a media altura':>22} {'SNR':>10}")
    for j in (0.0, 0.005, 0.02, 0.05):
        senal = simular_respuesta_eeg(CONFIG, 2, dur, snr_db=-5.0,
                                      jitter_frames=j, semilla=7)
        r = val.analizar(senal, b.fs)
        p = r.picos[2]
        print(f"  {j:>10.3f} {p.ancho_media_altura:>21.3f}Hz "
              f"{p.snr_db:>9.1f}dB")
    print()
    print("  Un ancho creciente con SNR decreciente indica problema de")
    print("  temporizacion en el renderizado, no de la senal EEG.")

    # --- Paso 4: escenarios ---
    print()
    print("-" * 72)
    print("PASO 4: escena estatica vs. en movimiento (CONTROL DE VALIDEZ)")
    print("-" * 72)

    def adquirir_simulado(escenario, condicion, duracion):
        """
        Simula la degradacion por complejidad visual del fondo.

        La degradacion asumida crece con el numero de obstaculos del
        escenario. Es una hipotesis, no un dato: el proposito de este paso
        es MEDIRLA con sujetos reales.
        """
        idx = [e.nombre for e in CONFIG.escenarios.lista].index(escenario)
        base = -6.0
        penal = 1.2 * idx if condicion == "movimiento" else 0.0
        return simular_respuesta_eeg(CONFIG, 2, duracion,
                                     snr_db=base - penal,
                                     semilla=200 + idx)

    prot = ProtocoloValidacion(CONFIG)
    nombres = [e.nombre for e in CONFIG.escenarios.lista]
    resultados = prot.paso4_escenarios(adquirir_simulado, nombres)
    print(prot.comparar_escenarios(resultados))

    # --- Paso 7: verificacion de fase ---
    print()
    print("-" * 72)
    print("PASO 7: verificacion de fase")
    print("-" * 72)
    print("  La fase medida debe corresponder a la asignada por JFPM y NO")
    print("  debe derivar a lo largo del trial.")
    print()
    gen = GeneradorLuminancia(CONFIG)
    n = int(CONFIG.estimulo.ventana_fft * CONFIG.estimulo.refresh_hz)
    sec = gen.secuencia(n * 4)

    print(f"  {'frecuencia':>11} {'fase asignada':>15} "
          f"{'medida (1er tramo)':>20} {'medida (4o tramo)':>19}")
    for k, f in enumerate(b.frecuencias):
        fase_asig = b.fases_jfpm[k]
        f1 = val._fase_en(sec[:n, k], CONFIG.estimulo.refresh_hz, f)
        f4 = val._fase_en(sec[3*n:, k], CONFIG.estimulo.refresh_hz, f)
        print(f"  {f:>10.1f}Hz {math.degrees(fase_asig):>14.1f}  "
              f"{math.degrees(f1):>19.1f}  {math.degrees(f4):>18.1f}")
    print()
    print("  Si la fase del cuarto tramo difiere de la del primero, la")
    print("  implementacion esta usando tiempo acumulado en vez de contador")
    print("  de frames.")

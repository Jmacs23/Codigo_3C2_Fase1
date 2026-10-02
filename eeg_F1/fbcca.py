"""
fbcca.py --- Clasificador FBCCA (Filter Bank Canonical Correlation Analysis)
Bloque 1: nucleo_F1

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
Responde a la pregunta "?cual de las cuatro frecuencias esta mirando el
usuario?" a partir de una ventana de EEG.

Es una pregunta DISTINTA de la que responde TASM. TASM decide si el usuario
esta mirando algo (estado cognitivo IC/TR/Idle); FBCCA decide cual. Ambos
corren en paralelo sobre la misma senal.

POR QUE FBCCA Y NO CCA SIMPLE
-----------------------------
La respuesta SSVEP no esta solo en la frecuencia fundamental: tambien
aparece en sus armonicos. Un CCA simple mira una sola banda y desperdicia
esa informacion.

FBCCA divide la senal en varias sub-bandas, calcula la correlacion en cada
una, y las combina con pesos decrecientes:

    rho_k = suma_m [ w_m * rho_k^(m)^2 ],    w_m = m^(-a) + b

Los pesos decrecen porque la amplitud del SSVEP cae rapido con el orden del
armonico. Segun la literatura de referencia, para densidad plena de pixeles
los ordenes 1 a 4 tienen amplitudes aproximadas de 1.05, 0.65, 0.23 y
0.13 uV: el cuarto ya aporta poco.

NO REQUIERE ENTRENAMIENTO
-------------------------
FBCCA compara la senal contra referencias sinusoidales generadas
analiticamente. No necesita datos de calibracion del sujeto, a diferencia
de TRCA. Esa es la razon por la que se emplea aqui: reduce el tiempo de
preparacion de cada sesion.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch
    _HAY_SCIPY = True
except ImportError:
    _HAY_SCIPY = False

from config import Config


# ===========================================================================
# RESULTADO
# ===========================================================================

@dataclass
class ResultadoFBCCA:
    """Salida de una clasificacion."""

    freq_idx: int
    """Indice de la frecuencia ganadora (0..K-1)."""

    freq_hz: float
    """Frecuencia ganadora en Hz."""

    rho: np.ndarray
    """Vector de correlaciones combinadas, una por frecuencia.

    Este vector es lo que TASM consume: no le interesa cual gana, sino la
    forma del vector completo y como cambia en el tiempo."""

    p_max: float
    """Confianza normalizada de la frecuencia ganadora, en [0,1].

    Se calcula como rho_max / suma(rho). NO es una probabilidad calibrada:
    es un indicador relativo. Para obtener probabilidades bien calibradas
    haria falta un paso de calibracion adicional (temperature scaling)."""

    @property
    def margen(self) -> float:
        """Diferencia entre la mejor y la segunda mejor correlacion.

        Un margen pequeno indica que dos frecuencias compiten, lo que suele
        ocurrir durante una transicion de mirada."""
        if len(self.rho) < 2:
            return 0.0
        orden = np.sort(self.rho)[::-1]
        return float(orden[0] - orden[1])


# ===========================================================================
# CLASIFICADOR
# ===========================================================================

class FBCCA:
    """
    Clasificador FBCCA.

    Uso:
        clf = FBCCA(CONFIG)
        res = clf.clasificar(ventana)   # ventana: (n_muestras, n_canales)
        print(res.freq_hz, res.rho)

    Todos los parametros vienen de `config.bci`. Este modulo no define
    ningun valor numerico propio.
    """

    def __init__(self, config: Config):
        self.cfg = config
        b = config.bci

        self.frecuencias = np.asarray(b.frecuencias, dtype=float)
        self.fs = b.fs
        self.n_armonicos = b.n_armonicos
        self.n_subbandas = b.n_subbandas

        # Pesos del banco: w_m = m^(-a) + b, con m = 1..M
        m = np.arange(1, self.n_subbandas + 1, dtype=float)
        self.pesos = m ** (-b.fbcca_a) + b.fbcca_b

        # Cache de referencias sinusoidales por longitud de ventana. Generar
        # las referencias es costoso y la longitud rara vez cambia, asi que
        # se reutilizan.
        self._cache_ref = {}

        # Filtros del banco, disenados una sola vez
        self._filtros = self._disenar_banco() if _HAY_SCIPY else None

    # -------------------------------------------------------------------
    def _disenar_banco(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Disena los filtros paso-banda del banco.

        La sub-banda m cubre desde m * f_min hasta el limite superior de la
        banda de paso, de modo que capture la fundamental y los armonicos a
        partir del orden m.
        """
        b = self.cfg.bci
        f_min = float(np.min(self.frecuencias))
        f_max_banda = b.banda_hz[1]
        nyq = self.fs / 2.0

        filtros = []
        for m in range(1, self.n_subbandas + 1):
            lo = m * f_min - 2.0          # margen inferior
            hi = min(f_max_banda, nyq * 0.95)

            lo = max(lo, 1.0)
            if lo >= hi:
                # Sub-banda vacia: se marca como None y se omitira
                filtros.append(None)
                continue

            try:
                sos_b, sos_a = butter(4, [lo / nyq, hi / nyq], btype="band")
                filtros.append((sos_b, sos_a))
            except ValueError:
                filtros.append(None)

        return filtros

    # -------------------------------------------------------------------
    def _referencias(self, n_muestras: int) -> np.ndarray:
        """
        Genera las senales de referencia sinusoidales.

        Devuelve un array (K, 2*n_armonicos, n_muestras) donde K es el numero
        de frecuencias. Para cada frecuencia se apilan pares seno/coseno de
        cada armonico:

            Y_f = [sin(2*pi*f*t), cos(2*pi*f*t), ...,
                   sin(2*pi*N*f*t), cos(2*pi*N*f*t)]

        El par seno/coseno es necesario porque la fase de la respuesta SSVEP
        es desconocida: con solo el seno, una respuesta desfasada 90 grados
        daria correlacion nula.
        """
        clave = n_muestras
        if clave in self._cache_ref:
            return self._cache_ref[clave]

        t = np.arange(n_muestras) / self.fs
        K = len(self.frecuencias)
        Y = np.zeros((K, 2 * self.n_armonicos, n_muestras))

        for k, f in enumerate(self.frecuencias):
            for h in range(1, self.n_armonicos + 1):
                ang = 2 * np.pi * h * f * t
                Y[k, 2 * (h - 1)]     = np.sin(ang)
                Y[k, 2 * (h - 1) + 1] = np.cos(ang)

        self._cache_ref[clave] = Y
        return Y

    # -------------------------------------------------------------------
    @staticmethod
    def _cca(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Maxima correlacion canonica entre dos conjuntos de senales.

        X : (n_muestras, n_canales)
        Y : (n_muestras, n_referencias)

        Se resuelve por descomposicion QR en lugar del problema de valores
        propios generalizado: es equivalente, mas estable numericamente y mas
        breve. Los valores singulares de Q_x^T Q_y son exactamente las
        correlaciones canonicas.
        """
        # Centrar
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Descartar canales o referencias constantes, que darian division
        # por cero al normalizar
        sx = X.std(axis=0)
        sy = Y.std(axis=0)
        X = X[:, sx > 1e-12]
        Y = Y[:, sy > 1e-12]

        if X.size == 0 or Y.size == 0:
            return 0.0

        try:
            Qx, _ = np.linalg.qr(X)
            Qy, _ = np.linalg.qr(Y)
            s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0.0

        if s.size == 0:
            return 0.0

        # Acotar a [0,1]: errores numericos pueden dar valores marginalmente
        # fuera del rango
        return float(np.clip(s[0], 0.0, 1.0))

    # -------------------------------------------------------------------
    def _filtrar(self, X: np.ndarray, m: int) -> Optional[np.ndarray]:
        """Aplica el filtro de la sub-banda m. Devuelve None si no aplica."""
        if self._filtros is None:
            # Sin scipy solo se usa la banda completa
            return X if m == 0 else None

        f = self._filtros[m]
        if f is None:
            return None

        b_coef, a_coef = f
        try:
            # filtfilt es de fase cero: valido porque aqui se procesa una
            # ventana completa ya adquirida, no en streaming causal.
            return filtfilt(b_coef, a_coef, X, axis=0)
        except ValueError:
            # Ventana demasiado corta para el orden del filtro
            return None

    # -------------------------------------------------------------------
    def clasificar(self, ventana: np.ndarray) -> ResultadoFBCCA:
        """
        Clasifica una ventana de EEG.

        Parametros
        ----------
        ventana : array (n_muestras, n_canales)

        Devuelve
        --------
        ResultadoFBCCA con la frecuencia ganadora y el vector completo de
        correlaciones.
        """
        X = np.asarray(ventana, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        n_muestras = X.shape[0]
        K = len(self.frecuencias)
        Y = self._referencias(n_muestras)

        rho = np.zeros(K)

        for m in range(self.n_subbandas):
            Xf = self._filtrar(X, m)
            if Xf is None:
                continue
            w = self.pesos[m]
            for k in range(K):
                r = self._cca(Xf, Y[k].T)
                rho[k] += w * (r ** 2)

        idx = int(np.argmax(rho))
        total = float(np.sum(rho))
        p_max = float(rho[idx] / total) if total > 1e-12 else 0.0

        return ResultadoFBCCA(
            freq_idx=idx,
            freq_hz=float(self.frecuencias[idx]),
            rho=rho,
            p_max=p_max,
        )

    # -------------------------------------------------------------------
    def gradiente(self, rho_actual: np.ndarray,
                  rho_anterior: np.ndarray) -> np.ndarray:
        """
        Gradiente temporal del vector de correlaciones.

        Es la entrada que TASM usa para detectar transiciones: durante un
        desplazamiento de mirada, la correlacion de la frecuencia que se
        abandona cae mientras la de la que se adquiere sube. Ese patron
        cruzado es la firma de una transicion.
        """
        return np.asarray(rho_actual) - np.asarray(rho_anterior)


# ===========================================================================
# GENERADOR DE SENAL SINTETICA (para probar sin EEG)
# ===========================================================================

def generar_ssvep(config: Config,
                  freq_idx: int,
                  duracion: float,
                  snr_db: float = -5.0,
                  semilla: Optional[int] = None) -> np.ndarray:
    """
    Genera una senal EEG sintetica con respuesta SSVEP a una frecuencia.

    Sirve para probar el clasificador sin hardware. NO pretende ser un modelo
    realista de EEG: no reproduce la estructura espacial entre canales ni el
    ruido 1/f. Es suficiente para verificar que el pipeline corre y que la
    frecuencia dominante se recupera.

    Parametros
    ----------
    freq_idx : indice de la frecuencia a inyectar; -1 para ruido puro (Idle)
    snr_db   : relacion senal-ruido en dB. Valores tipicos de SSVEP real
               estan entre -10 y 0 dB.
    """
    rng = np.random.default_rng(semilla)
    b = config.bci
    n = int(duracion * b.fs)
    t = np.arange(n) / b.fs
    n_ch = b.n_canales

    ruido = rng.standard_normal((n, n_ch))

    if freq_idx < 0:
        return ruido

    f = b.frecuencias[freq_idx]
    fase = b.fases_jfpm[freq_idx]

    # Amplitudes relativas por armonico, coherentes con lo reportado en la
    # literatura (decaimiento rapido con el orden)
    amp = np.array([1.00, 0.62, 0.22, 0.12, 0.06])[:b.n_armonicos]

    senal = np.zeros((n, n_ch))
    for h in range(1, b.n_armonicos + 1):
        senal += amp[h - 1] * np.sin(2 * np.pi * h * f * t + fase)[:, None]

    # Escalar segun la SNR pedida
    p_sig = np.mean(senal ** 2)
    p_ruido = np.mean(ruido ** 2)
    if p_sig > 1e-12:
        escala = np.sqrt(p_ruido * (10 ** (snr_db / 10.0)) / p_sig)
        senal *= escala

    return senal + ruido


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 70)
    print("DEMOSTRACION DEL CLASIFICADOR FBCCA")
    print("=" * 70)
    print()
    if not _HAY_SCIPY:
        print("AVISO: scipy no esta instalado. Se usara la banda completa sin")
        print("       banco de filtros. Instala scipy para el comportamiento")
        print("       correcto:  pip install scipy")
        print()

    clf = FBCCA(CONFIG)
    b = CONFIG.bci

    print(f"Frecuencias : {b.frecuencias} Hz")
    print(f"Ventana     : {b.tw_tasm*1000:.0f} ms "
          f"({b.muestras_ventana_tasm} muestras)")
    print(f"Sub-bandas  : {b.n_subbandas}, armonicos: {b.n_armonicos}")
    print(f"Pesos w_m   : {np.round(clf.pesos, 3)}")
    print()

    print("-" * 70)
    print("PRUEBA 1: recuperar la frecuencia inyectada")
    print("-" * 70)
    print(f"{'inyectada':>10} {'detectada':>10} {'p_max':>7} {'margen':>8}  ok")
    aciertos = 0
    for k in range(len(b.frecuencias)):
        x = generar_ssvep(CONFIG, k, duracion=b.tw_tasm,
                          snr_db=0.0, semilla=100 + k)
        r = clf.clasificar(x)
        ok = (r.freq_idx == k)
        aciertos += ok
        print(f"{b.frecuencias[k]:>10.1f} {r.freq_hz:>10.1f} "
              f"{r.p_max:>7.3f} {r.margen:>8.4f}  {'SI' if ok else 'NO'}")
    print(f"\n  Aciertos: {aciertos}/{len(b.frecuencias)}")
    print()

    print("-" * 70)
    print("PRUEBA 2: efecto de la longitud de ventana")
    print("-" * 70)
    print("  La ventana de TASM (750 ms) es la que se usa en el sistema.")
    print()
    print(f"{'ventana':>9} {'aciertos':>10} {'p_max medio':>13}")
    for dur in (0.25, 0.50, 0.75, 1.00):
        ac = 0
        pm = 0.0
        n_rep = 5
        for k in range(len(b.frecuencias)):
            for rep in range(n_rep):
                x = generar_ssvep(CONFIG, k, duracion=dur, snr_db=0.0,
                                  semilla=1000 + k * 10 + rep)
                r = clf.clasificar(x)
                ac += (r.freq_idx == k)
                pm += r.p_max
        tot = len(b.frecuencias) * n_rep
        print(f"{dur*1000:>7.0f}ms {ac:>6}/{tot:<3} {pm/tot:>13.3f}")
    print()

    print("-" * 70)
    print("PRUEBA 3: gradiente durante una transicion simulada")
    print("-" * 70)
    print("  Se pasa de mirar la frecuencia 0 a la frecuencia 2.")
    print("  Observa el patron cruzado: una correlacion baja y otra sube.")
    print()
    x0 = generar_ssvep(CONFIG, 0, b.tw_tasm, snr_db=0.0, semilla=1)
    x2 = generar_ssvep(CONFIG, 2, b.tw_tasm, snr_db=0.0, semilla=2)
    r0 = clf.clasificar(x0)
    r2 = clf.clasificar(x2)
    g = clf.gradiente(r2.rho, r0.rho)
    print(f"  rho antes  : {np.round(r0.rho, 4)}")
    print(f"  rho despues: {np.round(r2.rho, 4)}")
    print(f"  gradiente  : {np.round(g, 4)}")
    print()
    print(f"  Mayor caida  en indice {int(np.argmin(g))} "
          f"(frecuencia {b.frecuencias[int(np.argmin(g))]} Hz)")
    print(f"  Mayor subida en indice {int(np.argmax(g))} "
          f"(frecuencia {b.frecuencias[int(np.argmax(g))]} Hz)")

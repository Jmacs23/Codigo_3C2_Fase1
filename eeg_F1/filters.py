"""
filters.py --- Filtrado causal con estado persistente
Bloque 3: eeg_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado con el amplificador real.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Filtra la senal EEG en tiempo real: elimina la interferencia de red y limita
la banda a la region util.

Es el modulo mas delicado del bloque desde el punto de vista tecnico, por
dos razones que se explican abajo.


PROBLEMA 1: filtfilt NO SIRVE ONLINE
------------------------------------
La funcion `filtfilt` de scipy es la que casi todo el mundo usa para filtrar
EEG, porque no introduce distorsion de fase. Lo consigue filtrando la senal
dos veces: hacia adelante y hacia atras.

Y ahi esta el problema: para calcular la salida en el instante t, el pase
hacia atras necesita muestras POSTERIORES a t. En un analisis offline eso es
trivial porque toda la senal ya esta grabada. En tiempo real es imposible:
esas muestras aun no existen.

Usar filtfilt en un sistema online no da un error ni una excepcion. Da
resultados que parecen razonables pero que en realidad usan informacion del
futuro dentro del bloque, lo que produce un rendimiento offline
artificialmente bueno que no se reproduce en linea.

Este modulo usa `lfilter`, que es causal: cada muestra de salida depende
solo de muestras presentes y pasadas.


PROBLEMA 2: EL ESTADO DEBE PERSISTIR ENTRE BLOQUES
--------------------------------------------------
Un filtro IIR tiene memoria. Su salida en el instante t depende de las
entradas y salidas anteriores.

Si se llama a `lfilter` sobre cada bloque de 12 muestras sin conservar ese
estado interno, cada bloque arranca desde cero. El resultado es un
TRANSITORIO al principio de cada bloque: un escalon artificial que el
clasificador puede leer como senal.

A 20 bloques por segundo, eso serian 20 transitorios por segundo
contaminando la senal.

La solucion es pasar el estado (`zi`) de una llamada a la siguiente, que es
lo que hace la clase FiltroCausal. Es una linea de codigo, pero olvidarla
produce un fallo silencioso y dificil de diagnosticar.


COMO VERIFICARLO
----------------
`python run_eeg.py filtros` compara las tres variantes sobre la misma senal:
filtfilt, lfilter sin estado, y lfilter con estado. La diferencia entre la
segunda y la tercera es visible de inmediato.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

try:
    from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, filtfilt
    _HAY_SCIPY = True
except ImportError:
    _HAY_SCIPY = False

from config import Config


# ===========================================================================
# FILTRO CAUSAL CON ESTADO
# ===========================================================================

class FiltroCausal:
    """
    Filtro IIR causal que conserva su estado entre llamadas.

    Uso:
        filtro = FiltroCausal(b, a, n_canales=9)
        y1 = filtro.aplicar(bloque1)   # el estado se conserva
        y2 = filtro.aplicar(bloque2)   # continua donde quedo

    El estado se mantiene por canal: cada canal tiene su propia memoria del
    filtro, porque son senales independientes.
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, n_canales: int):
        if not _HAY_SCIPY:
            raise ImportError("FiltroCausal requiere scipy.")

        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.n_canales = n_canales

        # Estado inicial del filtro, uno por canal.
        # lfilter_zi devuelve el estado que corresponde a una entrada
        # constante de valor 1. Se escala luego con la primera muestra real
        # para que el filtro arranque ya "asentado" en vez de con un
        # transitorio desde cero.
        self._zi_base = lfilter_zi(self.b, self.a)
        self.reiniciar()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """
        Borra el estado. Llamar SOLO al empezar una sesion nueva.

        Llamarlo entre bloques anularia el proposito del filtro con estado.
        """
        self._zi = np.zeros((self.n_canales, len(self._zi_base)))
        self._inicializado = False

    # -------------------------------------------------------------------
    def aplicar(self, bloque: np.ndarray) -> np.ndarray:
        """
        Filtra un bloque, continuando desde el estado anterior.

        Parametros
        ----------
        bloque : array (n_muestras, n_canales)

        Devuelve
        --------
        Array de la misma forma, filtrado.
        """
        x = np.asarray(bloque, dtype=float)
        if x.ndim == 1:
            x = x[:, None]

        if x.shape[1] != self.n_canales:
            raise ValueError(
                f"El bloque tiene {x.shape[1]} canales, "
                f"el filtro espera {self.n_canales}."
            )

        # En el primer bloque se ajusta el estado al nivel de continua de la
        # senal. Sin esto, el filtro veria un escalon desde cero hasta el
        # valor real y respondería con un transitorio de varios cientos de
        # milisegundos.
        if not self._inicializado:
            for c in range(self.n_canales):
                self._zi[c] = self._zi_base * x[0, c]
            self._inicializado = True

        y = np.empty_like(x)
        for c in range(self.n_canales):
            y[:, c], self._zi[c] = lfilter(
                self.b, self.a, x[:, c], zi=self._zi[c])

        return y

    # -------------------------------------------------------------------
    @property
    def estado(self) -> np.ndarray:
        """Estado interno actual. Solo para diagnostico."""
        return self._zi.copy()


# ===========================================================================
# CADENA DE PREPROCESAMIENTO
# ===========================================================================

class CadenaFiltrado:
    """
    Cadena completa: notch de red seguido de paso-banda.

    El orden importa. El notch va primero porque la interferencia de red
    suele ser la componente de mayor amplitud, y eliminarla antes reduce el
    rango dinamico que el paso-banda tiene que manejar.

    Uso:
        cadena = CadenaFiltrado(CONFIG)
        y = cadena.aplicar(bloque)
    """

    def __init__(self, config: Config):
        if not _HAY_SCIPY:
            raise ImportError("CadenaFiltrado requiere scipy.")

        self.cfg = config
        b, a = config.bci, config.adquisicion

        fs = b.fs
        nyq = fs / 2.0
        n_ch = b.n_canales

        # --- Notch de red ---
        # Q alto para que la banda suprimida sea estrecha. Ver la nota en
        # config.py sobre por que este valor no es arbitrario.
        b_n, a_n = iirnotch(b.notch_hz, a.q_notch, fs)
        self.notch = FiltroCausal(b_n, a_n, n_ch)

        # --- Paso-banda ---
        lo, hi = b.banda_hz
        # El limite superior se acota por debajo de Nyquist para evitar
        # inestabilidad numerica del diseno del filtro.
        hi = min(hi, nyq * 0.98)
        b_p, a_p = butter(a.orden_pasabanda, [lo / nyq, hi / nyq],
                          btype="band")
        self.pasabanda = FiltroCausal(b_p, a_p, n_ch)

        # Guardados para diagnostico y para el calculo de respuesta
        self._coef_notch = (b_n, a_n)
        self._coef_banda = (b_p, a_p)

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Reinicia ambos filtros. Solo al empezar una sesion."""
        self.notch.reiniciar()
        self.pasabanda.reiniciar()

    # -------------------------------------------------------------------
    def aplicar(self, bloque: np.ndarray) -> np.ndarray:
        """Aplica la cadena completa a un bloque."""
        return self.pasabanda.aplicar(self.notch.aplicar(bloque))

    # -------------------------------------------------------------------
    def respuesta_frecuencia(self, n: int = 4096
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Respuesta en frecuencia de la cadena completa.

        Sirve para verificar que las frecuencias de interes y sus armonicos
        pasan, y que la red queda suprimida.
        """
        from scipy.signal import freqz

        fs = self.cfg.bci.fs
        w, h_n = freqz(*self._coef_notch, worN=n, fs=fs)
        _, h_p = freqz(*self._coef_banda, worN=n, fs=fs)
        return w, np.abs(h_n * h_p)

    # -------------------------------------------------------------------
    def ganancia_en(self, frecuencia: float) -> float:
        """Ganancia de la cadena a una frecuencia concreta."""
        w, h = self.respuesta_frecuencia()
        return float(np.interp(frecuencia, w, h))

    # -------------------------------------------------------------------
    def verificar(self) -> List[str]:
        """
        Comprueba que la cadena deja pasar lo que debe y suprime lo que debe.

        Devuelve una lista de problemas; vacia si todo esta bien.
        """
        problemas = []
        b = self.cfg.bci

        # La red debe quedar bien suprimida
        g_red = self.ganancia_en(b.notch_hz)
        if g_red > 0.15:
            problemas.append(
                f"La red ({b.notch_hz} Hz) solo se atenua a {g_red:.3f}. "
                f"Sube el orden o baja Q."
            )

        # Las fundamentales deben pasar casi intactas
        for f in b.frecuencias:
            g = self.ganancia_en(f)
            if g < 0.70:
                problemas.append(
                    f"La fundamental {f} Hz se atenua a {g:.3f}. "
                    f"Revisa la banda de paso."
                )

        # Los armonicos utiles deben pasar razonablemente
        for f in b.frecuencias:
            for orden in (2, 3, 4):
                arm = f * orden
                if arm >= b.fs / 2:
                    continue
                g = self.ganancia_en(arm)
                # El armonico de 15.2 Hz cerca de la red se atenua algo por
                # la falda del notch: se admite hasta 0.5
                cerca_red = abs(arm - b.notch_hz) < 3.0
                umbral = 0.50 if cerca_red else 0.70
                if g < umbral:
                    problemas.append(
                        f"El armonico {orden} de {f} Hz ({arm:.1f} Hz) se "
                        f"atenua a {g:.3f}."
                    )

        return problemas


# ===========================================================================
# DETECCION DE ARTEFACTOS
# ===========================================================================

@dataclass
class ResultadoArtefactos:
    """Resultado del control de calidad de una ventana."""
    valida: bool
    amplitud_max: float
    gradiente_max: float
    canales_saturados: List[int] = field(default_factory=list)
    motivo: str = ""


class DetectorArtefactos:
    """
    Marca ventanas contaminadas por artefactos.

    No intenta CORREGIR los artefactos, solo detectarlos. Una ventana marcada
    como invalida se propaga como is_valid=False en el contrato TASMState, y
    el consumidor la ignora manteniendo el comando enclavado.

    Esa es la decision correcta: es preferible no actualizar el comando a
    actualizarlo con datos corruptos.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def evaluar(self, ventana: np.ndarray) -> ResultadoArtefactos:
        """
        Evalua una ventana.

        Parametros
        ----------
        ventana : array (n_muestras, n_canales) en microvoltios
        """
        x = np.asarray(ventana, dtype=float)
        if x.ndim == 1:
            x = x[:, None]

        a = self.cfg.adquisicion

        # --- Amplitud ---
        amp = np.abs(x).max(axis=0)
        amp_max = float(amp.max())
        saturados = [int(i) for i in np.where(amp > a.umbral_amplitud)[0]]

        # --- Gradiente ---
        # Un salto mayor que la cota fisica de la banda no puede provenir de
        # senal cerebral: solo de una desconexion de electrodo o un artefacto
        # de gran amplitud. El umbral se DERIVA, no se fija a ojo.
        if x.shape[0] > 1:
            grad = np.abs(np.diff(x, axis=0)).max()
        else:
            grad = 0.0
        grad_max = float(grad)

        umbral_grad = a.umbral_gradiente(self.cfg.bci.banda_hz[1],
                                          self.cfg.bci.fs)

        # --- Veredicto ---
        motivo = ""
        valida = True
        if saturados:
            valida = False
            motivo = (f"amplitud {amp_max:.0f} uV supera el umbral "
                      f"({a.umbral_amplitud:.0f}) en los canales {saturados}")
        elif grad_max > umbral_grad:
            valida = False
            motivo = (f"gradiente {grad_max:.0f} uV entre muestras "
                      f"consecutivas supera el umbral "
                      f"({umbral_grad:.0f}); indica desconexion de electrodo")

        return ResultadoArtefactos(
            valida=valida,
            amplitud_max=amp_max,
            gradiente_max=grad_max,
            canales_saturados=saturados,
            motivo=motivo,
        )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DEL FILTRADO CAUSAL")
    print("=" * 72)

    if not _HAY_SCIPY:
        print()
        print("  scipy no esta instalado. Instala con: pip install scipy")
        raise SystemExit(1)

    b, a = CONFIG.bci, CONFIG.adquisicion
    cadena = CadenaFiltrado(CONFIG)

    # --- Respuesta en frecuencia ---
    print()
    print("-" * 72)
    print("RESPUESTA DE LA CADENA")
    print("-" * 72)
    print(f"  Notch     : {b.notch_hz} Hz, Q={a.q_notch} "
          f"(banda suprimida {b.notch_hz - b.notch_hz/a.q_notch/2:.2f}"
          f"-{b.notch_hz + b.notch_hz/a.q_notch/2:.2f} Hz)")
    print(f"  Pasabanda : {b.banda_hz[0]}-{b.banda_hz[1]} Hz, "
          f"orden {a.orden_pasabanda}")
    print()
    print(f"  {'frecuencia':>12} {'ganancia':>10}  que es")
    puntos = [(1.0, "muy baja, debe cortarse")]
    for f in b.frecuencias:
        puntos.append((f, f"fundamental"))
    puntos.append((b.notch_hz, "RED, debe suprimirse"))
    for f in b.frecuencias:
        puntos.append((f * 4, f"4o armonico de {f} Hz"))
    puntos.append((100.0, "fuera de banda"))

    for f, desc in sorted(puntos):
        if f >= b.fs / 2:
            continue
        g = cadena.ganancia_en(f)
        print(f"  {f:>11.1f}Hz {g:>10.4f}  {desc}")

    print()
    problemas = cadena.verificar()
    if problemas:
        print("  PROBLEMAS:")
        for p in problemas:
            print(f"    - {p}")
    else:
        print("  Verificacion de la cadena: sin problemas.")

    # --- El punto critico: 60.8 Hz ---
    print()
    print("-" * 72)
    print("EL PUNTO CRITICO: EL 4o ARMONICO DE 15.2 Hz")
    print("-" * 72)
    print("  Cae en 60.8 Hz, a solo 0.8 Hz de la red. Es la razon por la que")
    print("  el factor Q del notch no puede ser el habitual de 30.")
    print()
    print(f"  {'Q':>5} {'banda suprimida':>22} {'ganancia en 60.8 Hz':>21}")
    for q in (30, 45, 60, 90):
        cfg_q = CONFIG.copia_con(adquisicion=dict(q_notch=float(q)))
        c_q = CadenaFiltrado(cfg_q)
        ancho = b.notch_hz / q
        print(f"  {q:>5} {b.notch_hz-ancho/2:>10.2f}-"
              f"{b.notch_hz+ancho/2:<10.2f} "
              f"{c_q.ganancia_en(60.8):>21.4f}")

    # --- Comparacion de las tres variantes ---
    print()
    print("-" * 72)
    print("POR QUE HACE FALTA ESTADO PERSISTENTE")
    print("-" * 72)
    print("  Se filtra la MISMA senal de tres formas y se compara el")
    print("  resultado con el filtrado de referencia sobre la senal completa.")
    print()

    rng = np.random.default_rng(1)
    n_total = int(3.0 * b.fs)
    t = np.arange(n_total) / b.fs
    senal = (40 * np.sin(2 * np.pi * 14.0 * t) +
             25 * np.sin(2 * np.pi * b.notch_hz * t) +
             12 * rng.standard_normal(n_total))
    senal = senal[:, None] * np.ones((1, b.n_canales))

    # Referencia: filtrar la senal entera de una vez, causalmente
    ref_cadena = CadenaFiltrado(CONFIG)
    referencia = ref_cadena.aplicar(senal)

    # Variante A: por bloques, CON estado
    cad_a = CadenaFiltrado(CONFIG)
    nb = a.buffer_muestras
    salida_a = np.concatenate(
        [cad_a.aplicar(senal[i:i + nb]) for i in range(0, n_total, nb)])

    # Variante B: por bloques, SIN estado (se reinicia cada vez)
    cad_b = CadenaFiltrado(CONFIG)
    trozos = []
    for i in range(0, n_total, nb):
        cad_b.reiniciar()
        trozos.append(cad_b.aplicar(senal[i:i + nb]))
    salida_b = np.concatenate(trozos)

    err_a = float(np.abs(salida_a - referencia).max())
    err_b = float(np.abs(salida_b - referencia).max())
    rango = float(np.abs(referencia).max())

    print(f"  Rango de la senal filtrada      : {rango:.1f} uV")
    print(f"  Error maximo CON estado         : {err_a:.6f} uV "
          f"({err_a/rango*100:.4f}%)")
    print(f"  Error maximo SIN estado         : {err_b:.2f} uV "
          f"({err_b/rango*100:.1f}%)")
    print()
    print("  Con estado el resultado por bloques es identico al de filtrar")
    print("  la senal entera. Sin estado, cada bloque arranca con un")
    print("  transitorio que el clasificador leeria como senal.")

    # --- filtfilt no es causal ---
    print()
    print("-" * 72)
    print("POR QUE filtfilt NO SIRVE ONLINE")
    print("-" * 72)
    print("  Se aplica filtfilt a una senal que es CERO hasta la mitad y")
    print("  luego un escalon. Un filtro causal no puede reaccionar antes")
    print("  del escalon; filtfilt si lo hace.")
    print()

    n = 200
    escalon = np.zeros(n)
    escalon[n // 2:] = 1.0

    bp, ap = cadena._coef_banda
    y_ff = filtfilt(bp, ap, escalon)
    y_lf = lfilter(bp, ap, escalon)

    antes_ff = float(np.abs(y_ff[:n // 2]).max())
    antes_lf = float(np.abs(y_lf[:n // 2]).max())

    print(f"  Maxima respuesta ANTES del escalon:")
    print(f"    filtfilt (no causal) : {antes_ff:.6f}   <-- reacciona antes")
    print(f"    lfilter  (causal)    : {antes_lf:.6f}")
    print()
    print("  filtfilt usa muestras futuras. En analisis offline eso es")
    print("  legitimo. En tiempo real esas muestras aun no existen, y usar")
    print("  filtfilt daria un rendimiento offline que no se reproduce en")
    print("  linea.")

    # --- Artefactos ---
    print()
    print("-" * 72)
    print("DETECCION DE ARTEFACTOS")
    print("-" * 72)
    det = DetectorArtefactos(CONFIG)
    umbral_g = a.umbral_gradiente(b.banda_hz[1], b.fs)
    print(f"  Umbral de amplitud : {a.umbral_amplitud:.0f} uV")
    print(f"  Umbral de gradiente: {umbral_g:.0f} uV "
          f"(derivado de la banda, no fijado a ojo)")
    print()

    # La senal de prueba se filtra primero, como llegaria en el sistema real
    cad_demo = CadenaFiltrado(CONFIG)
    limpia = cad_demo.aplicar(
        20 * rng.standard_normal((int(b.tw_tasm * b.fs), b.n_canales)))
    r = det.evaluar(limpia)
    print(f"  Ventana limpia     : "
          f"{'valida' if r.valida else 'INVALIDA'}  "
          f"(amplitud {r.amplitud_max:.0f} uV)")

    con_parpadeo = limpia.copy()
    con_parpadeo[50:70, 0] += 300
    r = det.evaluar(con_parpadeo)
    print(f"  Ventana con parpadeo: "
          f"{'valida' if r.valida else 'INVALIDA'}  {r.motivo}")

    desconectado = limpia.copy()
    desconectado[100:, 3] += 200
    r = det.evaluar(desconectado)
    print(f"  Electrodo suelto    : "
          f"{'valida' if r.valida else 'INVALIDA'}  {r.motivo}")

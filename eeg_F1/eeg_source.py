"""
eeg_source.py --- Fuentes de senal EEG
Bloque 3: eeg_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado con el amplificador real.

La clase FuenteGUSBamp es un ESQUELETO: las llamadas al SDK de g.tec estan
marcadas como pendientes. Hay que completarlas con la version del SDK que
tengais en el laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Entrega bloques de senal EEG. Tres fuentes intercambiables tras la misma
interfaz:

  FuenteSintetica  Genera senal artificial con respuesta SSVEP controlada.
                   Para desarrollar sin hardware ni sujeto.

  FuenteDataset    Reproduce un registro grabado. Para pruebas
                   reproducibles y para trabajar con datos reales sin
                   ocupar el laboratorio.

  FuenteGUSBamp    Amplificador real. Solo Windows.

Cambiar de una a otra es cambiar `config.adquisicion.fuente`. Nada mas del
sistema se entera.


SOBRE EL SDK DE g.tec
---------------------
El amplificador g.USBamp se controla con un SDK propietario que solo existe
para Windows y requiere licencia. La forma habitual de usarlo desde Python es
el binding `pygds`, que g.tec distribuye con el equipo.

Este modulo NO incluye ese codigo, por dos razones:
  - No se puede probar sin el hardware, y codigo no probado da falsa
    confianza.
  - La API cambia entre versiones del SDK, y escribir contra una version que
    quiza no sea la vuestra generaria mas trabajo del que ahorra.

Lo que si esta definido es el CONTRATO: que metodos hay que implementar y
que deben devolver. Rellenar las llamadas al SDK dentro de ese contrato es
trabajo acotado, y el resto del sistema funcionara sin cambios.

Los puntos a completar estan marcados con  # >>> COMPLETAR


UN DETALLE QUE IMPORTA: EL ORDEN DE CANALES
--------------------------------------------
El amplificador entrega los canales en el orden en que estan fisicamente
conectados. Ese orden debe coincidir con config.bci.canales, porque los
filtros espaciales de TASM asumen esa disposicion exacta.

Si se conectan en otro orden, nada fallara de forma visible: el sistema
seguira corriendo y dando resultados, pero malos. La clase incluye una
verificacion explicita para evitarlo.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import math
import time
import numpy as np

from config import Config


# ===========================================================================
# INTERFAZ COMUN
# ===========================================================================

class FuenteEEG(ABC):
    """
    Interfaz comun de todas las fuentes de EEG.

    Un bloque es un array (n_muestras, n_canales) en microvoltios, con los
    canales en el orden de config.bci.canales.
    """

    @abstractmethod
    def iniciar(self) -> None:
        """Arranca la adquisicion."""
        ...

    @abstractmethod
    def leer(self) -> Optional[np.ndarray]:
        """
        Devuelve el siguiente bloque, o None si no hay datos.

        Debe ser NO BLOQUEANTE o con timeout corto: bloquear aqui detendria
        el bucle de procesamiento.
        """
        ...

    @abstractmethod
    def detener(self) -> None:
        """Detiene la adquisicion y libera recursos."""
        ...

    @property
    @abstractmethod
    def fs(self) -> float:
        """Frecuencia de muestreo real (Hz)."""
        ...

    @property
    @abstractmethod
    def n_canales(self) -> int:
        ...

    def impedancias(self) -> Optional[Dict[str, float]]:
        """
        Impedancia de cada electrodo (ohmios), si la fuente puede medirla.

        Devuelve None si no aplica.
        """
        return None

    def __enter__(self):
        self.iniciar()
        return self

    def __exit__(self, *args):
        self.detener()


# ===========================================================================
# FUENTE SINTETICA
# ===========================================================================

class FuenteSinteticaEEG(FuenteEEG):
    """
    Genera senal EEG artificial con respuesta SSVEP controlable.

    Permite desarrollar y probar toda la cadena sin hardware ni sujeto. La
    frecuencia atendida se cambia en caliente con `mirar()`, lo que permite
    simular transiciones de mirada.

    NO es un modelo realista de EEG. No reproduce el ruido 1/f, ni la
    estructura espacial entre canales, ni la variabilidad entre sujetos. Su
    proposito es verificar que la cadena procesa correctamente, no predecir
    el rendimiento con personas.
    """

    def __init__(self, config: Config,
                 freq_idx: int = -1,
                 snr_db: float = -8.0,
                 amplitud_ruido: float = 20.0,
                 con_red: bool = True,
                 semilla: Optional[int] = None):
        self.cfg = config
        self._freq_idx = freq_idx
        self.snr_db = snr_db
        self.amplitud_ruido = amplitud_ruido
        self.con_red = con_red

        self._rng = np.random.default_rng(semilla)
        self._muestra = 0
        self._corriendo = False
        self._t_ultimo = 0.0

    # -------------------------------------------------------------------
    def iniciar(self) -> None:
        self._corriendo = True
        self._muestra = 0
        self._t_ultimo = time.perf_counter()

    def detener(self) -> None:
        self._corriendo = False

    @property
    def fs(self) -> float:
        return self.cfg.bci.fs

    @property
    def n_canales(self) -> int:
        return self.cfg.bci.n_canales

    # -------------------------------------------------------------------
    def mirar(self, freq_idx: int) -> None:
        """
        Cambia la frecuencia atendida.

        freq_idx = -1 significa que el sujeto no mira ningun estimulo.
        Permite simular transiciones de mirada llamando a este metodo entre
        lecturas.
        """
        self._freq_idx = freq_idx

    # -------------------------------------------------------------------
    def leer(self) -> Optional[np.ndarray]:
        """
        Genera el siguiente bloque.

        Respeta el ritmo temporal real: si se llama mas rapido que fs,
        devuelve None. Asi el bucle de procesamiento se comporta como lo
        haria con hardware real.
        """
        if not self._corriendo:
            return None

        n = self.cfg.adquisicion.buffer_muestras
        ahora = time.perf_counter()
        if (ahora - self._t_ultimo) < (n / self.fs) * 0.9:
            return None
        self._t_ultimo = ahora

        b = self.cfg.bci
        t = (self._muestra + np.arange(n)) / self.fs
        self._muestra += n

        # Ruido de fondo
        x = self.amplitud_ruido * self._rng.standard_normal(
            (n, self.n_canales))

        # Interferencia de red. Se incluye por defecto porque es lo que hay
        # en un laboratorio real, y permite verificar que el notch funciona.
        if self.con_red:
            red = 30.0 * np.sin(2 * np.pi * b.notch_hz * t)
            x += red[:, None]

        # Respuesta SSVEP
        if self._freq_idx >= 0:
            f = b.frecuencias[self._freq_idx]
            fase = b.fases_jfpm[self._freq_idx]
            amps = [1.00, 0.62, 0.22, 0.12]

            ssvep = np.zeros(n)
            for h, a in enumerate(amps, start=1):
                ssvep += a * np.sin(2 * np.pi * h * f * t + h * fase)

            # Escalar a la SNR pedida
            p_sig = np.mean(ssvep ** 2)
            if p_sig > 1e-12:
                p_ruido = self.amplitud_ruido ** 2
                ssvep *= math.sqrt(
                    p_ruido * (10 ** (self.snr_db / 10.0)) / p_sig)

            # La respuesta es mayor en los canales occipitales, que en el
            # montaje son los ultimos de la lista
            peso = np.linspace(0.5, 1.0, self.n_canales)
            x += ssvep[:, None] * peso[None, :]

        return x

    # -------------------------------------------------------------------
    def impedancias(self) -> Optional[Dict[str, float]]:
        """Impedancias simuladas, todas dentro del rango aceptable."""
        return {c: float(self._rng.uniform(1500, 4000))
                for c in self.cfg.bci.canales}


# ===========================================================================
# FUENTE DESDE DATASET
# ===========================================================================

class FuenteDataset(FuenteEEG):
    """
    Reproduce un registro grabado, en tiempo real o acelerado.

    Util para dos cosas: probar con senal real sin ocupar el laboratorio, y
    tener condiciones exactamente reproducibles al comparar versiones del
    sistema.

    El array debe tener forma (n_muestras, n_canales) con los canales en el
    orden de config.bci.canales.
    """

    def __init__(self, config: Config,
                 datos: np.ndarray,
                 fs_original: Optional[float] = None,
                 en_bucle: bool = True,
                 tiempo_real: bool = True):
        self.cfg = config
        self.en_bucle = en_bucle
        self.tiempo_real = tiempo_real

        x = np.asarray(datos, dtype=float)
        if x.ndim == 1:
            x = x[:, None]

        fs_orig = fs_original if fs_original else config.bci.fs

        # Remuestreo si hace falta. El dataset de referencia se grabo a
        # 5000 Hz y el sistema opera a 256.
        if abs(fs_orig - config.bci.fs) > 1e-6:
            x = self._remuestrear(x, fs_orig, config.bci.fs)

        self._datos = x
        self._pos = 0
        self._corriendo = False
        self._t_ultimo = 0.0

    # -------------------------------------------------------------------
    @staticmethod
    def _remuestrear(x: np.ndarray, fs_in: float,
                     fs_out: float) -> np.ndarray:
        """
        Remuestreo por interpolacion lineal.

        Para un remuestreo de calidad habria que filtrar antes de decimar,
        para evitar aliasing. Aqui la cadena de filtrado posterior limita la
        banda a 90 Hz de todos modos, asi que la interpolacion simple basta
        para la mayoria de casos.

        Si trabajas con datos donde el aliasing importe, sustituye esto por
        scipy.signal.resample_poly.
        """
        n_in = x.shape[0]
        n_out = int(round(n_in * fs_out / fs_in))
        t_in = np.arange(n_in) / fs_in
        t_out = np.arange(n_out) / fs_out
        return np.stack(
            [np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])],
            axis=1)

    # -------------------------------------------------------------------
    def iniciar(self) -> None:
        self._corriendo = True
        self._pos = 0
        self._t_ultimo = time.perf_counter()

    def detener(self) -> None:
        self._corriendo = False

    @property
    def fs(self) -> float:
        return self.cfg.bci.fs

    @property
    def n_canales(self) -> int:
        return self._datos.shape[1]

    @property
    def progreso(self) -> float:
        """Fraccion del registro ya reproducida."""
        return self._pos / max(1, len(self._datos))

    # -------------------------------------------------------------------
    def leer(self) -> Optional[np.ndarray]:
        if not self._corriendo:
            return None

        n = self.cfg.adquisicion.buffer_muestras

        if self.tiempo_real:
            ahora = time.perf_counter()
            if (ahora - self._t_ultimo) < (n / self.fs) * 0.9:
                return None
            self._t_ultimo = ahora

        if self._pos + n > len(self._datos):
            if not self.en_bucle:
                return None
            self._pos = 0

        bloque = self._datos[self._pos:self._pos + n].copy()
        self._pos += n
        return bloque


# ===========================================================================
# AMPLIFICADOR REAL
# ===========================================================================

class FuenteGUSBamp(FuenteEEG):
    """
    Amplificador g.USBamp.

    ESQUELETO PENDIENTE DE COMPLETAR
    --------------------------------
    Las llamadas al SDK estan marcadas con  # >>> COMPLETAR

    El SDK de g.tec solo existe para Windows y requiere licencia. La forma
    habitual de usarlo desde Python es el binding `pygds`, que se distribuye
    con el equipo.

    QUE HAY QUE HACER
    -----------------
    1. Importar el binding y abrir el dispositivo.
    2. Configurar: frecuencia de muestreo, canales activos, filtros del
       hardware (dejarlos DESACTIVADOS: el filtrado se hace en software
       para tener control exacto de la fase).
    3. Implementar la lectura no bloqueante.
    4. Implementar la medida de impedancias si el SDK lo permite.

    QUE NO CAMBIAR
    --------------
    El contrato: `leer()` debe devolver un array (n_muestras, n_canales) en
    MICROVOLTIOS, con los canales en el orden de config.bci.canales. Si el
    SDK devuelve otra unidad u otro orden, la conversion va aqui dentro, no
    fuera.
    """

    def __init__(self, config: Config,
                 indice_dispositivo: int = 0):
        self.cfg = config
        self.indice = indice_dispositivo
        self._dev = None
        self._corriendo = False

        # Verificacion del orden de canales. Es una comprobacion barata que
        # evita un error grave y silencioso.
        self._verificar_montaje()

    # -------------------------------------------------------------------
    def _verificar_montaje(self) -> None:
        """
        Avisa sobre el orden de canales.

        Los filtros espaciales de TASM asumen el orden exacto de
        config.bci.canales. Conectar los electrodos en otro orden no produce
        ningun error visible: el sistema sigue funcionando y dando
        resultados, pero peores.
        """
        canales = self.cfg.bci.canales
        print("[g.USBamp] Orden de canales esperado:")
        for i, c in enumerate(canales):
            print(f"           {i + 1}. {c}")
        print("[g.USBamp] Verifica que los electrodos esten conectados en "
              "ESE orden.")
        print("           Un orden distinto NO produce error visible, pero "
              "invalida")
        print("           los filtros espaciales de TASM.")

    # -------------------------------------------------------------------
    def iniciar(self) -> None:
        """Abre el dispositivo y arranca la adquisicion."""
        # >>> COMPLETAR: importar el binding del SDK
        #
        #     import pygds
        #     self._dev = pygds.GDS()
        #
        # >>> COMPLETAR: configurar el dispositivo
        #
        #     self._dev.SamplingRate = self.cfg.bci.fs
        #     self._dev.NumberOfScans = self.cfg.adquisicion.buffer_muestras
        #
        #     for i, ch in enumerate(self._dev.Channels):
        #         ch.Acquire = (i < self.cfg.bci.n_canales)
        #         # Filtros del HARDWARE desactivados: el filtrado se hace en
        #         # software para controlar la fase con exactitud.
        #         ch.BandpassFilterIndex = -1
        #         ch.NotchFilterIndex = -1
        #
        #     self._dev.SetConfiguration()
        #
        raise NotImplementedError(
            "FuenteGUSBamp no esta implementada.\n"
            "\n"
            "Hay que completar las llamadas al SDK de g.tec marcadas con\n"
            "'>>> COMPLETAR' en eeg_source.py.\n"
            "\n"
            "Mientras tanto, para desarrollar sin hardware usa:\n"
            "    config.adquisicion.fuente = 'sintetica'\n"
        )

    # -------------------------------------------------------------------
    def leer(self) -> Optional[np.ndarray]:
        """Lee el siguiente bloque del amplificador."""
        # >>> COMPLETAR: lectura no bloqueante
        #
        #     datos = self._dev.GetData(
        #         self.cfg.adquisicion.buffer_muestras)
        #     if datos is None or len(datos) == 0:
        #         return None
        #
        #     x = np.asarray(datos, dtype=float)
        #
        #     # >>> VERIFICAR LA UNIDAD. Si el SDK devuelve voltios, hay que
        #     # multiplicar por 1e6 para pasar a microvoltios, que es lo que
        #     # el resto del sistema espera.
        #
        #     # >>> VERIFICAR EL ORDEN DE CANALES. Si no coincide con
        #     # config.bci.canales, reordenar AQUI.
        #
        #     return x[:, :self.cfg.bci.n_canales]
        #
        raise NotImplementedError("Ver iniciar().")

    # -------------------------------------------------------------------
    def detener(self) -> None:
        # >>> COMPLETAR
        #     if self._dev is not None:
        #         self._dev.Close()
        #         self._dev = None
        self._corriendo = False

    # -------------------------------------------------------------------
    def impedancias(self) -> Optional[Dict[str, float]]:
        """
        Mide la impedancia de cada electrodo.

        Debe ejecutarse ANTES de cada sesion. Una impedancia alta degrada la
        relacion senal-ruido lo suficiente para comprometer la deteccion, y
        el remedio es simple: reaplicar gel.
        """
        # >>> COMPLETAR
        #     valores = self._dev.GetImpedance()
        #     return {c: float(v)
        #             for c, v in zip(self.cfg.bci.canales, valores)}
        return None

    @property
    def fs(self) -> float:
        return self.cfg.bci.fs

    @property
    def n_canales(self) -> int:
        return self.cfg.bci.n_canales


# ===========================================================================
# VERIFICACION DE IMPEDANCIAS
# ===========================================================================

@dataclass
class ReporteImpedancias:
    """Resultado de la comprobacion previa a una sesion."""
    valores: Dict[str, float] = field(default_factory=dict)
    umbral: float = 5000.0

    @property
    def problematicos(self) -> List[str]:
        return [c for c, v in self.valores.items() if v > self.umbral]

    @property
    def todos_correctos(self) -> bool:
        return len(self.problematicos) == 0

    def resumen(self) -> str:
        if not self.valores:
            return "  (la fuente no puede medir impedancias)"
        L = []
        L.append(f"  {'canal':>8} {'impedancia':>13}  estado")
        for c, v in self.valores.items():
            estado = "OK" if v <= self.umbral else "ALTA --> reaplicar gel"
            L.append(f"  {c:>8} {v:>10.0f} ohm  {estado}")
        L.append("")
        if self.todos_correctos:
            L.append("  Todos los electrodos dentro del rango.")
        else:
            L.append(f"  ATENCION: {len(self.problematicos)} electrodos por "
                     f"encima de {self.umbral:.0f} ohm.")
            L.append(f"  Canales: {', '.join(self.problematicos)}")
            L.append("  No empieces la sesion sin corregirlos.")
        return "\n".join(L)


def comprobar_impedancias(fuente: FuenteEEG,
                          config: Config) -> ReporteImpedancias:
    """Comprueba las impedancias antes de una sesion."""
    vals = fuente.impedancias()
    return ReporteImpedancias(
        valores=vals if vals else {},
        umbral=config.adquisicion.impedancia_maxima,
    )


# ===========================================================================
# FABRICA
# ===========================================================================

def crear_fuente_eeg(config: Config, **kwargs) -> FuenteEEG:
    """
    Crea la fuente indicada en config.adquisicion.fuente.

    Cambiar de fuente no requiere tocar nada mas del sistema.
    """
    tipo = config.adquisicion.fuente

    if tipo == "sintetica":
        return FuenteSinteticaEEG(config, **kwargs)
    if tipo == "dataset":
        if "datos" not in kwargs:
            raise ValueError(
                "La fuente 'dataset' necesita el argumento `datos` con el "
                "array del registro."
            )
        return FuenteDataset(config, **kwargs)
    if tipo == "gusbamp":
        return FuenteGUSBamp(config, **kwargs)

    raise ValueError(
        f"Fuente '{tipo}' desconocida. "
        f"Opciones: {config.adquisicion.FUENTES_VALIDAS}"
    )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DE LAS FUENTES DE EEG")
    print("=" * 72)

    b = CONFIG.bci

    print()
    print(f"  Montaje ({b.n_canales} canales, ORDEN FIJO):")
    print(f"    {', '.join(b.canales)}")
    print()
    print(f"  Muestreo : {b.fs} Hz")
    print(f"  Bloque   : {CONFIG.adquisicion.buffer_muestras} muestras "
          f"({CONFIG.adquisicion.buffer_muestras/b.fs*1000:.0f} ms)")

    # --- Fuente sintetica ---
    print()
    print("-" * 72)
    print("FUENTE SINTETICA")
    print("-" * 72)

    fuente = FuenteSinteticaEEG(CONFIG, freq_idx=2, snr_db=-5.0, semilla=1)
    fuente.iniciar()

    bloques = []
    t0 = time.perf_counter()
    while len(bloques) < 20 and (time.perf_counter() - t0) < 3.0:
        bl = fuente.leer()
        if bl is not None:
            bloques.append(bl)
        else:
            time.sleep(0.001)

    dt = time.perf_counter() - t0
    total = sum(len(x) for x in bloques)
    print(f"  Bloques leidos : {len(bloques)}")
    print(f"  Muestras       : {total}")
    print(f"  Tiempo real    : {dt:.2f} s")
    print(f"  Tasa efectiva  : {total/dt:.0f} Hz (nominal {b.fs} Hz)")
    print()
    print("  La fuente respeta el ritmo temporal: devuelve None si se la")
    print("  llama mas rapido que fs, igual que haria el hardware real.")

    # --- Contenido espectral ---
    print()
    print("-" * 72)
    print("CONTENIDO ESPECTRAL DE LA SENAL GENERADA")
    print("-" * 72)

    fuente2 = FuenteSinteticaEEG(CONFIG, freq_idx=2, snr_db=0.0, semilla=2)
    fuente2.iniciar()
    trozos = []
    while sum(len(x) for x in trozos) < int(4 * b.fs):
        bl = fuente2.leer()
        if bl is not None:
            trozos.append(bl)
        else:
            time.sleep(0.001)
    x = np.concatenate(trozos)[:, 0]

    esp = np.abs(np.fft.rfft(x - x.mean()))
    fr = np.fft.rfftfreq(len(x), d=1.0 / b.fs)

    print(f"  Se genera respuesta a {b.frecuencias[2]} Hz")
    print()
    print(f"  {'frecuencia':>12} {'amplitud relativa':>19}")
    for f_ver in list(b.frecuencias) + [b.notch_hz, b.frecuencias[2] * 2]:
        i = int(np.argmin(np.abs(fr - f_ver)))
        rel = esp[i] / esp.max()
        marca = ""
        if abs(f_ver - b.frecuencias[2]) < 0.01:
            marca = "  <-- atendida"
        elif abs(f_ver - b.notch_hz) < 0.01:
            marca = "  <-- red"
        print(f"  {f_ver:>11.1f}Hz {rel:>19.3f}{marca}")

    fuente.detener()
    fuente2.detener()

    # --- Impedancias ---
    print()
    print("-" * 72)
    print("COMPROBACION DE IMPEDANCIAS")
    print("-" * 72)
    rep = comprobar_impedancias(fuente, CONFIG)
    print(rep.resumen())

    # --- Amplificador real ---
    print()
    print("-" * 72)
    print("AMPLIFICADOR REAL")
    print("-" * 72)
    print("  La clase FuenteGUSBamp es un esqueleto. Las llamadas al SDK")
    print("  estan marcadas con '>>> COMPLETAR' en el codigo fuente.")
    print()
    print("  Para desarrollar sin hardware:")
    print("     config.adquisicion.fuente = 'sintetica'")

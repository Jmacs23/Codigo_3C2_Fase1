"""
acquisition.py --- Pipeline completo de adquisicion
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
Encadena todo lo anterior:

    fuente -> filtrado causal -> buffer circular -> ventaneo ->
    control de artefactos -> FBCCA -> TASM -> mensaje TCP

Corre en la maquina Windows y envia el resultado a la maquina Linux, donde
ROS2 lo consume.


EL BUFFER CIRCULAR
------------------
Hay un desajuste que resolver: el amplificador entrega bloques de 12
muestras (47 ms), pero la ventana de analisis es de 750 ms (192 muestras) y
avanza cada 50 ms (13 muestras).

Es decir, cada bloque nuevo NO corresponde a una ventana: hay que acumular
varios bloques, y cada ventana solapa un 93% con la anterior.

El buffer circular resuelve esto. Se van escribiendo los bloques que
llegan, y cuando hay muestras suficientes se extrae una ventana. La ventana
siguiente empieza 13 muestras despues, no 192: por eso el solapamiento.

Se usa un buffer circular y no una lista que crece porque el sistema corre
durante minutos: una lista acumularia memoria indefinidamente.


DONDE ENCAJA TASM
-----------------
Este bloque NO implementa TASM. TASM es la Linea 1.

Lo que hace es dejar el hueco con un contrato claro: recibe una ventana ya
preprocesada de forma (n_muestras, n_canales) y devuelve un MensajeTASM.
Mientras TASM real no exista, se usa el mock del Bloque 1.

Escribir aqui un decodificador propio garantizaria que las dos versiones
divergieran, y que la del sistema robotico fuera peor.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple
import json
import socket
import time
import numpy as np

from config import Config
from filters import CadenaFiltrado, DetectorArtefactos, ResultadoArtefactos
from eeg_source import FuenteEEG, crear_fuente_eeg


# ===========================================================================
# BUFFER CIRCULAR
# ===========================================================================

class BufferCircular:
    """
    Acumula muestras y entrega ventanas solapadas.

    Uso:
        buf = BufferCircular(n_muestras=192, n_canales=9, paso=13)
        buf.escribir(bloque)
        while buf.hay_ventana():
            v = buf.extraer_ventana()
    """

    def __init__(self, n_muestras: int, n_canales: int, paso: int):
        self.n = n_muestras
        self.n_canales = n_canales
        self.paso = paso

        # Se reserva el doble de la ventana para tener holgura entre
        # extracciones sin tener que compactar el buffer en cada escritura.
        self._cap = max(n_muestras * 2, n_muestras + paso * 4)
        self._datos = np.zeros((self._cap, n_canales))
        self._escritas = 0        # total historico
        self._pos = 0             # posicion de escritura
        self._proxima = n_muestras  # muestra en que toca la proxima ventana

    # -------------------------------------------------------------------
    def escribir(self, bloque: np.ndarray) -> None:
        """Anade un bloque al buffer."""
        x = np.asarray(bloque, dtype=float)
        if x.ndim == 1:
            x = x[:, None]

        if x.shape[1] != self.n_canales:
            raise ValueError(
                f"El bloque tiene {x.shape[1]} canales, "
                f"el buffer espera {self.n_canales}."
            )

        n = x.shape[0]
        if n > self._cap:
            # Bloque mayor que el buffer: se queda con las ultimas muestras
            x = x[-self._cap:]
            n = self._cap

        fin = self._pos + n
        if fin <= self._cap:
            self._datos[self._pos:fin] = x
        else:
            # El bloque cruza el final: se parte en dos
            corte = self._cap - self._pos
            self._datos[self._pos:] = x[:corte]
            self._datos[:n - corte] = x[corte:]

        self._pos = fin % self._cap
        self._escritas += n

    # -------------------------------------------------------------------
    def hay_ventana(self) -> bool:
        """True si hay muestras suficientes para extraer una ventana."""
        return self._escritas >= self._proxima

    # -------------------------------------------------------------------
    def extraer_ventana(self) -> Optional[np.ndarray]:
        """
        Extrae la siguiente ventana y avanza la posicion un paso.

        Las ventanas consecutivas se solapan: la siguiente empieza `paso`
        muestras despues, no `n_muestras`.
        """
        if not self.hay_ventana():
            return None

        # Indice de la ultima muestra de esta ventana, en el buffer circular
        fin = (self._pos - (self._escritas - self._proxima)) % self._cap
        ini = (fin - self.n) % self._cap

        if ini < fin:
            v = self._datos[ini:fin].copy()
        else:
            v = np.concatenate([self._datos[ini:], self._datos[:fin]])

        self._proxima += self.paso
        return v

    # -------------------------------------------------------------------
    @property
    def muestras_escritas(self) -> int:
        return self._escritas

    def reiniciar(self) -> None:
        self._datos[:] = 0.0
        self._escritas = 0
        self._pos = 0
        self._proxima = self.n


# ===========================================================================
# MENSAJE
# ===========================================================================

@dataclass
class MensajeSalida:
    """
    Lo que se envia a la maquina Linux.

    Los campos coinciden con el contrato TASMState acordado con la Linea 1,
    de modo que el nodo ROS2 solo tenga que traducir el JSON a un mensaje
    ROS sin reinterpretar nada.
    """
    estado: str              # "IC", "TR" o "Idle"
    freq_idx: int            # -1 si el estado no es IC
    p_max: float
    lambda_bci: float
    valido: bool
    timestamp: float
    rho: Optional[List[float]] = None
    rho_grad: Optional[List[float]] = None
    secuencia: int = 0

    def a_json(self) -> str:
        d = {
            "estado": self.estado,
            "freq_idx": self.freq_idx,
            "p_max": round(self.p_max, 4),
            "lambda_bci": round(self.lambda_bci, 4),
            "valido": self.valido,
            "timestamp": round(self.timestamp, 6),
            "secuencia": self.secuencia,
        }
        if self.rho is not None:
            d["rho"] = [round(x, 5) for x in self.rho]
        if self.rho_grad is not None:
            d["rho_grad"] = [round(x, 5) for x in self.rho_grad]
        return json.dumps(d)


# ===========================================================================
# PUENTE TCP
# ===========================================================================

class PuenteTCP:
    """
    Envia los mensajes a la maquina Linux.

    PROTOCOLO
    ---------
    Un mensaje JSON por linea, terminado en salto de linea. Es simple, legible
    y suficiente: los mensajes son pequenos (unos 200 bytes) y van a 20 Hz,
    lo que da 4 kB/s.

    El salto de linea como delimitador funciona porque JSON nunca contiene
    saltos sin escapar.

    COMPORTAMIENTO ANTE FALLOS
    --------------------------
    Si la conexion se cae, `enviar` devuelve False en lugar de lanzar
    excepcion. El bucle de adquisicion sigue corriendo y reintenta conectar
    periodicamente.

    Es la decision correcta: perder la conexion con el robot no debe detener
    la adquisicion, porque los datos siguen siendo validos y la conexion
    puede volver. Del lado del robot, el watchdog se encarga de detenerlo.
    """

    def __init__(self, config: Config):
        self.cfg = config
        a = config.adquisicion
        self.host = a.tcp_host
        self.puerto = a.tcp_puerto
        self.timeout = a.tcp_timeout

        self._sock: Optional[socket.socket] = None
        self._conectado = False
        self._t_ultimo_intento = 0.0
        self._enviados = 0
        self._fallidos = 0

    # -------------------------------------------------------------------
    def conectar(self) -> bool:
        """Intenta conectar. Devuelve True si lo consigue."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.host, self.puerto))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            self._conectado = True
            return True
        except OSError:
            self._sock = None
            self._conectado = False
            return False

    # -------------------------------------------------------------------
    def enviar(self, msg: MensajeSalida) -> bool:
        """
        Envia un mensaje. Devuelve False si no se pudo.

        Reintenta conectar como maximo una vez por segundo, para no bloquear
        el bucle con intentos fallidos.
        """
        if not self._conectado:
            ahora = time.perf_counter()
            if ahora - self._t_ultimo_intento < 1.0:
                self._fallidos += 1
                return False
            self._t_ultimo_intento = ahora
            if not self.conectar():
                self._fallidos += 1
                return False

        try:
            self._sock.sendall((msg.a_json() + "\n").encode("utf-8"))
            self._enviados += 1
            return True
        except OSError:
            self._conectado = False
            self._sock = None
            self._fallidos += 1
            return False

    # -------------------------------------------------------------------
    def cerrar(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._conectado = False

    @property
    def estadisticas(self) -> dict:
        total = self._enviados + self._fallidos
        return {
            "enviados": self._enviados,
            "fallidos": self._fallidos,
            "tasa_exito": self._enviados / total if total else 0.0,
            "conectado": self._conectado,
        }


# ===========================================================================
# PIPELINE
# ===========================================================================

@dataclass
class EstadisticasPipeline:
    """Contadores de una sesion de adquisicion."""
    bloques_leidos: int = 0
    ventanas_procesadas: int = 0
    ventanas_invalidas: int = 0
    mensajes_enviados: int = 0
    tiempo_total: float = 0.0
    tiempo_computo: List[float] = field(default_factory=list)

    @property
    def tasa_ventanas(self) -> float:
        return (self.ventanas_procesadas / self.tiempo_total
                if self.tiempo_total > 0 else 0.0)

    @property
    def fraccion_invalidas(self) -> float:
        return (self.ventanas_invalidas / self.ventanas_procesadas
                if self.ventanas_procesadas else 0.0)

    @property
    def computo_medio_ms(self) -> float:
        return (float(np.mean(self.tiempo_computo)) * 1000
                if self.tiempo_computo else 0.0)

    @property
    def computo_p95_ms(self) -> float:
        """Percentil 95 del tiempo de computo.

        Mas informativo que la media: lo que importa es si los casos
        desfavorables caben en el periodo, no el caso medio."""
        return (float(np.percentile(self.tiempo_computo, 95)) * 1000
                if self.tiempo_computo else 0.0)

    def resumen(self, paso_ms: float) -> str:
        L = []
        L.append(f"  Bloques leidos      : {self.bloques_leidos}")
        L.append(f"  Ventanas procesadas : {self.ventanas_procesadas}")
        L.append(f"  Ventanas invalidas  : {self.ventanas_invalidas} "
                 f"({self.fraccion_invalidas*100:.1f}%)")
        L.append(f"  Mensajes enviados   : {self.mensajes_enviados}")
        L.append(f"  Duracion            : {self.tiempo_total:.2f} s")
        L.append(f"  Tasa de ventanas    : {self.tasa_ventanas:.1f} Hz "
                 f"(nominal {1000/paso_ms:.1f} Hz)")
        L.append("")
        L.append(f"  Computo medio       : {self.computo_medio_ms:.2f} ms")
        L.append(f"  Computo p95         : {self.computo_p95_ms:.2f} ms")
        L.append(f"  Periodo disponible  : {paso_ms:.1f} ms")
        margen = paso_ms - self.computo_p95_ms
        L.append(f"  Margen (p95)        : {margen:+.2f} ms "
                 f"{'OK' if margen > 0 else '<-- NO CABE'}")
        return "\n".join(L)


class PipelineAdquisicion:
    """
    Pipeline completo de adquisicion.

    Uso:
        pipe = PipelineAdquisicion(CONFIG, fuente, procesador)
        pipe.correr(duracion=60.0)
        print(pipe.stats.resumen(CONFIG.bci.paso_tasm * 1000))

    El `procesador` es la funcion que convierte una ventana en un
    MensajeSalida. Es donde encaja TASM.
    """

    def __init__(self, config: Config,
                 fuente: FuenteEEG,
                 procesador: Callable[[np.ndarray, bool], MensajeSalida],
                 puente: Optional[PuenteTCP] = None):
        self.cfg = config
        self.fuente = fuente
        self.procesador = procesador
        self.puente = puente

        b = config.bci
        self.cadena = CadenaFiltrado(config)
        self.detector = DetectorArtefactos(config)
        self.buffer = BufferCircular(
            n_muestras=b.muestras_ventana_tasm,
            n_canales=b.n_canales,
            paso=b.muestras_paso_tasm,
        )

        self.stats = EstadisticasPipeline()
        self._secuencia = 0
        self._parar = False

    # -------------------------------------------------------------------
    def detener(self) -> None:
        """Solicita la parada del bucle."""
        self._parar = True

    # -------------------------------------------------------------------
    def paso(self) -> Optional[MensajeSalida]:
        """
        Ejecuta un ciclo: lee, filtra, y procesa las ventanas disponibles.

        Devuelve el ultimo mensaje generado, o None si no hubo ventana lista.
        """
        bloque = self.fuente.leer()
        if bloque is None:
            return None

        self.stats.bloques_leidos += 1

        # Filtrado causal con estado. El estado persiste entre llamadas: por
        # eso la cadena es un atributo y no se crea aqui.
        filtrado = self.cadena.aplicar(bloque)
        self.buffer.escribir(filtrado)

        ultimo = None
        while self.buffer.hay_ventana():
            ventana = self.buffer.extraer_ventana()
            if ventana is None:
                break

            t0 = time.perf_counter()

            art = self.detector.evaluar(ventana)
            msg = self.procesador(ventana, art.valida)
            msg.secuencia = self._secuencia
            msg.timestamp = time.perf_counter()
            self._secuencia += 1

            self.stats.tiempo_computo.append(time.perf_counter() - t0)
            self.stats.ventanas_procesadas += 1
            if not art.valida:
                self.stats.ventanas_invalidas += 1

            if self.puente is not None and self.puente.enviar(msg):
                self.stats.mensajes_enviados += 1

            ultimo = msg

        return ultimo

    # -------------------------------------------------------------------
    def correr(self, duracion: float,
               callback: Optional[Callable[[MensajeSalida], None]] = None
               ) -> None:
        """
        Bucle principal durante `duracion` segundos.

        `callback` se llama con cada mensaje generado, si se da.
        """
        self._parar = False
        self.fuente.iniciar()
        t0 = time.perf_counter()

        try:
            while not self._parar:
                t = time.perf_counter() - t0
                if t >= duracion:
                    break

                msg = self.paso()
                if msg is not None and callback is not None:
                    callback(msg)

                if msg is None:
                    # Sin datos nuevos: ceder CPU brevemente en lugar de
                    # girar en vacio consumiendo un nucleo entero.
                    time.sleep(0.001)
        finally:
            self.stats.tiempo_total = time.perf_counter() - t0
            self.fuente.detener()

    # -------------------------------------------------------------------
    def reiniciar(self) -> None:
        """Reinicia el pipeline. Llamar entre trials."""
        self.cadena.reiniciar()
        self.buffer.reiniciar()
        self.stats = EstadisticasPipeline()
        self._secuencia = 0


# ===========================================================================
# PROCESADOR CON MOCK
# ===========================================================================

def crear_procesador_mock(config: Config, objetivo: int = 2,
                          semilla: int = 0):
    """
    Procesador que usa el mock de TASM del Bloque 1.

    Sirve para probar el pipeline completo sin TASM real. Ignora el contenido
    de la ventana y genera estados segun el guion del mock.

    IMPORTANTE: esto es andamiaje de desarrollo. En los experimentos del
    trabajo hay que usar TASM real sobre la senal real.
    """
    try:
        from tasm_mock import TASMMock
        from command_fsm import EstadoTASM
    except ImportError:
        raise ImportError(
            "El procesador mock necesita tasm_mock.py y command_fsm.py del "
            "Bloque 1. Copialos a este directorio o instala el paquete."
        )

    mock = TASMMock(config, objetivo=objetivo, semilla=semilla)

    def procesar(ventana: np.ndarray, valida: bool) -> MensajeSalida:
        m = mock.siguiente()
        return MensajeSalida(
            estado=m.estado.value,
            freq_idx=m.freq_idx,
            p_max=m.p_max,
            lambda_bci=m.lambda_bci,
            valido=valida and m.valido,
            timestamp=0.0,
        )

    return procesar


def crear_procesador_fbcca(config: Config):
    """
    Procesador que usa FBCCA real sobre la ventana, sin TASM.

    Clasifica la frecuencia pero no distingue estados cognitivos: siempre
    reporta IC. Sirve para verificar que la cadena de adquisicion entrega
    senal utilizable, antes de tener TASM.
    """
    try:
        from fbcca import FBCCA
    except ImportError:
        raise ImportError(
            "Este procesador necesita fbcca.py del Bloque 1."
        )

    clf = FBCCA(config)
    rho_prev = None

    def procesar(ventana: np.ndarray, valida: bool) -> MensajeSalida:
        nonlocal rho_prev
        r = clf.clasificar(ventana)
        grad = (clf.gradiente(r.rho, rho_prev) if rho_prev is not None
                else np.zeros_like(r.rho))
        rho_prev = r.rho.copy()

        return MensajeSalida(
            estado="IC",
            freq_idx=r.freq_idx,
            p_max=r.p_max,
            lambda_bci=r.p_max,
            valido=valida,
            timestamp=0.0,
            rho=r.rho.tolist() if config.adquisicion.enviar_diagnostico
            else None,
            rho_grad=grad.tolist() if config.adquisicion.enviar_diagnostico
            else None,
        )

    return procesar


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG
    from eeg_source import FuenteSinteticaEEG

    print("=" * 72)
    print("DEMOSTRACION DEL PIPELINE DE ADQUISICION")
    print("=" * 72)

    b = CONFIG.bci

    print()
    print("  Cadena: fuente -> filtrado causal -> buffer -> ventaneo ->")
    print("          artefactos -> clasificacion -> TCP")
    print()
    print(f"  Bloque de lectura : "
          f"{CONFIG.adquisicion.buffer_muestras} muestras "
          f"({CONFIG.adquisicion.buffer_muestras/b.fs*1000:.0f} ms)")
    print(f"  Ventana           : {b.muestras_ventana_tasm} muestras "
          f"({b.tw_tasm*1000:.0f} ms)")
    print(f"  Paso              : {b.muestras_paso_tasm} muestras "
          f"({b.paso_tasm*1000:.0f} ms)")
    solape = (1 - b.muestras_paso_tasm / b.muestras_ventana_tasm) * 100
    print(f"  Solapamiento      : {solape:.0f}%")

    # --- Buffer circular ---
    print()
    print("-" * 72)
    print("BUFFER CIRCULAR")
    print("-" * 72)
    buf = BufferCircular(b.muestras_ventana_tasm, b.n_canales,
                         b.muestras_paso_tasm)
    nb = CONFIG.adquisicion.buffer_muestras

    print(f"  {'bloque':>8} {'muestras':>10} {'ventanas listas':>17}")
    ventanas = 0
    for i in range(1, 41):
        buf.escribir(np.random.randn(nb, b.n_canales))
        n_v = 0
        while buf.hay_ventana():
            buf.extraer_ventana()
            n_v += 1
            ventanas += 1
        if i <= 5 or i % 10 == 0:
            print(f"  {i:>8} {buf.muestras_escritas:>10} {n_v:>17}")
    print(f"  ...")
    print(f"  Total de ventanas extraidas: {ventanas}")
    print()
    print("  Las primeras ventanas tardan en aparecer porque hay que")
    print("  acumular 192 muestras. Despues sale una cada 13 muestras.")

    # --- Pipeline con FBCCA real ---
    print()
    print("-" * 72)
    print("PIPELINE CON FBCCA REAL")
    print("-" * 72)
    print("  Se genera senal con respuesta a 14 Hz y se comprueba que la")
    print("  cadena la recupera despues de filtrar.")
    print()

    fuente = FuenteSinteticaEEG(CONFIG, freq_idx=2, snr_db=-3.0,
                                con_red=True, semilla=5)
    try:
        proc = crear_procesador_fbcca(CONFIG)
    except ImportError as ex:
        print(f"  {ex}")
        raise SystemExit(0)

    pipe = PipelineAdquisicion(CONFIG, fuente, proc)

    detectadas = []
    pipe.correr(duracion=6.0,
                callback=lambda m: detectadas.append(m.freq_idx))

    print(pipe.stats.resumen(b.paso_tasm * 1000))
    print()
    if detectadas:
        from collections import Counter
        cuenta = Counter(detectadas)
        print(f"  Frecuencia inyectada: {b.frecuencias[2]} Hz (indice 2)")
        print()
        print(f"  {'indice':>8} {'frecuencia':>12} {'detecciones':>13} "
              f"{'porcentaje':>12}")
        for idx in range(len(b.frecuencias)):
            n = cuenta.get(idx, 0)
            marca = "  <-- correcta" if idx == 2 else ""
            print(f"  {idx:>8} {b.frecuencias[idx]:>11.1f}Hz {n:>13} "
                  f"{n/len(detectadas)*100:>11.1f}%{marca}")

    # --- Verificacion del presupuesto de latencia ---
    print()
    print("-" * 72)
    print("PRESUPUESTO DE LATENCIA")
    print("-" * 72)
    l = CONFIG.latencia
    print(f"  Computo por ventana (medido) : "
          f"{pipe.stats.computo_p95_ms:.2f} ms (p95)")
    print(f"  Computo presupuestado        : {l.procesamiento_ms:.1f} ms")
    print(f"  Periodo de ventana           : {b.paso_tasm*1000:.1f} ms")
    print()
    print(f"  Extremo a extremo estimado   : "
          f"{l.extremo_a_extremo_ms:.1f} ms")
    print(f"  Watchdog                     : {b.watchdog_s*1000:.1f} ms")
    print(f"  Margen                       : "
          f"{l.margen_watchdog(b.watchdog_s):.1f} ms")
    print()
    print("  Son dos cosas distintas: el COMPUTO debe caber en el periodo")
    print("  (si no, se acumula retraso); la latencia EXTREMO A EXTREMO no")
    print("  tiene que caber, porque la cadena es un pipeline con etapas")
    print("  solapadas. Solo debe quedar por debajo del watchdog.")

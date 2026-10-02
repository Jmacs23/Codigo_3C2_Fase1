"""
tasm_interface.py --- Contrato, fabrica y verificacion de TASM
Bloque 6: integracion_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

El contrato, la fabrica y la verificacion estan verificados con pruebas
automatizadas usando el mock.

El adaptador a TASM real es un ESQUELETO: el paquete de la Linea 1 no estaba
disponible al escribir esto. Las llamadas estan marcadas ">>> COMPLETAR".

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.
===========================================================================

QUE HACE ESTE MODULO
--------------------
Define de donde viene el estado cognitivo y comprueba que llega bien
formado.

LA IDEA CENTRAL: UN SOLO SISTEMA, UN PARAMETRO
-----------------------------------------------
No hay una version "con mock" y otra "con TASM real". Hay un sistema, y un
parametro que decide de donde viene el estado:

    config.tasm.fuente = "mock"      desarrollo
    config.tasm.fuente = "dataset"   TASM real sobre datos grabados
    config.tasm.fuente = "real"      TASM real en linea

La FSM, el arbitraje, los nodos ROS2 y la Etapa 2 NO se enteran de cual esta
corriendo. Todos consumen el mismo contrato.

Esa indiferencia no es casual: es la razon por la que el contrato se definio
en el Bloque 1, antes de que TASM existiera. Si cada bloque hubiera inventado
su propio formato, integrarlos ahora seria un trabajo de semanas.


POR QUE HAY UN PASO INTERMEDIO ("dataset")
-------------------------------------------
Pasar directamente de mock a sujeto real mezclaria dos fuentes de error: si
algo falla, no se sabria si es la integracion o TASM.

Con datos grabados se verifica la integracion en condiciones reproducibles.
Cuando eso funciona, lo unico nuevo al traer un sujeto es el sujeto.


LA VERIFICACION DE CONTRATO NO ES BUROCRACIA
---------------------------------------------
Un desajuste de contrato produce fallos SILENCIOSOS. Ejemplos reales de
cosas que pueden pasar y no dan ningun error:

  - TASM devuelve el estado como entero (0,1,2) y el sistema espera cadenas
  - El indice de frecuencia empieza en 1 en vez de en 0
  - lambda_bci viene en porcentaje (0-100) en vez de en [0,1]
  - El orden de las frecuencias no coincide con config.bci.frecuencias

En todos esos casos el sistema corre, produce numeros plausibles, y ninguno
significa lo que uno cree. Por eso la verificacion se ejecuta ANTES de nada.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import os
import sys
import math
import numpy as np

from config import Config

# IMPORTACION CONSISTENTE --- no cambiar a `from logica.command_fsm import`
#
# El paquete logica anade su propio directorio al path, de modo que sus
# modulos se importan entre si como `from command_fsm import ...`.
#
# Si aqui se importara como `from logica.command_fsm import ...`, Python
# cargaria el MISMO archivo dos veces bajo nombres distintos, creando dos
# clases EstadoTASM que isinstance considera diferentes. El sintoma es un
# mensaje absurdo del tipo "estado es EstadoTASM, deberia ser EstadoTASM".
#
# Importar logica primero garantiza que las rutas esten puestas.
import logica  # noqa: F401
from command_fsm import EstadoTASM


# ===========================================================================
# CONTRATO
# ===========================================================================

@dataclass
class SalidaTASM:
    """
    Lo que TASM debe entregar en cada ventana.

    ES EL CONTRATO. Coincide campo a campo con el mensaje ROS2 TASMState y
    con el JSON que viaja de Windows a Linux.

    Cambiarlo obliga a cambiar los tres sitios, y ademas la Linea 2, que
    consume el mismo modulo.
    """

    estado: EstadoTASM
    """IC, TR o Idle. NUNCA una cadena ni un entero suelto: el tipo enumerado
    impide que un valor invalido pase inadvertido."""

    freq_idx: int
    """Indice en config.bci.frecuencias, base 0. Vale -1 si el estado no es
    IC."""

    p_max: float
    """Confianza sobre CUAL frecuencia, en [0,1]."""

    lambda_bci: float
    """P(IC | observaciones), salida del HMM, en [0,1].

    Es una pregunta DISTINTA de p_max: mide si hay intencion de comandar, no
    cual comando."""

    valido: bool
    """False si la ventana se rechazo por artefactos."""

    # --- Diagnostico, opcional ---
    rho: Optional[np.ndarray] = None
    rho_grad: Optional[np.ndarray] = None

    # --- Solo disponible con mock o con datos etiquetados ---
    estado_real: Optional[EstadoTASM] = None
    """Estado verdadero. Es lo que permite calcular N_FP_TR.

    Con un sujeto en operacion libre NO existe: hay que obtenerlo del
    protocolo de desvios inducidos."""


# ===========================================================================
# INTERFAZ
# ===========================================================================

class FuenteTASM(ABC):
    """
    Interfaz comun de todas las fuentes de estado cognitivo.

    Cualquier implementacion debe poder sustituir a otra sin que el resto
    del sistema note la diferencia.
    """

    @abstractmethod
    def siguiente(self, ventana: Optional[np.ndarray] = None) -> SalidaTASM:
        """
        Devuelve el estado de la siguiente ventana.

        Parametros
        ----------
        ventana : datos EEG (n_muestras, n_canales). El mock lo ignora; TASM
                  real lo necesita.
        """
        ...

    @abstractmethod
    def reiniciar(self) -> None:
        """Reinicia el estado interno. Llamar entre trials."""
        ...

    @property
    @abstractmethod
    def nombre(self) -> str:
        ...

    @property
    def tiene_ground_truth(self) -> bool:
        """
        True si la fuente conoce el estado verdadero.

        Solo el mock y los datos etiquetados lo tienen. Con un sujeto en
        operacion libre no existe, y las metricas que dependen de el
        (N_FP_TR, FPR en TR) no se pueden calcular sin el protocolo de
        desvios inducidos.
        """
        return False


# ===========================================================================
# FUENTE MOCK
# ===========================================================================

class FuenteMock(FuenteTASM):
    """
    Envoltorio del generador sintetico del Bloque 1.

    PROHIBIDO EN LOS EXPERIMENTOS FORMALES. Sirve para desarrollar el
    sistema robotico sin depender de la Linea 1, no para producir
    resultados.
    """

    def __init__(self, config: Config, objetivo: int = 2,
                 modo: Optional[str] = None, semilla: int = 0):
        from tasm_mock import TASMMock
        self.cfg = config
        self._mock = TASMMock(config, objetivo=objetivo, modo=modo,
                              semilla=semilla)

    def siguiente(self, ventana: Optional[np.ndarray] = None) -> SalidaTASM:
        m = self._mock.siguiente()
        return SalidaTASM(
            estado=m.estado,
            freq_idx=m.freq_idx,
            p_max=m.p_max,
            lambda_bci=m.lambda_bci,
            valido=m.valido,
            estado_real=m.estado_real,
        )

    def reiniciar(self) -> None:
        self._mock.reiniciar()

    @property
    def nombre(self) -> str:
        return "mock (sintetico)"

    @property
    def tiene_ground_truth(self) -> bool:
        return True

    @property
    def objetivo(self) -> int:
        return self._mock.objetivo

    @objetivo.setter
    def objetivo(self, v: int) -> None:
        self._mock.objetivo = v


# ===========================================================================
# FUENTE REAL
# ===========================================================================

class FuenteTASMReal(FuenteTASM):
    """
    Adaptador al modulo TASM de la Linea 1.

    ESQUELETO PENDIENTE DE COMPLETAR
    --------------------------------
    Las llamadas al paquete `tasm` estan marcadas ">>> COMPLETAR".

    QUE HAY QUE HACER
      1. Asegurar que el paquete `tasm` esta en el PYTHONPATH, o indicar su
         ruta en config.tasm.ruta_tasm.
      2. Cargar el modelo entrenado (LDA + HMM).
      3. Implementar la inferencia sobre una ventana.

    QUE NO CAMBIAR
      El contrato. siguiente() debe devolver un SalidaTASM con los campos en
      los tipos y rangos declarados. Si TASM devuelve otra cosa, la
      conversion va AQUI DENTRO, no fuera.

    LA CONVERSION ES EL PUNTO CRITICO
      Los desajustes tipicos son: estado como entero en vez de enumerado,
      indice de frecuencia en base 1, lambda_bci en porcentaje, orden de
      frecuencias distinto.

      Ninguno da error. Todos producen un sistema que corre y miente. Por eso
      existe VerificadorContrato: ejecutalo ANTES de correr nada.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._modelo = None
        self._procesador = None

        # Anadir la ruta del paquete si se indico
        ruta = config.tasm.ruta_tasm or os.environ.get("TASM_PATH", "")
        if ruta and os.path.isdir(ruta) and ruta not in sys.path:
            sys.path.insert(0, ruta)

        self._cargar()

    # -------------------------------------------------------------------
    def _cargar(self) -> None:
        """Carga el modulo y el modelo entrenado."""
        # >>> COMPLETAR: importar el paquete de la Linea 1
        #
        #     from tasm import Config as TASMCfg, TASMProcessor
        #
        #     cfg_tasm = TASMCfg()
        #     cfg_tasm.feat.win_ms = self.cfg.bci.tw_tasm * 1000
        #     cfg_tasm.feat.step_ms = self.cfg.bci.paso_tasm * 1000
        #
        #     self._procesador = TASMProcessor(cfg_tasm)
        #
        #     if self.cfg.tasm.modelo_entrenado:
        #         self._procesador.cargar_modelo(
        #             self.cfg.tasm.modelo_entrenado)
        #
        raise NotImplementedError(
            "FuenteTASMReal no esta implementada.\n"
            "\n"
            "Hay que completar las llamadas al paquete `tasm` de la Linea 1,\n"
            "marcadas '>>> COMPLETAR' en tasm_interface.py.\n"
            "\n"
            "Mientras tanto, para desarrollar:\n"
            "    config.tasm.fuente = 'mock'\n"
        )

    # -------------------------------------------------------------------
    def siguiente(self, ventana: Optional[np.ndarray] = None) -> SalidaTASM:
        """Infiere el estado de una ventana de EEG."""
        if ventana is None:
            raise ValueError(
                "TASM real necesita la ventana de EEG. Que llegue None "
                "indica que la cadena de adquisicion no la esta pasando.")

        # >>> COMPLETAR: inferencia
        #
        #     salida = self._procesador.procesar(ventana)
        #
        #     # AQUI VA LA CONVERSION AL CONTRATO. Verificar cada campo:
        #     #
        #     #   estado     : si TASM devuelve 0/1/2, mapear al enumerado
        #     #   freq_idx   : si es base 1, restar 1
        #     #   lambda_bci : si viene en 0-100, dividir por 100
        #     #   orden      : comprobar que coincide con
        #     #                config.bci.frecuencias
        #
        #     return SalidaTASM(
        #         estado=EstadoTASM(salida.estado),
        #         freq_idx=int(salida.freq_idx),
        #         p_max=float(salida.p_max),
        #         lambda_bci=float(salida.lambda_bci),
        #         valido=bool(salida.valido),
        #         rho=salida.rho,
        #         rho_grad=salida.rho_grad,
        #     )
        #
        raise NotImplementedError("Ver _cargar().")

    def reiniciar(self) -> None:
        # >>> COMPLETAR
        #     if self._procesador is not None:
        #         self._procesador.reset()
        pass

    @property
    def nombre(self) -> str:
        return f"TASM real ({self.cfg.tasm.fuente})"


# ===========================================================================
# FABRICA
# ===========================================================================

def crear_fuente_tasm(config: Config, **kwargs) -> FuenteTASM:
    """
    Crea la fuente indicada en config.tasm.fuente.

    ES EL UNICO PUNTO donde el sistema decide de donde viene el estado
    cognitivo. Cambiar de mock a real es cambiar un parametro; nada mas del
    sistema se entera.
    """
    f = config.tasm.fuente

    if f == "mock":
        return FuenteMock(config, **kwargs)
    if f in ("dataset", "real"):
        return FuenteTASMReal(config)

    raise ValueError(
        f"tasm.fuente='{f}' desconocida. "
        f"Opciones: {config.tasm.FUENTES_VALIDAS}")


# ===========================================================================
# VERIFICACION DEL CONTRATO
# ===========================================================================

@dataclass
class ResultadoVerificacion:
    """Resultado de comprobar que una fuente respeta el contrato."""
    correcto: bool
    n_muestras: int
    problemas: List[str] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)
    estadisticas: Dict[str, Any] = field(default_factory=dict)

    def resumen(self) -> str:
        L = []
        L.append(f"  Muestras verificadas : {self.n_muestras}")
        L.append("")
        if self.problemas:
            L.append("  PROBLEMAS (el sistema NO puede usar esta fuente):")
            for p in self.problemas:
                L.append(f"    - {p}")
            L.append("")
        if self.avisos:
            L.append("  Avisos:")
            for a in self.avisos:
                L.append(f"    - {a}")
            L.append("")
        if self.estadisticas:
            L.append("  Distribucion observada:")
            for k, v in self.estadisticas.items():
                if isinstance(v, float):
                    L.append(f"    {k:<22}: {v:.4f}")
                else:
                    L.append(f"    {k:<22}: {v}")
        L.append("")
        L.append(f"  VEREDICTO: "
                 f"{'CONTRATO CORRECTO' if self.correcto else 'CONTRATO VIOLADO'}")
        return "\n".join(L)


class VerificadorContrato:
    """
    Comprueba que una fuente de TASM entrega lo que el sistema espera.

    EJECUTAR ANTES DE CORRER NADA con TASM real.

    Los desajustes de contrato no dan error: producen un sistema que corre y
    da numeros plausibles que no significan lo que uno cree. Esta clase los
    detecta en segundos.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def verificar(self, fuente: FuenteTASM,
                  n_muestras: int = 500,
                  generar_ventana=None) -> ResultadoVerificacion:
        """
        Ejercita la fuente y comprueba cada campo.

        Parametros
        ----------
        generar_ventana : funcion que produce una ventana de EEG. Necesaria
                          para TASM real; el mock la ignora.
        """
        problemas: List[str] = []
        avisos: List[str] = []

        n_estados = {"IC": 0, "TR": 0, "Idle": 0}
        freqs_vistas = set()
        p_max_min, p_max_max = 1.0, 0.0
        lam_min, lam_max = 1.0, 0.0
        n_invalidas = 0
        n_ok = 0

        for i in range(n_muestras):
            ventana = generar_ventana() if generar_ventana else None

            try:
                s = fuente.siguiente(ventana)
            except NotImplementedError:
                problemas.append(
                    "La fuente no esta implementada (esqueleto pendiente).")
                return ResultadoVerificacion(False, 0, problemas)
            except Exception as ex:
                problemas.append(
                    f"Excepcion en la muestra {i}: {type(ex).__name__}: {ex}")
                return ResultadoVerificacion(False, i, problemas)

            n_ok += 1

            # --- Tipo del estado ---
            if not isinstance(s.estado, EstadoTASM):
                problemas.append(
                    f"muestra {i}: estado es {type(s.estado).__name__}, "
                    f"deberia ser EstadoTASM. Si TASM devuelve enteros o "
                    f"cadenas, la conversion va en el adaptador.")
                break

            n_estados[s.estado.value] += 1

            # --- Coherencia estado / freq_idx ---
            if s.estado == EstadoTASM.IC:
                if not (0 <= s.freq_idx < len(self.cfg.bci.frecuencias)):
                    problemas.append(
                        f"muestra {i}: estado IC con freq_idx={s.freq_idx}, "
                        f"fuera de [0, {len(self.cfg.bci.frecuencias)-1}]. "
                        f"Si TASM usa base 1, restar 1 en el adaptador.")
                    break
                freqs_vistas.add(s.freq_idx)
            else:
                if s.freq_idx != -1:
                    avisos.append(
                        f"muestra {i}: estado {s.estado.value} con "
                        f"freq_idx={s.freq_idx}; deberia ser -1.")

            # --- Rangos ---
            if not (0.0 <= s.p_max <= 1.0):
                problemas.append(
                    f"muestra {i}: p_max={s.p_max:.3f} fuera de [0,1]. "
                    f"Si viene en porcentaje, dividir por 100.")
                break

            if not (0.0 <= s.lambda_bci <= 1.0):
                problemas.append(
                    f"muestra {i}: lambda_bci={s.lambda_bci:.3f} fuera de "
                    f"[0,1]. Si viene en porcentaje, dividir por 100.")
                break

            p_max_min = min(p_max_min, s.p_max)
            p_max_max = max(p_max_max, s.p_max)
            lam_min = min(lam_min, s.lambda_bci)
            lam_max = max(lam_max, s.lambda_bci)

            if not isinstance(s.valido, bool):
                problemas.append(
                    f"muestra {i}: valido es {type(s.valido).__name__}, "
                    f"deberia ser bool.")
                break
            if not s.valido:
                n_invalidas += 1

            # --- rho, si viene ---
            if s.rho is not None:
                if len(s.rho) != len(self.cfg.bci.frecuencias):
                    problemas.append(
                        f"muestra {i}: rho tiene {len(s.rho)} elementos, "
                        f"deberian ser {len(self.cfg.bci.frecuencias)}. "
                        f"Posible desajuste en el numero de frecuencias.")
                    break

        # --- Comprobaciones globales ---
        if n_ok > 50:
            if n_estados["TR"] == 0:
                problemas.append(
                    "NUNCA se observo el estado TR. Es la clase central del "
                    "trabajo: si no aparece, o el modelo no la aprendio o el "
                    "adaptador la esta perdiendo.")

            if n_estados["IC"] == 0:
                problemas.append(
                    "NUNCA se observo el estado IC. El sistema no podria "
                    "aceptar ningun comando.")

            if n_estados["Idle"] == 0:
                avisos.append(
                    "Nunca se observo Idle. Bajo enclavamiento deberia ser "
                    "el estado dominante.")

            if len(freqs_vistas) == 1 and n_estados["IC"] > 20:
                avisos.append(
                    f"Solo se observo la frecuencia {list(freqs_vistas)[0]}. "
                    f"Normal si el guion asi lo pide; sospechoso si no.")

            if abs(lam_max - lam_min) < 0.05:
                avisos.append(
                    f"lambda_bci apenas varia ({lam_min:.3f}-{lam_max:.3f}). "
                    f"En la Fase 2 se usa como ganancia continua; "
                    f"un valor casi constante la haria inutil.")

        est = {
            "IC": n_estados["IC"],
            "TR": n_estados["TR"],
            "Idle": n_estados["Idle"],
            "invalidas": n_invalidas,
            "frecuencias_vistas": sorted(freqs_vistas),
            "p_max_rango": f"[{p_max_min:.3f}, {p_max_max:.3f}]",
            "lambda_rango": f"[{lam_min:.3f}, {lam_max:.3f}]",
            "tiene_ground_truth": fuente.tiene_ground_truth,
        }

        return ResultadoVerificacion(
            correcto=(len(problemas) == 0),
            n_muestras=n_ok,
            problemas=problemas,
            avisos=avisos,
            estadisticas=est,
        )


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 72)
    print("DEMOSTRACION DE LA INTERFAZ DE TASM")
    print("=" * 72)

    print()
    print(f"  Fuente configurada: {CONFIG.tasm.fuente}")
    print()
    print("  Cambiar de mock a real es cambiar UN parametro:")
    print("     config.tasm.fuente = 'real'")
    print()
    print("  La FSM, el arbitraje, los nodos ROS2 y la Etapa 2 no se enteran")
    print("  de cual esta corriendo: todos consumen el mismo contrato.")

    # --- Fabrica ---
    print()
    print("-" * 72)
    print("FABRICA")
    print("-" * 72)
    fuente = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
    print(f"  Creada: {fuente.nombre}")
    print(f"  Ground truth disponible: "
          f"{'si' if fuente.tiene_ground_truth else 'no'}")

    # --- Contrato ---
    print()
    print("-" * 72)
    print("VERIFICACION DEL CONTRATO")
    print("-" * 72)
    print("  Se ejercita la fuente y se comprueba cada campo: tipos, rangos,")
    print("  coherencia entre estado y freq_idx, y presencia de las tres")
    print("  clases.")
    print()

    ver = VerificadorContrato(CONFIG)
    res = ver.verificar(fuente, n_muestras=500)
    print(res.resumen())

    # --- Que detecta ---
    print()
    print("-" * 72)
    print("QUE DETECTA LA VERIFICACION")
    print("-" * 72)
    print("  Se simulan tres adaptadores mal escritos, de los que producen")
    print("  fallos silenciosos.")
    print()

    class FuenteRota(FuenteTASM):
        def __init__(self, fallo):
            self.fallo = fallo
            self.i = 0

        def siguiente(self, ventana=None):
            self.i += 1
            base = SalidaTASM(EstadoTASM.IC, 2, 0.9, 0.9, True)
            if self.fallo == "indice_base_1":
                base.freq_idx = 4          # base 1 con 4 frecuencias
            elif self.fallo == "porcentaje":
                base.lambda_bci = 95.0     # 0-100 en vez de [0,1]
            elif self.fallo == "sin_tr":
                base.estado = EstadoTASM.IC
            return base

        def reiniciar(self):
            self.i = 0

        @property
        def nombre(self):
            return f"rota ({self.fallo})"

    for fallo, desc in (
            ("indice_base_1", "indice de frecuencia en base 1"),
            ("porcentaje", "lambda_bci en porcentaje"),
            ("sin_tr", "el modelo nunca reporta TR")):
        r = ver.verificar(FuenteRota(fallo), n_muestras=100)
        print(f"  {desc:>34}: "
              f"{'DETECTADO' if not r.correcto else 'no detectado'}")
        if r.problemas:
            print(f"      {r.problemas[0][:70]}")

    print()
    print("  Ninguno de estos fallos daria un error en ejecucion. Todos")
    print("  producirian un sistema que corre y da numeros que no significan")
    print("  lo que uno cree.")

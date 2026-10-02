"""
config.py --- FUENTE UNICA DE VERDAD DEL SISTEMA
Bloque 1: nucleo_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo, pero NO ha sido probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando que fallaba y, si lo resolviste, como.

El debugging es parte normal del trabajo de investigacion, no una senal de
que algo este mal disenado.
===========================================================================

REGLA FUNDAMENTAL DE ESTE ARCHIVO
---------------------------------
Este es el UNICO lugar donde se definen valores numericos del sistema.

Ningun otro modulo debe contener constantes propias. Todos reciben el objeto
`Config` por parametro y leen de el. Si cambias un valor aqui, el cambio se
propaga a TODO el sistema sin excepcion.

Para garantizarlo hay dos mecanismos:
  - Los dataclasses son `frozen=True`: no se pueden modificar en caliente.
    Un modulo que intente `cfg.bci.fs = 512` lanzara excepcion.
  - El test `tests/test_config_unica_fuente.py` recorre todos los modulos
    buscando numeros magicos y falla si encuentra alguno.

Como usarlo:
    from config import CONFIG
    fsm = CommandFSM(CONFIG)        # se pasa el objeto completo
    fs = CONFIG.bci.fs              # se lee, nunca se escribe

Como probar una variante sin tocar este archivo:
    cfg = CONFIG.copia_con(bci=dict(n_conf=4))
    fsm = CommandFSM(cfg)
"""

from dataclasses import dataclass, field, replace, asdict
from typing import Tuple, Dict, List
import math


# ===========================================================================
# BCI: adquisicion, clasificacion y ventanas
# ===========================================================================

@dataclass(frozen=True)
class BCIConfig:
    """Parametros de adquisicion EEG y del clasificador FBCCA."""

    # ---- Adquisicion ----
    fs: float = 256.0
    """Frecuencia de muestreo del amplificador (Hz)."""

    canales: Tuple[str, ...] = (
        "Pz", "PO5", "PO3", "POz", "PO4", "PO6", "Oz", "O1", "O2"
    )
    """
    Montaje occipital, 9 canales, ORDEN FIJO.

    ATENCION: este orden es parte del contrato con el modulo TASM. Los filtros
    espaciales entrenados en la Linea 1 asumen esta disposicion exacta.
    Permutarlo o sustituir canales los invalida SIN QUE NADA FALLE DE FORMA
    VISIBLE: el sistema seguira corriendo y dando resultados, pero malos.
    """

    notch_hz: float = 60.0
    """
    Frecuencia del filtro notch (Hz).

    60 Hz corresponde a la red electrica peruana. La mayoria de pipelines
    publicados usan 50 Hz (Europa, China) y es un error facil de heredar al
    reutilizar codigo ajeno.
    """

    banda_hz: Tuple[float, float] = (5.0, 90.0)
    """
    Banda de paso del filtro (Hz).

    El limite superior de 90 Hz NO es arbitrario: el decodificador usa filter
    bank y explota armonicos hasta el cuarto orden. El cuarto armonico de
    15.2 Hz cae en 60.8 Hz. Una banda que terminase en 50 Hz lo eliminaria.
    """

    # ---- Frecuencias de estimulacion ----
    freq_giro_izq: float = 8.0
    freq_giro_der: float = 12.0
    freq_avanzar: float = 14.0
    freq_parar: float = 15.2
    """
    Conjunto de frecuencias (Hz). Verificado contra cuatro criterios:
      - Todas pertenecen a la grilla del dataset Benchmark (8.0 a 15.8, paso 0.2)
      - Ningun armonico de orden <= 4 cae sobre la red de 60 Hz
      - No hay colision armonica entre fundamentales
      - Todas caen en bins enteros con ventana FFT de 5 s (resolucion 0.2 Hz)

    El valor 15.2 (y no 15.0) evita que el cuarto armonico caiga exactamente
    en 60 Hz, donde seria indistinguible de la interferencia de red.
    """

    # ---- Ventanas ----
    tw_tasm: float = 0.750
    """
    Ventana de analisis de TASM (s). Valor tomado de la Bitacora TASM
    (cfg.feat.win_ms = 750).

    NO subir a 1000 ms. La Linea 1 barrio 250/500/750/1000 ms y encontro que,
    aunque AUC, accuracy y kappa mejoran con 1000 ms, el F1 de la clase TR se
    desploma de 0.313 a 0.150. La causa: con TR de ~460 ms, una ventana de
    1000 ms es mas larga que la clase que pretende medir, y el clasificador
    deja de detectar el transitorio para detectar solo que hubo un onset.
    Ademas la latencia subiria a 1050 ms, casi el doble de los 592 ms del
    estado del arte.
    """

    paso_tasm: float = 0.050
    """Paso de la ventana deslizante de TASM (s). Valor de la Bitacora TASM
    (cfg.feat.step_ms = 50). Da 20 decisiones/s."""

    n_conf: int = 8
    """
    Ventanas consecutivas en el mismo estado IC necesarias para confirmar un
    comando y enclavarlo.

    8 ventanas x 50 ms = 400 ms.

    ESTE VALOR ES PROVISIONAL Y DEBE FIJARSE CON LOS DATOS DE LA LINEA 1.

    El criterio de diseno es que la racha sea del ORDEN de la duracion de una
    transicion (140-600 ms segun cfg.states.tr_win de TASM), no mayor. Si la
    racha supera la duracion tipica de una TR, ninguna transicion alcanza a
    completarla y el mecanismo de supresion se vuelve irrelevante: el sistema
    quedaria protegido, pero el experimento no podria medir el aporte de TASM
    porque el baseline binario tampoco dejaria pasar comandos espurios.

    Una version anterior de este archivo usaba 8 ventanas de 125 ms = 1.0 s.
    Esa configuracion equivale a la ventana de 1000 ms que la Linea 1 descarto
    explicitamente, y en simulacion producia diferencia nula entre condiciones.

    Rango razonable a barrer: 4 a 12 ventanas (200 a 600 ms).
    """

    tr_timeout_ventanas: int = 40
    """Ventanas en TR antes de forzar Idle. 40 x 125 ms = 5.0 s. Cubre el
    caso patologico de una transicion que nunca se resuelve."""

    watchdog_s: float = 0.400
    """Tiempo sin mensaje de TASM antes de degradar a estado seguro (s)."""

    # ---- FBCCA ----
    n_subbandas: int = 5
    n_armonicos: int = 5
    fbcca_a: float = 1.25
    fbcca_b: float = 0.25
    """Pesos del banco de filtros: w_m = m^(-a) + b."""

    @property
    def frecuencias(self) -> Tuple[float, ...]:
        """Las cuatro frecuencias en orden de indice fijo (0..3)."""
        return (self.freq_giro_izq, self.freq_giro_der,
                self.freq_avanzar, self.freq_parar)

    @property
    def fases_jfpm(self) -> Tuple[float, ...]:
        """
        Fases JFPM, paso pi/2.

        Con CUATRO objetivos el paso pi/2 es valido porque phi_0 != phi_3.
        La Linea 2, con cinco objetivos, necesita paso 2*pi/5 porque con pi/2
        se tendria phi_0 == phi_4 (2*pi equivale a 0).
        """
        return tuple((math.pi / 2) * k for k in range(len(self.frecuencias)))

    @property
    def muestras_ventana_tasm(self) -> int:
        return int(round(self.tw_tasm * self.fs))

    @property
    def muestras_paso_tasm(self) -> int:
        return int(round(self.paso_tasm * self.fs))

    @property
    def n_canales(self) -> int:
        return len(self.canales)

    @property
    def t_confirmacion(self) -> float:
        """Tiempo real que tarda en confirmarse un comando (s)."""
        return self.n_conf * self.paso_tasm


# ===========================================================================
# ROBOT: geometria y cinematica
# ===========================================================================

@dataclass(frozen=True)
class RobotConfig:
    """Parametros del TurtleBot3 Waffle Pi."""

    ancho: float = 0.281
    largo: float = 0.306
    """Dimensiones del chasis (m). Datasheet ROBOTIS."""

    u_max: float = 0.15
    """
    Velocidad lineal (m/s).

    NO AUMENTAR sin medir el delay real del sistema. El valor se deriva de
    u * T_d = 0.168 m de desplazamiento por ciclo de decision, que es la cota
    de seguridad del sistema.
    """

    omega_max: float = 0.50
    """Velocidad angular (rad/s). Giro de 90 grados en ~3.1 s."""

    u_emergencia: float = -0.10
    """Velocidad de retroceso en modo emergencia (m/s). Negativa."""

    rho_safe: float = 0.15
    """
    Distancia de activacion del override de emergencia (m).

    NO CAMBIAR sin justificacion tecnica documentada. Es el ultimo mecanismo
    de proteccion antes de una colision.
    """

    rho_hist: float = 0.05
    """Histeresis de recuperacion (m). Se sale de emergencia cuando
    rho_min > rho_safe + rho_hist."""

    n_hist: int = 5
    """Ciclos consecutivos por encima del umbral para salir de emergencia."""

    ts_control: float = 0.05
    """Periodo del lazo de control (s). 20 Hz."""

    lidar_altura: float = 0.18
    """Altura del plano de escaneo del LiDAR LDS-01 (m).

    Determina la altura minima de los obstaculos: cualquier objeto por debajo
    de este plano es INVISIBLE para el sistema de seguridad."""

    lidar_n_rayos: int = 72
    lidar_rango_max: float = 5.0
    """Configuracion del LiDAR simulado. 72 rayos dan 5 grados de resolucion."""

    @property
    def diagonal(self) -> float:
        """
        Diagonal del chasis (m) = diametro del circulo barrido al rotar sobre
        el propio eje.

        Es el numero que justifica que este sistema use cuatro comandos y no
        cinco: con 0.415 m y un gap minimo de 0.70 m, el robot SIEMPRE puede
        reorientarse en sitio, por lo que no necesita comando de retroceso.
        """
        return math.hypot(self.ancho, self.largo)


# ===========================================================================
# CONTROL: PID y campo potencial
# ===========================================================================

@dataclass(frozen=True)
class ControlConfig:
    """Parametros del controlador y del planificador."""

    # ---- PID ----
    kp: float = 1.20
    ki: float = 0.10
    kd: float = 0.05
    antiwindup: float = 1.0
    """
    Ganancias del PID discreto (punto de partida por Ziegler-Nichols).

    Deben reajustarse en hardware real. El termino integral es la mejora
    principal sobre un control proporcional puro: elimina el error de estado
    estacionario que un P puro deja siempre presente.
    """

    # ---- Campo potencial artificial ----
    apf_xi: float = 1.0
    """Ganancia del campo atractivo."""

    apf_d0: float = 0.50
    """Distancia de transicion cuadratico/lineal del campo atractivo (m)."""

    apf_eta: float = 2.0
    """Ganancia del campo repulsivo."""

    apf_rho0: float = 0.40
    """Radio de influencia del campo repulsivo (m). Mas alla, F_rep = 0."""

    apf_epsilon_stall: float = 0.02
    apf_t_stall: float = 2.0
    """Deteccion de minimo local: |F| < epsilon durante t_stall segundos."""


# ===========================================================================
# FSM: maquina de estados de comando
# ===========================================================================

@dataclass(frozen=True)
class FSMConfig:
    """Salvaguardas de la maquina de estados con enclavamiento."""

    tope_rotacion_grados: float = 400.0
    """Rotacion acumulada maxima antes de forzar DETENIDO (grados).

    Cubre el modo de fallo en que el BCI queda enganchado en un comando de
    giro. En operacion normal nunca se alcanza."""

    tope_traslacion_s: float = 15.0
    """Tiempo maximo de avance continuo antes de forzar DETENIDO (s)."""


# ===========================================================================
# ETAPA 2: seleccion por particion binaria
# ===========================================================================

@dataclass(frozen=True)
class Etapa2Config:
    """Parametros de la fase de seleccion de objeto."""

    objeto_ancho: float = 0.20
    """Ancho de los objetos seleccionables (m). Uniforme en todos los niveles
    de N para no introducir sesgo de visibilidad entre condiciones."""

    objeto_separacion: float = 0.25
    """Separacion centro a centro entre objetos (m). CONSTANTE en todos los
    niveles: lo que varia es la distancia a la que se detiene el robot."""

    niveles_n: Tuple[int, ...] = (2, 4, 8)
    """
    Numero de objetos a evaluar en el experimento de escalabilidad.

    Son potencias de dos para que el arbol de decision este balanceado y el
    numero de decisiones sea identico para todos los objetivos. Con N=6, por
    ejemplo, algunos objetos se alcanzan en 2 decisiones y otros en 3, lo que
    mezclaria dos condiciones bajo la misma etiqueta al comparar tiempos.
    """

    umbral_transicion: float = 1.80
    """Distancia maxima a los objetos para pasar de Etapa 1 a Etapa 2 (m)."""

    permanencia_detenido_s: float = 2.0
    """Tiempo minimo en DETENIDO antes de permitir la transicion (s).

    Evita que una parada momentanea para evaluar el entorno dispare el cambio
    de etapa."""

    t_max_iteracion: float = 8.0
    """Timeout de una iteracion de la busqueda binaria (s). Al agotarse, la
    iteracion se repite en vez de avanzar con un voto dudoso."""

    distancia_aproximacion: float = 0.30
    """Distancia final al objeto seleccionado en la fase de aproximacion (m)."""

    # ---- Camara ----
    fov_horizontal: float = 62.2
    """Campo visual horizontal de la camara (grados). Raspberry Pi Camera v2.1."""

    ancho_imagen_px: int = 1280
    """Resolucion horizontal (px). A 640 px la deteccion de marcadores a
    distancia se degrada notablemente."""

    px_minimo_deteccion: int = 30
    """Tamano minimo en pixeles para detectar un objeto de forma fiable."""

    def distancia_minima(self, n_objetos: int) -> float:
        """
        Distancia minima a la que el robot debe situarse para que N objetos
        quepan en el campo visual (m).

        Deriva de: ancho_visible(D) = 2 * D * tan(FOV/2)
        """
        ancho_fila = (n_objetos - 1) * self.objeto_separacion + self.objeto_ancho
        return ancho_fila / (2 * math.tan(math.radians(self.fov_horizontal / 2)))

    def pixeles_por_objeto(self, n_objetos: int) -> float:
        """Tamano aparente de cada objeto (px) a la distancia minima."""
        d = self.distancia_minima(n_objetos)
        ang = 2 * math.degrees(math.atan(self.objeto_ancho / (2 * d)))
        return ang / self.fov_horizontal * self.ancho_imagen_px

    def decisiones_necesarias(self, n_objetos: int) -> int:
        """Numero de decisiones BCI para seleccionar entre N objetos."""
        if n_objetos <= 1:
            return 0
        return math.ceil(math.log2(n_objetos))


# ===========================================================================
# TASM MOCK: generador sintetico para desarrollo
# ===========================================================================

@dataclass(frozen=True)
class TASMMockConfig:
    """
    Parametros del generador sintetico que sustituye a TASM durante el
    desarrollo.

    ATENCION: el mock es andamiaje de desarrollo. Esta PROHIBIDO usarlo en los
    experimentos formales: alli la senal debe ser EEG real. Usar senal
    sintetica destruiria uno de los diferenciadores frente a trabajos previos
    que si emplearon EEG simulado.
    """

    dur_ic_min: float = 1.0
    dur_ic_max: float = 4.0
    """Duracion del estado IC (s). Valores de la Bitacora TASM
    (cfg.states.ic_win = (1.00, 4.00))."""

    dur_tr_min: float = 0.14
    dur_tr_max: float = 0.60
    """
    Duracion del estado TR (s).

    Valores de la Bitacora TASM (cfg.states.tr_win = (0.14, 0.60)), obtenidos
    del etiquetado del dataset. Son mas cortos que el rango generico de la
    literatura sobre gaze shift (0.5-1.0 s), porque aqui se etiqueta el
    transitorio de la respuesta SSVEP, no el movimiento ocular completo.

    Con paso de 50 ms esto equivale a 3-12 ventanas.
    """

    dur_idle_min: float = 2.0
    dur_idle_max: float = 6.0
    """Duracion del estado Idle (s). Es el estado dominante bajo el paradigma
    de enclavamiento: el usuario navega mirando el video, no los estimulos."""

    p_episodio_fp_binary: float = 0.35
    """
    Probabilidad de que un EPISODIO completo de TR produzca falsos positivos
    sostenidos, con el detector binario.

    MODELO DE ERROR POR EPISODIOS, NO POR VENTANAS
    ----------------------------------------------
    Una version anterior de este archivo modelaba los errores como
    independientes ventana a ventana. Eso es fisicamente incorrecto y hacia
    imposible medir nada.

    Razon: durante una transicion del estimulo A al B, el SSVEP no es ruido
    aleatorio. Es una mezcla decreciente de A y creciente de B. Si el
    clasificador se confunde, se confunde de forma CONSISTENTE durante todo
    el transitorio, reportando A o B de manera sostenida, no saltando entre
    las cuatro frecuencias al azar.

    Con errores independientes, la probabilidad de acumular n ventanas
    seguidas con la misma frecuencia es p^n * (1/K)^(n-1), que para n=4 y
    K=4 da 1.3e-4: nunca ocurre. Con errores por episodio, un solo episodio
    mal clasificado puede completar la racha entera.
    """

    p_episodio_fp_tasm: float = 0.10
    """
    Idem para TASM (LDA de 3 clases + HMM).

    El valor es coherente con el benchmark de la Linea 1: FPR sobre ventanas
    TR de 0.141 con LDA solo y 0.089 anadiendo el HMM.

    ESTA ES LA DIFERENCIA QUE EL EXPERIMENTO MIDE. Si al sustituir el mock
    por TASM real este numero resultara parecido al del baseline, el efecto
    del trabajo desapareceria. Por eso la PoC de la Linea 1 es dependencia
    critica.
    """

    p_ventana_dentro_episodio: float = 0.80
    """
    Dentro de un episodio marcado como falso positivo, probabilidad de que
    cada ventana concreta reporte IC.

    No es 1.0 porque incluso en un episodio mal clasificado hay ventanas que
    caen del lado correcto.
    """

    p_error_idle_como_ic_binary: float = 0.08
    p_error_idle_como_ic_tasm: float = 0.05
    """
    Probabilidad de reportar IC durante Idle real.

    La diferencia entre modos aqui es menor que en TR: distinguir reposo de
    control es un problema mas facil que distinguir transicion de control, y
    ambos detectores lo resuelven razonablemente bien. El aporte de TASM se
    concentra en TR, no en Idle.
    """

    p_acierto_frecuencia: float = 0.90
    """Probabilidad de identificar correctamente la frecuencia en estado IC."""

    semilla: int = 42
    """Semilla del generador. Fijarla hace los experimentos reproducibles."""


# ===========================================================================
# ESCENARIOS
# ===========================================================================

@dataclass(frozen=True)
class Escenario:
    """Definicion geometrica de un escenario de navegacion."""
    nombre: str
    gap: float
    """Separacion libre entre obstaculos en la zona critica (m)."""
    n_obstaculos: int
    n_tr_minimo: int
    """Transiciones de mirada minimas que el escenario induce."""
    timeout_s: float
    sala: Tuple[float, float] = (6.0, 4.0)

    def margen_libre(self, ancho_robot: float) -> float:
        """Holgura lateral efectiva tras descontar el ancho del robot (m).

        Es la variable que operacionaliza la dificultad del escenario."""
        return self.gap - ancho_robot

    def margen_tras_seguridad(self, ancho_robot: float,
                               rho_safe: float) -> float:
        """Holgura tras descontar tambien los margenes de seguridad (m).

        Debe ser positiva para que el escenario sea navegable."""
        return self.margen_libre(ancho_robot) - 2 * rho_safe


@dataclass(frozen=True)
class EscenariosConfig:
    """Los tres escenarios de dificultad creciente."""

    lista: Tuple[Escenario, ...] = (
        Escenario("Escenario 1 - Corredor recto",
                  gap=1.20, n_obstaculos=2, n_tr_minimo=6, timeout_s=180.0),
        Escenario("Escenario 2 - Corredor en L",
                  gap=0.90, n_obstaculos=3, n_tr_minimo=10, timeout_s=240.0),
        Escenario("Escenario 3 - Sala abierta",
                  gap=0.70, n_obstaculos=5, n_tr_minimo=14, timeout_s=300.0),
    )

    usados_en_fase1: Tuple[int, ...] = (0, 1)
    """Indices de los escenarios del experimento de Fase 1.

    El escenario 3 se reserva para la Fase 2."""


# ===========================================================================
# EXPERIMENTO
# ===========================================================================

@dataclass(frozen=True)
class ExperimentoConfig:
    """Parametros del diseno experimental."""

    modo_bci: str = "tasm"
    """
    Condicion experimental. Valores admitidos:
      "tasm"   : deteccion de 3 clases (IC / TR / Idle). Propuesta.
      "binary" : deteccion binaria (IC / no-control). Baseline.

    El baseline NO es un sistema sincrono: es un detector que opera de forma
    continua y permite navegacion libre, pero que colapsa TR e Idle en una
    sola clase, que es lo que hace el estado del arte.
    """

    trials_por_celda: int = 5
    n_sujetos_minimo: int = 8
    """Con efecto grande esperado (d ~ 0.8), potencia 0.80 y alpha 0.05, 8
    sujetos bastan. Reclutar 10 para cubrir exclusiones por analfabetismo
    BCI."""

    fatiga_cada_n_trials: int = 15
    """Cada cuantos trials se aplica la escala de fatiga visual 1-10."""

    MODOS_VALIDOS: Tuple[str, ...] = ("tasm", "binary")




# ===========================================================================
# ESTIMULO VISUAL  (anadido en el Bloque 2)
# ===========================================================================

@dataclass(frozen=True)
class EstimuloConfig:
    """
    Parametros del estimulo SSVEP y de la interfaz visual.

    Los valores de forma, distribucion y densidad provienen del estudio de
    Meng et al. (2023), que barrio esas tres variables con 12
    sujetos y decodificacion FB-eTRCA.
    """

    # ---- Pantalla ----
    refresh_hz: float = 60.0
    """
    Tasa de refresco del monitor (Hz).

    Es la restriccion que obliga a usar generacion sinusoidal muestreada. Con
    onda cuadrada solo serian realizables frecuencias 60/n con n par, y
    ninguna de las cuatro del sistema lo cumple.
    """

    resolucion: Tuple[int, int] = (1920, 1080)
    distancia_vision: float = 0.70
    """Distancia ojo-pantalla (m). Determina el tamano en pixeles que
    corresponde a un tamano angular dado."""

    ancho_pantalla_m: float = 0.52
    """Ancho fisico del area visible del monitor (m)."""

    # ---- Estimulo ----
    tamano_angular: float = 3.0
    """Tamano angular del estimulo (grados). Valor estandar en la
    literatura de SSVEP."""

    densidad_pixeles: float = 0.60
    """
    Fraccion de celdas encendidas dentro del cuadrado.

    Meng et al. barrieron 100/90/80/70/60/40/20%. Entre 100% y 60% la
    precision no cambia significativamente (92.7% a 90.1%, p=0.148) mientras
    la fatiga cae de 8.75 a 5.58 en escala 1-10. Por debajo de 60% la
    precision se desploma (p<0.005).

    El 60% es por tanto el BORDE INFERIOR de la meseta, no su centro.

    Ventaja adicional para este sistema: los huecos dejan ver el video de la
    camara a traves del estimulo, lo que resuelve la tension entre estimulo
    eficaz y visibilidad del entorno.
    """

    celdas_por_lado: int = 16
    """Resolucion de la retICUla del estimulo. Con 16x16 = 256 celdas, al
    60% se encienden 153."""

    distribucion_aleatoria: bool = True
    """
    Distribucion de las celdas encendidas.

    Meng et al.: la distribucion ALEATORIA supera a la uniforme tanto en
    precision como en fatiga. El articulo contiene una errata en su seccion
    de Discusion (dice "square-uniform"), contradicha por todas sus tablas y
    su conclusion.
    """

    color_activo: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_fondo: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Blanco sobre negro. Maximiza el contraste, que determina la amplitud
    de la respuesta SSVEP."""

    # ---- Disposicion en cruz ----
    posiciones_cruz: Tuple[Tuple[float, float], ...] = (
        (-0.42,  0.00),   # 0: giro izquierda  -> izquierda de la pantalla
        ( 0.42,  0.00),   # 1: giro derecha    -> derecha
        ( 0.00,  0.38),   # 2: avanzar         -> arriba
        ( 0.00, -0.38),   # 3: parar           -> abajo
    )
    """
    Posicion de cada estimulo en coordenadas normalizadas [-1, 1].

    El orden corresponde al de config.bci.frecuencias. La disposicion es
    semanticamente directa: el estimulo de girar a la izquierda esta a la
    izquierda, el de avanzar arriba. Eso reduce la carga de aprendizaje.
    """

    # ---- Indicador de estado TASM ----
    mostrar_indicador_tasm: bool = True
    """
    Borde de color que informa al usuario del estado cognitivo detectado.

    Existe porque en un sistema asincrono el robot puede legitimamente no
    aceptar un comando durante una transicion. Sin realimentacion, el usuario
    lo interpretaria como una falla. Relevante para la medida de carga
    cognitiva.
    """

    grosor_indicador_px: int = 8
    opacidad_indicador: float = 0.25
    """Tenue a proposito: informa sin competir por la atencion con los
    estimulos ni con el video."""

    color_ic: Tuple[float, float, float] = (0.20, 0.80, 0.35)
    color_tr: Tuple[float, float, float] = (0.95, 0.70, 0.15)
    color_idle: Tuple[float, float, float] = (0.55, 0.55, 0.55)

    # ---- Etapa 2 ----
    grosor_bbox_px: int = 4
    """Grosor del recuadro que encuadra cada objeto en la fase de seleccion.

    El recuadro encuadra el OBJETO completo, no el marcador ArUco: el
    marcador es infraestructura invisible para el usuario."""

    # ---- Validacion ----
    ventana_fft: float = 5.0
    """
    Duracion de la ventana para la validacion espectral (s).

    Con 5 s la resolucion es 0.2 Hz y las cuatro frecuencias caen cada una en
    su propio bin (40, 60, 70, 76). Con 4 s la resolucion seria 0.25 Hz y
    15.2 Hz quedaria repartida entre dos bins, con lo que el error de fase
    medido seria un artefacto de la ventana y no del renderizado.
    """

    max_frames_perdidos: float = 0.01
    """Fraccion maxima de frames perdidos admisible en un trial. Por encima,
    el trial se descarta. Criterio declarado a priori."""

    def grados_a_pixeles(self, grados: float) -> float:
        """Convierte un tamano angular a pixeles en la pantalla."""
        m_por_grado = 2 * self.distancia_vision * math.tan(
            math.radians(grados / 2))
        px_por_m = self.resolucion[0] / self.ancho_pantalla_m
        return m_por_grado * px_por_m

    @property
    def lado_estimulo_px(self) -> int:
        """Lado del cuadrado del estimulo en pixeles."""
        return int(round(self.grados_a_pixeles(self.tamano_angular)))

    @property
    def n_celdas_activas(self) -> int:
        """Numero de celdas encendidas dentro del cuadrado."""
        total = self.celdas_por_lado ** 2
        return int(round(total * self.densidad_pixeles))

    def bin_fft(self, frecuencia: float) -> float:
        """Indice de bin espectral de una frecuencia con la ventana de
        validacion."""
        return frecuencia * self.ventana_fft


# ===========================================================================
# CONFIGURACION RAIZ
# ===========================================================================

@dataclass(frozen=True)
class Config:
    """
    Configuracion completa del sistema.

    Se instancia una sola vez como CONFIG y se pasa por parametro a todos los
    modulos. Ningun modulo debe crear su propia instancia ni definir valores
    numericos propios.
    """
    bci: BCIConfig = field(default_factory=BCIConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    etapa2: Etapa2Config = field(default_factory=Etapa2Config)
    mock: TASMMockConfig = field(default_factory=TASMMockConfig)
    escenarios: EscenariosConfig = field(default_factory=EscenariosConfig)
    experimento: ExperimentoConfig = field(default_factory=ExperimentoConfig)
    estimulo: EstimuloConfig = field(default_factory=EstimuloConfig)

    # -------------------------------------------------------------------
    def copia_con(self, **cambios) -> "Config":
        """
        Devuelve una copia con algunos parametros modificados, sin tocar este
        archivo ni la instancia original.

        Uso:
            cfg = CONFIG.copia_con(bci=dict(n_conf=4))
            cfg = CONFIG.copia_con(experimento=dict(modo_bci="binary"))

        Es la forma correcta de barrer parametros en un experimento.
        """
        nuevos = {}
        for seccion, valores in cambios.items():
            if not hasattr(self, seccion):
                raise ValueError(
                    f"Seccion '{seccion}' no existe en Config. "
                    f"Secciones validas: {list(asdict(self).keys())}"
                )
            actual = getattr(self, seccion)
            if not isinstance(valores, dict):
                raise TypeError(
                    f"Los cambios de '{seccion}' deben ser un dict, "
                    f"se recibio {type(valores).__name__}"
                )
            for clave in valores:
                if not hasattr(actual, clave):
                    raise ValueError(
                        f"'{clave}' no es un parametro de {seccion}"
                    )
            nuevos[seccion] = replace(actual, **valores)
        return replace(self, **nuevos)

    # -------------------------------------------------------------------
    def validar(self) -> List[str]:
        """
        Comprueba la coherencia interna de los parametros.

        Devuelve una lista de problemas encontrados. Lista vacia = todo bien.
        Se ejecuta automaticamente al importar el modulo.
        """
        p: List[str] = []
        b, r, e2 = self.bci, self.robot, self.etapa2

        # --- Frecuencias ---
        fr = b.frecuencias
        if len(set(fr)) != len(fr):
            p.append("Hay frecuencias repetidas en el conjunto.")

        for f in fr:
            for orden in range(1, 5):
                if abs(f * orden - b.notch_hz) < 0.15:
                    p.append(
                        f"El armonico de orden {orden} de {f} Hz cae sobre el "
                        f"notch de {b.notch_hz} Hz y sera eliminado."
                    )

        for f in fr:
            for orden in range(2, 6):
                for g in fr:
                    if abs(f * orden - g) < 0.15:
                        p.append(
                            f"Colision armonica: {orden}x{f} Hz coincide con "
                            f"la fundamental {g} Hz."
                        )

        if max(fr) * 4 > b.banda_hz[1]:
            p.append(
                f"El cuarto armonico de {max(fr)} Hz ({max(fr)*4:.1f} Hz) "
                f"queda fuera de la banda de paso {b.banda_hz}."
            )

        if min(fr) < b.banda_hz[0]:
            p.append(f"La frecuencia {min(fr)} Hz esta por debajo de la banda.")

        # --- Ventanas ---
        if b.paso_tasm > b.tw_tasm:
            p.append("El paso de ventana es mayor que la ventana: no habria "
                     "solapamiento y se perderian muestras.")

        if b.muestras_ventana_tasm < 32:
            p.append(f"La ventana TASM tiene solo {b.muestras_ventana_tasm} "
                     f"muestras; muy pocas para estimar correlaciones.")

        # --- Racha vs duracion de TR ---
        vent_tr_max = self.mock.dur_tr_max / b.paso_tasm
        if b.n_conf <= vent_tr_max:
            p.append(
                f"AVISO DE DISENO: n_conf={b.n_conf} ventanas es menor o igual "
                f"que la duracion maxima de una TR ({vent_tr_max:.1f} ventanas). "
                f"Una transicion podria completar la racha y enclavar un "
                f"comando espurio."
            )

        # --- Robot ---
        if r.u_emergencia >= 0:
            p.append("La velocidad de emergencia debe ser negativa (retroceso).")

        if r.rho_safe <= 0:
            p.append("rho_safe debe ser positivo.")

        for esc in self.escenarios.lista:
            m = esc.margen_tras_seguridad(r.ancho, r.rho_safe)
            if m <= 0:
                p.append(
                    f"'{esc.nombre}': tras descontar el robot y los margenes "
                    f"de seguridad quedan {m:.3f} m. El escenario no es "
                    f"navegable."
                )
            if esc.gap < r.diagonal:
                p.append(
                    f"'{esc.nombre}': el gap ({esc.gap} m) es menor que la "
                    f"diagonal del robot ({r.diagonal:.3f} m); no podria "
                    f"reorientarse en sitio."
                )

        # --- Etapa 2 ---
        for n in e2.niveles_n:
            if n < 2:
                p.append(f"N={n} no tiene sentido: hacen falta al menos 2 "
                         f"objetos para una decision binaria.")
                continue

            if (n & (n - 1)) != 0:
                p.append(
                    f"AVISO: N={n} no es potencia de dos. El arbol quedara "
                    f"desbalanceado y el numero de decisiones variara segun "
                    f"el objetivo."
                )

            d = e2.distancia_minima(n)
            if d > e2.umbral_transicion:
                p.append(
                    f"N={n} requiere que el robot este a {d:.2f} m para ver "
                    f"todos los objetos, pero el umbral de transicion es "
                    f"{e2.umbral_transicion} m. La transicion nunca se "
                    f"disparara."
                )

            px = e2.pixeles_por_objeto(n)
            if px < e2.px_minimo_deteccion:
                p.append(
                    f"N={n}: cada objeto ocuparia {px:.0f} px, por debajo del "
                    f"minimo de {e2.px_minimo_deteccion} px."
                )

        # --- Estimulo ---
        e = self.estimulo

        if not (0.0 < e.densidad_pixeles <= 1.0):
            p.append(f"densidad_pixeles={e.densidad_pixeles} fuera de (0,1].")

        if e.densidad_pixeles < 0.60:
            p.append(
                f"AVISO: densidad_pixeles={e.densidad_pixeles} esta por debajo "
                f"del 60%, donde la precision se desploma segun Meng 2023."
            )

        if len(e.posiciones_cruz) != len(b.frecuencias):
            p.append(
                f"Hay {len(e.posiciones_cruz)} posiciones para "
                f"{len(b.frecuencias)} frecuencias."
            )

        # Nyquist del muestreo de pantalla
        for f in b.frecuencias:
            if f >= e.refresh_hz / 2:
                p.append(
                    f"La frecuencia {f} Hz alcanza o supera Nyquist de la "
                    f"pantalla ({e.refresh_hz/2} Hz): no es representable."
                )

        # Bins de la FFT de validacion
        for f in b.frecuencias:
            bin_f = e.bin_fft(f)
            if abs(bin_f - round(bin_f)) > 1e-6:
                p.append(
                    f"AVISO: {f} Hz no cae en un bin entero con ventana de "
                    f"{e.ventana_fft} s (bin {bin_f:.2f})."
                )

        if e.lado_estimulo_px < e.celdas_por_lado:
            p.append(
                f"El estimulo mide {e.lado_estimulo_px} px pero tiene "
                f"{e.celdas_por_lado} celdas por lado: cada celda seria menor "
                f"que un pixel."
            )

        # --- Experimento ---
        if self.experimento.modo_bci not in self.experimento.MODOS_VALIDOS:
            p.append(
                f"modo_bci='{self.experimento.modo_bci}' no es valido. "
                f"Opciones: {self.experimento.MODOS_VALIDOS}"
            )

        return p

    # -------------------------------------------------------------------
    def resumen(self) -> str:
        """Vuelca los parametros principales, para incluir en los logs de
        cada experimento y saber con que configuracion se corrio."""
        b, r, e2 = self.bci, self.robot, self.etapa2
        L = []
        L.append("=" * 70)
        L.append("CONFIGURACION DEL SISTEMA")
        L.append("=" * 70)
        L.append(f"  Modo BCI            : {self.experimento.modo_bci}")
        L.append("")
        L.append(f"  Frecuencias (Hz)    : {b.frecuencias}")
        L.append(f"  Fases JFPM (rad)    : "
                 f"{tuple(round(x, 3) for x in b.fases_jfpm)}")
        L.append(f"  Muestreo            : {b.fs} Hz, {b.n_canales} canales")
        L.append(f"  Notch / banda       : {b.notch_hz} Hz / {b.banda_hz} Hz")
        L.append(f"  Ventana TASM        : {b.tw_tasm*1000:.0f} ms "
                 f"({b.muestras_ventana_tasm} muestras), "
                 f"paso {b.paso_tasm*1000:.0f} ms")
        L.append(f"  Racha de confirmac. : {b.n_conf} ventanas "
                 f"= {b.t_confirmacion:.2f} s")
        L.append("")
        L.append(f"  Robot               : {r.ancho:.3f} x {r.largo:.3f} m, "
                 f"diagonal {r.diagonal:.3f} m")
        L.append(f"  Velocidades         : u={r.u_max} m/s, "
                 f"w={r.omega_max} rad/s")
        L.append(f"  rho_safe            : {r.rho_safe} m")
        L.append("")
        L.append("  Escenarios:")
        for esc in self.escenarios.lista:
            L.append(f"    {esc.nombre:<32} gap={esc.gap:.2f} m  "
                     f"margen libre={esc.margen_libre(r.ancho):.3f} m")
        L.append("")
        L.append("  Etapa 2:")
        for n in e2.niveles_n:
            L.append(f"    N={n:<3} D_min={e2.distancia_minima(n):.2f} m  "
                     f"px/objeto={e2.pixeles_por_objeto(n):.0f}  "
                     f"decisiones={e2.decisiones_necesarias(n)}")
        L.append("")
        L.append("  Estimulo:")
        L.append(f"    Refresco          : {self.estimulo.refresh_hz} Hz")
        L.append(f"    Lado del estimulo : "
                 f"{self.estimulo.lado_estimulo_px} px "
                 f"({self.estimulo.tamano_angular} grados)")
        L.append(f"    Densidad          : "
                 f"{self.estimulo.densidad_pixeles*100:.0f}% "
                 f"({self.estimulo.n_celdas_activas} de "
                 f"{self.estimulo.celdas_por_lado**2} celdas)")
        L.append(f"    Ventana FFT       : {self.estimulo.ventana_fft} s "
                 f"(resolucion {1/self.estimulo.ventana_fft:.2f} Hz)")
        L.append("=" * 70)
        return "\n".join(L)


# ===========================================================================
# INSTANCIA GLOBAL
# ===========================================================================

CONFIG = Config()
"""
Instancia unica de la configuracion.

Importar SIEMPRE esta, no crear instancias nuevas:
    from config import CONFIG
"""


# Validacion automatica al importar. Si algo esta mal, se sabe de inmediato y
# no doce pasos mas adelante con un sintoma confuso.
_problemas = CONFIG.validar()
if _problemas:
    import warnings
    warnings.warn(
        "\n\n*** PROBLEMAS DE CONFIGURACION DETECTADOS ***\n" +
        "\n".join(f"  - {x}" for x in _problemas) +
        "\n\nRevisa config.py antes de continuar.\n",
        stacklevel=2,
    )


if __name__ == "__main__":
    print(CONFIG.resumen())
    print()
    problemas = CONFIG.validar()
    if problemas:
        print("PROBLEMAS DETECTADOS:")
        for x in problemas:
            print(f"  - {x}")
    else:
        print("Validacion: sin problemas.")

"""
binary_search.py --- Seleccion de objeto por particion binaria (Etapa 2)
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
Selecciona un objeto entre N detectados, usando solo DOS frecuencias y
ceil(log2 N) decisiones.

Los objetos se ordenan por su posicion horizontal en la imagen. En cada
iteracion se parten en dos mitades: la izquierda parpadea a una frecuencia y
la derecha a otra. El usuario elige un lado, y la mitad descartada se apaga.
Se repite sobre la mitad superviviente hasta quedar un solo objeto.

EL ROBOT PERMANECE ESTATICO
---------------------------
La camara no se mueve durante las iteraciones. La imagen es la misma; lo
unico que cambia es CUALES objetos parpadean. Un objeto que ocupa 146 px en
la primera iteracion sigue ocupando 146 px en la ultima.

Esto tiene dos consecuencias importantes:
  - El numero de decisiones es exactamente ceil(log2 N), independiente de la
    geometria de la escena.
  - No hay degradacion de la senal SSVEP por vibracion de camara o cambio de
    perspectiva entre decisiones.

El desplazamiento se reserva para una fase de aproximacion posterior, ya
seleccionado el objeto.

POR QUE POTENCIAS DE DOS EN EL EXPERIMENTO
------------------------------------------
El algoritmo funciona con cualquier N. Con N=6 la particion es 3/3, luego
2/1: algunos objetivos se alcanzan en 2 decisiones y otros en 3.

Se usan potencias de dos en el diseno experimental para que el arbol este
balanceado y todos los objetivos requieran el mismo numero de decisiones.
Sin eso, al comparar tiempos de completitud se estarian mezclando dos
condiciones distintas bajo la misma etiqueta.

COMPUERTA TASM
--------------
Cada voto requiere estado IC confirmado durante n_conf ventanas, igual que
en la Etapa 1. Durante TR e Idle la iteracion espera sin registrar voto.
Si se agota el timeout, la iteracion se repite en lugar de avanzar con un
voto dudoso.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import math

from config import Config
from command_fsm import EstadoTASM


# ===========================================================================
# TIPOS
# ===========================================================================

class LadoSeleccion(Enum):
    """Grupo elegido en una iteracion."""
    IZQUIERDA = "izquierda"
    DERECHA = "derecha"


@dataclass
class Objeto:
    """
    Objeto detectado en la escena.

    En el sistema real estos datos vienen de YOLOv8 (bounding box) y de
    ArUco (identidad y pose 3D). Aqui se representan de forma abstracta para
    que el nucleo no dependa de la capa de vision.
    """
    id_aruco: int
    x_center: float
    """Posicion horizontal del centro en la imagen (px). Es el criterio de
    ordenamiento."""
    y_center: float = 0.0
    ancho_px: float = 0.0
    alto_px: float = 0.0
    distancia: float = 0.0
    """Distancia al robot (m), obtenida de la pose del marcador."""
    etiqueta: str = ""


@dataclass
class Iteracion:
    """Registro de una iteracion de la busqueda."""
    numero: int
    candidatos: List[int]
    """Indices de los objetos aun en juego, en orden."""
    grupo_izq: List[int]
    grupo_der: List[int]
    lado_elegido: Optional[LadoSeleccion] = None
    n_ventanas: int = 0
    """Ventanas de TASM consumidas en esta iteracion."""
    n_ventanas_tr: int = 0
    """De esas, cuantas estuvieron en TR real. Alimenta la metrica T_TR."""
    timeout: bool = False
    duracion: float = 0.0


@dataclass
class ResultadoSeleccion:
    """Resultado completo de una seleccion."""
    objeto_elegido: Optional[int]
    """Indice del objeto seleccionado, o None si se agoto sin converger."""
    acierto: bool
    n_decisiones: int
    n_decisiones_teoricas: int
    """ceil(log2 N). Debe coincidir con n_decisiones si no hubo timeouts."""
    n_timeouts: int
    duracion_total: float
    iteraciones: List[Iteracion] = field(default_factory=list)

    @property
    def coincide_teoria(self) -> bool:
        """True si el numero de decisiones fue el predicho por la teoria.

        Es la verificacion directa de la hipotesis de complejidad
        logaritmica."""
        return self.n_decisiones == self.n_decisiones_teoricas


# ===========================================================================
# BUSQUEDA BINARIA
# ===========================================================================

class BusquedaBinaria:
    """
    Motor de seleccion por particion binaria con compuerta TASM.

    Uso tipico (paso a paso, controlado desde fuera):
        bb = BusquedaBinaria(CONFIG, objetos)
        while not bb.terminada:
            izq, der = bb.grupos_actuales()
            # ... presentar los grupos parpadeando, esperar decision ...
            bb.votar(LadoSeleccion.IZQUIERDA)
        print(bb.resultado())
    """

    def __init__(self, config: Config, objetos: List[Objeto]):
        self.cfg = config

        if len(objetos) < 2:
            raise ValueError(
                f"Hacen falta al menos 2 objetos para una decision binaria; "
                f"se recibieron {len(objetos)}."
            )

        # Ordenamiento por posicion horizontal. Es el criterio que hace que
        # la particion sea espacialmente coherente: el grupo izquierdo esta
        # realmente a la izquierda en la pantalla.
        self.objetos = sorted(objetos, key=lambda o: o.x_center)
        self.n = len(self.objetos)

        self._lo = 0
        self._hi = self.n - 1
        self._iteraciones: List[Iteracion] = []
        self._n_timeouts = 0
        self._t = 0.0

        self._abrir_iteracion()

    # -------------------------------------------------------------------
    @property
    def terminada(self) -> bool:
        return self._lo >= self._hi

    @property
    def n_candidatos(self) -> int:
        return self._hi - self._lo + 1

    @property
    def decisiones_teoricas(self) -> int:
        return math.ceil(math.log2(self.n)) if self.n > 1 else 0

    # -------------------------------------------------------------------
    def _abrir_iteracion(self) -> None:
        """Prepara la siguiente iteracion partiendo el rango vigente."""
        if self.terminada:
            return

        mid = (self._lo + self._hi) // 2
        izq = list(range(self._lo, mid + 1))
        der = list(range(mid + 1, self._hi + 1))

        self._iteraciones.append(Iteracion(
            numero=len(self._iteraciones) + 1,
            candidatos=list(range(self._lo, self._hi + 1)),
            grupo_izq=izq,
            grupo_der=der,
        ))

    # -------------------------------------------------------------------
    def grupos_actuales(self) -> Tuple[List[int], List[int]]:
        """
        Indices de los objetos de cada grupo en la iteracion vigente.

        El grupo izquierdo parpadea a config.bci.freq_giro_izq y el derecho a
        config.bci.freq_giro_der. Se reutilizan las mismas frecuencias que en
        la Etapa 1, lo que es coherente semanticamente: el usuario aprende
        "esta frecuencia significa izquierda" y le vale en ambas etapas.
        """
        if self.terminada:
            return ([], [])
        it = self._iteraciones[-1]
        return (it.grupo_izq, it.grupo_der)

    # -------------------------------------------------------------------
    def votar(self, lado: LadoSeleccion,
              n_ventanas: int = 0,
              n_ventanas_tr: int = 0,
              duracion: float = 0.0) -> None:
        """
        Registra el voto de una iteracion y avanza la busqueda.

        Los parametros de conteo son opcionales y sirven para las metricas:
        cuantas ventanas consumio la decision y cuantas de ellas fueron TR.
        """
        if self.terminada:
            raise RuntimeError("La busqueda ya termino; no admite mas votos.")

        it = self._iteraciones[-1]
        it.lado_elegido = lado
        it.n_ventanas = n_ventanas
        it.n_ventanas_tr = n_ventanas_tr
        it.duracion = duracion
        self._t += duracion

        mid = (self._lo + self._hi) // 2
        if lado == LadoSeleccion.IZQUIERDA:
            self._hi = mid
        else:
            self._lo = mid + 1

        self._abrir_iteracion()

    # -------------------------------------------------------------------
    def registrar_timeout(self, duracion: float = 0.0) -> None:
        """
        Registra que la iteracion agoto el tiempo sin voto valido.

        La iteracion NO avanza: se repite. Es preferible repetir a avanzar
        con un voto que TASM no confirmo, porque un error aqui es
        irrecuperable (el objeto correcto sale de la lista).
        """
        if self.terminada:
            return
        it = self._iteraciones[-1]
        it.timeout = True
        it.duracion += duracion
        self._t += duracion
        self._n_timeouts += 1

        # Se abre una iteracion nueva con el MISMO rango
        self._iteraciones.append(Iteracion(
            numero=len(self._iteraciones) + 1,
            candidatos=list(range(self._lo, self._hi + 1)),
            grupo_izq=it.grupo_izq,
            grupo_der=it.grupo_der,
        ))

    # -------------------------------------------------------------------
    def resultado(self, objetivo: Optional[int] = None) -> ResultadoSeleccion:
        """
        Resultado de la seleccion.

        Parametros
        ----------
        objetivo : indice (en la lista ORDENADA) del objeto que el usuario
                   queria. Si se da, se calcula si hubo acierto.
        """
        elegido = self._lo if self.terminada else None
        decisiones = sum(1 for it in self._iteraciones
                         if it.lado_elegido is not None)

        return ResultadoSeleccion(
            objeto_elegido=elegido,
            acierto=(elegido == objetivo) if objetivo is not None else False,
            n_decisiones=decisiones,
            n_decisiones_teoricas=self.decisiones_teoricas,
            n_timeouts=self._n_timeouts,
            duracion_total=self._t,
            iteraciones=list(self._iteraciones),
        )

    # -------------------------------------------------------------------
    def lado_correcto(self, objetivo: int) -> Optional[LadoSeleccion]:
        """
        Cual seria el voto correcto en la iteracion vigente para alcanzar el
        objetivo dado.

        Se usa en simulacion para modelar un usuario que sabe lo que quiere.
        En operacion real esto lo decide la persona.
        """
        if self.terminada:
            return None
        it = self._iteraciones[-1]
        if objetivo in it.grupo_izq:
            return LadoSeleccion.IZQUIERDA
        if objetivo in it.grupo_der:
            return LadoSeleccion.DERECHA
        return None


# ===========================================================================
# UTILIDADES
# ===========================================================================

def generar_objetos_en_fila(config: Config, n: int,
                             ancho_imagen: Optional[int] = None
                             ) -> List[Objeto]:
    """
    Genera N objetos dispuestos en fila, como en el montaje experimental.

    Las posiciones en pixeles se derivan de la geometria real: N objetos de
    `objeto_ancho` separados `objeto_separacion`, vistos desde la distancia
    minima a la que caben todos en el campo visual.
    """
    e2 = config.etapa2
    W = ancho_imagen if ancho_imagen is not None else e2.ancho_imagen_px

    # Distribucion uniforme a lo ancho de la imagen, con margen en los bordes
    margen = W * 0.08
    util = W - 2 * margen
    paso = util / max(n - 1, 1) if n > 1 else 0.0

    d = e2.distancia_minima(n)
    px = e2.pixeles_por_objeto(n)

    objetos = []
    for i in range(n):
        objetos.append(Objeto(
            id_aruco=i,
            x_center=margen + i * paso,
            y_center=W * 0.4,
            ancho_px=px,
            alto_px=px,
            distancia=d,
            etiqueta=f"obj_{i}",
        ))
    return objetos


# ===========================================================================
# DEMOSTRACION
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG

    print("=" * 70)
    print("DEMOSTRACION DE LA SELECCION POR PARTICION BINARIA")
    print("=" * 70)
    print()

    e2 = CONFIG.etapa2

    print("Configuraciones del experimento de escalabilidad:")
    print(f"{'N':>4} {'decisiones':>11} {'D_min':>8} {'px/objeto':>11}  viable")
    for n in e2.niveles_n:
        d = e2.distancia_minima(n)
        px = e2.pixeles_por_objeto(n)
        ok = (d <= e2.umbral_transicion and px >= e2.px_minimo_deteccion)
        print(f"{n:>4} {e2.decisiones_necesarias(n):>11} {d:>7.2f}m "
              f"{px:>10.0f}  {'SI' if ok else 'NO'}")
    print()

    # --- Recorrido completo con N=8 ---
    print("-" * 70)
    print("RECORRIDO COMPLETO CON N=8, objetivo = objeto 4")
    print("-" * 70)
    objetos = generar_objetos_en_fila(CONFIG, 8)
    bb = BusquedaBinaria(CONFIG, objetos)
    objetivo = 4

    while not bb.terminada:
        izq, der = bb.grupos_actuales()
        lado = bb.lado_correcto(objetivo)
        it_num = len(bb._iteraciones)
        print(f"  Iteracion {it_num}:")
        print(f"    izquierda ({CONFIG.bci.freq_giro_izq} Hz): {izq}")
        print(f"    derecha   ({CONFIG.bci.freq_giro_der} Hz): {der}")
        print(f"    voto -> {lado.value}")
        bb.votar(lado, n_ventanas=12, n_ventanas_tr=4, duracion=1.4)

    r = bb.resultado(objetivo=objetivo)
    print()
    print(f"  Objeto elegido      : {r.objeto_elegido}")
    print(f"  Acierto             : {'SI' if r.acierto else 'NO'}")
    print(f"  Decisiones          : {r.n_decisiones}")
    print(f"  Decisiones teoricas : {r.n_decisiones_teoricas}")
    print(f"  Coincide con teoria : {'SI' if r.coincide_teoria else 'NO'}")
    print(f"  Duracion            : {r.duracion_total:.1f} s")
    print()

    # --- Verificacion sobre todos los objetivos y todos los N ---
    print("-" * 70)
    print("VERIFICACION: decisiones para CADA objetivo posible")
    print("-" * 70)
    print("  Con N potencia de dos, todos los objetivos deben requerir el")
    print("  mismo numero de decisiones (arbol balanceado).")
    print()
    for n in (2, 4, 8):
        objetos = generar_objetos_en_fila(CONFIG, n)
        conteos = []
        aciertos = 0
        for obj in range(n):
            bb = BusquedaBinaria(CONFIG, objetos)
            while not bb.terminada:
                bb.votar(bb.lado_correcto(obj))
            r = bb.resultado(objetivo=obj)
            conteos.append(r.n_decisiones)
            aciertos += r.acierto
        teorico = math.ceil(math.log2(n))
        uniforme = (len(set(conteos)) == 1)
        print(f"  N={n:<3} decisiones por objetivo: {conteos}")
        print(f"        teorico={teorico}  uniforme={'SI' if uniforme else 'NO'}"
              f"  aciertos={aciertos}/{n}")
    print()

    # --- Caso no potencia de dos ---
    print("-" * 70)
    print("CASO N=6 (no potencia de dos)")
    print("-" * 70)
    objetos = generar_objetos_en_fila(CONFIG, 6)
    conteos = []
    for obj in range(6):
        bb = BusquedaBinaria(CONFIG, objetos)
        while not bb.terminada:
            bb.votar(bb.lado_correcto(obj))
        conteos.append(bb.resultado(objetivo=obj).n_decisiones)
    print(f"  Decisiones por objetivo: {conteos}")
    print(f"  Cota teorica ceil(log2 6) = {math.ceil(math.log2(6))}")
    print()
    print("  El algoritmo funciona, pero el numero de decisiones VARIA segun")
    print("  el objetivo. Por eso el experimento usa potencias de dos: si no,")
    print("  al comparar tiempos se mezclarian dos condiciones distintas.")

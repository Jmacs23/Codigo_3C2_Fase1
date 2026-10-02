#!/usr/bin/env python3
"""
logica_bridge.py --- Puente entre el paquete ROS2 y la logica verificada
Paquete: turtlebot3_bci_3c2

===========================================================================
POR QUE EXISTE ESTE ARCHIVO

La logica de decision vive en logica/nodos_logica.py, fuera del paquete
ROS2, para poder probarla sin arrancar ROS2 ni el robot.

Este archivo la importa y la reexporta, de modo que los nodos hagan:

    from turtlebot3_bci_3c2.logica_bridge import LogicaMision

en lugar de manipular sys.path en cada nodo.

Si al desplegar el paquete la ruta cambia, se corrige AQUI y en un solo
sitio, no en siete nodos.
===========================================================================
"""

import os
import sys

# Rutas donde puede estar la logica, en orden de preferencia:
#   1. Variable de entorno, para despliegues no estandar
#   2. Junto al workspace, que es la disposicion de desarrollo
#   3. Instalada como paquete
_CANDIDATAS = []

if os.environ.get("BCI3C2_LOGICA"):
    _CANDIDATAS.append(os.environ["BCI3C2_LOGICA"])

_aqui = os.path.dirname(os.path.abspath(__file__))
_CANDIDATAS.append(os.path.abspath(
    os.path.join(_aqui, "..", "..", "..", "..", "..", "logica")))
_CANDIDATAS.append(os.path.abspath(
    os.path.join(_aqui, "..", "..", "..", "..", "..")))

for _r in _CANDIDATAS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

try:
    from config import CONFIG, Config
    from nodos_logica import (
        LogicaMision, LogicaSeguridad, LogicaNavegacion, LogicaControl,
        ReceptorTASM, MensajeTASMRecibido, ModoOperacion, Etapa, Velocidad,
        EstadoSeguridad, DecisionMision,
    )
except ImportError as ex:
    raise ImportError(
        f"No se encontro la logica del sistema.\n"
        f"\n"
        f"Rutas probadas:\n" +
        "\n".join(f"  {r}" for r in _CANDIDATAS) +
        f"\n\n"
        f"Solucion: define la variable de entorno BCI3C2_LOGICA apuntando\n"
        f"al directorio que contiene config.py y nodos_logica.py.\n"
        f"\n"
        f"Error original: {ex}"
    ) from ex


def cargar_config() -> Config:
    """Devuelve la configuracion del sistema.

    Es la MISMA instancia que usan los demas bloques. No se crea una copia
    para ROS2: eso reintroduciria el problema de tener dos fuentes de verdad.
    """
    return CONFIG


__all__ = [
    "cargar_config", "CONFIG", "Config",
    "LogicaMision", "LogicaSeguridad", "LogicaNavegacion", "LogicaControl",
    "ReceptorTASM", "MensajeTASMRecibido",
    "ModoOperacion", "Etapa", "Velocidad",
    "EstadoSeguridad", "DecisionMision",
]

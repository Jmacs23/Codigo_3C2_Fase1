#!/usr/bin/env python3
"""
vision_bridge.py --- Puente entre el paquete ROS2 y los modulos de vision
Paquete: turtlebot3_bci_3c2

Mismo proposito que logica_bridge.py del Bloque 4: los modulos de vision
viven fuera del paquete ROS2 para poder probarlos sin arrancar ROS2, y este
archivo los importa y reexporta.

Si la ruta cambia al desplegar, se corrige AQUI y en un solo sitio.
"""

import os
import sys

_CANDIDATAS = []
if os.environ.get("BCI3C2_VISION"):
    _CANDIDATAS.append(os.environ["BCI3C2_VISION"])

_aqui = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "logica"):
    _CANDIDATAS.append(os.path.abspath(
        os.path.join(_aqui, "..", "..", "..", "..", "..", _sub)))

for _r in _CANDIDATAS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

try:
    from config import CONFIG, Config
    from aruco_detector import DetectorArUco, MarcadorDetectado
    from yolo_detector import (DetectorYOLO, FusionVision,
                               ObjetoSeleccionable, DeteccionYOLO)
    from etapa2_logica import (DetectorTransicion, ControlAproximacion,
                               EstadoAproximacion, ComandoAproximacion,
                               EvaluacionTransicion)
    from binary_search import (BusquedaBinaria, LadoSeleccion, Objeto,
                               generar_objetos_en_fila)
except ImportError as ex:
    raise ImportError(
        "No se encontraron los modulos de vision.\n\n"
        "Rutas probadas:\n" +
        "\n".join(f"  {r}" for r in _CANDIDATAS) +
        "\n\nSolucion: define BCI3C2_VISION apuntando al directorio que "
        "contiene aruco_detector.py y yolo_detector.py.\n\n"
        f"Error original: {ex}"
    ) from ex


def cargar_config() -> Config:
    return CONFIG


__all__ = [
    "cargar_config", "CONFIG", "Config",
    "DetectorArUco", "MarcadorDetectado",
    "DetectorYOLO", "FusionVision", "ObjetoSeleccionable", "DeteccionYOLO",
    "DetectorTransicion", "ControlAproximacion", "EstadoAproximacion",
    "ComandoAproximacion", "EvaluacionTransicion",
    "BusquedaBinaria", "LadoSeleccion", "Objeto", "generar_objetos_en_fila",
]

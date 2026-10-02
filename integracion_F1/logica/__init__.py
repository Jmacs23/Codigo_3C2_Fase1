"""
logica --- Modulos heredados de los bloques anteriores
Bloque 6: integracion_F1

Estos modulos vienen verificados de sus bloques de origen. Se incluyen para
que este bloque corra de forma autonoma.

El anadido de rutas de abajo permite que se importen entre si tanto desde
fuera del paquete (`from logica.command_fsm import ...`) como desde dentro
(`from command_fsm import ...`), que es como estan escritos en su bloque
original. Asi no hay que reescribir sus imports, lo que los haria divergir
de la version verificada.
"""

import os as _os
import sys as _sys

_aqui = _os.path.dirname(_os.path.abspath(__file__))
_raiz = _os.path.dirname(_aqui)

for _r in (_raiz, _aqui):
    if _r not in _sys.path:
        _sys.path.insert(0, _r)

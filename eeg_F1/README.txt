===========================================================================
  BLOQUE 3 --- eeg_F1
  Adquisicion EEG, filtrado y transporte a la maquina Linux
===========================================================================

AVISO
-----
Este codigo es un punto de partida funcional. Las 44 pruebas automatizadas
pasan, pero NO ha sido probado con el amplificador real.

Hay una parte deliberadamente incompleta: la clase FuenteGUSBamp es un
ESQUELETO. Las llamadas al SDK de g.tec estan marcadas con ">>> COMPLETAR".
Ver la seccion correspondiente mas abajo.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo, y que intentaste.


QUE CONTIENE
------------
  config.py         Bloques 1 y 2 MAS las secciones Adquisicion y Latencia
  filters.py        Filtrado causal con estado persistente
  eeg_source.py     Fuentes: sintetica, dataset, g.USBamp (esqueleto)
  acquisition.py    Pipeline completo, buffer circular, puente TCP
  run_eeg.py        Punto de entrada
  tests/            44 pruebas

  fbcca.py, tasm_mock.py, command_fsm.py vienen del Bloque 1 y se incluyen
  para que el bloque corra de forma autonoma.


INSTALACION
-----------
  pip install numpy scipy pytest

En la maquina Windows del laboratorio hace falta ademas el SDK de g.tec con
su binding de Python, que se distribuye con el equipo.


ORDEN DE TRABAJO
----------------
  python run_eeg.py config       Que parametros hay
  python run_eeg.py filtros      La cadena deja pasar lo que debe
  python run_eeg.py causal       POR QUE el filtrado es causal con estado
  python run_eeg.py buffer       Como funciona el ventaneo solapado
  python run_eeg.py pipeline     COMPUERTA del bloque
  python run_eeg.py latencia     El computo cabe en el periodo
  python run_eeg.py test         Las 44 pruebas

Antes de cada sesion con sujeto:
  python run_eeg.py impedancias


LOS DOS ERRORES QUE ESTE BLOQUE EVITA
--------------------------------------

ERROR 1: USAR filtfilt EN TIEMPO REAL

`filtfilt` es la funcion que casi todo el mundo usa para filtrar EEG, porque
no introduce distorsion de fase. Lo consigue filtrando dos veces: hacia
adelante y hacia atras.

Y ahi esta el problema: para calcular la salida en el instante t, el pase
hacia atras necesita muestras POSTERIORES a t. Offline es trivial porque
toda la senal ya esta grabada. En tiempo real esas muestras no existen.

Usar filtfilt online no da error ni excepcion. Da resultados que parecen
razonables pero que usan informacion del futuro dentro de cada bloque, lo
que produce un rendimiento en analisis que NO se reproduce en linea.

Ejecuta `python run_eeg.py causal` para verlo medido.


ERROR 2: NO CONSERVAR EL ESTADO DEL FILTRO

Un filtro IIR tiene memoria: su salida depende de entradas y salidas
anteriores.

Si se llama a lfilter sobre cada bloque de 12 muestras sin conservar ese
estado, cada bloque arranca desde cero y produce un TRANSITORIO. A 21
bloques por segundo, eso serian 21 transitorios por segundo contaminando la
senal.

Medido en este bloque: sin estado, el error respecto al filtrado correcto
alcanza el 95% del rango de la senal. Con estado, es exactamente cero.

La solucion es una linea de codigo (pasar `zi` de una llamada a la
siguiente), pero olvidarla produce un fallo silencioso y dificil de
diagnosticar.


EL PUNTO CRITICO: EL FACTOR Q DEL NOTCH
----------------------------------------
El cuarto armonico de 15.2 Hz cae en 60.8 Hz, a solo 0.8 Hz de la red
electrica.

Con el valor habitual de Q=30, el notch mide 2 Hz de ancho (59 a 61 Hz) y SE
COMERIA ESE ARMONICO, anulando el beneficio de haber movido la frecuencia de
parada de 15.0 a 15.2 Hz.

Por eso Q=60, que estrecha el notch a 1 Hz (59.5 a 60.5) y deja el armonico
fuera con 0.3 Hz de margen.

    Q=30 -> ganancia en 60.8 Hz = 0.62
    Q=60 -> ganancia en 60.8 Hz = 0.85

La validacion de config.py comprueba esto automaticamente. Si en algun
momento se cambian las frecuencias, avisara.


DOS LATENCIAS DISTINTAS
-----------------------
Conviene no confundirlas, porque solo una tiene que caber en el periodo.

  TIEMPO DE COMPUTO (medido: 5 ms, presupuestado: 32 ms)
  Es lo que tarda procesar una ventana. DEBE caber en el periodo de 50 ms:
  si tardara mas, la cola de trabajo creceria sin limite y el sistema
  acumularia retraso indefinidamente.

  LATENCIA EXTREMO A EXTREMO (82 ms)
  Retardo entre que ocurre algo en el cerebro y el robot reacciona. NO tiene
  que caber en el periodo: la cadena es un PIPELINE y las etapas se solapan.
  Mientras una muestra viaja por la red, la siguiente ya se esta filtrando.

  Solo debe quedar por debajo del watchdog (400 ms), y queda con 318 ms de
  margen.

Es el mismo principio que el retardo de propagacion en una tuberia: importa
para el control, no para el caudal.


COMPLETAR EL SDK DE g.tec
-------------------------
La clase FuenteGUSBamp en eeg_source.py es un esqueleto. Los puntos a
rellenar estan marcados con ">>> COMPLETAR".

QUE HAY QUE HACER
  1. Importar el binding (normalmente `pygds`) y abrir el dispositivo.
  2. Configurar frecuencia de muestreo, canales activos y DESACTIVAR los
     filtros del hardware: el filtrado se hace en software para controlar la
     fase con exactitud.
  3. Implementar la lectura no bloqueante.
  4. Implementar la medida de impedancias si el SDK lo permite.

QUE NO CAMBIAR
  El contrato. `leer()` debe devolver un array (n_muestras, n_canales) en
  MICROVOLTIOS con los canales en el orden de config.bci.canales. Si el SDK
  devuelve voltios o los canales en otro orden, la conversion va DENTRO de
  esa clase, no fuera.

POR QUE NO ESTA ESCRITO
  No se puede probar sin el hardware, y codigo no probado da falsa
  confianza. Ademas la API cambia entre versiones del SDK, y escribir contra
  una version que quiza no sea la vuestra generaria mas trabajo del que
  ahorra.

Mientras tanto, para desarrollar sin hardware:
    config.adquisicion.fuente = "sintetica"


EL ORDEN DE CANALES
-------------------
    Pz, PO5, PO3, POz, PO4, PO6, Oz, O1, O2

Los filtros espaciales entrenados en la Linea 1 asumen ese orden EXACTO.
Conectar los electrodos en otro orden no produce ningun error visible: el
sistema sigue corriendo y dando resultados, pero peores.

FuenteGUSBamp imprime el orden esperado al arrancar, precisamente para que
se verifique antes de empezar.


EL VENTANEO SOLAPADO
--------------------
Hay un desajuste que resolver:

  el amplificador entrega bloques de   12 muestras (47 ms)
  la ventana de analisis es de        192 muestras (750 ms)
  y avanza                             13 muestras (50 ms)

Cada ventana solapa un 93% con la anterior. El buffer circular resuelve
esto: acumula los bloques que llegan y extrae ventanas cuando hay muestras
suficientes.

Se usa un buffer circular y no una lista que crece porque el sistema corre
durante minutos: una lista acumularia memoria indefinidamente.


DONDE ENCAJA TASM
-----------------
Este bloque NO implementa TASM. TASM es la Linea 1.

Lo que hace es dejar el hueco con un contrato claro: el `procesador` recibe
una ventana preprocesada de forma (n_muestras, n_canales) y devuelve un
MensajeSalida. Mientras TASM real no exista, se usa el mock del Bloque 1 o
el procesador de FBCCA solo.

Escribir aqui un decodificador propio garantizaria que las dos versiones
divergieran, y que la del sistema robotico fuera la peor.


PARAMETROS QUE NO DEBEN CAMBIARSE SIN JUSTIFICACION
---------------------------------------------------
  q_notch = 60             Con 30 se pierde el 4o armonico de 15.2 Hz
  banda = [5, 90] Hz       Preserva los armonicos utiles
  notch = 60 Hz            Red peruana, no 50
  canales (orden)          Los filtros espaciales de TASM lo asumen
  usar_filtro_causal       NUNCA False en operacion online
  buffer_muestras = 12     Debe ser <= paso de ventana (13)


LO QUE FALTA VERIFICAR EN EL LABORATORIO
-----------------------------------------
  1. Que el SDK entrega los datos en microvoltios. Si fueran voltios, todos
     los umbrales de artefactos estarian mal por un factor de un millon.

  2. Que el orden de canales del amplificador coincide con el configurado.

  3. Que las impedancias bajan de 5 kOhm antes de cada sesion.

  4. Que la latencia real de red entre las dos maquinas se parece a los
     20 ms presupuestados. Si fuera mucho mayor, hay que revisar el
     watchdog.

  5. Que los picos espectrales aparecen con un sujeto real. Eso es el
     protocolo de validacion del Bloque 2, que ahora ya puede ejecutarse
     con senal de verdad.


QUE VIENE DESPUES
-----------------
  Bloque 4: capa ROS2 (nodos, mensajes, emisor de video)
  Bloque 5: vision (YOLOv8 + ArUco)
  Bloque 6: integracion con TASM real


SOBRE LAS FASES DEL TRABAJO
---------------------------
Este codigo corresponde a la FASE 1.

La FASE 2 es una extension posterior que no esta implementada. Lo que si
esta preparado es su punto de conexion:

  - El campo lambda_bci (probabilidad continua de control intencional) YA
    VIAJA en el mensaje TASMState y SE REGISTRA en cada trial, aunque en
    Fase 1 no interviene en el control.

  - En ctrl_node.py hay un bloque delimitado por los comentarios
        ===== INICIO BLOQUE FASE 2 =====
        ===== FIN BLOQUE FASE 2 =====
    con la ecuacion de ganancia variable ya escrita, desactivada por el
    parametro usar_lambda_bci de params.yaml.

Quien desarrolle la Fase 2 solo tiene que activar ese parametro y validar la
ecuacion. No hace falta tocar la cadena de comunicacion, porque el dato que
necesita ya se esta transportando y guardando.

Esa es la razon de que lambda_bci se registre desde el principio: si no se
registrara ahora, los datos de Fase 1 no servirian para desarrollar Fase 2.

===========================================================================

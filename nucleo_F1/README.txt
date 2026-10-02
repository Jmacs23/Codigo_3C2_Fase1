===========================================================================
  BLOQUE 1 --- nucleo_F1
  Nucleo algoritmico del sistema BCI-SSVEP asincrono (Fase 1)
===========================================================================

AVISO
-----
Este codigo es un punto de partida funcional. Se ha verificado que corre de
extremo a extremo y las 50 pruebas automatizadas pasan, pero NO ha sido
probado en el hardware del laboratorio.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto que
     ejecutaste, el traceback completo, y que intentaste.

El debugging es parte normal del trabajo de investigacion, no una senal de
que algo este mal disenado.


QUE CONTIENE ESTE BLOQUE
------------------------
Todo el nucleo algoritmico en Python puro. Sin ROS2, sin hardware, sin EEG,
sin Unity. Corre en cualquier maquina con Python 3.

  config.py         FUENTE UNICA DE VERDAD. Todos los parametros.
  fbcca.py          Clasificador de frecuencia (que estimulo mira)
  command_fsm.py    Maquina de estados con enclavamiento y compuerta TASM
  tasm_mock.py      Generador sintetico que sustituye a TASM
  binary_search.py  Seleccion de objeto por particion binaria (Etapa 2)
  metrics.py        Metricas, incluida N_FP_TR
  simulator2d.py    Simulador headless de un trial completo
  run.py            Punto de entrada unico
  tests/            50 pruebas automatizadas


INSTALACION
-----------
  pip install numpy scipy pytest

scipy es opcional pero recomendado: sin el, FBCCA funciona con la banda
completa en vez del banco de filtros, y la precision baja.


COMO EMPEZAR
------------
Ejecuta los comandos en este orden. No pases al siguiente hasta que el
anterior este en verde.

  python run.py config      Vuelca y valida la configuracion
  python run.py smoke       Prueba rapida de todos los modulos
  python run.py test        Las 50 pruebas
  python run.py fsm         Compuerta 1: supresion durante transiciones
  python run.py fbcca       Compuerta 2: recuperacion de frecuencia
  python run.py seleccion   Compuerta 3: complejidad logaritmica
  python run.py trial       Un trial completo
  python run.py viabilidad  HITO DE VIABILIDAD
  python run.py barrido     Barrido del parametro n_conf


LA REGLA MAS IMPORTANTE: config.py
----------------------------------
config.py es el UNICO lugar donde se definen valores numericos. Ningun otro
modulo tiene constantes propias: todos reciben el objeto Config por
parametro.

Si cambias un valor ahi, el cambio se propaga a todo el sistema. No hay
manera de que un modulo lo sobreescriba en silencio, por dos mecanismos:

  - Los dataclasses son frozen: intentar `CONFIG.bci.fs = 512` lanza
    excepcion.
  - La validacion automatica al importar avisa de incoherencias.

Para probar una variante sin tocar el archivo:

    cfg = CONFIG.copia_con(bci=dict(n_conf=4))


LAS DOS VENTANAS --- no confundirlas
------------------------------------
El sistema responde dos preguntas distintas, en paralelo, sobre la misma
senal:

  TASM  (750 ms, paso 50 ms)  ->  ?el usuario esta mirando algun estimulo?
                                  Responde IC / TR / Idle

  FBCCA (750 ms)              ->  ?cual de los cuatro esta mirando?

La "racha de confirmacion" son n_conf ventanas consecutivas de TASM en
estado IC. Con n_conf = 8 y paso de 50 ms son 400 ms.


HALLAZGOS DEL DESARROLLO
------------------------
Durante la construccion de este bloque se detectaron tres errores de
formulacion. Se documentan porque explican por que el codigo esta como esta.

1. PARAMETROS DE VENTANA INCORRECTOS
   Una version inicial usaba ventana de 250 ms y paso de 125 ms. Los valores
   correctos, tomados de la Bitacora TASM, son 750 ms y 50 ms.

   Importa porque la Linea 1 descarto explicitamente la ventana de 1000 ms:
   aunque mejora AUC y accuracy, el F1 de la clase TR se desploma de 0.313 a
   0.150. Con TR de ~460 ms, una ventana de 1000 ms es mas larga que la
   clase que pretende medir.

2. LA METRICA CONTABA LAS SALVAGUARDAS
   N_FP_TR sumaba las transiciones forzadas por el tope de traslacion y el
   watchdog. Como esas se disparan igual en ambas condiciones
   experimentales, enmascaraban por completo el efecto que se quiere medir.
   Ahora solo se cuentan transiciones causadas por comando BCI.

3. EL MODELO DE ERROR ERA FISICAMENTE INCORRECTO
   El mock sorteaba errores independientes por ventana, con frecuencia
   elegida al azar entre las cuatro. Con eso, la probabilidad de acumular
   una racha espuria completa es p^n * (1/K)^(n-1): practicamente cero, y
   por tanto no habia diferencia posible entre condiciones.

   Durante una transicion real la senal es una mezcla decreciente del
   estimulo que se abandona y creciente del que se adquiere. Un clasificador
   que se confunda lo hace de forma SOSTENIDA y sobre una frecuencia
   CONSISTENTE. El mock ahora modela episodios, no ventanas sueltas.


LIMITACION CONOCIDA DEL HITO DE VIABILIDAD
------------------------------------------
Al correr `run.py viabilidad` observaras que el FPR en TR si difiere
claramente entre condiciones (del orden de 9% con TASM frente a 33% con el
baseline), pero el numero de comandos espurios enclavados es cero o casi
cero en AMBAS.

La causa: los trials del simulador duran unos 40 segundos y el usuario
simulado es eficiente, de modo que se producen pocas transiciones de mirada.
Pocas transiciones significan pocas oportunidades de error, y una diferencia
que existe pero no llega a manifestarse en eventos contables.

Esto NO significa que TASM no sirva. Significa que el diseno experimental
debe garantizar suficientes transiciones por trial. Dos consecuencias
practicas:

  - El protocolo de desvios inducidos (que el experimentador instruya
    desvios deliberados de mirada en instantes marcados) no es un extra: es
    necesario para que la metrica tenga eventos que contar.

  - Conviene verificar con datos reales de la Linea 1 si n_conf = 8 es el
    valor adecuado. Usa `run.py barrido` para ver el compromiso.

Este es exactamente el tipo de problema que un hito de viabilidad temprana
existe para detectar: se descubrio en simulacion, en horas, y no despues de
meses de trabajo en hardware y sesiones con sujetos.


PARAMETROS QUE NO DEBEN CAMBIARSE SIN JUSTIFICACION
---------------------------------------------------
  rho_safe = 0.15 m        Ultimo mecanismo de proteccion antes de colision
  u_max = 0.15 m/s         Derivado de u * T_d como cota de seguridad
  canales (orden)          Los filtros espaciales de TASM asumen ese orden
  notch = 60 Hz            Red peruana, no 50 Hz
  banda = [5, 90] Hz       Preserva el 4o armonico de 15.2 Hz


PARAMETROS ABIERTOS
-------------------
  n_conf                   Provisional. Debe fijarse con datos de la Linea 1
  TAU_PLANTA (simulator2d) Orientativo. Identificar en el robot real


QUE VIENE DESPUES
-----------------
  Bloque 2: estimulo SSVEP (PsychoPy, generacion sinusoidal a 60 Hz)
  Bloque 3: adquisicion EEG (g.USBamp)
  Bloque 4: capa ROS2
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

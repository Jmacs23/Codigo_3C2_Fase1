===========================================================================
  BLOQUE 6 --- integracion_F1
  Integracion con TASM real, regresion y experimentos
===========================================================================

AVISO
-----
Este bloque tiene dos partes con distinto grado de verificacion:

  Interfaz, fabrica, verificacion    VERIFICADAS. 41 pruebas automatizadas.
  de contrato, regresion,
  experimentos y registro

  Adaptador a TASM real              ESQUELETO. El paquete de la Linea 1 no
                                     estaba disponible. Las llamadas estan
                                     marcadas ">>> COMPLETAR".

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo, y que intentaste.


LA IDEA CENTRAL: UN SISTEMA, UN PARAMETRO
------------------------------------------
No hay una version "con mock" y otra "con TASM real". Hay UN sistema y un
parametro que decide de donde viene el estado cognitivo:

    config.tasm.fuente = "mock"      desarrollo
    config.tasm.fuente = "dataset"   TASM real sobre datos grabados
    config.tasm.fuente = "real"      TASM real en linea

La FSM, el arbitraje, los nodos ROS2 y la Etapa 2 NO se enteran de cual esta
corriendo. Todos consumen el mismo contrato.

Esa indiferencia no es casual: es la razon por la que el contrato TASMState
se definio en el Bloque 1, antes de que TASM existiera. Si cada bloque
hubiera inventado su propio formato, integrarlos ahora seria un trabajo de
semanas.


LAS TRES ETAPAS DE USO
----------------------
  1. AHORA (mock)
     Desarrollar y probar el sistema robotico completo sin EEG, sin sujeto
     y sin depender de la Linea 1. Es lo que se ha usado en los bloques 1
     a 5.

  2. CUANDO LA PoC ENTREGUE AUC > 0.75 (dataset)
     TASM real sobre datos grabados. Verifica la INTEGRACION en condiciones
     reproducibles.

  3. DESPUES (real)
     Con sujeto y amplificador.

EL PASO 2 NO ES OPCIONAL. Pasar directamente de mock a sujeto mezclaria dos
fuentes de error: si algo falla, no se sabria si es la integracion o TASM.
Con datos grabados se aisla la integracion; cuando eso funciona, lo unico
nuevo al traer un sujeto es el sujeto.


QUE CONTIENE
------------
  config.py             Bloques 1 a 5 MAS la seccion TASM
  tasm_interface.py     Contrato, fabrica y verificacion
  experimentos.py       Regresion, experimentos A y B, registro
  logica/               Modulos heredados de los bloques anteriores
  run_integracion.py    Punto de entrada
  tests/                41 pruebas


ORDEN DE TRABAJO
----------------
  python run_integracion.py config      Estado de la integracion
  python run_integracion.py contrato    COMPUERTA del bloque
  python run_integracion.py regresion   Compara dos fuentes
  python run_integracion.py protocolo   Protocolo de sesion con sujeto
  python run_integracion.py migracion   Como pasar de mock a real
  python run_integracion.py expA        Experimento A (simulado)
  python run_integracion.py expB        Experimento B (simulado)
  python run_integracion.py test        Las 41 pruebas


LA VERIFICACION DE CONTRATO NO ES BUROCRACIA
---------------------------------------------
Un desajuste de contrato NO da error. Produce un sistema que corre, da
numeros plausibles, y ninguno significa lo que uno cree.

Ejemplos reales de lo que puede pasar al conectar TASM:

  - TASM devuelve el estado como entero (0,1,2) y el sistema espera el
    tipo enumerado
  - El indice de frecuencia empieza en 1 en vez de en 0, y todos los
    comandos apuntan a la frecuencia equivocada
  - lambda_bci viene en porcentaje (0-100) en vez de en [0,1]
  - El orden de las frecuencias no coincide con config.bci.frecuencias
  - El modelo nunca reporta TR, y nadie se da cuenta

El verificador detecta los cinco. Ejecutalo ANTES de correr nada:

    python run_integracion.py contrato

Y con TASM real, pasandole ventanas de EEG de verdad:

    ver.verificar(fuente, generar_ventana=lector_de_eeg)


POR QUE HACEN FALTA TESTS DE REGRESION
---------------------------------------
Cuando se sustituya el mock por TASM real, los resultados van a cambiar. La
pregunta es POR QUE.

Hay dos causas y hay que poder distinguirlas:

  (a) TASM real se comporta distinto del mock. Es lo esperado y lo
      interesante: el mock era una hipotesis sobre como se comportaria.

  (b) La integracion rompio algo. Un campo mal convertido, un indice
      desplazado, un reinicio que falta.

Sin regresion, un fallo de tipo (b) aparece como "TASM funciona peor de lo
esperado", y se buscaria el problema en el sitio equivocado durante semanas.

EL TEST NO COMPARA VALORES, que cambian legitimamente. Compara PROPIEDADES
ESTRUCTURALES que deben cumplirse con cualquier fuente:

  1. La FSM no transiciona con TR sostenido
     (es la propiedad central del trabajo)
  2. La FSM enclava tras n_conf ventanas de IC
     (si no, el sistema no aceptaria ningun comando)
  3. La Etapa 2 sigue requiriendo ceil(log2 N) decisiones
  4. La fuente produce las tres clases

Si esas propiedades se rompen, el problema es de INTEGRACION.


QUE ESPERAR QUE CAMBIE AL PASAR A TASM REAL
--------------------------------------------
  FPR en TR       Probablemente SUBA. El mock usa 0.089, tomado del
                  benchmark de la Linea 1 en condiciones controladas. En
                  linea, con un sujeto moviendose y artefactos, sera peor.

  N_FP_TR         Depende de la estructura temporal de los errores. Si
                  vienen en racha (que es lo que se espera durante una
                  transicion real), subira mas de lo que sugiere el FPR.

  Decisiones E2   NO DEBE CAMBIAR. Es determinista.

Ese ultimo punto es el mas util para diagnosticar: si el numero de
decisiones deja de ser ceil(log2 N), el problema es de integracion, no de
TASM.


LOS DOS EXPERIMENTOS SON SEPARADOS
-----------------------------------
  A: TASM vs baseline binario, en 2 o 3 escenarios, con N fijo en 4.
     Responde: reduce TASM los comandos espurios? Crece el efecto con la
     dificultad?

  B: escalabilidad con N en {2,4,8}, solo con TASM, escenario fijo.
     Responde: se sostiene ceil(log2 N)? Se degrada la precision?

NO SE CRUZAN. Cruzarlos daria 90 trials por sujeto, unas 5 horas de sesion,
inviable por fatiga visual. Y no aportaria: la interaccion entre dificultad
de navegacion y numero de objetos no es una pregunta del trabajo.

Principio: variar una cosa a la vez.


LA CALIBRACION DE TASM TIENE UN PUNTO CRITICO
----------------------------------------------
IC e Idle son directos: se le pide al sujeto que mire un estimulo, o que
mire la escena.

TR no. En operacion libre no se sabe CUANDO el sujeto esta desplazando la
mirada, y sin saberlo no se puede etiquetar.

Por eso hay que INDUCIR las transiciones: se le indica al sujeto que mire el
estimulo A, y en un instante marcado se le pide que pase al B. Ese intervalo
marcado es el segmento de TR.

Sin esos segmentos no hay forma de entrenar la clase que da nombre al
trabajo. Por eso config.tasm.inducir_transiciones no debe ponerse a False.


EL REGISTRO SE ESCRIBE DESPUES DE CADA TRIAL
---------------------------------------------
No al final de la sesion. Si el programa se cae en el trial 30, los 29
anteriores estan a salvo.

El formato es JSON, una linea por trial, legible con cualquier herramienta.
Un formato binario seria mas compacto pero imposible de inspeccionar cuando
algo va mal a las once de la noche.

La cabecera guarda la configuracion completa. Sin eso, dentro de seis meses
no se sabra con que parametros se corrio.


MIGRACION PASO A PASO
---------------------
  1. Completar FuenteTASMReal en tasm_interface.py (dos puntos marcados)
  2. config.tasm.fuente = "dataset"
     python run_integracion.py contrato        <- NO seguir sin CORRECTO
  3. python run_integracion.py regresion
  4. python run_integracion.py expA / expB      <- con datos grabados
  5. config.tasm.fuente = "real"                <- ya con sujeto

El detalle completo esta en:
    python run_integracion.py migracion


PARAMETROS QUE NO DEBEN CAMBIARSE
----------------------------------
  verificar_contrato = True       Un desajuste produce fallos silenciosos
  inducir_transiciones = True     Sin TR etiquetado no hay trabajo
  fuente = "mock"                 Solo para desarrollo, NUNCA para los experimentos formales


LO QUE FALTA VERIFICAR
----------------------
  1. Que el paquete `tasm` de la Linea 1 expone la interfaz que el
     adaptador asume. Si no, adaptar el esqueleto.

  2. Que la conversion al contrato es correcta. El verificador lo comprueba,
     pero solo puede detectar lo que sabe buscar.

  3. Que el tiempo de inferencia de TASM real cabe en el periodo de ventana
     (50 ms). El presupuesto del Bloque 3 asume 30 ms para clasificacion.

  4. Que el protocolo de transiciones inducidas produce segmentos limpios.
     Si el sujeto tarda mas de lo previsto en desplazar la mirada, el
     segmento etiquetado como TR contendria parte de IC.


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

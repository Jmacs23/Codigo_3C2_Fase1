===========================================================================
  BLOQUE 4 --- ros2_F1
  Capa ROS2: nodos, mensajes y comunicacion entre las dos maquinas
===========================================================================

AVISO
-----
Este bloque tiene dos partes con distinto grado de verificacion:

  LOGICA (logica/nodos_logica.py)
    Verificada. 48 pruebas automatizadas que corren en 0.2 segundos.
    Es donde estan las decisiones: cuando activar la emergencia, como
    modular el comando, que prioridad tiene cada cosa.

  NODOS ROS2 (ros2_ws/)
    NO verificados por ejecucion: ROS2 no estaba disponible en el entorno
    donde se escribieron. Son envoltorios finos sobre la logica anterior.
    Los errores probables aqui son de plomeria (nombres de topics, tipos de
    mensaje, QoS), no de decision.

Si algo falla en el robot:
  1. Comprueba primero si el problema esta en la logica o en la plomeria.
     `python run_ros2.py mision` ejercita la logica sin ROS2.
  2. Si la logica funciona, el problema esta en el nodo.
  3. Reportalo indicando QUE TEST falla o que comando de ROS2 da error, el
     traceback completo, y la version de ROS2.


POR QUE LA LOGICA ESTA SEPARADA DE LOS NODOS
---------------------------------------------
Cada nodo de ROS2 hace dos cosas: hablar con otros nodos y decidir algo.

Separarlas tiene una ventaja practica inmediata: las 48 pruebas corren en
cualquier maquina, sin ROS2, sin Gazebo y sin robot, en menos de un segundo.
Si la logica estuviera dentro de los nodos, probar un cambio requeriria
levantar todo el sistema.

Y hay una razon de fondo: los errores dificiles no estan en la plomeria de
ROS2, que es codigo repetitivo y bien documentado. Estan en las decisiones.


QUE CONTIENE
------------
  config.py                 Bloques 1 a 3 MAS la seccion ROS2
  logica/nodos_logica.py    Logica de decision (VERIFICADA)
  logica/*.py               Modulos heredados del Bloque 1
  ros2_ws/                  Paquete ROS2 (NO verificado)
  run_ros2.py               Punto de entrada
  tests/                    48 pruebas


ORDEN DE TRABAJO
----------------
  python run_ros2.py config       Coherencia de frecuencias
  python run_ros2.py grafo        Estructura de nodos y topics
  python run_ros2.py seguridad    Histeresis del override
  python run_ros2.py navegacion   Campo potencial y verificacion de signo
  python run_ros2.py receptor     Robustez ante mensajes corruptos
  python run_ros2.py mision       COMPUERTA del bloque
  python run_ros2.py integracion  Simulacion del grafo completo
  python run_ros2.py despliegue   Instrucciones de compilacion
  python run_ros2.py test         Las 48 pruebas


EL ORDEN DE PRIORIDADES DEL ARBITRAJE
--------------------------------------
Es lo mas importante de este bloque y no es negociable:

  1. EMERGENCIA   Se impone sobre todo, incluso sobre un comando con
                  confianza maxima. Un obstaculo a 15 cm es peligroso
                  independientemente de lo que el usuario quiera.

  2. WATCHDOG     Sin mensajes de TASM, parada segura. NO se mantiene el
                  comando enclavado: si no sabemos si el usuario sigue ahi,
                  seguir moviendose no es aceptable.

  3. ETAPA 2      Robot estatico durante la seleccion.

  4. ETAPA 1      Comando del usuario, compuertado por TASM y asistido por
                  el campo potencial.

Si la emergencia no fuera lo primero, un error de TASM podria impedir que se
activara la proteccion.


LAS DOS DECISIONES DE DISENO QUE CONVIENE ENTENDER
---------------------------------------------------

1. safety_node ES REDUNDANTE A PROPOSITO

mission_node ya evalua la seguridad como parte de su arbitraje. Aun asi hay
un nodo separado que vigila el LiDAR y publica directamente en /cmd_vel.

La razon: si mission_node se colgara, dejara de publicar o tuviera un error
de logica, el robot quedaria sin proteccion. safety_node es simple a
proposito (lee, compara, publica) para que haya poco que pueda fallar.

Y NO se suscribe a /tasm/state, deliberadamente. Acoplarlo introduciria un
modo de fallo nuevo: un error del detector podria desactivar la proteccion.


2. QoS BEST-EFFORT Y COLA CORTA

En control en tiempo real, un mensaje viejo es PEOR que ningun mensaje. Si
la cola creciera, el robot actuaria sobre comandos obsoletos.

Por eso profundidad 1 y best-effort: si un mensaje se pierde, el siguiente
llega 50 ms despues y es mas reciente. Reintentar entregaria informacion
caduca.

La excepcion es el comando de emergencia de safety_node, que va por
transporte FIABLE: un comando de parada perdido es mucho peor que uno con
20 ms de retraso.


LAS DOS MAQUINAS
----------------
  WINDOWS   g.USBamp (su SDK solo existe alli), filtrado, FBCCA, TASM,
            PsychoPy con la pantalla del usuario

  LINUX     ROS2, los nodos, YOLOv8, ArUco, control del robot

Tres canales TCP entre ellas:

  :5556   Windows -> Linux   estado cognitivo (JSON por linea)
  :5555   Linux -> Windows   video de la camara (JPEG con prefijo de longitud)
  :5557   Linux -> Windows   estado del sistema para la interfaz (JSON)

UNA PROPIEDAD IMPORTANTE: el estimulo SSVEP NO depende del video. Se genera
con el contador de frames de la pantalla de Windows. Un tiron de red degrada
la experiencia visual del usuario pero no la senal.


params.yaml NO ES LA FUENTE DE VERDAD
--------------------------------------
Los valores del sistema estan en config.py. params.yaml solo contiene lo que
ROS2 necesita antes de arrancar: puertos, frecuencias de nodo y conmutadores
de fase.

Si un valor aparece en los dos sitios, MANDA config.py. Duplicar parametros
entre dos archivos es la forma mas rapida de que el sistema haga algo
distinto de lo que uno cree.


EL CONMUTADOR DE FASE
---------------------
En params.yaml, ctrl_node tiene:

    usar_lambda_bci: false

  false = Fase 1. PID simple, la referencia entra tal cual.
  true  = Fase 2. Ganancia de seguimiento ponderada por lambda_bci.

El campo lambda_bci YA VIAJA en el mensaje TASMState y se registra en ambos
casos. Cambiar esto a true no requiere tocar la cadena de comunicacion: el
bloque de codigo correspondiente esta marcado en ctrl_node.py.


DESPLIEGUE
----------
Disposicion esperada en la maquina Linux:

    ~/bci3c2/
      ros2_ws/
      logica/
      config.py

Si prefieres otra, define la variable de entorno:

    export BCI3C2_LOGICA=/ruta/a/logica

Compilacion:

    cd ~/bci3c2/ros2_ws
    colcon build --packages-select turtlebot3_bci_3c2
    source install/setup.bash

Verifica que los mensajes se generaron:

    ros2 interface show turtlebot3_bci_3c2/msg/TASMState

Si esto falla, nada mas va a arrancar.

Prueba sin robot:

    ros2 launch turtlebot3_bci_3c2 test_sin_robot.launch.py

Sistema completo:

    ros2 launch turtlebot3_bci_3c2 bci_navigation.launch.py \
        windows_ip:=192.168.1.50

safety_node arranca PRIMERO y sin retardo. Si el robot ya esta encendido y
hay un obstaculo cerca, queremos proteccion desde el primer instante.


PROBLEMAS FRECUENTES
--------------------
  "No module named turtlebot3_bci_3c2.msg"
     Falta source install/setup.bash, o la compilacion de mensajes fallo.

  "No se encontro la logica del sistema"
     El directorio logica/ no esta donde logica_bridge.py lo busca.
     Usa la variable BCI3C2_LOGICA.

  El robot no se mueve
     Comprueba en este orden:
       ros2 topic hz /tasm/state       ~20 Hz?
       ros2 topic echo /bci/command    que modo reporta?
     Si el modo es detenido_seguro, no llegan mensajes de Windows.
     Si es emergencia, el LiDAR ve algo cerca.

  El robot se PEGA a las paredes en vez de evitarlas
     El signo del gradiente del campo potencial esta invertido. Ver
     LogicaNavegacion.fuerza_repulsiva. Es un error facil de cometer al
     derivarlo, y la Linea 2 lo tuvo en su formulacion publicada.

  El movimiento va a tirones cerca de un obstaculo
     Probablemente se desactivo la histeresis del override. Sin ella, un
     obstaculo justo en el umbral hace oscilar el sistema varias veces por
     segundo.


LO QUE FALTA VERIFICAR EN EL ROBOT
-----------------------------------
  1. Que los topics se publican con los nombres y tipos correctos.
  2. Que la frecuencia real de cada nodo es la esperada
     (ros2 topic hz /tasm/state).
  3. Que la latencia entre las dos maquinas se parece a los 20 ms
     presupuestados. Si fuera mucho mayor, revisar el watchdog.
  4. Que el orden de arranque funciona: safety_node debe estar activo antes
     que ctrl_node.
  5. Que el signo del campo potencial es correcto sobre el robot real.


QUE VIENE DESPUES
-----------------
  Bloque 5: vision (YOLOv8 + ArUco, transicion de etapa, aproximacion)
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

===========================================================================
  BLOQUE 5 --- vision_F1
  Vision: deteccion de objetos, transicion de etapa y aproximacion
===========================================================================

AVISO
-----
Este bloque tiene tres partes con distinto grado de verificacion:

  ArUco                 VERIFICADO POR EJECUCION. OpenCV con soporte aruco
                        estaba disponible, asi que las pruebas generan
                        marcadores, los detectan y comprueban la pose 3D.
                        Es el unico modulo de todo el proyecto donde la
                        libreria real se ha ejercitado de verdad.

  Fusion, transicion,   VERIFICADAS con 52 pruebas automatizadas.
  aproximacion

  YOLO                  NO verificado. ultralytics no estaba disponible.
                        La clase es un esqueleto con las llamadas marcadas
                        ">>> COMPLETAR".

  e2_node (ROS2)        NO verificado, como el resto de nodos.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo, y que intentaste.


QUE CONTIENE
------------
  config.py            Bloques 1 a 4 MAS la seccion Vision
  aruco_detector.py    Deteccion de marcadores y pose 3D (VERIFICADO)
  vision_fusion.py     Fusion YOLO+ArUco, transicion, aproximacion
  ros2_nodo/e2_node.py Nodo ROS2 (copiar al paquete del Bloque 4)
  logica/              Modulos heredados de los bloques anteriores
  run_vision.py        Punto de entrada
  tests/               52 pruebas


INSTALACION
-----------
  pip install numpy scipy pytest opencv-contrib-python

IMPORTANTE: opencv-contrib-python, no opencv-python a secas. El modulo aruco
esta en el paquete contrib.

  pip install ultralytics     (para YOLO, opcional)


ORDEN DE TRABAJO
----------------
  python run_vision.py config       Coherencia con el resto del sistema
  python run_vision.py marcadores   Genera los marcadores para IMPRIMIR
  python run_vision.py deteccion    Detecta sobre escena sintetica
  python run_vision.py alcance      Precision segun distancia
  python run_vision.py fusion       Fusion ArUco + YOLO
  python run_vision.py transicion   COMPUERTA del bloque
  python run_vision.py aproximacion Control de aproximacion final
  python run_vision.py calibracion  Como calibrar la camara
  python run_vision.py test         Las 52 pruebas


POR QUE DOS DETECTORES Y NO UNO
--------------------------------
No es redundancia. Resuelven problemas distintos y ninguno cubre al otro.

  ArUco aporta:
    - IDENTIDAD UNIVOCA. Cada marcador tiene un numero; dos objetos
      identicos no se confunden nunca.
    - POSE 3D sin camara de profundidad. Es lo que permite la condicion de
      transicion y la aproximacion final.
    - DISCRIMINACION objeto/obstaculo. Los obstaculos no llevan marcador.
      Sin ambiguedad y sin entrenar ninguna clase.

  YOLO aporta:
    - RECUADROS DE OBJETOS REALES, que es lo que el usuario ve parpadear.
      Conserva la validez ecologica: la persona ve munecos, no cuadrados en
      blanco y negro.

Sin YOLO el sistema FUNCIONA igual: la identidad y la pose las da el
marcador, y el recuadro se estima ampliandolo. Lo que se pierde es que el
recuadro encuadre el objeto completo.

Para la Fase 1 eso es aceptable. El trabajo no versa sobre
vision por computadora, y controlar esa variable es preferible a introducir
la variabilidad de un detector afinado a medias.


LAS TRES CONDICIONES DE TRANSICION
-----------------------------------
Pasar de navegacion a seleccion exige las tres SIMULTANEAMENTE:

  1. DISTANCIA        Al menos 2 objetos a menos de 1.8 m.
  2. SIN OBSTACULOS   Nada intermedio en el sector frontal de 40 grados.
  3. PERMANENCIA      El robot lleva 2 s detenido.

LA TERCERA ES LA MAS IMPORTANTE y conviene entender por que. Sin ella, el
falso positivo mas probable seria este: el usuario para a mitad de camino
para evaluar el entorno, y el sistema lo interpreta como que ha llegado.
Con la exigencia de permanencia, una parada momentanea no basta.

Ademas, el contador se REINICIA si el robot se mueve. Detenerse un segundo,
avanzar un poco y volver a detenerse no acumula.

La transicion es AUTOMATICA, sin confirmacion adicional: el usuario ya
expreso su intencion al detenerse frente a los objetos, y el cambio de
interfaz es autoevidente. Pero es REVERSIBLE: el comando de parar en la
etapa 2 devuelve el sistema a navegacion.


LO QUE HAY QUE MEDIR BIEN (Y CASI NADIE MIDE)
----------------------------------------------

1. EL LADO DEL MARCADOR IMPRESO

Las impresoras escalan por defecto. Al imprimir, elige "tamano real" o
"100%", NUNCA "ajustar a pagina".

Despues MIDE con una regla el marcador impreso, sin contar el borde blanco.
Si no coincide con config.vision.lado_marcador, corrige el valor de config.

Un error del 5% aqui se traduce en un error del 5% en TODAS las distancias
que el sistema calcule. Y de esas distancias depende la condicion de
transicion.

2. LOS INTRINSECOS DE LA CAMARA

Los valores por defecto son NOMINALES, derivados del campo visual declarado.
Sirven para desarrollar, pero la distorsion de lente de las camaras pequenas
es apreciable y produce error SISTEMATICO.

`python run_vision.py calibracion` da el procedimiento completo y el codigo
de referencia. Tras calibrar, poner calibracion_verificada = True para que
el sistema deje de avisar.

COMO SABER SI QUEDO BIEN: coloca un marcador a 1.00 m medido con cinta y
comprueba que el sistema reporta esa distancia con menos de 2 cm de error.
Repitelo a 0.5 y 1.5 m. Si el error crece de forma sistematica con la
distancia, el problema esta en lado_marcador, no en la calibracion.


PRECISION MEDIDA (sintetica, sin calibrar)
-------------------------------------------
    distancia    error
      0.5 m       6 mm
      1.0 m      13 mm
      1.5 m      36 mm
      1.8 m     104 mm

El error crece con la distancia y no es casual: a mayor distancia el
marcador ocupa menos pixeles, y un error de un pixel en la deteccion de
esquinas se traduce en mas milimetros de profundidad.

Con la camara real habra ademas error sistematico por la distorsion. Por eso
la calibracion no es opcional.

Un efecto util: la fisica de la deteccion actua como FILTRO DE DISTANCIA.
Mas alla de 1.9 m el marcador de 45 mm simplemente no se decodifica, antes
de aplicar ningun umbral.


COMPLETAR YOLO
--------------
En vision_fusion.py, la clase DetectorYOLO tiene dos puntos marcados
">>> COMPLETAR":

  1. Cargar el modelo en __init__:
       from ultralytics import YOLO
       self._modelo = YOLO(config.vision.modelo_yolo)

  2. La inferencia en detectar(). El codigo esta escrito en comentarios.

QUE NO CAMBIAR: el contrato. detectar() devuelve una lista de DeteccionYOLO
con bbox en pixeles de la imagen ORIGINAL. Si el modelo trabaja a otra
resolucion, la conversion va DENTRO de la clase.

SOBRE EL ENTRENAMIENTO: el modelo preentrenado reconoce las 80 clases de
COCO, que incluyen algunas utiles (bottle, cup, teddy bear). Para objetos
fuera de esas clases hay dos opciones:

  a) Afinar con imagenes propias. Da generalidad pero exige etiquetar unas
     cien imagenes.
  b) Aceptar que YOLO no reconozca la clase y usar el recuadro generico.
     Como la identidad la da el ArUco, es suficiente.

Para Fase 1 basta la opcion (b).


DESPLEGAR EL NODO ROS2
-----------------------
1. Copia ros2_nodo/e2_node.py a:
     ros2_ws/src/turtlebot3_bci_3c2/turtlebot3_bci_3c2/

2. Copia aruco_detector.py y vision_fusion.py junto a logica/

3. Anade e2_node.py a la lista de install(PROGRAMS ...) del CMakeLists.txt

4. Recompila:
     colcon build --packages-select turtlebot3_bci_3c2

5. Anade el nodo al launch, o arrancalo suelto:
     ros2 run turtlebot3_bci_3c2 e2_node.py


LA COMPUERTA DE TASM TAMBIEN APLICA EN LA ETAPA 2
--------------------------------------------------
Cada voto de la busqueda binaria exige estado IC confirmado durante n_conf
ventanas, igual que en la Etapa 1.

Durante TR e Idle la iteracion espera sin registrar voto. Si se agota el
timeout, la iteracion SE REPITE en lugar de avanzar con un voto dudoso.

Esa decision importa: un error aqui es IRRECUPERABLE. Si el sistema elige el
grupo equivocado, el objeto correcto desaparece de la lista y ya no se puede
alcanzar. Es preferible repetir.


LA APROXIMACION USA ODOMETRIA, NO VISION CONTINUA
--------------------------------------------------
La pose del objetivo se mide UNA VEZ, al seleccionarlo, y se navega hacia
ese punto con realimentacion de odometria.

Refinar continuamente con realimentacion visual (visual servoing)
compensaria la deriva odometrica y daria mejor precision final. Se identifica
como extension natural y NO se implementa: el trabajo versa sobre
arbitracion cognitiva, y anadirlo mezclaria dos contribuciones distintas.

El controlador es PROPORCIONAL, no PID. La maniobra es corta y termina en
reposo; un termino integral acumularia error durante la aproximacion y daria
sobreimpulso justo al final, que es lo peor cuando el robot se acerca a un
objeto.


PARAMETROS QUE NO DEBEN CAMBIARSE SIN JUSTIFICACION
---------------------------------------------------
  lado_marcador              Debe coincidir con el marcador IMPRESO medido
  umbral_distancia_transicion  Debe superar la distancia que exige N=8
  permanencia_detenido       Evita el falso positivo de la parada momentanea
  min_objetos_transicion     Con menos de 2 no hay decision binaria posible


LO QUE FALTA VERIFICAR EN EL LABORATORIO
-----------------------------------------
  1. Que los marcadores se detectan con la camara real, con la iluminacion
     del laboratorio.

  2. Que la ALTURA del marcador cae dentro del campo visual. La camara del
     TurtleBot3 va montada baja (~15-20 cm). Hay que medir a que altura
     queda el centro del encuadre a 1.8 m y colocar los marcadores ahi.
     Si los objetos de 20 cm apoyados en el suelo quedaran fuera, la
     solucion es elevarlos sobre una plataforma, no mover la camara.

  3. Que la distancia medida coincide con la real. Calibrar primero.

  4. Que el desenfoque de movimiento no impide la deteccion cuando el robot
     se acerca. Si fuera un problema, bajar la velocidad de aproximacion.

  5. Que los obstaculos NO producen detecciones espurias de marcador. No
     deberia haber nada en el entorno que se parezca a un ArUco.


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

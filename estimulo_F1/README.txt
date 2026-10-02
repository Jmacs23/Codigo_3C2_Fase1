===========================================================================
  BLOQUE 2 --- estimulo_F1
  Estimulo SSVEP e interfaz visual
===========================================================================

AVISO
-----
Este codigo es un punto de partida funcional. Las 44 pruebas automatizadas
pasan, pero NO ha sido probado en el hardware del laboratorio.

Hay una excepcion importante: `interface.py` es el unico archivo del bloque
que NO se ha podido verificar por ejecucion, porque PsychoPy necesita una
pantalla real. Es por tanto el archivo con mayor probabilidad de necesitar
ajustes.

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo, y que intentaste. Si es de PsychoPy, incluye la
     version.


QUE CONTIENE
------------
  config.py           Igual que el Bloque 1 MAS la seccion EstimuloConfig
  stimulus.py         Generacion de luminancia y textura (numpy puro)
  video_source.py     Fuente de video: sintetica, archivo o TCP
  interface.py        Interfaz PsychoPy (Etapa 1 y Etapa 2)
  validacion_fft.py   Protocolo de validacion espectral
  run_estimulo.py     Punto de entrada
  tests/              44 pruebas automatizadas


INSTALACION
-----------
  pip install numpy scipy pytest
  pip install psychopy          (solo en la maquina del laboratorio)
  pip install opencv-python     (para video por archivo o por red)

Todo el bloque funciona sin PsychoPy salvo la demostracion grafica. Puedes
desarrollar y verificar la logica temporal en cualquier maquina.


ORDEN DE TRABAJO
----------------
  python run_estimulo.py config       Que parametros hay
  python run_estimulo.py realizable   POR QUE se usa sinusoidal y no cuadrada
  python run_estimulo.py espectro     La senal generada es correcta
  python run_estimulo.py deriva       El problema del reloj, con numeros
  python run_estimulo.py timing       COMPUERTA del bloque
  python run_estimulo.py test         Las 44 pruebas

Y ya en la maquina del laboratorio, con pantalla:

  python run_estimulo.py demo


POR QUE MODULACION SINUSOIDAL Y NO ONDA CUADRADA
------------------------------------------------
Con encendido/apagado por frame solo son realizables las frecuencias de la
forma 60/n con n PAR:

    8.0 Hz  -> 60/8.0  = 7.50   no entero
    12.0 Hz -> 60/12.0 = 5.00   entero pero IMPAR
    14.0 Hz -> 60/14.0 = 4.29   no entero
    15.2 Hz -> 60/15.2 = 3.95   no entero

NINGUNA de las cuatro frecuencias del sistema es realizable asi.

La solucion es modular la luminancia de forma continua:

    s_k(i) = 0.5 * [1 + sin(2*pi*f_k*i/R + phi_k)]

Con eso cualquier frecuencia bajo Nyquist es realizable, porque la
informacion esta en la envolvente de luminancia y no en el instante de
conmutacion. Es el mismo metodo con que se grabo el dataset Benchmark, lo
que garantiza que los componentes preentrenados con el transfieran.

Ejecuta `python run_estimulo.py realizable` para ver la tabla completa.


EL ERROR MAS FACIL DE COMETER
-----------------------------
Calcular la fase con el tiempo transcurrido en lugar del contador de frames.

    MAL:   angulo = 2*pi*f*time.time()
    MAL:   t += deltaTime;  angulo = 2*pi*f*t
    BIEN:  angulo = 2*pi*f*(frame/refresh)

Con un error de solo 0.1 ms por frame, la deriva de fase alcanza 265 grados
en medio minuto. Un trial de cinco minutos tendria la senal completamente
descorrelacionada de la referencia.

Ejecuta `python run_estimulo.py deriva` para ver la magnitud del problema.

El generador de este bloque solo expone la version con contador. Si en el
codigo de PsychoPy ves algo parecido a la version incorrecta, esta mal.


DISENO DEL ESTIMULO
-------------------
Los valores provienen del estudio de Meng et al. (2023), que
barrio forma, distribucion y densidad con 12 sujetos:

  Forma        Cuadrado, NO damero. El cuadrado supera al damero en unos 30
               puntos de precision (p<0.001). La causa es la cancelacion de
               fase entre celdas vecinas desfasadas pi en el damero: sus
               respuestas se restan en corteza.

  Distribucion Aleatoria, no uniforme. Menor fatiga Y mayor precision.

  Densidad     60%. Entre 100% y 60% la precision no cambia
               significativamente (92.7% a 90.1%, p=0.148) mientras la
               fatiga cae de 8.75 a 5.58 en escala 1-10. Por debajo del 60%
               la precision se desploma (p<0.005). Es el BORDE INFERIOR de
               la meseta.

Ventaja adicional para este sistema: al 60% los huecos entre celdas dejan
ver el video de la camara a traves del estimulo. Eso resuelve la tension
entre estimulo eficaz y visibilidad del entorno mejor que bajar la opacidad
global, que reduciria el contraste y con el la amplitud de la respuesta.


LAS DOS MAQUINAS
----------------
  Windows   g.USBamp (su SDK solo existe para Windows), FBCCA, TASM,
            PsychoPy con la pantalla que ve el usuario

  Linux     ROS2, los siete nodos, YOLOv8, ArUco, control del robot

El video de la camara viaja de Linux a Windows por TCP. La clase FuenteTCP
implementa el lado receptor; el emisor va en el Bloque 4.

UNA PROPIEDAD IMPORTANTE DEL DISENO: el video de fondo y el estimulo son
procesos INDEPENDIENTES. El estimulo se calcula del contador de frames de la
pantalla, no de la llegada del video. Consecuencia: un tiron de red degrada
la experiencia visual del usuario pero NO la senal SSVEP. Si estuvieran
acoplados, cada paquete perdido corromperia el estimulo.


PROTOCOLO DE VALIDACION
-----------------------
Se ejecuta UNA VEZ por sujeto, al caracterizar el estimulo. Mide la cadena
completa pantalla -> ojo -> corteza -> EEG, que es lo que importa.

  1. Barrido de frecuencia   Pico en f_k y en 2*f_k
  2. Amplitud relativa       Ninguna frecuencia sistematicamente debil
  3. Ancho de banda          Un pico ensanchado delata jitter
  4. Estatica vs movimiento  CRITICO. En los TRES escenarios
  5. Frames perdidos         Por escenario. Exclusion si >1%
  6. Interferencia           Los otros estimulos no deben dominar
  7. Fase                    Una fase que deriva delata uso de reloj

EL PASO 4 ES EL MAS IMPORTANTE Y CONVIENE ENTENDER POR QUE. En el estudio de
Meng et al. los pixeles apagados eran fondo negro CONTROLADO. Aqui son
transparentes y dejan pasar video brillante, texturado y en movimiento.

Eso convierte el contraste de borde en variable no controlada. Si la calidad
del SSVEP se degradara mas en un escenario visualmente denso que en uno
vacio, parte del efecto atribuido a la arbitracion cognitiva seria en
realidad un artefacto del estimulo, y la hipotesis de escalamiento quedaria
comprometida.

Por eso el paso 4 se ejecuta en los tres escenarios y se REPORTA LA TABLA.
Si las SNR son comparables, el asunto queda cerrado y se cita como control
de validez, que es un dato que ningun trabajo comparable ofrece. Si
difieren, la mitigacion es renderizar los pixeles inactivos en negro con
opacidad 70-80% en lugar de totalmente transparentes.


LO QUE FALTA VERIFICAR EN EL LABORATORIO
-----------------------------------------
Estas cosas no se pueden comprobar sin el hardware:

  1. Que el monitor va REALMENTE a 60 Hz. La interfaz lo mide al arrancar y
     avisa si no coincide. Si el monitor fuera de 144 Hz y el codigo asume
     60, todas las frecuencias generadas estarian mal por un factor 2.4.

  2. Frames perdidos con video real de fondo. La carga de renderizado
     depende de la escena, y un escenario denso puede provocar perdidas que
     uno vacio no provoca.

  3. Que el tamano angular de 3 grados es correcto para la distancia y el
     monitor concretos. El calculo usa distancia_vision y ancho_pantalla_m
     de config; verifica esos dos valores con una cinta metrica.

  4. Que los cuatro estimulos son comodos de mirar y no se solapan
     visualmente con los objetos de la escena.


PARAMETROS QUE NO DEBEN CAMBIARSE SIN JUSTIFICACION
---------------------------------------------------
  refresh_hz = 60          Debe coincidir con el monitor REAL
  densidad_pixeles = 0.60  Borde inferior de la meseta de precision
  ventana_fft = 5.0 s      Con 4 s, 15.2 Hz no cae en un bin entero
  frecuencias              Estan en la grilla del Benchmark; cambiarlas
                           rompe la transferencia de TASM


QUE VIENE DESPUES
-----------------
  Bloque 3: adquisicion EEG (g.USBamp, filtrado, ventaneo)
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

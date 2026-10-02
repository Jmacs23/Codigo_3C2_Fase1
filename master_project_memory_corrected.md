# MASTER_PROJECT_MEMORY_CORRECT.md

> **Documento Maestro de Transferencia de Contexto y Continuidad Técnica Definitivo**  
> **Proyecto de Tesis Capstone:** BCI-SSVEP Asíncrono para Telepresencia Robótica  
> **Curso:** MT5003 – Proyecto Final de Ingeniería Mecatrónica II (UTEC)  
> **Fecha de Consolidación:** 2026-10-01 (Semana 9 de 16)  
> **Propósito:** Proporcionar una referencia técnica, académica, operativa y cinemática completa e inmutable para que cualquier asistente de IA o investigador pueda continuar el proyecto con cero pérdida de contexto y alineación estricta con los criterios de evaluación.

---

# 1. Resumen Ejecutivo

El presente proyecto abarca el diseño, desarrollo, integración y validación experimental de un **sistema BCI-SSVEP (Brain-Computer Interface basada en Potenciales Evocados Visuales en Estado Estacionario) asíncrono e inmune a falsos positivos durante transiciones de mirada**, destinado al control de telepresencia robótica de una plataforma móvil TurtleBot3 en entornos semiestructurados.

El problema central que aborda la investigación es la discriminación del estado cognitivo durante los intervalos de transición ocular (140 ms a 600 ms). En los BCI síncronos convencionales, el sistema impone ventanas fijas de decisión. En la conducción asíncrona de un robot, el usuario desplaza libremente la mirada entre estímulos visuales y el feed de video. Durante dicho movimiento, la señal EEG representa una combinación transitoria que los clasificadores tradicionales de dos estados (*Order / No-Order*) interpretan erróneamente como comandos válidos, enganchando órdenes espurias que causan maniobras erráticas o colisiones.

Para resolver este dilema sin requerir electrodos biológicos adicionales (EOG/EMG) ni introducir pausas artificiales, el sistema integra el módulo **TASM (Transition-Aware State Monitor)**. TASM discrimina tres estados cognitivos directamente desde la señal SSVEP: **IC** (*Intentional Control*), **TR** (*Transition*) e **Idle** (*Reposo*).

El proyecto abarca una arquitectura distribuida en dos máquinas (Windows para BCI/PsychoPy y Linux Ubuntu para ROS2/Visión), organizada en seis paquetes de código Python (`*_F1`) con **279 pruebas unitarias y de integración automáticas** que ejecutan exitosamente en entornos sintéticos.

**Estado académico de emergencia:** El proyecto se encuentra en la **Semana 9 del ciclo de 16 semanas**. El primer entregable (P1, 10%) no fue realizado y se asume una calificación de $0.0$. La prioridad crítica e inmediata es el **Entregable P2 (25% del curso, correspondiente a la Semana 9)**. El objetivo inmediato es generar la mayor cantidad de avance técnico real, cuantitativo, verificable y defendible sobre el código existente para respaldar la entrega de P2 antes de realizar las pruebas físicas en laboratorio.

---

# 2. Objetivos del Proyecto

## 2.1 Objetivo General
Desarrollar, integrar y validar experimentalmente un sistema BCI-SSVEP asíncrono e inmune a falsos positivos durante transiciones de mirada para el control de telepresencia robótica de un TurtleBot3 en entornos semiestructurados.

## 2.2 Objetivos Específicos
1. **Núcleo Algorítmico y Máquina de Estados:** Diseñar e implementar una máquina de estados finitos (FSM) con enclavamiento de comandos protegida por compuerta TASM y racha de confirmación ($n_{\text{conf}} = 8$ ventanas consecutivas IC = 400 ms), así como el algoritmo de selección de objetos por partición binaria.
2. **Estimulación Visual de Alta Precisión:** Diseñar un sistema de estimulación visual basado en modulación sinusoidal continua de luminancia a 60 Hz (densidad de píxeles del 60%) que elimine la deriva de fase temporal y permita superponer los estímulos con la transmisión de video sin corromperse ante retardos de red.
3. **Adquisición y Filtrado EEG Causal:** Construir un pipeline de filtrado de señales EEG estrictamente causal con memoria de estado persistente entre bloques (`zi`), integrando un filtro Notch ultra-estrecho ($Q = 60$) a 60 Hz para conservar el $4.^\circ$ armónico de la frecuencia de 15.2 Hz (60.8 Hz) y garantizando una latencia computacional inferior al periodo de ventana (50 ms).
4. **Capa de Control Robótico y Seguridad en ROS2:** Desarrollar el sistema de control del robot en ROS2 mediante una jerarquía de arbitraje estricta donde la seguridad física (`safety_node` con histéresis 15 cm / 20 cm) prevalezca autónomamente sobre cualquier comando BCI, incorporando asistencia de evasión mediante campos potenciales artificiales.
5. **Visión Computacional y Transición de Etapa:** Implementar la fusión de detectores visuales (ArUco para pose 3D e identidad unívoca + YOLOv8 para delimitación de objetos reales) y evaluar las 3 condiciones simultáneas para la transición automática de etapa (Distancia $< 1.8\text{ m}$, Sector frontal libre de $40^\circ$, Permanencia de $2\text{ s}$).
6. **Validación Experimental:** Diseñar y ejecutar dos protocolos experimentales formalmente separados (Experimento A: Comparación TASM vs Baseline sin transiciones; Experimento B: Escalabilidad en selección de objetos) incorporando el protocolo de desvíos inducidos de mirada para garantizar significancia estadística.

---

# 3. Contexto Académico y Situación Operativa

## 3.1 Ficha Técnica Administrativa
- **Institución:** Universidad de Ingeniería y Tecnología (UTEC).
- **Carrera:** Ingeniería Mecatrónica.
- **Curso:** MT5003 – Proyecto Final de Ingeniería Mecatrónica II (10.° Ciclo, 4 Créditos).
- **Entorno de Trabajo Local (Ruta Absoluta):** `/home/estudiante/proyecto_tesis/` (o equivalente en Windows: `C:\Users\Estudiante\proyecto_tesis\`).

## 3.2 Esquema de Evaluación y Ponderaciones

| Evaluación | Semana | Peso | Estado Real | Acción Requerida |
| :--- | :---: | :---: | :--- | :--- |
| **P1 – Primer Entregable** | 5 | 10% | **No entregado ($0.0$)** | Asumido como pérdida preventiva; enfocarse en el 90% restante. |
| **P2 – Segundo Entregable** | 9 | 25% | **EN CURSO (PRIORIDAD 1)** | Generar informe con evidencias cuantitativas de `nucleo_F1`. |
| **P3 – Informe Escrito Final**| 15 | 35% | Pendiente | Redacción en formato IMRyC / IEEE Capstone. |
| **P4 – Presentación Oral Final**| 16 | 25% | Pendiente | Sustentación ante jurado con demostración. |
| **P5 – Portafolio de Proyecto** | 16 | 5% | Pendiente | Archivos, repositorio y documentación final. |

## 3.3 Matriz de Rúbrica para Entregable P2 (Total: 20 puntos / 25% de la nota final)

```
[C1: Matemáticas y Ciencias] (5 pts)  ---> Evaluación de FBCCA, FSM, matrices de confusión y latencias.
[C2: Herramientas Modernas]  (5 pts)  ---> Uso de SciPy, PsychoPy, ROS2, pytest (279 PASS) y simulador 2D.
[C3: Normas y Estándares]    (5 pts)  ---> Arquitectura de 6 paquetes, contratos JSON (TASMState), ISO/IEEE.
[C4: Comunicación Escrita]   (5 pts)  ---> Estructura IMRyC, claridad narrativa, gráficos de resultados reales.
```

---

# 4. Especificaciones del Entorno y Hardware Real

Para eliminar ambigüedades técnicas ante futuras ejecuciones, se definen los parámetros del entorno físico del laboratorio:

## 4.1 Entorno de Software
- **Python:** 3.8.10 o superior (64-bit).
- **Librerías Clave:** `scipy>=1.7.0`, `numpy>=1.20.0`, `psychopy>=2022.2.4`, `torch>=1.12.0`, `ultralytics>=8.0.0`, `opencv-contrib-python>=4.6.0`.
- **Middleware Robótico:** ROS2 Humble Hawksbill en Ubuntu 22.04 LTS.

## 4.2 Hardware del Laboratorio
- **Monitor de Estimulación Visual:** Pantalla LCD de $24''$ con tasa de refresco fija de $60.0\text{ Hz}$ ($16.66\text{ ms}$ por cuadro).
- **Sistema EEG:** Amplificador biosensorial g.USBamp (g.tec medical engineering) con SDK `pygds` bajo Windows. 9 electrodos pasivos/activos de $Ag/AgCl$ dispuestos en la zona occipital-parietal: `[Pz, PO5, PO3, POz, PO4, PO6, Oz, O1, O2]`. Tasa de muestreo: $256\text{ Hz}$.
- **Plataforma Robótica Mobile:** TurtleBot3 Burger / Waffle equipado con Raspberry Pi 4B ($4\text{ GB}$ RAM), LiDAR LDS-01 / LDS-02 ($360^\circ$, rango $0.12\text{ m} - 3.5\text{ m}$), y cámara USB Gran Angular ($1280 \times 720$ a $30\text{ fps}$).

## 4.3 Constantes Cinemáticas de la Planta Robótica
- **Velocidad Lineal Máxima ($v_{\text{max}}$):** $0.22\text{ m/s}$.
- **Velocidad Angular Máxima ($\omega_{\text{max}}$):** $2.84\text{ rad/s}$.
- **Constante de Tiempo de la Planta ($\tau_{\text{planta}}$):** $0.15\text{ s}$.
- **Ganancias PID Angular (`ctrl_node`):** $K_p = 1.2$, $K_i = 0.05$, $K_d = 0.1$.
- **Umbrales de Freno de Seguridad:** $15\text{ cm}$ (activación de parada obligatoria) y $20\text{ cm}$ (liberación por histéresis).

---

# 5. Arquitectura Distribuida y Comunicaciones

El sistema opera distribuyendo las cargas de cómputo en dos nodos independientes conectados por red Ethernet/Wi-Fi local de alta velocidad.

## 5.1 Diagrama de Arquitectura de Red

```text
+----------------------------------------------------+       Red TCP/IP Local       +----------------------------------------------------+
|                  MÁQUINA WINDOWS                   |                              |                   MÁQUINA LINUX                    |
|                                                    |   :5556 (TASMState JSON)     |                                                    |
|  [Amplificador g.USBamp] (g.tec SDK)               | ---------------------------> |  [Nodos ROS2]                                      |
|            |                                       |                              |   - safety_node (Sobrescribe /cmd_vel)            |
|  [Filtrado Causal EEG] (SciPy sosfilt)             |   :5555 (MJPEG Camera Video) |   - mission_node (Arbitraje & FSM)                 |
|            |                                       | <--------------------------- |   - ctrl_node (PID + Campos Potenciales)            |
|  [Clasificador FBCCA + Módulo TASM]                |                              |            |                                       |
|            |                                       |   :5557 (Robot State JSON)   |  [Visión Computacional]                            |
|  [Interfaz PsychoPy] (Pantalla Usuario 60 Hz)      | <--------------------------- |   - ArUco Detector 3D (OpenCV)                     |
|    - Estímulos independientes de red               |                              |   - YOLOv8 (Detector de Objetos)                   |
+----------------------------------------------------+                              +----------------------------------------------------+
```

## 5.2 Puertos de Comunicación TCP
1. **Puerto `:5556` (Windows $\rightarrow$ Linux):** Transmisión continua de comandos BCI y estados cognitivos serializados como JSON línea por línea a una frecuencia de $20\text{ Hz}$ ($50\text{ ms}$).
2. **Puerto `:5555` (Linux $\rightarrow$ Windows):** Transmisión de la señal de video de la cámara en formato MJPEG comprimido.
3. **Puerto `:5557` (Linux $\rightarrow$ Windows):** Telemetría del robot (fase de misión, distancia a obstáculos, alertas) para renderizado de superposición (overlay) en la interfaz PsychoPy.

## 5.3 Contrato Inmutable de Estado Cognitivo (`TASMState`)
La estructura JSON intercambiada en el puerto `:5556` cumple la siguiente especificación:

```json
{
  "estado": "IC",       
  "freq_idx": 0,        
  "p_max": 0.8851,      
  "lambda_bci": 0.9210, 
  "valido": true        
}
```

- `"estado"`: Cadena restringida a `"IC"` (Intentional Control), `"TR"` (Transition), o `"Idle"` (Reposo).
- `"freq_idx"`: Índice entero del estímulo detectado ($0: 8\text{ Hz}, 1: 12\text{ Hz}, 2: 14\text{ Hz}, 3: 15.2\text{ Hz}$) o $-1$ en caso de Idle/TR.
- `"p_max"`: Confianza espectral del clasificador FBCCA en la ventana actual ($[0.0, 1.0]$).
- `"lambda_bci"`: Métrica continua ponderada de la intencionalidad del usuario ($[0.0, 1.0]$).
- `"valido"`: Booleano que indica si la ventana superó los criterios de rechazo por artefacción de amplitud.

---

# 6. Descripción Detallada de Módulos del Proyecto

El código fuente está estructurado en seis paquetes Python dentro de la raíz del proyecto:

```text
proyecto_tesis/
├── nucleo_F1/         # Algoritmos puros: FSM, FBCCA, TASM Mock, Simulador 2D
├── estimulo_F1/       # Generación de estímulos PsychoPy a 60 Hz y servidor TCP
├── eeg_F1/            # Filtrado causal IIR, buffers circulares y drivers EEG
├── ros2_F1/           # Lógica pura de control robótico y wrappers ROS2 Humble
├── vision_F1/         # Detección ArUco 3D, YOLOv8 y lógica de fusión de etapas
├── integracion_F1/    # Conector de modelo TASM real, suite de experimentos y tests
├── resultados/        # Logs de salida, tablas JSON y evidencias cuantitativas
└── reportes/          # Documentos académicos e informes IMRyC
```

### Funcionalidad por Paquete:

1. **`nucleo_F1/`**:
   - `config.py`: **Fuente Única de Verdad (SSOT).** Define frecuencias, umbrales, puertos TCP, constantes PID y rutas de archivos.
   - `fbcca.py`: Clasificador FBCCA (*Filter Bank Canonical Correlation Analysis*) con $M = 3$ subbandas espectrales.
   - `command_fsm.py`: FSM con racha de confirmación ($n_{\text{conf}} = 8$ ventanas), enclavamiento (*latch*) y paso obligatorio por `DETENIDO`.
   - `tasm_mock.py`: Generador sintético de estados cognitivos para simulación determinista de episodios.
   - `binary_search.py`: Árbol de decisión por partición binaria $O(\log_2 N)$ para selección de objetos en Etapa 2.
   - `simulator2d.py`: Entorno kinemático sintético de alta velocidad (*headless*) para evaluar ejecuciones completas de misión.

2. **`estimulo_F1/`**:
   - `stimulus.py`: Renderizado de estímulos mediante modulación sinusoidal continua basada en cuadro de pantalla ($frame$).
   - `interface.py`: Interfaz de usuario PsychoPy que superpone los 4 estímulos rectangulares con el video MJPEG del robot.

3. **`eeg_F1/`**:
   - `filters.py`: Filtro pasa-banda IIR Butterworth de $4.^\circ$ orden $[5, 90]\text{ Hz}$ con preservación de estados iniciales `zi`. Filtro Notch centrado en $60\text{ Hz}$ ($Q = 60$).
   - `acquisition.py`: Gestor de buffer circular deslizante de $750\text{ ms}$ ($192$ muestras a $256\text{ Hz}$) con avance de ventana de $50\text{ ms}$ ($12$ muestras).
   - `eeg_source.py`: Fábrica de fuentes de datos EEG (`sintetica`, `dataset`, `gUSBamp`).

4. **`ros2_F1/`**:
   - `logica/nodos_logica.py`: Lógica de navegación, cálculo de campos potenciales y arbitraje en Python puro (desacoplado de ROS2 para testing rápido).
   - `ros2_ws/src/turtlebot3_bci_3c2/`:
     - `safety_node`: Nodo de máxima prioridad que interrumpe `/cmd_vel` al detectar obstáculos $< 15\text{ cm}$.
     - `ctrl_node`: Controlador proporcional-integrativo-derivativo (PID) con anti-windup para el seguimiento de orientaciones.
     - `mission_node`: Gestor de la misión global y transiciones entre Etapa 1 y Etapa 2.

5. **`vision_F1/`**:
   - `aruco_detector.py`: Estimación de pose $3\text{D}$ ($x, y, z$, roll, pitch, yaw) mediante marcadores ArUco de la familia $4 \times 4$.
   - `vision_fusion.py`: Módulo que combina las cajas delimitadoras de YOLOv8 con la posición precisa de ArUco y valida las tres condiciones de transición de etapa.

6. **`integracion_F1/`**:
   - `tasm_interface.py`: Adaptador `FuenteTASMReal` que conecta las salidas del modelo de clasificación TASM entrenado por el laboratorio con el contrato `TASMState`.
   - `experimentos.py`: Motor automatizado para la ejecución sintética repetible de los Experimentos A y B.

---

# 7. Justificación Científica y Mecatrónica de Decisiones

1. **Selección de Frecuencias SSVEP ($8\text{ Hz}, 12\text{ Hz}, 14\text{ Hz}, 15.2\text{ Hz}$):**
   - *Fundamento:* Se escogieron para evitar interferencias por armónicos cruzados ($\text{MCD}(f_i, f_j) = 1$) y garantizar compatibilidad exacta con un monitor de $60.0\text{ Hz}$. Los periodos en fotogramas de pantalla son enteros o semienteros exactos ($8\text{ Hz} = 7.5\text{ frames}$, $12\text{ Hz} = 5\text{ frames}$, $14\text{ Hz} \approx 4.28\text{ frames}$, $15.2\text{ Hz} \approx 3.94\text{ frames}$), maximizando la respuesta en la corteza visual primaria (V1).
2. **Densidad de Píxeles del 60% en Estímulos:**
   - *Fundamento:* Utilizar una retícula con $60\%$ de píxeles activos en lugar de un bloque sólido al $100\%$ reduce drásticamente la fatiga visual (*asthenopia*) del usuario durante sesiones prolongadas y permite la transparencia (*overlay*) sobre el video en vivo sin ocluir los obstáculos del entorno.
3. **Filtro Notch Ultra-Estrecho ($Q = 60$) a 60 Hz:**
   - *Fundamento:* El cuarto armónico de la frecuencia de comando de $15.2\text{ Hz}$ es $4 \times 15.2\text{ Hz} = 60.8\text{ Hz}$. Un filtro Notch convencional ($Q = 30$) posee un ancho de banda de $2\text{ Hz}$ ($59 - 61\text{ Hz}$), lo cual atenuaría severamente el armónico evocada de $60.8\text{ Hz}$. Elevar el factor de calidad a $Q = 60$ reduce el ancho de banda a $1\text{ Hz}$ ($59.5 - 60.5\text{ Hz}$), filtrando la red eléctrica sin distorsionar la respuesta SSVEP.
4. **Ventana de Confirmación de Racha ($n_{\text{conf}} = 8$ ventanas = 400 ms):**
   - *Fundamento:* Las transiciones de mirada involuntarias duran fisiológicamente entre $140\text{ ms}$ y $600\text{ ms}$ ($3$ a $12$ ventanas de $50\text{ ms}$). Exigir $8$ ventanas consecutivas clasificadas como IC asegura que cualquier clasificación espuria que ocurra a mitad del movimiento ocular sea descartada reiniciando el contador, eliminando falsos positivos.
5. **Comportamiento de Enclavamiento (*Latch*) y Conducción en *Idle*:**
   - *Fundamento:* Para evitar la carga cognitiva de mantener fijación visual continua en las luces mientras el robot avanza, la FSM engancha la orden de movimiento al confirmar la racha. Posteriormente, cuando el usuario desvía la mirada para monitorear el camino en el video (estado *Idle*), el robot **mantiene su velocidad continua**. El movimiento solo cesa si se detecta la frecuencia de Parar ($15.2\text{ Hz}$) o una alerta de seguridad del LiDAR.
6. **Paso Obligatorio por Estado `DETENIDO`:**
   - *Fundamento:* La FSM rechaza transiciones directas entre direcciones opuestas (ej. de Avanzar a Girar). Es obligatorio pasar por `DETENIDO`. Esta restricción protege la transmisión mecánica del TurtleBot3 frente a cambios bruscos de inercia y fuerza la existencia de eventos de transición claros para la evaluación experimental.
7. **Histéresis del Freno de Seguridad (15 cm / 20 cm):**
   - *Fundamento:* Dada la velocidad lineal $v_{\text{max}} = 0.22\text{ m/s}$ y la constante de tiempo de frenado $\tau_{\text{planta}} = 0.15\text{ s}$, la distancia de parada dinámica es $d_{\text{frenado}} \approx 0.22 \times 0.15 \approx 3.3\text{ cm}$. Un umbral de $15\text{ cm}$ provee un factor de seguridad $> 4\times$. La histéresis hasta $20\text{ cm}$ evita oscilaciones encendido/apagado (*chattering*) ante ruido numérico en la nube de puntos del LiDAR.

---

# 8. Ecuaciones Matemáticas Fundamentales

### 8.1 Modulación Visual Sinusoidal Continua
La luminancia $s_k(i)$ del $k$-ésimo estímulo visual en el fotograma $i$ se calcula como:

$$s_k(i) = 0.5 \cdot \left[ 1 + \sin\left(2\pi \cdot f_k \cdot \frac{i}{f_{\text{refresh}}} + \phi_k\right) \right]$$

donde $f_k \in \{8, 12, 14, 15.2\}\text{ Hz}$, $f_{\text{refresh}} = 60.0\text{ Hz}$, y $\phi_k$ es la fase inicial.

### 8.2 Clasificación FBCCA (Filter Bank Canonical Correlation Analysis)
Para cada subbanda $m \in \{1, \dots, M\}$, se computa la correlación canónica $\rho_k^{(m)}$ entre la señal EEG filtrada $X^{(m)} \in \mathbb{R}^{N_c \times N_s}$ y la matriz de referencia armónica $Y_k \in \mathbb{R}^{2N_h \times N_s}$:

$$\rho_k^{(m)} = \max_{W_x, W_y} \frac{W_x^T X^{(m)} Y_k^T W_y}{\sqrt{W_x^T X^{(m)} (X^{(m)})^T W_x \cdot W_y^T Y_k Y_k^T W_y}}$$

La métrica combinada $R(f_k)$ se obtiene mediante la suma ponderada de subbandas:

$$R(f_k) = \sum_{m=1}^{M} w(m) \cdot \left( \rho_k^{(m)} \right)^2, \quad \text{con } w(m) = m^{-1.25} + 0.25$$

### 8.3 Involución del Filtro Causal IIR con Conservación de Estado
La ecuación en diferencias para el filtrado paso-banda causal $H(z)$ sobre la muestra $n$ usando condiciones de contorno persistentes `zi` es:

$$y[n] = \frac{1}{a_0} \left( \sum_{m=0}^{P} b_m x[n-m] - \sum_{m=1}^{Q} a_m y[n-m] \right), \quad \text{con } z_{\text{init}} = \text{sosfilt\_zi}(b, a)$$

---

# 9. Plan de Ataque y Ejecución para Entregable P2

Dado que la prioridad absoluta es la calificación de **P2 (Semana 9, 25% del curso)**, el plan de trabajo se enfoca en ejecutar el código existente en `nucleo_F1`, extraer la evidencia cuantitativa de las 279 pruebas y consolidar el informe.

## 9.1 Lista Maestra de Comandos de Ejecución CLI

Para generar todos los logs y métricas necesarias para el informe de P2, se deben ejecutar los siguientes comandos desde la terminal dentro de la máquina local:

```bash
# 1. Verificación de integridad de entorno y suite de pruebas unitarias
pytest --verbose

# 2. Navegar al paquete núcleo
cd nucleo_F1

# 3. Verificación de configuración y prueba humo de comunicación
python run.py config
python run.py smoke
python run.py test

# 4. Generación de evidencias cuantitativas de la FSM y FBCCA (Criterio C1)
python run.py fsm
python run.py fbcca
python run.py seleccion

# 5. Ejecución de simulaciones de trayectorias y análisis de viabilidad (Criterio C2)
python run.py trial
python run.py viabilidad
python run.py barrido
```

Todos los resultados tabulares y JSON generados automáticamente por estos comandos se guardarán en la carpeta `/resultados/E1/`.

## 9.2 Esquema de Mapeo entre Resultados y Rúbrica P2

1. **Evidencia para Criterio C1 (Matemáticas y Ciencias):**
   - Incluir en el informe la matriz de confusión de FBCCA, las curvas de sensibilidad/especificidad bajo distintas SNR y los tiempos de latencia de la FSM extraídos de `python run.py fbcca` y `python run.py fsm`.
2. **Evidencia para Criterio C2 (Herramientas Modernas):**
   - Adjuntar captura y datos de la tasa de acierto del $100\%$ sobre las $279$ pruebas de `pytest`, junto con los gráficos de trayectoria del robot en el simulador $2\text{D}$ obtenidos de `python run.py trial`.
3. **Evidencia para Criterio C3 (Normas y Estándares):**
   - Presentar el diagrama de arquitectura de $6$ paquetes, el esquema del contrato JSON `TASMState` y la justificación de la separación de capas ROS2 respetando la norma IEEE para sistemas embebidos de tiempo real.
4. **Evidencia para Criterio C4 (Comunicación Escrita):**
   - Redactar el informe siguiendo la estructura formal IMRyC (Introducción, Métodos, Resultados y Discusión) con gráficos exportados en formato vectorial o PNG de alta resolución.

---

# 10. Análisis de Riesgos y Trampas Técnicas

| Riesgo / Trampa Técnica | Impacto | Causa Raíz | Estrategia de Mitigación |
| :--- | :---: | :--- | :--- |
| **Acumulación de Desfase Temporal** | **Crítico** | Calcular la fase del estímulo visual con `time.time()` en lugar del contador de cuadros del monitor ($frame / f_{\text{refresh}}$). | Renderizado numérico estricto en PsychoPy usando el contador entero de refrescos de pantalla. |
| **Frecuencia de Refresco Incompatible** | **Alto** | Ejecutar la estimulación en monitores de $144\text{ Hz}$ o $75\text{ Hz}$ sin ajustar $f_{\text{refresh}}$ en `config.py`. | Validar en tiempo de arranque con `win.getActualFrameRate()` que la pantalla opere a $60.0\text{ Hz} \pm 0.5\text{ Hz}$. |
| **Inversión de Escala EEG ($\mu\text{V}$ vs $\text{V}$)** | **Alto** | La librería de adquisición retorna voltios ($10^{-6}$), provocando que los umbrales de rechazo de artefacción ($100\text{ }\mu\text{V}$) descarten todas las ventanas. | Normalización explícita en `acquisition.py`: $X_{\mu\text{V}} = X_{\text{raw}} \times 10^6$ si la amplitud máxima es $< 1.0$. |
| **Discontinuidad por Filtrado por Bloques** | **Medio** | Filtrar cada ventana de $50\text{ ms}$ de forma independiente sin pasar el estado final del filtro IIR. | Mantener de forma persistente la variable de estado `zi` en `filters.py` con `scipy.signal.sosfilt(..., zi=self.zi)`. |
| **Bloqueo por Parálisis de Análisis** | **Crítico** | Intentar desplegar en el hardware físico del laboratorio antes de tener redactado el informe P2. | Congelar el desarrollo físico hasta entregar P2 el día de la fecha límite; priorizar los datos de simulación sintética. |

---

# 11. Reglas Invariables de Actuación para el Asistente de IA (Claude / Gemini)

Cuando un nuevo agente de IA asuma la guía de este proyecto, **debe regirse estrictamente por los siguientes 5 principios operativos**:

1. **Acompañamiento Progresivo Paso a Paso:**
   - Prohibido generar código masivo sin solicitud explícita.
   - Entregar **un solo comando o instrucción a la vez**. Explicar el propósito, solicitar el resultado de la terminal y esperar la respuesta del estudiante antes de avanzar.
2. **Principio de Veracidad Inflexible:**
   - Prohibido inventar funciones, archivos, constantes, comandos o métricas no documentadas.
   - Si un dato no se encuentra en el repositorio, declararlo explícitamente y solicitar la verificación física al usuario.
3. **Respeto Absoluto al Código Base Integrado:**
   - No realizar refactorizaciones "estéticas" o estructurales sobre los paquetes `*_F1`. El código fue diseñado para alinearse con los estándares académicos del curso.
4. **Filosofía de Configuración Centralizada:**
   - Jamás declarar variables o umbrales en scripts secundarios. Toda modificación debe canalizarse a través de `nucleo_F1/config.py`.
5. **Separación Clara entre Simulación y Hardware:**
   - Reconocer siempre la diferencia entre el entorno sintético (279 tests PASS) y el entorno físico real (0% de pruebas en robot físico), orientando el trabajo actual a la maximización de la nota académica de P2.

---

# 12. Referencias a Documentos del Repositorio

1. `LEEME.txt`: Guía rápida de inicio y mapeo de dependencias del proyecto.
2. `PROJECT_MANAGEMENT_MEMORY.md`: Memoria detallada de gestión, riesgos y rúbrica de evaluación.
3. `nucleo_F1/config.py`: Fuente Única de Verdad para todos los parámetros del sistema.
4. `Guia_Asistente_3C2_Fase1_v2.0_bas_Bit_v3.1.pdf`: Documento técnico de especificaciones BCI y guía de laboratorio.
5. `SILABO_MT5003_2026-2.pdf`: Sílabo oficial del curso Proyecto Final de Ingeniería Mecatrónica II.

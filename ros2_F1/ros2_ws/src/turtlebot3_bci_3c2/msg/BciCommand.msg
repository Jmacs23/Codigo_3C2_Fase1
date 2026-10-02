# BciCommand.msg
# Comando resultante tras el arbitraje de mision.
#
# Publicado por : mission_node
# Consumido por : ctrl_node, interface_node, metrics_logger

std_msgs/Header header

# Modo: 0 = bci_manual, 1 = emergencia, 2 = detenido_seguro
int8 modo

# Etapa: 1 = navegacion, 2 = seleccion
int8 etapa

float32 u_ref
float32 omega_ref

# Estado de la maquina de comando enclavado:
# 0 = DETENIDO, 1 = AVANZANDO, 2 = ROTANDO_IZQ, 3 = ROTANDO_DER
int8 estado_fsm

# true si la FSM cambio de estado POR UN COMANDO BCI.
#
# Las transiciones forzadas por salvaguarda (topes, watchdog) ponen esto en
# false. La distincion es esencial: una salvaguarda no es un comando
# espurio, y contarla como tal enmascararia por completo la metrica que el
# experimento mide.
bool transiciono

float32 rho_min
string motivo

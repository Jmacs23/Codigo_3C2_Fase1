# TASMState.msg
# Estado cognitivo detectado por TASM.
#
# CONTRATO COMPARTIDO. Este mensaje es identico en la Linea 2 (silla de
# ruedas) porque ambas lineas consumen el mismo modulo TASM. Cambiarlo aqui
# obliga a cambiarlo alli.
#
# Publicado por : bci_node   (a ~20 Hz)
# Consumido por : mission_node, ctrl_node, interface_node, e2_node

std_msgs/Header header

# Estado cognitivo: 0 = Idle, 1 = TR, 2 = IC
int8 state

# Indice de la frecuencia detectada (0..K-1). Vale -1 si state != IC.
int8 freq_idx

# Confianza del clasificador sobre CUAL frecuencia es. [0,1]
float32 p_max

# P(IC | observaciones), salida del HMM. Confianza sobre si HAY intencion de
# comandar, que es una pregunta distinta de p_max. [0,1]
# En la Fase 1 se transporta pero no se usa en el control.
float32 lambda_bci

# Vectores de diagnostico. No intervienen en el control; se registran para
# poder reanalizar un trial despues sin repetirlo.
float32[] rho
float32[] rho_grad

# false si la ventana se rechazo por artefactos. El consumidor debe tratar
# un mensaje invalido como ausencia de mensaje.
bool is_valid

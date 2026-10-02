"""
run_integracion.py --- Script principal del Bloque 6
Bloque 6: integracion_F1

===========================================================================
AVISO SOBRE ESTE CODIGO

La interfaz, la fabrica, la verificacion de contrato, la regresion y los
experimentos estan verificados con pruebas automatizadas usando el mock.

El adaptador a TASM real es un ESQUELETO: el paquete de la Linea 1 no estaba
disponible. Las llamadas estan marcadas ">>> COMPLETAR".

Si encuentras un error:
  1. Estudia el modulo y trata de entender por que falla.
  2. Intenta resolverlo tu mismo.
  3. Reportalo al profesor indicando QUE TEST falla, el comando exacto, el
     traceback completo, y que intentaste.
===========================================================================

COMANDOS
--------
    python run_integracion.py config       Estado de la integracion
    python run_integracion.py contrato     COMPUERTA: verifica el contrato
    python run_integracion.py regresion    Compara dos fuentes
    python run_integracion.py protocolo    Protocolo de sesion con sujeto
    python run_integracion.py expA         Experimento A (simulado)
    python run_integracion.py expB         Experimento B (simulado)
    python run_integracion.py migracion    Como pasar de mock a TASM real
    python run_integracion.py test         Bateria de tests

EL CAMBIO DE MOCK A REAL ES UN PARAMETRO
-----------------------------------------
    config.tasm.fuente = "mock"      desarrollo
    config.tasm.fuente = "dataset"   TASM real sobre datos grabados
    config.tasm.fuente = "real"      TASM real en linea

Nada mas del sistema cambia. La FSM, el arbitraje, los nodos ROS2 y la
Etapa 2 no se enteran de cual esta corriendo.

LA COMPUERTA DEL BLOQUE
-----------------------
`python run_integracion.py contrato` comprueba que la fuente entrega
exactamente lo que el sistema espera. EJECUTARLO ANTES DE CORRER NADA con
TASM real: un desajuste de contrato produce fallos silenciosos.
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
import logica  # noqa: F401


def titulo(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def seccion(t: str) -> None:
    print()
    print("-" * 72)
    print(t)
    print("-" * 72)


def veredicto(ok: bool, msg: str) -> bool:
    print(f"  {'[ OK ]' if ok else '[FALLO]'} {msg}")
    return ok


# ===========================================================================

def cmd_config() -> int:
    titulo("ESTADO DE LA INTEGRACION")
    t = CONFIG.tasm

    print()
    print(f"  Fuente actual : {t.fuente}")
    if t.usa_mock:
        print()
        print("    Generador sintetico. Correcto para desarrollar el sistema")
        print("    robotico sin depender de la Linea 1.")
        print()
        print("    PROHIBIDO en los experimentos formales: usar senal")
        print("    sintetica destruiria uno de los diferenciadores frente a")
        print("    trabajos previos que si la emplearon.")
    else:
        print(f"    Ruta del paquete tasm : "
              f"{t.ruta_tasm or '(PYTHONPATH)'}")
        print(f"    Modelo entrenado      : "
              f"{t.modelo_entrenado or '(entrenar en calibracion)'}")

    print()
    print(f"  Verificar contrato    : {'si' if t.verificar_contrato else 'NO'}")
    print(f"  Transiciones inducidas: "
          f"{'si' if t.inducir_transiciones else 'NO'}")
    print(f"  Tolerancia regresion  : {t.tolerancia_regresion*100:.0f}%")

    seccion("LAS TRES ETAPAS DE USO")
    print("  1. AHORA (mock)")
    print("     Desarrollar y probar el sistema completo sin EEG, sin sujeto")
    print("     y sin depender de la Linea 1.")
    print()
    print("  2. CUANDO LA PoC ENTREGUE AUC > 0.75 (dataset)")
    print("     TASM real sobre datos grabados. Verifica la INTEGRACION en")
    print("     condiciones reproducibles, antes de traer sujetos.")
    print()
    print("  3. DESPUES (real)")
    print("     Con sujeto y amplificador.")
    print()
    print("  El paso 2 existe para no mezclar dos fuentes de error. Si algo")
    print("  falla al pasar directamente de mock a sujeto, no se sabria si")
    print("  es la integracion o TASM.")

    seccion("VALIDACION")
    probs = [p for p in CONFIG.validar()
             if any(k in p.lower() for k in
                    ("tasm", "contrato", "transic", "mock", "fuente"))]
    if probs:
        for p in probs:
            print(f"  - {p}")
    else:
        print("  Sin problemas.")
    return 0


def cmd_contrato() -> int:
    """Compuerta del bloque."""
    from tasm_interface import (crear_fuente_tasm, VerificadorContrato,
                                FuenteTASM, SalidaTASM)
    from command_fsm import EstadoTASM

    titulo("COMPUERTA DEL BLOQUE 6 --- Verificacion del contrato")
    print()
    print("  Un desajuste de contrato NO da error. Produce un sistema que")
    print("  corre, da numeros plausibles, y ninguno significa lo que uno")
    print("  cree.")
    print()
    print("  Por eso esta verificacion se ejecuta ANTES de correr nada.")

    ok = True
    ver = VerificadorContrato(CONFIG)

    # --- Fuente configurada ---
    seccion(f"Fuente configurada: {CONFIG.tasm.fuente}")
    try:
        fuente = crear_fuente_tasm(CONFIG, objetivo=2, semilla=1)
        print(f"  Creada: {fuente.nombre}")
        res = ver.verificar(fuente, n_muestras=800)
        print()
        print(res.resumen())
        ok &= res.correcto
    except NotImplementedError as ex:
        print(f"  La fuente no esta implementada:")
        print(f"    {str(ex).splitlines()[0]}")
        print()
        print("  Para desarrollar, usar config.tasm.fuente = 'mock'")
        return 1

    # --- Deteccion de fallos ---
    seccion("Fallos que la verificacion detecta")
    print("  Se simulan adaptadores mal escritos, de los que producirian")
    print("  fallos silenciosos con TASM real.")
    print()

    class Rota(FuenteTASM):
        def __init__(self, fallo):
            self.fallo = fallo

        def siguiente(self, ventana=None):
            s = SalidaTASM(EstadoTASM.IC, 2, 0.9, 0.9, True)
            if self.fallo == "base_1":
                s.freq_idx = len(CONFIG.bci.frecuencias)
            elif self.fallo == "porcentaje":
                s.lambda_bci = 92.0
            elif self.fallo == "p_max_fuera":
                s.p_max = 1.5
            elif self.fallo == "sin_tr":
                pass
            elif self.fallo == "rho_corto":
                s.rho = np.zeros(2)
            return s

        def reiniciar(self):
            pass

        @property
        def nombre(self):
            return self.fallo

    casos = [
        ("base_1", "indice de frecuencia en base 1"),
        ("porcentaje", "lambda_bci en porcentaje (0-100)"),
        ("p_max_fuera", "p_max fuera de [0,1]"),
        ("sin_tr", "el modelo nunca reporta TR"),
        ("rho_corto", "vector rho de longitud incorrecta"),
    ]

    print(f"  {'fallo simulado':>36} {'detectado':>12}")
    for fallo, desc in casos:
        r = ver.verificar(Rota(fallo), n_muestras=200)
        detectado = not r.correcto
        ok &= detectado
        print(f"  {desc:>36} {'SI' if detectado else 'NO':>12}")

    print()
    print("  Ninguno de estos daria un error en ejecucion.")

    seccion("RESULTADO")
    print(f"  {'COMPUERTA SUPERADA' if ok else 'COMPUERTA NO SUPERADA'}")
    print()
    print("  RECORDATORIO: con TASM real hay que ejecutar esto ANTES de")
    print("  cualquier experimento, y con ventanas de EEG reales:")
    print()
    print("      ver.verificar(fuente, generar_ventana=lector_de_eeg)")
    return 0 if ok else 1


def cmd_regresion() -> int:
    from tasm_interface import FuenteMock
    from experimentos import TestRegresion

    titulo("TEST DE REGRESION")
    print()
    print("  Cuando se sustituya el mock por TASM real, los resultados van a")
    print("  cambiar. La pregunta es POR QUE.")
    print()
    print("    (a) TASM real se comporta distinto del mock. Es lo esperado.")
    print("    (b) La integracion rompio algo.")
    print()
    print("  Sin regresion, un fallo de tipo (b) aparece como 'TASM funciona")
    print("  peor de lo esperado', y se buscaria el problema en el sitio")
    print("  equivocado durante semanas.")
    print()
    print("  El test NO compara valores (que cambian legitimamente) sino")
    print("  PROPIEDADES ESTRUCTURALES que deben cumplirse siempre.")

    f_a = FuenteMock(CONFIG, objetivo=2, modo="tasm", semilla=1)
    cfg_b = CONFIG.copia_con(
        mock=dict(p_episodio_fp_tasm=0.20,
                  p_error_idle_como_ic_tasm=0.10))
    f_b = FuenteMock(cfg_b, objetivo=2, modo="tasm", semilla=2)

    seccion("Comparacion")
    reg = TestRegresion(CONFIG)
    res = reg.comparar(f_a, f_b, n_ventanas=2000)
    print(res.resumen())

    seccion("PROPIEDADES QUE SE COMPRUEBAN")
    print("  1. La FSM no transiciona con TR sostenido")
    print("     (es la propiedad central del trabajo)")
    print("  2. La FSM enclava tras n_conf ventanas de IC")
    print("     (si no, el sistema no aceptaria ningun comando)")
    print("  3. La Etapa 2 sigue requiriendo ceil(log2 N) decisiones")
    print("  4. La fuente produce las tres clases")
    print()
    print("  Son invariantes del DISENO, no del clasificador. Si se rompen,")
    print("  el problema esta en la integracion.")
    return 0 if res.propiedades_ok else 1


def cmd_protocolo() -> int:
    titulo("PROTOCOLO DE SESION CON SUJETO")
    b, t, e = CONFIG.bci, CONFIG.tasm, CONFIG.experimento

    seccion("ANTES DE LA SESION")
    print("  1. Verificar el contrato de TASM")
    print("       python run_integracion.py contrato")
    print()
    print("  2. Comprobar impedancias (Bloque 3)")
    print("       python run_eeg.py impedancias")
    print("     Todas por debajo de 5 kOhm. Si alguna no baja, reaplicar gel")
    print("     antes de seguir.")
    print()
    print("  3. Verificar el estimulo (Bloque 2)")
    print("       python run_estimulo.py timing")
    print()
    print("  4. Comprobar que el monitor va a 60 Hz de verdad. La interfaz lo")
    print("     mide al arrancar y avisa si no coincide.")

    seccion("CALIBRACION (unos 18 minutos)")
    print(f"  a) FBCCA: {e.trials_por_celda} bloques x 10 trials  (~10 min)")
    print()
    print(f"  b) TASM  (~8 min):")
    print(f"       IC   : {t.duracion_calibracion_ic} s por frecuencia, "
          f"{t.n_bloques_calibracion} bloques")
    print(f"       Idle : {t.duracion_calibracion_idle} s, protocolo NS3")
    print(f"              (mirando la escena con los estimulos ENCENDIDOS)")
    print(f"       TR   : transiciones INDUCIDAS")
    print()
    print("  LOS SEGMENTOS DE TR SON EL PUNTO CRITICO.")
    print()
    print("  IC e Idle son directos: se le pide al sujeto que mire un")
    print("  estimulo, o que mire la escena.")
    print()
    print("  TR no: en operacion libre no se sabe CUANDO el sujeto esta")
    print("  desplazando la mirada, y sin saberlo no se puede etiquetar.")
    print()
    print("  Por eso hay que INDUCIRLAS: se le indica al sujeto que mire el")
    print("  estimulo A, y en un instante marcado se le pide que pase al B.")
    print("  Ese intervalo marcado es el segmento de TR.")
    print()
    print("  Sin esos segmentos no hay forma de entrenar la clase que da")
    print("  nombre al trabajo.")

    seccion("EXPERIMENTO A --- efecto de TASM")
    escs = len(CONFIG.escenarios.usados_en_fase1)
    nA = escs * 2 * e.trials_por_celda
    print(f"  {escs} escenarios x 2 condiciones x "
          f"{e.trials_por_celda} trials = {nA} trials")
    print(f"  N fijo en 4. Duracion aproximada: {nA*2.5:.0f} min")
    print()
    print("  El orden de las condiciones debe CONTRABALANCEARSE entre")
    print("  sujetos: si todos hicieran primero TASM y luego baseline, el")
    print("  aprendizaje del sujeto se confundiria con el efecto medido.")

    seccion("EXPERIMENTO B --- escalabilidad")
    nB = len(CONFIG.etapa2.niveles_n) * e.trials_por_celda
    print(f"  {len(CONFIG.etapa2.niveles_n)} niveles de N x "
          f"{e.trials_por_celda} trials = {nB} trials")
    print(f"  Solo con TASM, escenario fijo. "
          f"Duracion aproximada: {nB*2:.0f} min")

    seccion("DURANTE LA SESION")
    print(f"  - Escala de fatiga visual (1-10) cada "
          f"{e.fatiga_cada_n_trials} trials")
    print("  - NASA-TLX al terminar cada condicion del Experimento A")
    print("  - Descanso de 10 minutos cada 15 trials")
    print()
    print("  Sesion unica como objetivo, dos como respaldo. La decision se")
    print("  toma con los datos de fatiga del piloto: si a los 45 trials la")
    print("  puntuacion se mantiene por debajo de 6, cabe en una sesion.")

    seccion("CRITERIOS DE EXCLUSION (declarar a priori)")
    print("  - Accuracy FBCCA offline < 80% en calibracion")
    print("  - Mas del 1% de frames perdidos en el estimulo")
    print("  - Impedancias por encima de 5 kOhm que no bajen")
    print("  - AUC(TR vs IC) del sujeto por debajo de 0.65")
    print()
    print("  Declararlos ANTES de recoger datos, no despues de verlos.")
    return 0


def cmd_expA() -> int:
    from experimentos import Experimento, analizar_experimento_A

    titulo("EXPERIMENTO A (simulado)")
    print()
    print("  Se ejecuta con el simulador 2D, lo que permite ensayar el")
    print("  protocolo completo sin robot ni sujeto.")
    print()
    print(f"  Fuente de TASM: {CONFIG.tasm.fuente}")

    exp = Experimento(CONFIG, sujeto="simulado_A",
                      ruta_salida="/tmp/resultados")
    trials = exp.experimento_A(n_trials=3)

    print()
    print(analizar_experimento_A(trials))
    print()
    print(f"  Registro: {exp.ruta} ({exp.registrador.n_trials} trials)")
    return 0


def cmd_expB() -> int:
    from experimentos import Experimento, analizar_experimento_B

    titulo("EXPERIMENTO B (simulado)")
    print()
    print(f"  Fuente de TASM: {CONFIG.tasm.fuente}")

    exp = Experimento(CONFIG, sujeto="simulado_B",
                      ruta_salida="/tmp/resultados")
    trials = exp.experimento_B(n_trials=3)

    print()
    print(analizar_experimento_B(trials))
    print()
    print(f"  Registro: {exp.ruta} ({exp.registrador.n_trials} trials)")
    return 0


def cmd_migracion() -> int:
    titulo("MIGRACION DE MOCK A TASM REAL")

    seccion("PASO 1 --- Completar el adaptador")
    print("  En tasm_interface.py, la clase FuenteTASMReal tiene dos puntos")
    print("  marcados '>>> COMPLETAR':")
    print()
    print("    _cargar()    importar el paquete tasm y cargar el modelo")
    print("    siguiente()  la inferencia y LA CONVERSION AL CONTRATO")
    print()
    print("  La conversion es el punto critico. Verificar campo a campo:")
    print()
    print("    estado      si TASM devuelve 0/1/2, mapear al enumerado")
    print("    freq_idx    si es base 1, restar 1")
    print("    lambda_bci  si viene en 0-100, dividir por 100")
    print("    orden       comprobar que coincide con config.bci.frecuencias")

    seccion("PASO 2 --- Verificar el contrato")
    print("       config.tasm.fuente = 'dataset'")
    print("       python run_integracion.py contrato")
    print()
    print("  NO SEGUIR hasta que esto de CONTRATO CORRECTO.")

    seccion("PASO 3 --- Regresion")
    print("       python run_integracion.py regresion")
    print()
    print("  Si las propiedades estructurales se mantienen, la integracion")
    print("  esta bien y las diferencias son atribuibles a TASM.")
    print()
    print("  Si se rompen, el problema es de integracion. Revisar la")
    print("  conversion del paso 1.")

    seccion("PASO 4 --- Experimentos con datos grabados")
    print("       python run_integracion.py expA")
    print("       python run_integracion.py expB")
    print()
    print("  Todavia sin sujeto. Aqui se verifica que el pipeline completo")
    print("  produce metricas coherentes.")

    seccion("PASO 5 --- En linea")
    print("       config.tasm.fuente = 'real'")
    print()
    print("  Ahora si, con sujeto y amplificador. Seguir el protocolo:")
    print("       python run_integracion.py protocolo")

    seccion("QUE ESPERAR QUE CAMBIE")
    print("  Al pasar de mock a TASM real:")
    print()
    print("    FPR en TR      probablemente SUBA. El mock usa 0.089, tomado")
    print("                   del benchmark de la Linea 1 en condiciones")
    print("                   controladas. En linea sera peor.")
    print()
    print("    N_FP_TR        depende de la estructura temporal de los")
    print("                   errores. Si vienen en racha, subira mas de lo")
    print("                   que sugiere el FPR.")
    print()
    print("    Decisiones E2  NO debe cambiar. Es determinista.")
    print()
    print("  Ese ultimo punto es el mas util para diagnosticar: si el numero")
    print("  de decisiones deja de ser ceil(log2 N), hay un problema de")
    print("  integracion, no de TASM.")
    return 0


def cmd_test() -> int:
    import subprocess
    titulo("BATERIA DE TESTS")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v",
                        "--tb=short", "-W", "ignore"])
    return r.returncode


# ===========================================================================

COMANDOS = {
    "config": cmd_config,
    "contrato": cmd_contrato,
    "regresion": cmd_regresion,
    "protocolo": cmd_protocolo,
    "expA": cmd_expA,
    "expB": cmd_expB,
    "migracion": cmd_migracion,
    "test": cmd_test,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print(__doc__)
        print("Comandos:")
        for k in COMANDOS:
            print(f"  {k}")
        return 1
    try:
        return COMANDOS[sys.argv[1]]()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

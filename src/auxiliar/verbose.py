def _verbose(mensaje, verbose, nivel=1, tipo="info"):
    '''
    Funcion auxiliar creada para ayudar tanto en el desarrollo como en el uso del paquete

    verbose=0 -> Nada
    verbose=1 -> Informacion basica
    verbose=2 -> Informacion detallada
    '''

    if verbose >= nivel:

        if tipo == "info":
            if nivel == 1:
                print(f"\033[94m{mensaje}\033[0m")  # azul
            elif nivel == 2:
                print(f"\033[90m{mensaje}\033[0m")  # gris

        elif tipo == "warning":
            print(f"\033[93mWARNING: {mensaje}\033[0m")  # amarillo
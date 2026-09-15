def _verbose(mensaje, verbose, nivel=1, tipo="info"):
    """
    Muestra mensajes informativos durante la ejecución.

    Niveles:
        verbose=0 -> No muestra mensajes.
        verbose=1 -> Información principal.
        verbose=2 -> Información detallada.
        verbose=3 -> Información de diagnóstico.

    Tipos:
        info
        warning
        error
        success
        debug
    """

    if verbose < nivel:
        return

    if tipo == "warning":
        print(f"\033[93mWARNING: {mensaje}\033[0m")

    elif tipo == "error":
        print(f"\033[91mERROR: {mensaje}\033[0m")

    elif tipo == "success":
        print(f"\033[92m{mensaje}\033[0m")

    elif nivel == 1:
        # Azul fuerte
        print(f"\033[94m{mensaje}\033[0m")

    elif nivel == 2:
        # Gris
        print(f"\033[90m{mensaje}\033[0m")

    elif nivel == 3:
        # Gris oscuro
        print(f"\033[38;5;240m{mensaje}\033[0m")

    else:
        print(mensaje)
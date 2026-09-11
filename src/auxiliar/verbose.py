def _verbose(mensaje, verbose, nivel=1, tipo="info"):
    """
    Muestra mensajes informativos durante la ejecución.

    Niveles:
        verbose=0 -> No muestra mensajes.
        verbose=1 -> Información básica sobre los pasos principales.
        verbose=2 -> Información detallada y resultados intermedios.
        verbose=3 -> Información de diagnóstico / depuración.

    Tipos:
        info
        warning
        error
        success
        debug
    """

    if verbose < nivel:
        return

    colores = {
        "info": "\033[94m",      # azul
        "warning": "\033[93m",   # amarillo
        "error": "\033[91m",     # rojo
        "success": "\033[92m",   # verde
        "debug": "\033[90m",     # gris
    }

    color = colores.get(tipo, "\033[94m")

    prefijos = {
        "info": "",
        "warning": "WARNING: ",
        "error": "ERROR: ",
        "success": "",
        "debug": "DEBUG: ",
    }

    prefijo = prefijos.get(tipo, "")

    print(f"{color}{prefijo}{mensaje}\033[0m")
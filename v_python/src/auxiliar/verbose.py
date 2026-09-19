def _verbose(mensaje, verbose, nivel=1, tipo="info"):
    """
    Muestra mensajes informativos durante la ejecución.

    Los mensajes se filtran por nivel: solo se imprimen aquellos cuyo
    ``nivel`` es menor o igual al valor de ``verbose`` con el que se
    llamó a la función. El color depende del ``nivel`` (azul, gris,
    gris oscuro) salvo cuando ``tipo`` indica explícitamente una
    categoría especial (warning/error/success).

    Niveles
    -------
    verbose=0 -> No muestra mensajes.
    verbose=1 -> Información principal (azul fuerte).
    verbose=2 -> Información detallada (gris).
    verbose=3 -> Información de diagnóstico (gris oscuro). Esto lo hemos metido por si hiciera falta, esta por si acaso, no se usa.

    Tipos
    -----
    ``warning`` -> amarillo; además levanta un ``RuntimeError``.
    ``error`` -> rojo; además levanta un ``RuntimeError``.
    ``success`` -> verde.
    Cualquier otro valor (incluido ``info``) -> se colorea según el
    ``nivel``.

    Parameters
    ----------
    mensaje : str
        Texto a mostrar.

    verbose : int
        Umbral global de visibilidad. Los mensajes con ``nivel`` mayor
        que este valor no se muestran.

    nivel : int, opcional
        Nivel del mensaje (1-3). Por defecto 1.

    tipo : str, opcional
        Categoría del mensaje. Por defecto ``"info"``.

    Returns
    -------
    None
    """

    if verbose < nivel:
        return

    if tipo == "warning":
        print(f"\033[93mWARNING: {mensaje}\033[0m")

    elif tipo == "error":
        print(f"\033[91mERROR: {mensaje}\033[0m")
        raise RuntimeError(mensaje)

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
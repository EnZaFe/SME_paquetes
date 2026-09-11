from auxiliar.verbose import _verbose
from auxiliar.auxiliar_es import (
    _es_dataset,
    _es_variable,
    _es_discreta,
    _es_continua
)

import pandas as pd
import numpy as np


def calcular_correlacion(
    dataset,
    atributos=None,
    normalizar=False,
    verbose=1
):
    """
    API PUBLICA.

    Calcula una matriz de correlación/asociación entre los atributos
    seleccionados del dataset.

    Casos contemplados:

        numérica + numérica
            -> Correlación de Pearson

        categórica + categórica
            -> Información Mutua (Mutual Information)

        categórica + numérica
            -> ANOVA (Welch)

    Parámetros
    ----------
    dataset : pandas.DataFrame
        Dataset sobre el que realizar el cálculo.

    atributos : list o None
        Lista de nombres de columnas que se quieren analizar.
        Si es None, se utilizan todas las columnas del dataset.

    normalizar : bool
        Indica si se deben normalizar los resultados a una escala 0-1.

        TODO:
        Actualmente esta funcionalidad no está implementada.

    verbose : int
        Nivel de información que se muestra durante la ejecución.

    Returns
    -------
    matriz : pandas.DataFrame
        Matriz NUMÉRICA de correlaciones/asociaciones entre los
        atributos (solo el valor escalar, lista para heatmaps,
        exportar a csv, operar con numpy, etc.)

    detalles : dict
        Diccionario con toda la información adicional de cada par de
        atributos, indexado por la tupla (atributo1, atributo2) (y
        también por (atributo2, atributo1), para no tener que
        preocuparse del orden al consultarlo).

        Cada valor es a su vez un diccionario devuelto por
        _calc_corr_num / _calc_corr_catg / _calc_corr_catg_num, listo
        para pasarse directamente a las funciones graficar_* sin
        tener que recalcular nada.
    """

    _verbose("Iniciando cálculo de correlaciones...", verbose)

    if not _es_dataset(dataset):
        raise TypeError(
            "dataset debe ser un DataFrame."
        )

    if atributos is not None and not isinstance(atributos, list):
        raise TypeError(
            "atributos debe ser una lista o None."
        )

    # Si no se especifican atributos, utilizamos todas las columnas.
    if atributos is None:
        atributos = list(dataset.columns)
        _verbose(
            f"Warning: no se han indicado atributos "
            "se usaran todas las columnas del dataset.",
            verbose
        )

    for atributo in atributos:

        if atributo not in dataset.columns:
            raise ValueError(
                f"La columna '{atributo}' no existe en el dataset."
            )

    tipos = {}

    for atributo in atributos:

        columna = dataset[atributo]

        if _es_continua(columna):

            tipos[atributo] = "continua"

        elif _es_discreta(columna):

            tipos[atributo] = "discreta"

        else:

            _verbose(
                f"Warning: no se utilizará '{atributo}' porque "
                "no se detecta como continuo ni discreto.",
                verbose
            )

    atributos_validos = list(tipos.keys())

    _verbose(
        f"Se analizarán {len(atributos_validos)} atributos.",
        verbose
    )

    matriz = pd.DataFrame(
        index=atributos_validos,
        columns=atributos_validos,
        dtype=float
    )

    detalles = {}

    # Una variable comparada consigo misma tiene asociación máxima 1.
    for atributo in atributos_validos:
        matriz.loc[atributo, atributo] = 1.0

    # Las medidas utilizadas son simétricas:
    #
    #     Pearson(A, B) = Pearson(B, A)
    #     MI(A, B)      = MI(B, A)
    #     Welch(A, B)   = Welch(B, A)
    #
    for i, atributo1 in enumerate(atributos_validos):

        for atributo2 in atributos_validos[i + 1:]:

            resultado = _calc_atributo(
                dataset[atributo1],
                dataset[atributo2],
                tipos[atributo1],
                tipos[atributo2],
                normalizar,
                verbose
            )

            # En la matriz numérica solo guardamos el escalar.
            matriz.loc[atributo1, atributo2] = resultado["valor"]
            matriz.loc[atributo2, atributo1] = resultado["valor"]

            # En detalles guardamos TODO lo calculado, indexado en
            # ambos sentidos para que sea cómodo de consultar.
            detalles[(atributo1, atributo2)] = resultado
            detalles[(atributo2, atributo1)] = resultado

    _verbose(
        "Cálculo de correlaciones finalizado.",
        verbose
    )

    return matriz, detalles


def _calc_atributo(
    atributo1,
    atributo2,
    tipo1,
    tipo2,
    normalizar=False,
    verbose=1
):
    """
    Decide qué método utilizar en función del tipo de los atributos.
    Siempre devuelve un diccionario con, como mínimo, la clave
    "valor" (el escalar de la matriz) y "tipo" (qué método se usó).
    """

    if tipo1 == "discreta" and tipo2 == "discreta":

        return _calc_corr_catg(
            atributo1,
            atributo2,
            normalizar,
            verbose
        )

    elif tipo1 == "continua" and tipo2 == "continua":

        return _calc_corr_num(
            atributo1,
            atributo2,
            normalizar,
            verbose
        )

    elif (
        (tipo1 == "discreta" and tipo2 == "continua")
        or
        (tipo1 == "continua" and tipo2 == "discreta")
    ):

        return _calc_corr_catg_num(
            atributo1,
            atributo2,
            normalizar,
            verbose
        )

    else:

        raise ValueError(
            "Los atributos no son ni continuos ni discretos."
        )


def _calc_corr_num(
    atributo1,
    atributo2,
    normalizar=False,
    verbose=1
):
    """
    Correlación de Pearson.

    Pearson mide la intensidad y dirección de la relación LINEAL
    entre dos variables numéricas.

    El resultado está entre -1 y 1:

        +1 -> relación lineal positiva perfecta
         0 -> ausencia de relación lineal
        -1 -> relación lineal negativa perfecta

    Fórmula:
        r = (n * Σ(xy) - Σx * Σy) /
            √[(n * Σ(x²) - (Σx)²) * (n * Σ(y²) - (Σy)²)]

    Devuelve un diccionario con el coeficiente y todo lo necesario
    para pintar el gráfico de dispersión + recta de regresión sin
    tener que recalcular nada (pendiente, intercepto, nombres...).
    """

    nombre1 = getattr(atributo1, "name", None) or "Atributo 1"
    nombre2 = getattr(atributo2, "name", None) or "Atributo 2"

    datos = pd.DataFrame({
        "x": pd.Series(atributo1).reset_index(drop=True),
        "y": pd.Series(atributo2).reset_index(drop=True)
    })

    datos = datos.dropna()

    if len(datos) == 0:
        raise ValueError(
            "No hay suficientes datos para calcular la correlación."
        )

    x = datos["x"]
    y = datos["y"]

    n = len(datos)

    suma_x = x.sum()
    suma_y = y.sum()

    suma_x2 = (x ** 2).sum()
    suma_y2 = (y ** 2).sum()

    suma_xy = (x * y).sum()

    numerador = (
        n * suma_xy - suma_x * suma_y
    )

    parte_x = (
        n * suma_x2 - suma_x ** 2
    )
    parte_y = (
        n * suma_y2 - suma_y ** 2
    )

    denominador = (parte_x * parte_y) ** 0.5

    if denominador == 0:
        raise ValueError(
            "No se puede calcular la correlación de Pearson: "
            "alguna de las variables es constante."
        )

    r = numerador / denominador

    # Recta de regresión y = pendiente * x + intercepto, útil para
    # el gráfico sin tener que volver a tocar los datos.
    pendiente, intercepto = np.polyfit(x, y, 1)

    if normalizar:
        # TODO:
        # Normalizar el resultado de Pearson a [0, 1].
        pass

    return {
        "tipo": "pearson",
        "valor": r,
        "r2": r ** 2,
        "n": n,
        "pendiente": pendiente,
        "intercepto": intercepto,
        "nombre1": nombre1,
        "nombre2": nombre2,
        "x": x,
        "y": y
    }


def _calc_corr_catg(
    atributo1,
    atributo2,
    normalizar=False,
    verbose=1
):
    """
    Mutual Information (Información Mutua).

    La información mutua mide cuánta información aporta conocer
    una variable sobre la otra.

    Si MI(X, Y) = 0, entonces X e Y son independientes. Cuanto mayor
    sea la MI, mayor es la dependencia entre ambas variables.

    MI(X,Y) = sumatorio_y sumatorio_x [ p(x,y) * log(p(x,y)/(p(x)*p(y))) ]

    Devuelve un diccionario con la MI y la tabla de contingencia ya
    calculada, para poder pintar el heatmap sin recalcularla.
    """

    nombre1 = getattr(atributo1, "name", None) or "Atributo 1"
    nombre2 = getattr(atributo2, "name", None) or "Atributo 2"

    datos = pd.DataFrame({
        "x": pd.Series(atributo1).reset_index(drop=True),
        "y": pd.Series(atributo2).reset_index(drop=True)
    })

    datos = datos.dropna()

    if len(datos) == 0:
        raise ValueError(
            "No hay suficientes datos para calcular la información mutua."
        )

    x = datos["x"]
    y = datos["y"]

    n = len(datos)

    contando_x = x.value_counts()
    contando_y = y.value_counts()

    tabla = pd.crosstab(x, y)

    resultado = 0

    for valor_x in contando_x.index:
        for valor_y in contando_y.index:

            pareja_count = tabla.loc[valor_x, valor_y]

            if pareja_count == 0:
                continue

            p_xy = pareja_count / n
            p_x = contando_x[valor_x] / n
            p_y = contando_y[valor_y] / n

            resultado += p_xy * np.log(p_xy / (p_x * p_y))

    if normalizar:
        # TODO:
        # Normalizar MI a [0, 1].
        pass

    return {
        "tipo": "informacion_mutua",
        "valor": resultado,
        "n": n,
        "tabla": tabla,
        "nombre1": nombre1,
        "nombre2": nombre2
    }


def _calc_corr_catg_num(
    atributo1,
    atributo2,
    normalizar=False,
    verbose=1
):
    """
    Welch ANOVA de un factor.

    Comprueba si existen diferencias entre las medias de una
    variable continua para los distintos grupos de una variable
    discreta. Welch ANOVA es una variante del ANOVA de un factor que
    no requiere que las varianzas de los grupos sean iguales.

    Devuelve un diccionario con el estadístico F y toda la info por
    grupo (medias, tamaños, varianzas) para poder pintar el boxplot
    sin volver a tocar los datos originales.
    """

    datos = pd.DataFrame({
        "x": pd.Series(atributo1).reset_index(drop=True),
        "y": pd.Series(atributo2).reset_index(drop=True)
    })

    datos = datos.dropna()

    x = datos["x"]
    y = datos["y"]

    nombre_categorico = getattr(atributo1, "name", None) or "Grupo"
    nombre_numerico = getattr(atributo2, "name", None) or "Valor"

    # Nos aseguramos de que x sea la discreta e y la continua.
    if not (_es_discreta(x) and _es_continua(y)):
        x, y = y, x
        nombre_categorico, nombre_numerico = nombre_numerico, nombre_categorico

    grupos = list(x.unique())

    if len(grupos) < 2:
        raise ValueError(
            "Welch ANOVA necesita al menos dos grupos."
        )

    valores_grupos = {}
    n_grupos = {}
    medias = {}
    varianzas = {}

    for grupo in grupos:

        valores = y[x == grupo]

        if len(valores) < 2:
            raise ValueError(
                f"El grupo '{grupo}' no tiene suficientes observaciones."
            )

        valores_grupos[grupo] = valores
        n_grupos[grupo] = len(valores)
        medias[grupo] = valores.mean()
        varianzas[grupo] = valores.var()

    pesos = {}

    for grupo in grupos:

        if varianzas[grupo] == 0:
            raise ValueError(
                f"La varianza del grupo '{grupo}' es 0."
            )

        pesos[grupo] = n_grupos[grupo] / varianzas[grupo]

    suma_pesos = sum(pesos.values())
    suma_pesos_media = sum(
        pesos[grupo] * medias[grupo] for grupo in grupos
    )

    media_ponderada = suma_pesos_media / suma_pesos

    k = len(grupos)

    suma_entre = sum(
        pesos[grupo] * (medias[grupo] - media_ponderada) ** 2
        for grupo in grupos
    )

    numerador = suma_entre / (k - 1)

    correccion = 0

    for grupo in grupos:
        parte = 1 - pesos[grupo] / suma_pesos
        correccion += parte ** 2 / (n_grupos[grupo] - 1)

    correccion = 1 + (2 * (k - 2) / (k ** 2 - 1)) * correccion

    f_welch = numerador / correccion

    gl_entre = k - 1

    # TODO:
    # Calcular los grados de libertad del denominador y el p-value
    # a partir de la distribución F.

    if normalizar:
        # TODO:
        # Normalizar el resultado a [0, 1].
        pass

    return {
        "tipo": "welch",
        "valor": f_welch,
        "gl_entre": gl_entre,
        "grupos": grupos,
        "n_grupos": n_grupos,
        "medias": medias,
        "varianzas": varianzas,
        "media_general": y.mean(),
        "valores_grupos": valores_grupos,
        "nombre_categorico": nombre_categorico,
        "nombre_numerico": nombre_numerico
    }
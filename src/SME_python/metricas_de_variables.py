'''
enunciado:
Cálculo de métricas para los atributos de un dataset: varianza y AUC para las variables contínuas y entropía para las discretas. La función deberá reconocer el tipo de atributo y actuar en consecuencia. Notese que en el caso del AUC, el dataset debe ser supervisado, es decir, es necesario especificar una variable clase binaria con la que evaluar el AUC de los atributos numéricos.
'''
from auxiliar.verbose import _verbose 
from auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta,_es_continua
import pandas as pd
from math import log2

def calcular_metricas(dataset, clases=None, atributos=None, verbose=1):
    '''API PUBLICA
    
    dataset para tener la info
    clases para AUC
    atributos, por si se quieren calcular solo algunos atributos
    '''

    _verbose( "Iniciando cálculo de métricas...", verbose ) 
    _verbose( f"""Parámetros elegidos: \n
                atributos: {atributos} \n
                clases: {clases}
                """, 
             verbose, nivel=2 )


    if atributos is not None and not isinstance(atributos, list):
        _verbose(
            "El parámetro 'atributos' no es una lista.",
            verbose,
            nivel=1,
            tipo="warning"
        )
        raise TypeError("columnas debe ser una lista o None.")

    # Comprobación de clases para AUC
    if clases is None:
        _verbose(
            "No se han proporcionado clases. "
            "El AUC no se calculará para las variables continuas.",
            verbose,
            nivel=1,
            tipo="warning"
        )
    else:
        _verbose(
            "Se han proporcionado clases. "
            "Se calculará el AUC para las variables continuas.",
            verbose,
            nivel=2
        )

    if _es_variable(dataset):

            _verbose(
                f"Se ha recibido una variable: '{dataset.name}'.",
                verbose,
                nivel=1
            )

            metricas = _calcular_metricas_variable(
                dataset,
                clases,
                verbose
            )

            _verbose(
                "Cálculo de métricas finalizado.",
                verbose,
                nivel=1,
                tipo="success"
            )

            return metricas

        # Si es un dataset
    elif _es_dataset(dataset):

        _verbose(
            f"Se ha recibido un dataset con "
            f"{dataset.shape[0]} filas y {dataset.shape[1]} columnas.",
            verbose,
            nivel=1
        )

        metricas = _calcular_metricas_dataset(
            dataset,
            clases,
            atributos,
            verbose
        )

        _verbose(
            "Cálculo de métricas finalizado correctamente.",
            verbose,
            nivel=1,
            tipo="success"
        )

        return metricas

    else:

        _verbose(
            "El objeto recibido no es una Series ni un DataFrame.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise TypeError(
            "dataset debe ser una Series o un DataFrame."
        )


def _calcular_metricas_dataset(dataset, clases, atributos_clm, verbose=1):
    """Calcular métricas para las columnas de un dataset entero."""

    metricas = {}

    if atributos_clm is not None:

        _verbose(
            f"Se han seleccionado {len(atributos_clm)} atributos "
            "para calcular sus métricas.",
            verbose,
            nivel=1
        )

        for col in atributos_clm:

            if col not in dataset.columns:

                _verbose(
                    f"La columna '{col}' no existe en el dataset.",
                    verbose,
                    nivel=1,
                    tipo="warning"
                )

                raise ValueError(
                    f"La columna '{col}' no existe en el dataset."
                )

            _verbose(
                f"Analizando atributo '{col}'...",
                verbose,
                nivel=1
            )

            metricas[col] = _calcular_metricas_variable(
                dataset[col],
                clases,
                verbose
            )

        return metricas

    # Si no se especifican columnas, calculamos para todas las columnas

    _verbose(
        f"No se han especificado atributos. "
        f"Se analizarán las {len(dataset.columns)} columnas del dataset.",
        verbose,
        nivel=1
    )

    for col in dataset.columns:

        _verbose(
            f"Analizando atributo '{col}'...",
            verbose,
            nivel=1
        )

        metricas[col] = _calcular_metricas_variable(
            dataset[col],
            clases,
            verbose
        )

    return metricas


def _calcular_metricas_variable(columna, clases, verbose=1):
    """Router para calcular las métricas de una variable."""

    if _es_discreta(columna):

        _verbose(
            f"Se ha detectado que '{columna.name}' es discreta. "
            "Calculando entropía...",
            verbose,
            nivel=1
        )

        return _calcular_metricas_discreta(
            columna,
            verbose
        )

    elif _es_continua(columna):

        _verbose(
            f"Se ha detectado que '{columna.name}' es continua. "
            "Calculando varianza y AUC...",
            verbose,
            nivel=1
        )

        return _calcular_metricas_continua(
            columna,
            clases,
            verbose
        )

    else:

        _verbose(
            f"No se puede determinar el tipo de la columna "
            f"'{columna.name}'.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise TypeError(
            f"No se puede determinar el tipo de la columna "
            f"'{columna.name}'."
        )


def _calcular_metricas_discreta(columna, verbose=1):
    """Calcula las métricas de una variable discreta."""

    _verbose(
        f"Calculando entropía de '{columna.name}'...",
        verbose,
        nivel=2
    )

    entropia = _calcular_entropia(
        columna,
        verbose
    )

    return {
        "entropia": entropia
    }


def _calcular_metricas_continua(columna, clases, verbose=1):
    """Calcula las métricas de una variable continua."""

    _verbose(
        f"Calculando varianza de '{columna.name}'...",
        verbose,
        nivel=2
    )

    varianza = _calcular_varianza(
        columna,
        verbose
    )

    _verbose(
        f"Varianza calculada: {round(varianza, 4)}",
        verbose,
        nivel=2
    )

    if clases is None:

        _verbose(
            "No se han proporcionado clases. "
            "No se calculará el AUC.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        return {
            "varianza": varianza
        }

    _verbose(
        f"Calculando AUC de '{columna.name}'...",
        verbose,
        nivel=2
    )

    auc = _calcular_AUC(
        columna,
        clases,
        verbose
    )

    _verbose(
        f"AUC calculado: {round(auc, 4)}",
        verbose,
        nivel=2
    )

    return {
        "varianza": varianza,
        "auc": auc
    }

#######################
# Variables Discretas #
#######################
def _calcular_entropia(atributo_clm, verbose):
    ''' Explicacion para saber exacto que estamos haciendo:
    La entropía mide cuánta incertidumbre o diversidad hay respecto a las clases de una variable objetivo. Si todos los ejemplos pertenecen a la misma clase, la entropía es 0 porque no existe incertidumbre. Cuanto más repartidos estén los ejemplos entre las distintas clases, mayor será la entropía.

    Para calcular la entropía esperada de un atributo A:

    E(D,A) = sumatorio(v pertenece a Valores(A))
            (|Dv| / |D|) * E(Dv)

    Valores(A) son los distintos valores que puede tomar el atributo A.

    Dv es el subconjunto de datos formado por los ejemplos cuyo atributo A
    toma el valor v.

    (|Dv| / |D|) es el peso proporcional de ese subconjunto respecto al
    dataset completo.

    E(Dv) es la entropía de las clases dentro del subconjunto Dv.


    La entropía del dataset D se calcula como:

    E(D) = - sumatorio(c pertenece a Clases)
        (|Dc| / |D|) * log2(|Dc| / |D|)

    Clases son los distintos valores que puede tomar la variable objetivo.

    Dc es el subconjunto de datos que pertenece a la clase c.

    (|Dc| / |D|) es la proporción de ejemplos que pertenecen a la clase c.

    El signo negativo se utiliza porque el logaritmo de una probabilidad
    entre 0 y 1 es negativo, y queremos obtener una entropía positiva.
    '''

    _verbose(
        f"Preparando datos para calcular la entropía "
        f"de '{atributo_clm.name}'...",
        verbose,
        nivel=2
    )

    if pd.isna(atributo_clm).any():

        num_nulos = pd.isna(atributo_clm).sum()

        _verbose(
            f"Se han detectado {num_nulos} valores nulos. "
            "Se eliminarán antes de calcular la entropía.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        atributo_clm = atributo_clm.dropna()

    if len(atributo_clm) == 0:

        _verbose(
            "El atributo no contiene valores válidos.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "No se puede calcular la entropía: "
            "el atributo no contiene valores válidos."
        )

    atributo_clm = atributo_clm.dropna()

    valores_unicos = set(atributo_clm)

    _verbose(
        f"""Datos preparados para la entropía:
    Número de casos: {len(atributo_clm)}
    Valores únicos: {valores_unicos}""",
        verbose,
        nivel=2
    )

    entropia = 0

    for valor in valores_unicos:

        num_rep = 0

        for elemento in atributo_clm:

            if elemento == valor:
                num_rep += 1

        probabilidad = num_rep / len(atributo_clm)

        _verbose(
            f"Valor '{valor}': {num_rep} apariciones "
            f"({probabilidad:.2%})",
            verbose,
            nivel=3
        )

        entropia -= probabilidad * log2(probabilidad)

    _verbose(
        f"Entropía final: {entropia:.4f}",
        verbose,
        nivel=2
    )

    return entropia

#######################
# Variables Continuas #
#######################

def _calcular_varianza(atributo_clm, verbose):
    '''
    Mide cuánto se dispersan los valores respecto a su media.
    Formula:
        Var(X) = (1/n) * sumatorio((xi - media)^2)
    '''

    _verbose(
        f"Preparando datos para calcular la varianza "
        f"de '{atributo_clm.name}'...",
        verbose,
        nivel=2
    )

    if pd.isna(atributo_clm).any():

        num_nulos = pd.isna(atributo_clm).sum()

        _verbose(
            f"Se han detectado {num_nulos} valores nulos. "
            "Se eliminarán antes de calcular la varianza.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        atributo_clm = atributo_clm.dropna()

    if len(atributo_clm) == 0:

        _verbose(
            "El atributo no contiene valores válidos.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "No se puede calcular la varianza: "
            "el atributo no contiene valores válidos."
        )

    suma = 0

    for valor in atributo_clm:
        suma += valor

    media = suma / len(atributo_clm)

    _verbose(
        f"Media: {media:.4f}",
        verbose,
        nivel=2
    )

    suma_diferencias = 0

    for valor in atributo_clm:

        diferencia = valor - media
        diferencia_cuadrado = diferencia ** 2
        suma_diferencias += diferencia_cuadrado

    varianza = suma_diferencias / len(atributo_clm)

    _verbose(
        f"Varianza: {varianza:.4f}",
        verbose,
        nivel=2
    )

    return varianza


def _calcular_AUC(atributo_clm, clases, verbose):
    '''
    AUC (Area Under the ROC Curve)...
    '''

    _verbose(
        "Iniciando cálculo de AUC...",
        verbose,
        nivel=2
    )

    if len(atributo_clm) != len(clases):

        _verbose(
            "El atributo y las clases tienen diferente número "
            "de elementos.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "El atributo y las clases deben tener el mismo número de elementos."
        )

    if len(atributo_clm) == 0:

        _verbose(
            "No se puede calcular el AUC de un atributo vacío.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "No se puede calcular el AUC de un atributo vacío."
        )

    if not _es_continua(atributo_clm):

        _verbose(
            "El atributo no es numérico y no puede utilizarse "
            "para calcular el AUC.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise TypeError(
            "El atributo debe ser numérico para calcular el AUC."
        )

    valores_clases = set(clases)

    if len(valores_clases) != 2:

        _verbose(
            f"Se han detectado {len(valores_clases)} clases. "
            "El AUC requiere exactamente dos.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "El atributo clases debe contener exactamente dos clases."
        )

    _verbose(
        f"Clases detectadas: {valores_clases}",
        verbose,
        nivel=2
    )

    clase_1 = list(valores_clases)[0]
    clase_2 = list(valores_clases)[1]

    valores_clase_1 = []
    valores_clase_2 = []

    for i in range(len(atributo_clm)):

        if clases.iloc[i] == clase_1:
            valores_clase_1.append(atributo_clm.iloc[i])

        elif clases.iloc[i] == clase_2:
            valores_clase_2.append(atributo_clm.iloc[i])

    _verbose(
        f"Elementos de la clase '{clase_1}': {len(valores_clase_1)}",
        verbose,
        nivel=2
    )

    _verbose(
        f"Elementos de la clase '{clase_2}': {len(valores_clase_2)}",
        verbose,
        nivel=2
    )

    comparaciones_correctas = 0
    empates = 0

    for valor_1 in valores_clase_1:

        for valor_2 in valores_clase_2:

            if valor_1 > valor_2:
                comparaciones_correctas += 1

            elif valor_1 == valor_2:
                empates += 1

    total_comparaciones = (
        len(valores_clase_1) * len(valores_clase_2)
    )

    if total_comparaciones == 0:

        _verbose(
            "Una de las clases no tiene valores para comparar.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        raise ValueError(
            "No se puede calcular el AUC porque una de las clases no tiene valores."
        )

    auc = (
        comparaciones_correctas + 0.5 * empates
    ) / total_comparaciones

    auc = max(auc, 1 - auc)

    _verbose(
        f"Comparaciones correctas: {comparaciones_correctas}",
        verbose,
        nivel=2
    )

    _verbose(
        f"Empates: {empates}",
        verbose,
        nivel=2
    )

    _verbose(
        f"AUC: {auc:.4f}",
        verbose,
        nivel=2
    )

    return auc
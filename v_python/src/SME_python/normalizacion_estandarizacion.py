from v_python.src.auxiliar.verbose import _verbose
from v_python.src.auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta,_es_continua
import pandas as pd

def normalizar(dataset, atributos=None, verbose=1):
    """Normaliza una variable o un dataset a escala [0, 1] (min-max).

        normalizar(series) -> pd.Series en [0, 1]
        normalizar(df, atributos=[...]) -> pd.DataFrame con columnas transformadas
    """
    
    _verbose("Iniciando normalización...", verbose)

    if atributos is not None and not isinstance(atributos, list):
        raise TypeError("atributos debe ser una lista o None.")

    return _transformar_dataset(
        dataset,
        atributos,
        _normalizar_variable,
        verbose
    )


def estandarizar(dataset, atributos=None, verbose=1):
    """Estandariza una variable o un dataset con puntuación z (media 0, desviación 1).

        estandarizar(series) -> pd.Series
        estandarizar(df, atributos=[...]) -> pd.DataFrame con columnas transformadas
    """

    _verbose("Iniciando estandarización...", verbose)

    if atributos is not None and not isinstance(atributos, list):
        raise TypeError("atributos debe ser una lista o None.")

    return _transformar_dataset(
        dataset,
        atributos,
        _estandarizar_variable,
        verbose
    )

def _transformar_dataset(dataset, atributos, funcion, verbose=1):
    """Aplica ``funcion`` a cada variable continua de un dataset.

    Para una serie se aplica directamente; para un DataFrame se transforma
    solo las columnas continuas (las discretas quedan sin cambios).
    """

    if _es_variable(dataset):
        return funcion(dataset, verbose)

    elif _es_dataset(dataset):
        resultado = dataset.copy()

        if atributos is not None:
            columnas = atributos
        else:
            columnas = dataset.columns

        for col in columnas:

            if col not in dataset.columns:
                raise ValueError(
                    f"La columna '{col}' no existe en el dataset."
                )

            if _es_continua(dataset[col]):
                resultado[col] = funcion(dataset[col], verbose)

        return resultado

    else:
        raise TypeError(
            "dataset debe ser una Series o un DataFrame."
        )
    
def _normalizar_variable(columna, verbose=1):
    """Normaliza una variable numérica."""

    if not _es_continua(columna):
        raise TypeError(
            f"La columna '{columna.name}' no es numérica."
        )

    minimo = columna.min()
    maximo = columna.max()
    _verbose(
        f"Se han detectado: mínimo={minimo}, máximo={maximo}",
        verbose,
        nivel=2
    )
    if maximo == minimo:
        raise ValueError(
            f"No se puede normalizar '{columna.name}': "
            "todos sus valores son iguales."
        )

    return (columna - minimo) / (maximo - minimo)


def _estandarizar_variable(columna, verbose=1):
    """Estandariza una variable numérica."""

    if not pd.api.types.is_numeric_dtype(columna):
        raise TypeError(
            f"La columna '{columna.name}' no es numérica."
        )

    media = columna.mean()
    desviacion = columna.std()
    _verbose(
        f"Se han detectado: media={media}, desviacion={desviacion}",
        verbose,
        nivel=2
    )
    if desviacion == 0:
        raise ValueError(
            f"No se puede estandarizar '{columna.name}': "
            "la desviación estándar es 0."
        )

    return (columna - media) / desviacion
# Try to load any type of dataset.
#
# Por ahora la pipeline se conecta con el dataset del iris (in-memory, vía
# scikit-learn) para poder verificar que load_dataset() -> discretizacion()
# están conectadas sin depender de archivos externos.
#
# TODO: implementar los loaders de archivos (csv, tsv, excel...) cuando se
#       disponga de la lógica de lectura.

import pandas as pd
from sklearn.datasets import load_iris
from python.src.auxiliar.verbose import _verbose

def load_dataset(dataset="iris", verbose=1):
    """Cargar un dataset y devolverlo como DataFrame.

    Parámetros
    ----------
    dataset : str, por defecto "iris"
        Nombre del dataset a cargar. Por ahora solo se soporta el iris
        (generado en memoria con scikit-learn).
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.

    Raises
    ------
    ValueError
        Si el dataset solicitado no está soportado todavía.
    """
    _verbose(f"Cargando dataset '{dataset}'...", verbose)

    if dataset == "iris":
        data = load_iris(as_frame=True)
        df = data.frame
        _verbose(
            f"Dataset '{dataset}' cargado: {df.shape[0]} filas x {df.shape[1]} columnas.",
            verbose,
        )
        return df

    # TODO: implementar carga de datasets por nombre (ej. "wine", "breast_cancer")
    raise ValueError(f"Dataset '{dataset}' no soportado todavía.")


def load_csv(ruta, verbose=1):
    """Cargar un dataset desde un archivo CSV.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV a leer.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado (no implementado todavía).

    Raises
    ------
    NotImplementedError
        Si la lectura de CSV no ha sido implementada.
    """
    _verbose(f"Cargando CSV desde '{ruta}'...", verbose)
    # TODO: df = pd.read_csv(ruta)
    raise NotImplementedError("load_csv() no implementado todavía.")


def load_tsv(ruta, verbose=1):
    """Cargar un dataset desde un archivo TSV (tab-separated values).

    Parámetros
    ----------
    ruta : str
        Ruta al archivo TSV a leer.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado (no implementado todavía).

    Raises
    ------
    NotImplementedError
        Si la lectura de TSV no ha sido implementada.
    """
    _verbose(f"Cargando TSV desde '{ruta}'...", verbose)
    # TODO: df = pd.read_csv(ruta, sep="\t")
    raise NotImplementedError("load_tsv() no implementado todavía.")


def load_excel(ruta, hoja=0, verbose=1):
    """Cargar un dataset desde un archivo Excel.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo Excel a leer.
    hoja : int, por defecto 0
        Índice de la hoja a leer (0 = primera hoja).
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado (no implementado todavía).

    Raises
    ------
    NotImplementedError
        Si la lectura de Excel no ha sido implementada.
    """
    _verbose(f"Cargando Excel desde '{ruta}' (hoja {hoja})...", verbose)
    # TODO: df = pd.read_excel(ruta, sheet_name=hoja)
    raise NotImplementedError("load_excel() no implementado todavía.")


def router(dataset="iris", verbose=1):
    """Enrutador simple: elige el loader según el tipo de dataset.

    Parámetros
    ----------
    dataset : str, por defecto "iris"
        Nombre del dataset a cargar. Por ahora solo se soporta el iris
        (in-memory vía scikit-learn).
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.
    """
    return load_dataset(dataset, verbose)



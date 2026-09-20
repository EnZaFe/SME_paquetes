# Cargar cualquier tipo de dataset y devolverlo como pandas.DataFrame.
#
# Formatos soportados:
#   - Datasets en memoria de scikit-learn: "iris", "wine", "breast_cancer"
#   - CSV  (.csv, .txt)  -> separador autodetectado (, ; tab |)
#   - TSV  (.tsv, .tab)
#   - Excel (.xlsx, .xlsm, .xls, .ods)
#
# Uso rápido:
#   df = router("iris")
#   df = router("datos/ventas.csv")
#   df = router("datos/ventas.xlsx", hoja="Enero")

import csv
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from v_python.src.auxiliar.verbose import _verbose

# Datasets incluidos en scikit-learn que se pueden pedir por nombre.
_SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


# ---------------------------------------------------------------------------
# Datasets por nombre
# ---------------------------------------------------------------------------
def load_dataset(dataset="iris", verbose=1):
    """Cargar un dataset de scikit-learn por nombre y devolverlo como DataFrame.

    Parámetros
    ----------
    dataset : str, por defecto "iris"
        Nombre del dataset: "iris", "wine" o "breast_cancer".
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.

    Raises
    ------
    ValueError
        Si el dataset solicitado no está soportado.
    """
    _verbose(f"Cargando dataset '{dataset}'...", verbose)

    loader = _SKLEARN_DATASETS.get(dataset)
    if loader is None:
        soportados = ", ".join(sorted(_SKLEARN_DATASETS))
        raise ValueError(
            f"Dataset '{dataset}' no soportado. Disponibles: {soportados}."
        )

    df = loader(as_frame=True).frame
    _verbose(
        f"Dataset '{dataset}' cargado: {df.shape[0]} filas x {df.shape[1]} columnas.",
        verbose,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _comprobar_archivo(ruta):
    """Lanza FileNotFoundError si la ruta no es un archivo existente."""
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe el archivo: '{ruta}'")


def _detectar_separador(ruta, encoding):
    """Detecta el separador de un archivo de texto mirando su primer bloque."""
    with open(ruta, "r", encoding=encoding, newline="") as f:
        muestra = f.read(65536)
    try:
        return csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","  # si no se puede detectar, se asume coma


def _leer_texto(ruta, sep=None, encoding=None, **kwargs):
    """Lee un archivo de texto delimitado probando varias codificaciones.

    Si `encoding` no se indica, prueba utf-8 (con o sin BOM) y después latin-1,
    que es lo habitual en CSV exportados desde Excel en Windows.
    Si `sep` es None, el separador se autodetecta.
    """
    codificaciones = [encoding] if encoding else ["utf-8-sig", "latin-1"]
    ultimo_error = None

    for enc in codificaciones:
        try:
            separador = sep if sep is not None else _detectar_separador(ruta, enc)
            return pd.read_csv(ruta, sep=separador, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            ultimo_error = e

    raise ultimo_error


def _resumen(df, verbose):
    _verbose(f"Cargado: {df.shape[0]} filas x {df.shape[1]} columnas.", verbose)


# ---------------------------------------------------------------------------
# Loaders de archivos
# ---------------------------------------------------------------------------
def load_csv(ruta, sep=None, encoding=None, verbose=1, **kwargs):
    """Cargar un dataset desde un archivo CSV.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV a leer.
    sep : str, opcional
        Separador. Si es None se autodetecta entre `,` `;` tab y `|`
        (útil para CSV "a la española", que usan `;`).
    encoding : str, opcional
        Codificación. Si es None se prueba utf-8 y luego latin-1.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).
    **kwargs
        Argumentos extra que se pasan a `pandas.read_csv`
        (por ejemplo `decimal=","`, `na_values=[...]`).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.
    """
    _verbose(f"Cargando CSV desde '{ruta}'...", verbose)
    _comprobar_archivo(ruta)
    df = _leer_texto(ruta, sep=sep, encoding=encoding, **kwargs)
    _resumen(df, verbose)
    return df


def load_tsv(ruta, encoding=None, verbose=1, **kwargs):
    """Cargar un dataset desde un archivo TSV (tab-separated values).

    Parámetros
    ----------
    ruta : str
        Ruta al archivo TSV a leer.
    encoding : str, opcional
        Codificación. Si es None se prueba utf-8 y luego latin-1.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).
    **kwargs
        Argumentos extra que se pasan a `pandas.read_csv`.

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.
    """
    _verbose(f"Cargando TSV desde '{ruta}'...", verbose)
    _comprobar_archivo(ruta)
    df = _leer_texto(ruta, sep="\t", encoding=encoding, **kwargs)
    _resumen(df, verbose)
    return df


def load_excel(ruta, hoja=0, verbose=1, **kwargs):
    """Cargar un dataset desde un archivo Excel.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo Excel a leer (.xlsx, .xlsm, .xls, .ods).
    hoja : int | str, por defecto 0
        Índice (0 = primera hoja) o nombre de la hoja a leer.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).
    **kwargs
        Argumentos extra que se pasan a `pandas.read_excel`.

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.

    Raises
    ------
    ImportError
        Si falta el motor de lectura (openpyxl para .xlsx, xlrd para .xls,
        odfpy para .ods).
    """
    _verbose(f"Cargando Excel desde '{ruta}' (hoja {hoja})...", verbose)
    _comprobar_archivo(ruta)
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"{e}\nInstala el motor necesario: `pip install openpyxl` "
            "(.xlsx/.xlsm), `pip install xlrd` (.xls) o `pip install odfpy` (.ods)."
        ) from e
    _resumen(df, verbose)
    return df


# ---------------------------------------------------------------------------
# Enrutador
# ---------------------------------------------------------------------------
_LOADERS_POR_EXTENSION = {
    ".csv": load_csv,
    ".txt": load_csv,
    ".tsv": load_tsv,
    ".tab": load_tsv,
    ".xlsx": load_excel,
    ".xlsm": load_excel,
    ".xls": load_excel,
    ".ods": load_excel,
}


def router(dataset="iris", verbose=1, **kwargs):
    """Enrutador: elige el loader según lo que se le pase y devuelve un DataFrame.

    Parámetros
    ----------
    dataset : str | os.PathLike | pandas.DataFrame, por defecto "iris"
        - Nombre de un dataset de scikit-learn ("iris", "wine", "breast_cancer").
        - Ruta a un archivo .csv/.txt, .tsv/.tab o .xlsx/.xlsm/.xls/.ods.
        - Un DataFrame (se devuelve tal cual).
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).
    **kwargs
        Opciones extra para el loader de archivos (por ejemplo `sep=";"`,
        `encoding="latin-1"` o `hoja="Datos"`). Se ignoran para datasets por nombre.

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.

    Raises
    ------
    FileNotFoundError
        Si parece una ruta a un archivo soportado pero no existe.
    ValueError
        Si la extensión no está soportada o el nombre de dataset es desconocido.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset

    dataset = os.fspath(dataset)
    extension = os.path.splitext(dataset)[1].lower()

    # 1) Archivo existente o con extensión conocida -> loader por extensión
    if extension in _LOADERS_POR_EXTENSION:
        return _LOADERS_POR_EXTENSION[extension](dataset, verbose=verbose, **kwargs)

    # 2) Archivo existente con extensión no soportada
    if os.path.isfile(dataset):
        soportadas = ", ".join(sorted(_LOADERS_POR_EXTENSION))
        raise ValueError(
            f"Extensión '{extension}' no soportada. Soportadas: {soportadas}."
        )

    # 3) Si no, se interpreta como nombre de dataset
    return load_dataset(dataset, verbose)
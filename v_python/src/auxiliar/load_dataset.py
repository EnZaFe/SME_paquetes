import csv
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from auxiliar.verbose import _verbose

# Datasets incluidos en scikit-learn que se pueden pedir por nombre.
_SKLEARN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

_EXCEL_EXT = (".xlsx", ".xlsm", ".xls", ".ods")
_TEXTO_EXT = (".csv", ".txt", ".tsv", ".tab")


def load_dataset(
    dataset="iris", sep=None, encoding=None, hoja=0, verbose=1, **kwargs
):
    """Cargar un dataset (por nombre, ruta o DataFrame) y devolverlo como DataFrame.

    Parámetros
    ----------
    dataset : str | os.PathLike | pandas.DataFrame, por defecto "iris"
        - Nombre de un dataset de scikit-learn ("iris", "wine", "breast_cancer").
        - Ruta a un archivo .csv/.txt, .tsv/.tab o .xlsx/.xlsm/.xls/.ods.
        - Un DataFrame (se devuelve tal cual).
    sep : str, opcional
        Separador para archivos de texto. Si es None se autodetecta entre
        `,` `;` tab y `|`.
    encoding : str, opcional
        Codificación. Si es None se prueba utf-8 (con BOM) y luego latin-1,
        que es lo habitual en CSV exportados desde Excel en Windows.
    hoja : int | str, por defecto 0
        Índice (0 = primera hoja) o nombre de la hoja, solo para Excel.
    verbose : int
        Nivel de verbosidad (0 = nada, 1 = básico).
    **kwargs
        Argumentos extra que se pasan a `pandas.read_csv` o `pandas.read_excel`
        (por ejemplo `decimal=","`, `na_values=[...]`).

    Devuelve
    -------
    pandas.DataFrame
        El dataset cargado.

    Raises
    ------
    FileNotFoundError
        Si la ruta tiene una extensión de archivo pero no existe.
    ValueError
        Si la extensión no está soportada o el nombre de dataset es desconocido.
    ImportError
        Si falta el motor de lectura de Excel (openpyxl, xlrd u odfpy).
    """
    # 0) Ya es un DataFrame: se devuelve tal cual.
    if isinstance(dataset, pd.DataFrame):
        _verbose("Ya es un DataFrame, se devuelve tal cual.", verbose)
        return dataset

    ruta = os.fspath(dataset)
    ext = os.path.splitext(ruta)[1].lower()

    # 1) Sin extensión -> nombre de dataset de scikit-learn.
    if not ext:
        _verbose(f"Cargando dataset '{ruta}'...", verbose)
        loader = _SKLEARN_DATASETS.get(ruta)
        if loader is None:
            soportados = ", ".join(sorted(_SKLEARN_DATASETS))
            raise ValueError(
                f"Dataset '{ruta}' no soportado. Disponibles: {soportados}."
            )
        df = loader(as_frame=True).frame

    # 2) Con extensión -> archivo. Tiene que existir.
    else:
        if not os.path.isfile(ruta):
            raise FileNotFoundError(f"No existe el archivo: '{ruta}'")

        # 2a) Excel (autodetectado por extensión).
        if ext in _EXCEL_EXT:
            _verbose(f"Cargando Excel desde '{ruta}' (hoja {hoja})...", verbose)
            try:
                df = pd.read_excel(ruta, sheet_name=hoja, **kwargs)
            except ImportError as e:
                raise ImportError(
                    f"{e}\nInstala el motor necesario: `pip install openpyxl` "
                    "(.xlsx/.xlsm), `pip install xlrd` (.xls) o "
                    "`pip install odfpy` (.ods)."
                ) from e

        # 2b) Texto delimitado (CSV/TSV).
        elif ext in _TEXTO_EXT:
            _verbose(f"Cargando texto delimitado desde '{ruta}'...", verbose)
            if sep is None and ext in (".tsv", ".tab"):
                sep = "\t"
            codificaciones = [encoding] if encoding else ["utf-8-sig", "latin-1"]

            for i, enc in enumerate(codificaciones):
                try:
                    separador = sep
                    if separador is None:
                        with open(ruta, "r", encoding=enc, newline="") as f:
                            muestra = f.read(65536)
                        try:
                            separador = (
                                csv.Sniffer()
                                .sniff(muestra, delimiters=",;\t|")
                                .delimiter
                            )
                        except csv.Error:
                            separador = ","  # si no se detecta, se asume coma
                        _verbose(f"Separador detectado: {separador!r}", verbose)
                    df = pd.read_csv(ruta, sep=separador, encoding=enc, **kwargs)
                    break
                except UnicodeDecodeError:
                    _verbose(f"Codificación '{enc}' no válida, probando otra...", verbose)
                    if i == len(codificaciones) - 1:
                        raise

        # 2c) Extensión desconocida.
        else:
            soportadas = ", ".join(_TEXTO_EXT + _EXCEL_EXT)
            raise ValueError(
                f"Extensión '{ext}' no soportada. Soportadas: {soportadas}."
            )

    _verbose(f"Cargado: {df.shape[0]} filas x {df.shape[1]} columnas.", verbose)
    return df
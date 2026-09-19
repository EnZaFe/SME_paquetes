#Codigo auiliar con metodos que se usan en varias funcionalidades
import pandas as pd


def _es_variable(datos):
    """
    Determina si `datos` es una variable numérica.

    Una variable es un ``pd.Series`` cuyo tipo de dato es numérico.
    No basta con ser un ``Series``: debe contener números (enteros o
    decimales).

    Parameters
    ----------
    datos : object
        Valor a comprobar.

    Returns
    -------
    bool
        ``True`` si es un ``pd.Series`` numérico, ``False`` en caso
        contrario.
    """
    return (
        isinstance(datos, pd.Series)
        and pd.api.types.is_numeric_dtype(datos)
    )


def _es_dataset(datos):
    """
    Determina si `datos` es un dataset.

    Un dataset es cualquier ``pd.DataFrame`` (tabla con filas y columnas).

    Parameters
    ----------
    datos : object
        Valor a comprobar.

    Returns
    -------
    bool
        ``True`` si es un ``pd.DataFrame``, ``False`` en caso contrario.
    """
    return isinstance(datos, pd.DataFrame)


def _es_continua(columna):
    """
    Determina si una columna representa una variable continua.

    Es continua cuando pandas la reconoce como numérica o cuando sus
    valores ``object`` pueden convertirse a números sin error (por
    ejemplo una columna de enteros leídos como texto).

    WARNING: Esto puede ser peligroso si se utilizan numeros enteros para representar variables categoricas '1': Rojo, '2': Verde...

    Parameters
    ----------
    columna : pandas.Series
        Columna a comprobar.

    Returns
    -------
    bool
        ``True`` si la columna es numérica o convertible a numérica,
        ``False`` en caso contrario.
    """

    # Si pandas ya sabe que es numérica.
    if pd.api.types.is_numeric_dtype(columna):
        return True

    # Si es object, intentamos convertir sus valores a números.
    if pd.api.types.is_object_dtype(columna):

        valores = columna.dropna()

        if len(valores) == 0:
            return False

        try:
            pd.to_numeric(valores)
            return True

        except (TypeError, ValueError):
            return False

    return False


def _es_discreta(columna):
    """
    Determina si una columna representa una variable discreta/categórica.

    Es discreta cuando es booleana, categórica (``CategoricalDtype``),
    de tipo string, o ``object`` con valores simples (no listas,
    diccionarios ni conjuntos). Una columna que sea continua nunca se
    considera discreta.

    Parameters
    ----------
    columna : pandas.Series
        Columna a comprobar.

    Returns
    -------
    bool
        ``True`` si la columna contiene valores categóricos,
        ``False`` en caso contrario.
    """

    # Booleanos.
    if pd.api.types.is_bool_dtype(columna):
        return True

    # Categorical de pandas.
    if isinstance(columna.dtype, pd.CategoricalDtype):
        return True

    # Si ya sabemos que es continua, no puede ser discreta.
    if _es_continua(columna):
        return False

    # Para object tenemos que mirar los valores reales.
    if pd.api.types.is_object_dtype(columna):

        valores = columna.dropna()

        if len(valores) == 0:
            return False

        for valor in valores:

            # Estos tipos no son valores categóricos simples.
            if isinstance(
                valor,
                (list, dict, set, tuple)
            ):
                return False

        # Si son valores simples, los consideramos categóricos.
        return True

    # Strings normales.
    if pd.api.types.is_string_dtype(columna):
        return True

    return False
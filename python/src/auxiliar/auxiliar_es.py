#Codigo auiliar con metodos que se usan en varias funcionalidades
import pandas as pd
def _es_variable(datos):
    """Devuelve True si `datos` es una variable numérica (pd.Series)."""
    return (
        isinstance(datos, pd.Series)
        and pd.api.types.is_numeric_dtype(datos)
    )

def _es_dataset(datos):
    """Devuelve True si `datos` es un dataset (DataFrame)."""
    return isinstance(datos, pd.DataFrame)

def _es_continua(columna):
    """Devuelve True si una columna contiene valores numéricos."""

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
    """Devuelve True si una columna contiene valores discretos/categóricos."""

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
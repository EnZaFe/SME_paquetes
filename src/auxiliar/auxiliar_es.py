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

def _es_discreta(columna): 
    """Devuelve True si una columna es discreta/categórica.""" 
def _es_discreta(columna):
    """Devuelve True si una columna es discreta/categórica."""
    return (
        pd.api.types.is_object_dtype(columna)
        or pd.api.types.is_string_dtype(columna) # Bestela pd.Series(["M", "F", "M", "F"]) no detecta.
        or isinstance(columna.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(columna)
    )
def _es_continua(columna): 
    """Devuelve True si una columna es numérica.""" 
    return pd.api.types.is_numeric_dtype(columna)

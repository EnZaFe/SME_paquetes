'''
FILTRADO DE VARIABLES

`filtrar_variables` recibe un dataset y filtra sus variables según los
valores de sus métricas.

Parámetros:
- `dataset`: DataFrame que queremos filtrar.
- `entropia`: `(min, max)` para filtrar variables discretas.
- `varianza`: `(min, max)` para filtrar variables continuas.
- `auc`: `(min, max)` para filtrar variables continuas.
- `clases`: variable clase binaria necesaria para calcular el AUC.
- 'modo': 
    "dentro" (5,7) → [5,7]
    "fuera"  (5,7) → (-∞,5) ∪ (7,∞)
- `atributos`: lista opcional de variables a evaluar.
- `verbose`: nivel de información mostrado.



Se pueden combinar varias métricas. Si se especifican varias, deben cumplirse
todas las condiciones correspondientes a cada variable.

Devuelve un nuevo DataFrame con las variables que cumplen los requisitos.
'''
from SME_python.metricas_de_variables import calcular_metricas
from auxiliar.verbose import _verbose
from auxiliar.auxiliar_es import _es_continua, _es_discreta
import pandas as pd

def filtrar_variables(
    dataset,
    entropia=None,
    varianza=None,
    auc=None,
    modo="dentro",
    clases=None,
    atributos=None,
    metricas=None,
    verbose=1
):
    """API PUBLICA"""

    _verbose("Iniciando filtrado de variables...", verbose)

    if not isinstance(dataset, pd.DataFrame):
        _verbose(
            "El objeto recibido no es un dataset.",
            verbose,
            tipo="warning"
        )
        raise TypeError("dataset debe ser un DataFrame.")
    
    if entropia is None and varianza is None and auc is None:
        _verbose(
            "No se ha especificado ningún criterio de filtrado.",
            verbose,
            tipo="warning"
        )

    if metricas is None:
        _verbose(
            "No se han recibido métricas. "
            "Se calcularán automáticamente.",
            verbose,
            nivel=2
        )


        if clases is None and auc is not None:
            _verbose(
                "No se han proporcionado clases. "
                "El filtro AUC no será evaluado.",
                verbose,
                tipo="warning"
            )

        if atributos is None:
            _verbose(
                "No se han especificado atributos. "
                "Se evaluarán todas las variables.",
                verbose,
                nivel=2
            )

        metricas = calcular_metricas(
            dataset,
            clases=clases,
            atributos=atributos,
            verbose=verbose
        )

    elif not isinstance(metricas, dict):
        _verbose(
            "El objeto recibido no contiene métricas válidas.",
            verbose,
            tipo="warning"
        )
        raise TypeError("metricas debe ser un diccionario.")
    
    columnas_validas = []

    for columna, valores in metricas.items():

        if _es_discreta(dataset[columna]):

            if _cumple_filtro(valores.get("entropia"), entropia, modo):
                columnas_validas.append(columna)

        elif _es_continua(dataset[columna]):

            if (_cumple_filtro(valores.get("varianza"), varianza, modo)
                    and _cumple_filtro(valores.get("auc"), auc, modo)):
                columnas_validas.append(columna)

    return dataset[columnas_validas].copy()

    
def _cumple_filtro(valor, intervalo, modo):
    """Comprueba si un valor cumple un intervalo.
    
    Ejemplos:
    _cumple_filtro(5, (None, 7), "dentro")  # True
    _cumple_filtro(8, (None, 7), "dentro")  # False

    _cumple_filtro(5, (None, 7), "fuera")   # False
    _cumple_filtro(8, (None, 7), "fuera")   # True
    
    
    
    """

    if intervalo is None:
        return True

    if valor is None:
        return True

    minimo, maximo = intervalo

    if modo == "dentro":
        if minimo is not None and valor < minimo:
            return False

        if maximo is not None and valor > maximo:
            return False

        return True

    elif modo == "fuera":
        if minimo is not None and maximo is not None:
            return valor < minimo or valor > maximo

        if minimo is not None:
            return valor < minimo

        if maximo is not None:
            return valor > maximo

        return True

    else:
        raise ValueError(
            "modo debe ser 'dentro' o 'fuera'."
        )
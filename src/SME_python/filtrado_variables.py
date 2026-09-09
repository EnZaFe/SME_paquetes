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
- `atributos`: lista opcional de variables a evaluar.
- `verbose`: nivel de información mostrado.

`None` indica que no hay límite:
- `(0.5, None)` → >= 0.5
- `(None, 1.5)` → <= 1.5
- `(0.5, 1.5)` → entre 0.5 y 1.5
- `None` → no se aplica ese filtro.

Se pueden combinar varias métricas. Si se especifican varias, deben cumplirse
todas las condiciones correspondientes a cada variable.

Devuelve un nuevo DataFrame con las variables que cumplen los requisitos.
'''
from SME_python.metricas_de_variables import calcular_metricas
from auxiliar.verbose import _verbose
from auxiliar.auxiliar_es import _es_continua, _es_discreta


def filtrar_variables(
    dataset,
    entropia=None,
    varianza=None,
    auc=None,
    clases=None,
    atributos=None,
    verbose=1
):
    """API PUBLICA"""

    _verbose("Iniciando filtrado de variables...", verbose)

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

    if entropia is None and varianza is None and auc is None:
        _verbose(
            "No se ha especificado ningún criterio de filtrado.",
            verbose,
            tipo="warning"
        )

    metricas = calcular_metricas(
        dataset,
        clases=clases,
        atributos=atributos,
        verbose=verbose
    )


    columnas_validas = []

    for columna, valores in metricas.items():

        if _es_discreta(dataset[columna]):

            if _cumple_filtro(valores.get("entropia"), entropia):
                columnas_validas.append(columna)

        elif _es_continua(dataset[columna]):

            if (_cumple_filtro(valores.get("varianza"), varianza)
                    and _cumple_filtro(valores.get("auc"), auc)):
                columnas_validas.append(columna)

    return dataset[columnas_validas].copy()

    
def _cumple_filtro(valor, intervalo):
        """Comprueba si un valor cumple un intervalo (min, max)."""

        if intervalo is None:
            return True

        minimo, maximo = intervalo

        if valor is None:
            return True

        if minimo is not None and valor < minimo:
            return False

        if maximo is not None and valor > maximo:
            return False

        return True
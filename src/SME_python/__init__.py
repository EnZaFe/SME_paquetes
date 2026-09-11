"""SME_python.

Librería para la gestión y análisis de datasets.

Módulos públicos:
    - discretizacion
    - metricas_de_variables
    - normalizacion_estandarizacion
    - filtrado_variables
    - correlacion_info
    - visualizacion
"""

__version__ = "0.1.0"


# Discretización
from .discretizacion import discretizar

# Métricas
from .metricas_de_variables import calcular_metricas

# Normalización y estandarización
from .normalizacion_estandarizacion import normalizar, estandarizar

# Filtrado de variables
from .filtrado_variables import filtrar_variables

# Correlación
from .correlacion_info import calcular_correlacion

# Visualización
from .visualizacion import (
    graficar_auc,
    graficar_pearson,
    graficar_informacion_mutua,
    graficar_welch,
)


__all__ = [
    "__version__",
    "discretizar",
    "calcular_metricas",
    "normalizar",
    "estandarizar",
    "filtrar_variables",
    "calcular_correlacion",
    "graficar_auc",
    "graficar_pearson",
    "graficar_informacion_mutua",
    "graficar_welch",
]
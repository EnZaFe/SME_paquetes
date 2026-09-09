"""SME_python.

Librería para la gestión y análisis de datasets.

Módulos públicos (una funcionalidad por módulo):

    - discretizacion:            discretización por igual frecuencia / igual anchura.
    - metricas_de_variables:     varianza, entropía y AUC por variable.
    - normalizacion_estandarizacion: normalización (min-max) y estandarización (z-score).
    - filtrado_variables:        selección de variables según una métrica y umbral.
    - correlacion_info:          correlación e información mutua entre variables.
    - visualizacion:             heatmaps, plots de AUC, etc.

Ejemplo de uso::

    import SME_python as sme

    from sme import discretizacion
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "discretizacion",
    "metricas_de_variables",
    "normalizacion_estandarizacion",
    "filtrado_variables",
    "correlacion_info",
    "visualizacion",
]

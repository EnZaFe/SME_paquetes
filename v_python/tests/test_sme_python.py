"""Pruebas de importación y estructura del paquete SME_python.

comprueban que el paquete y todos sus módulos públicos se pueden importar correctamente una vez instalado en modo editable.

***Las pruebas de cada función se añadirán a medida que se implementen.
"""

import pytest

from python.src.SME_python import correlacion_info, discretizacion, filtrado_variables, metricas_de_variables, normalizacion_estandarizacion


def test_paquete_importable():
    import python.src.SME_python as sme

    assert sme.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "modulo",
    [
        "discretizacion",
        "metricas_de_variables",
        "normalizacion_estandarizacion",
        "filtrado_variables",
        "correlacion_info",
        "visualizacion",
    ],
)
def test_modulos_publicos_importables(modulo):
    import python.src.SME_python as sme

    assert hasattr(sme, modulo), f"Módulo '{modulo}' no es accesible desde el paquete."


def test_import_directo_de_cada_modulo():
    from python.src.SME_python import (  # noqa: F401
        visualizacion,
    )

"""
Tests de la discretización.

Objetivos:
- Probar la discretización de una variable numérica.
- Probar la discretización de un DataFrame.
- Mostrar los intervalos creados y su anchura.
- Comprobar qué columnas se discretizan.
- Comprobar los errores de los parámetros inválidos.
"""

import pandas as pd
import pytest
from SME_python.discretizacion import discretizar
from auxiliar.load_dataset import load_dataset

# ============================================================
# TESTS DE DISCRETIZACIÓN POR IGUAL ANCHURA
# ============================================================

def test_anchura_n_intervalos():
    """
    Comprueba que se crean correctamente los intervalos
    indicando directamente n_intervalos.
    """

    columna = pd.Series([
        10, 20, 30, 40,
        50, 60, 70, 80
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        n_intervalos=4,
        verbose=0
    )

    # Debe devolver una lista
    assert isinstance(resultado, list)

    # Debe haber un intervalo por cada valor
    assert len(resultado) == len(columna)

    # Deben existir exactamente 4 intervalos distintos
    intervalos = list(dict.fromkeys(resultado))

    assert len(intervalos) == 4


def test_anchura_intervalos_correctos():
    """
    Comprueba que los límites de los intervalos
    son los esperados.
    """

    columna = pd.Series([
        10, 20, 30, 40,
        50, 60, 70, 80
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        n_intervalos=4,
        verbose=0
    )

    intervalos = list(dict.fromkeys(resultado))

    # Anchura:
    #
    # (80 - 10) / 4 = 17.5
    #
    # Límites:
    # 10
    # 27.5
    # 45
    # 62.5
    # 80

    intervalos_esperados = [
        (10.0, 27.5),
        (27.5, 45.0),
        (45.0, 62.5),
        (62.5, 80.0)
    ]

    assert intervalos == intervalos_esperados


def test_anchura_frecuencia_elementos():
    """
    Comprueba cuántos elementos caen dentro de cada
    intervalo de igual anchura.
    """

    columna = pd.Series([
        10, 20, 30, 40,
        50, 60, 70, 80
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        n_intervalos=4,
        verbose=0
    )

    intervalos = list(dict.fromkeys(resultado))

    frecuencias = [
        resultado.count(intervalo)
        for intervalo in intervalos
    ]

    assert frecuencias == [2, 2, 2, 2]


def test_anchura_cubre_todos_los_datos():
    """
    Comprueba que el primer intervalo empieza en el mínimo
    y el último termina en el máximo.
    """

    columna = pd.Series([
        10, 20, 30, 40,
        50, 60, 70, 80
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        n_intervalos=4,
        verbose=0
    )

    intervalos = list(dict.fromkeys(resultado))

    minimo_global = min(
        intervalo[0]
        for intervalo in intervalos
    )

    maximo_global = max(
        intervalo[1]
        for intervalo in intervalos
    )

    assert minimo_global == columna.min()
    assert maximo_global == columna.max()


def test_anchura_intervalos_son_tuplas():
    """
    Comprueba que los intervalos devueltos tienen el formato
    esperado: tuplas de dos números.
    """

    columna = pd.Series([
        10, 20, 30,
        40, 50, 60
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        n_intervalos=3,
        verbose=0
    )

    intervalos = list(dict.fromkeys(resultado))

    for intervalo in intervalos:

        assert isinstance(intervalo, tuple)
        assert len(intervalo) == 2

        minimo, maximo = intervalo

        assert isinstance(minimo, (int, float))
        assert isinstance(maximo, (int, float))

        assert minimo <= maximo


def test_anchura_por_defecto():
    """
    Comprueba que anchura utiliza 5 intervalos cuando
    no se especifica n_intervalos.
    """

    columna = pd.Series([
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100
    ])

    resultado = discretizar(
        datos=columna,
        metodo="anchura",
        verbose=0
    )

    assert isinstance(resultado, list)
    assert len(resultado) == len(columna)

    intervalos = list(dict.fromkeys(resultado))

    assert len(intervalos) == 5


def test_error_anchura_n_intervalos_cero():
    """
    n_intervalos no puede ser 0.
    """

    datos = pd.Series([
        10, 20, 30
    ])

    with pytest.raises(
        ValueError,
        match=r"n_intervalos debe ser mayor que 0"
    ):
        discretizar(
            datos=datos,
            metodo="anchura",
            n_intervalos=0
        )


def test_error_anchura_n_intervalos_negativo():
    """
    n_intervalos no puede ser negativo.
    """

    datos = pd.Series([
        10, 20, 30
    ])

    with pytest.raises(
        ValueError,
        match=r"n_intervalos debe ser mayor que 0"
    ):
        discretizar(
            datos=datos,
            metodo="anchura",
            n_intervalos=-2
        )


def test_error_anchura_n_intervalos_demasiados():
    """
    No puede haber más intervalos que casos.
    """

    datos = pd.Series([
        10, 20, 30
    ])

    with pytest.raises(
        ValueError,
        match=r"n_intervalos \d+ no puede ser mayor "
              r"que el número de casos \d+"
    ):
        discretizar(
            datos=datos,
            metodo="anchura",
            n_intervalos=5
        )


def test_error_anchura_valores_iguales():
    """
    No se puede calcular una anchura si todos los valores
    de la variable son iguales.
    """

    datos = pd.Series([
        10, 10, 10, 10
    ])

    with pytest.raises(
        ValueError,
        match="todos los valores son iguales"
    ):
        discretizar(
            datos=datos,
            metodo="anchura",
            n_intervalos=3
        )


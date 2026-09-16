import pytest
import pandas as pd
import numpy as np

from python.src.SME_python.correlacion_info import (
    calcular_correlacion,
    _calc_corr_num,
    _calc_corr_catg,
    _calc_corr_catg_num
)


# ============================================================
# TESTS DE PEARSON
# ============================================================

def test_pearson_positivo():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])

    resultado = _calc_corr_num(x, y)

    assert resultado == pytest.approx(1.0)


def test_pearson_negativo():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([10, 8, 6, 4, 2])

    resultado = _calc_corr_num(x, y)

    assert resultado == pytest.approx(-1.0)


def test_pearson_sin_correlacion_lineal():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([1, -1, 1, -1, 1])

    resultado = _calc_corr_num(x, y)

    assert resultado == pytest.approx(0.0)


def test_pearson_variable_constante():
    x = pd.Series([1, 1, 1, 1, 1])
    y = pd.Series([1, 2, 3, 4, 5])

    with pytest.raises(ValueError):
        _calc_corr_num(x, y)


def test_pearson_con_nan():
    x = pd.Series([1, 2, np.nan, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])

    resultado = _calc_corr_num(x, y)

    assert resultado == pytest.approx(1.0)


# ============================================================
# TESTS DE MUTUAL INFORMATION
# ============================================================

def test_mi_dependencia_perfecta():
    x = pd.Series(["A", "A", "B", "B"])
    y = pd.Series(["X", "X", "Y", "Y"])

    resultado = _calc_corr_catg(x, y)

    assert resultado > 0


def test_mi_independencia():
    x = pd.Series(["A", "A", "B", "B"])
    y = pd.Series(["X", "Y", "X", "Y"])

    resultado = _calc_corr_catg(x, y)

    assert resultado == pytest.approx(0.0)


def test_mi_con_nan():
    x = pd.Series(["A", "A", "B", None])
    y = pd.Series(["X", "X", "Y", "Y"])

    resultado = _calc_corr_catg(x, y)

    assert resultado > 0


# ============================================================
# TESTS DE WELCH ANOVA
# ============================================================

def test_welch_anova_diferencias():
    grupos = pd.Series([
        "A", "A", "A",
        "B", "B", "B",
        "C", "C", "C"
    ])

    valores = pd.Series([
        1, 2, 3,
        10, 11, 12,
        20, 21, 22
    ])

    resultado = _calc_corr_catg_num(
        grupos,
        valores
    )

    assert resultado > 0


def test_welch_anova_medias_iguales():
    grupos = pd.Series([
        "A", "A", "A",
        "B", "B", "B",
        "C", "C", "C"
    ])

    valores = pd.Series([
        1, 2, 3,
        1, 2, 3,
        1, 2, 3
    ])

    resultado = _calc_corr_catg_num(
        grupos,
        valores
    )

    assert resultado == pytest.approx(0.0)


def test_welch_anova_un_solo_grupo():
    grupos = pd.Series([
        "A", "A", "A"
    ])

    valores = pd.Series([
        1, 2, 3
    ])

    with pytest.raises(ValueError):
        _calc_corr_catg_num(
            grupos,
            valores
        )


def test_welch_anova_varianza_cero():
    grupos = pd.Series([
        "A", "A", "A",
        "B", "B", "B"
    ])

    valores = pd.Series([
        1, 1, 1,
        2, 3, 4
    ])

    with pytest.raises(ValueError):
        _calc_corr_catg_num(
            grupos,
            valores
        )


def test_welch_anova_con_nan():
    grupos = pd.Series([
        "A", "A", "A",
        "B", "B", "B",
        None
    ])

    valores = pd.Series([
        1, 2, 3,
        10, 11, 12,
        20
    ])

    resultado = _calc_corr_catg_num(
        grupos,
        valores
    )

    assert resultado > 0


# ============================================================
# TESTS DE CALCULAR_CORRELACION
# ============================================================

def test_calcular_correlacion_dataset_mixto():

    dataset = pd.DataFrame({
        "edad": [20, 21, 22, 23, 24, 25],
        "salario": [20000, 21000, 22000, 23000, 24000, 25000],
        "departamento": [
            "A", "A", "B", "B", "C", "C"
        ]
    })

    resultado = calcular_correlacion(dataset)

    assert isinstance(resultado, pd.DataFrame)

    assert list(resultado.columns) == [
        "edad",
        "salario",
        "departamento"
    ]

    assert list(resultado.index) == [
        "edad",
        "salario",
        "departamento"
    ]


def test_calcular_correlacion_pearson():

    dataset = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    resultado = calcular_correlacion(dataset)

    assert resultado.loc["x", "y"] == pytest.approx(1.0)
    assert resultado.loc["y", "x"] == pytest.approx(1.0)


def test_matriz_simetrica():

    dataset = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    resultado = calcular_correlacion(dataset)

    assert resultado.equals(resultado.T)


def test_diagonal_uno():

    dataset = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1]
    })

    resultado = calcular_correlacion(
        dataset,
        atributos=["x", "y"]
    )

    assert resultado.loc["x", "x"] == 1.0
    assert resultado.loc["y", "y"] == 1.0


# ============================================================
# TESTS DE SELECCION DE ATRIBUTOS
# ============================================================

def test_seleccion_atributos():

    dataset = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1],
        "z": [10, 20, 30]
    })

    resultado = calcular_correlacion(
        dataset,
        atributos=["x", "y"]
    )

    assert list(resultado.columns) == ["x", "y"]
    assert list(resultado.index) == ["x", "y"]


def test_atributo_inexistente():

    dataset = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1]
    })

    with pytest.raises(ValueError):
        calcular_correlacion(
            dataset,
            atributos=["x", "z"]
        )


# ============================================================
# TESTS DE VALIDACION
# ============================================================

def test_dataset_invalido():

    with pytest.raises(TypeError):
        calcular_correlacion(
            [1, 2, 3]
        )


def test_atributos_debe_ser_lista():

    dataset = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1]
    })

    with pytest.raises(TypeError):
        calcular_correlacion(
            dataset,
            atributos="x"
        )


# ============================================================
# TEST DE ATRIBUTOS INVALIDOS
# ============================================================
def test_atributo_no_numerico_ni_discreto():

    dataset = pd.DataFrame({
        "x": [1, 2, 3],
        "texto": [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
    })

    resultado = calcular_correlacion(dataset)

    assert "x" in resultado.columns
    assert "x" in resultado.index

    assert "texto" not in resultado.columns
    assert "texto" not in resultado.index
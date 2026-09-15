
import pytest
import pandas as pd
import numpy as np

from python.src.SME_python.normalizacion_estandarizacion import normalizar, estandarizar


# ============================================================
# DATOS DE PRUEBA
# ============================================================

@pytest.fixture
def variable():
    """Variable numérica para las pruebas."""
    return pd.Series([1, 2, 3, 4, 5], name="edad")


@pytest.fixture
def variable_constante():
    """Variable numérica cuyos valores son todos iguales."""
    return pd.Series([5, 5, 5, 5], name="constante")


@pytest.fixture
def dataset():
    """Dataset con variables numéricas y una categórica."""
    return pd.DataFrame({
        "edad": [10, 20, 30, 40, 50],
        "altura": [150, 160, 170, 180, 190],
        "sexo": ["M", "F", "M", "F", "M"]
    })


# ============================================================
# NORMALIZACIÓN
# ============================================================

def test_normalizar_variable(variable):
    """Normalizar una variable numérica individual."""

    resultado = normalizar(variable)

    esperado = pd.Series(
        [0, 0.25, 0.5, 0.75, 1],
        name="edad"
    )

    pd.testing.assert_series_equal(
        resultado,
        esperado
    )


def test_normalizar_dataset(dataset):
    """Normalizar todas las variables numéricas del dataset."""

    resultado = normalizar(dataset)

    # Las variables numéricas deben quedar entre 0 y 1
    assert resultado["edad"].min() == 0
    assert resultado["edad"].max() == 1

    assert resultado["altura"].min() == 0
    assert resultado["altura"].max() == 1

    # La variable categórica no debe modificarse
    pd.testing.assert_series_equal(
        resultado["sexo"],
        dataset["sexo"]
    )


def test_normalizar_columnas_seleccionadas(dataset):
    """Normalizar únicamente las columnas indicadas."""

    resultado = normalizar(
        dataset,
        atributos=["edad"]
    )

    # edad sí se transforma
    assert resultado["edad"].min() == 0
    assert resultado["edad"].max() == 1

    # altura no se transforma
    pd.testing.assert_series_equal(
        resultado["altura"],
        dataset["altura"]
    )

    # sexo tampoco
    pd.testing.assert_series_equal(
        resultado["sexo"],
        dataset["sexo"]
    )


def test_normalizar_dataset_no_modifica_original(dataset):
    """Comprobar que la transformación no modifica el DataFrame original."""

    original = dataset.copy()

    normalizar(dataset)

    pd.testing.assert_frame_equal(
        dataset,
        original
    )


def test_normalizar_variable_no_numerica():
    """Una variable no numérica no puede normalizarse directamente."""

    variable = pd.Series(
        ["M", "F", "M"],
        name="sexo"
    )

    with pytest.raises(TypeError):
        normalizar(variable)


def test_normalizar_constante():
    """No se puede normalizar una variable constante."""

    variable = pd.Series(
        [5, 5, 5],
        name="constante"
    )

    with pytest.raises(ValueError):
        normalizar(variable)


# ============================================================
# ESTANDARIZACIÓN
# ============================================================

def test_estandarizar_variable(variable):
    """Estandarizar una variable numérica individual."""

    resultado = estandarizar(variable)

    # La media debe ser aproximadamente 0
    assert resultado.mean() == pytest.approx(0)

    # La desviación estándar debe ser aproximadamente 1
    assert resultado.std() == pytest.approx(1)


def test_estandarizar_dataset(dataset):
    """Estandarizar todas las variables numéricas del dataset."""

    resultado = estandarizar(dataset)

    # Las variables numéricas deben tener media 0
    assert resultado["edad"].mean() == pytest.approx(0)
    assert resultado["altura"].mean() == pytest.approx(0)

    # Y desviación estándar 1
    assert resultado["edad"].std() == pytest.approx(1)
    assert resultado["altura"].std() == pytest.approx(1)

    # La variable categórica no debe modificarse
    pd.testing.assert_series_equal(
        resultado["sexo"],
        dataset["sexo"]
    )


def test_estandarizar_columnas_seleccionadas(dataset):
    """Estandarizar únicamente las columnas indicadas."""

    resultado = estandarizar(
        dataset,
        atributos=["edad"]
    )

    # edad sí se transforma
    assert resultado["edad"].mean() == pytest.approx(0)
    assert resultado["edad"].std() == pytest.approx(1)

    # altura no se transforma
    pd.testing.assert_series_equal(
        resultado["altura"],
        dataset["altura"]
    )

    # sexo tampoco
    pd.testing.assert_series_equal(
        resultado["sexo"],
        dataset["sexo"]
    )


def test_estandarizar_dataset_no_modifica_original(dataset):
    """Comprobar que no se modifica el DataFrame original."""

    original = dataset.copy()

    estandarizar(dataset)

    pd.testing.assert_frame_equal(
        dataset,
        original
    )


def test_estandarizar_variable_no_numerica():
    """Una variable no numérica no puede estandarizarse directamente."""

    variable = pd.Series(
        ["M", "F", "M"],
        name="sexo"
    )

    with pytest.raises(TypeError):
        estandarizar(variable)


def test_estandarizar_constante():
    """No se puede estandarizar una variable con desviación 0."""

    variable = pd.Series(
        [5, 5, 5],
        name="constante"
    )

    with pytest.raises(ValueError):
        estandarizar(variable)


# ============================================================
# VALIDACIÓN DE ATRIBUTOS
# ============================================================

def test_atributos_debe_ser_lista_normalizar(dataset):

    with pytest.raises(TypeError):
        normalizar(
            dataset,
            atributos="edad"
        )


def test_atributos_debe_ser_lista_estandarizar(dataset):

    with pytest.raises(TypeError):
        estandarizar(
            dataset,
            atributos="edad"
        )


def test_columna_inexistente_normalizar(dataset):

    with pytest.raises(ValueError):
        normalizar(
            dataset,
            atributos=["altura_inexistente"]
        )


def test_columna_inexistente_estandarizar(dataset):

    with pytest.raises(ValueError):
        estandarizar(
            dataset,
            atributos=["altura_inexistente"]
        )


# ============================================================
# VALIDACIÓN DEL TIPO DE ENTRADA
# ============================================================

def test_normalizar_tipo_incorrecto():

    with pytest.raises(TypeError):
        normalizar([1, 2, 3, 4])


def test_estandarizar_tipo_incorrecto():

    with pytest.raises(TypeError):
        estandarizar([1, 2, 3, 4])


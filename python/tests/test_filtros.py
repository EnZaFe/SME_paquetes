import pandas as pd
import pytest

from python.src.SME_python.filtrado_variables import filtrar_variables


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "edad": [10, 20, 30, 40],
        "altura": [1.0, 2.0, 3.0, 4.0],
        "sexo": ["H", "H", "M", "M"]
    })


@pytest.fixture
def clases():
    return pd.Series([0, 0, 1, 1])


# ---------------------------------------------------------
# SIN FILTROS
# ---------------------------------------------------------

def test_sin_filtros(dataset):
    resultado = filtrar_variables(dataset, verbose=0)

    pd.testing.assert_frame_equal(resultado, dataset)


# ---------------------------------------------------------
# DISCRETAS: SOLO ENTROPÍA
# ---------------------------------------------------------

def test_filtro_entropia_no_afecta_a_continuas(dataset):
    resultado = filtrar_variables(
        dataset,
        entropia=(1, None),
        verbose=0
    )

    assert "sexo" in resultado.columns
    assert "edad" in resultado.columns
    assert "altura" in resultado.columns


def test_filtro_entropia_descarta_discreta(dataset):
    resultado = filtrar_variables(
        dataset,
        entropia=(2, None),
        verbose=0
    )

    assert "sexo" not in resultado.columns
    assert "edad" in resultado.columns
    assert "altura" in resultado.columns


# ---------------------------------------------------------
# CONTINUAS: SOLO VARIANZA
# ---------------------------------------------------------

def test_filtro_varianza_no_afecta_a_discretas(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, None),
        verbose=0
    )

    assert "sexo" in resultado.columns
    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns


def test_filtro_varianza_descarta_continua(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(1000, None),
        verbose=0
    )

    assert "edad" not in resultado.columns
    assert "altura" not in resultado.columns
    assert "sexo" in resultado.columns


# ---------------------------------------------------------
# INTERVALOS
# ---------------------------------------------------------

def test_varianza_intervalo(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, 200),
        verbose=0
    )

    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns
    assert "sexo" in resultado.columns


def test_entropia_intervalo(dataset):
    resultado = filtrar_variables(
        dataset,
        entropia=(0.5, 1.5),
        verbose=0
    )

    assert "sexo" in resultado.columns
    assert "edad" in resultado.columns
    assert "altura" in resultado.columns

# ---------------------------------------------------------
# MODO: DENTRO / FUERA
# ---------------------------------------------------------

def test_varianza_modo_dentro(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, 200),
        modo="dentro",
        verbose=0
    )

    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns
    assert "sexo" in resultado.columns


def test_varianza_modo_fuera(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, 200),
        modo="fuera",
        verbose=0
    )

    assert "edad" not in resultado.columns
    assert "altura" in resultado.columns
    assert "sexo" in resultado.columns


def test_varianza_modo_dentro_sin_maximo(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, None),
        modo="dentro",
        verbose=0
    )

    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns
    assert "sexo" in resultado.columns


def test_varianza_modo_fuera_sin_maximo(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(100, None),
        modo="fuera",
        verbose=0
    )

    assert "edad" not in resultado.columns
    assert "altura" in resultado.columns
    assert "sexo" in resultado.columns


def test_varianza_modo_dentro_sin_minimo(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(None, 100),
        modo="dentro",
        verbose=0
    )

    assert "edad" not in resultado.columns
    assert "altura" in resultado.columns
    assert "sexo" in resultado.columns


def test_varianza_modo_fuera_sin_minimo(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(None, 100),
        modo="fuera",
        verbose=0
    )

    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns
    assert "sexo" in resultado.columns
    
# ---------------------------------------------------------
# AUC
# ---------------------------------------------------------

def test_filtro_auc(clases):
    dataset = pd.DataFrame({
        "x1": [1, 2, 10, 20],
        "x2": [1, 10, 2, 20]
    })

    resultado = filtrar_variables(
        dataset,
        auc=(0.8, None),
        clases=clases,
        verbose=0
    )

    assert "x1" in resultado.columns


def test_auc_sin_clases_se_ignora(dataset):
    resultado = filtrar_variables(
        dataset,
        auc=(0.9, None),
        clases=None,
        verbose=0
    )

    # Al no haber clases, el AUC no se utiliza.
    assert list(resultado.columns) == ["edad", "altura", "sexo"]


# ---------------------------------------------------------
# COMBINACIÓN DE MÉTRICAS
# ---------------------------------------------------------

def test_entropia_y_varianza(dataset):
    resultado = filtrar_variables(
        dataset,
        entropia=(1, None),
        varianza=(100, None),
        verbose=0
    )

    # La discreta solo necesita cumplir entropía.
    # Las continuas solo necesitan cumplir varianza.
    assert "sexo" in resultado.columns
    assert "edad" in resultado.columns
    assert "altura" not in resultado.columns


def test_varianza_y_auc(clases):
    dataset = pd.DataFrame({
        "x1": [1, 2, 10, 20],
        "x2": [1, 10, 2, 20]
    })

    resultado = filtrar_variables(
        dataset,
        varianza=(0, None),
        auc=(0.8, None),
        clases=clases,
        verbose=0
    )

    assert "x1" in resultado.columns


# ---------------------------------------------------------
# ATRIBUTOS
# ---------------------------------------------------------

def test_filtrar_atributos(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(0, None),
        atributos=["edad"],
        verbose=0
    )

    assert list(resultado.columns) == ["edad"]


def test_atributo_inexistente(dataset):
    with pytest.raises(ValueError):
        filtrar_variables(
            dataset,
            varianza=(0, None),
            atributos=["peso"],
            verbose=0
        )


def test_atributos_no_es_lista(dataset):
    with pytest.raises(TypeError):
        filtrar_variables(
            dataset,
            varianza=(0, None),
            atributos="edad",
            verbose=0
        )


# ---------------------------------------------------------
# NO MODIFICA EL ORIGINAL
# ---------------------------------------------------------

def test_no_modifica_dataset(dataset):
    original = dataset.copy()

    filtrar_variables(
        dataset,
        varianza=(1000, None),
        verbose=0
    )

    pd.testing.assert_frame_equal(dataset, original)


# ---------------------------------------------------------
# NINGUNA CONTINUA CUMPLE
# ---------------------------------------------------------

def test_ninguna_continua_cumple(dataset):
    resultado = filtrar_variables(
        dataset,
        varianza=(10000, None),
        verbose=0
    )

    # La discreta no se ve afectada por el filtro de varianza.
    assert "sexo" in resultado.columns
    assert "edad" not in resultado.columns
    assert "altura" not in resultado.columns
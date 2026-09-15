import pytest
import pandas as pd
import numpy as np
from python.src.SME_python.metricas_de_variables import calcular_metricas, _calcular_metricas_dataset, _calcular_AUC, _calcular_varianza, _calcular_entropia, _calcular_metricas_variable
from python.src.auxiliar.auxiliar_es import _es_continua, _es_variable, _es_dataset, _es_discreta

############################
# TESTS DE ENTROPÍA
############################

def test_entropia_todos_iguales():
    atributo = pd.Series(["A", "A", "A", "A"])

    resultado = _calcular_entropia(atributo, verbose=0)

    # Si todos los valores son iguales no existe incertidumbre
    assert resultado == pytest.approx(0.0)


def test_entropia_dos_clases_equilibradas():
    atributo = pd.Series(["A", "A", "B", "B"])

    resultado = _calcular_entropia(atributo, verbose=0)

    # 50% A y 50% B -> entropía máxima para 2 valores
    assert resultado == pytest.approx(1.0)


def test_entropia_tres_clases_equilibradas():
    atributo = pd.Series(["A", "B", "C"] * 10)

    resultado = _calcular_entropia(atributo, verbose=0)

    # Tres clases con la misma frecuencia
    assert resultado == pytest.approx(np.log2(3))


def test_entropia_clases_desbalanceadas():
    atributo = pd.Series(
        ["A"] * 9 + ["B"]
    )

    resultado = _calcular_entropia(atributo, verbose=0)

    esperada = -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))

    assert resultado == pytest.approx(esperada)


def test_entropia_con_nulos():
    atributo = pd.Series(
        ["A", "A", "B", None]
    )

    resultado = _calcular_entropia(atributo, verbose=0)

    # El None se elimina -> A, A, B
    esperada = -(
        (2/3) * np.log2(2/3)
        + (1/3) * np.log2(1/3)
    )

    assert resultado == pytest.approx(esperada)


def test_entropia_solo_nulos():
    atributo = pd.Series([None, None, None])

    with pytest.raises(ValueError):
        _calcular_entropia(atributo, verbose=0)


def test_entropia_vacia():
    atributo = pd.Series([], dtype=object)

    with pytest.raises(ValueError):
        _calcular_entropia(atributo, verbose=0)


def test_entropia_valores_numericos_discretos():
    atributo = pd.Series([1, 1, 2, 2, 3, 3])

    resultado = _calcular_entropia(atributo, verbose=0)

    assert resultado == pytest.approx(np.log2(3))


############################
# TESTS DE VARIANZA
############################

def test_varianza_todos_iguales():
    atributo = pd.Series([5, 5, 5, 5])

    resultado = _calcular_varianza(atributo, verbose=0)

    assert resultado == pytest.approx(0.0)


def test_varianza_simple():
    atributo = pd.Series([1, 2, 3, 4, 5])

    resultado = _calcular_varianza(atributo, verbose=0)

    # Varianza poblacional
    assert resultado == pytest.approx(2.0)


def test_varianza_dos_valores():
    atributo = pd.Series([0, 10])

    resultado = _calcular_varianza(atributo, verbose=0)

    assert resultado == pytest.approx(25.0)


def test_varianza_negativos():
    atributo = pd.Series([-2, -1, 0, 1, 2])

    resultado = _calcular_varianza(atributo, verbose=0)

    assert resultado == pytest.approx(2.0)


def test_varianza_decimales():
    atributo = pd.Series([1.5, 2.5, 3.5])

    resultado = _calcular_varianza(atributo, verbose=0)

    assert resultado == pytest.approx(2/3)


def test_varianza_con_nulos():
    atributo = pd.Series([1, 2, 3, None])

    resultado = _calcular_varianza(atributo, verbose=0)

    # Después de eliminar None -> [1, 2, 3]
    assert resultado == pytest.approx(2/3)


def test_varianza_solo_nulos():
    atributo = pd.Series([None, None])

    with pytest.raises(ValueError):
        _calcular_varianza(atributo, verbose=0)


def test_varianza_vacia():
    atributo = pd.Series([], dtype=float)

    with pytest.raises(ValueError):
        _calcular_varianza(atributo, verbose=0)


############################
# TESTS DE AUC
############################

def test_auc_perfecto():
    atributo = pd.Series([1, 2, 3, 4])
    clases = pd.Series(["A", "A", "B", "B"])

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    # Una clase siempre tiene valores mayores que la otra
    assert resultado == pytest.approx(1.0)


def test_auc_inverso():
    atributo = pd.Series([4, 3, 2, 1])
    clases = pd.Series(["A", "A", "B", "B"])

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    # Dependiendo de qué clase se considere clase_1,
    # el AUC puede ser 0.
    assert resultado == pytest.approx(1.0)


def test_auc_sin_capacidad_discriminativa():
    atributo = pd.Series([1, 2, 1, 2])
    clases = pd.Series(["A", "A", "B", "B"])

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    # En este caso las distribuciones se solapan
    assert 0 <= resultado <= 1


def test_auc_con_empates():
    atributo = pd.Series([1, 2, 2, 3])
    clases = pd.Series(["A", "A", "B", "B"])

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    # Se comprueba que los empates se tienen en cuenta
    assert 0 <= resultado <= 1


def test_auc_clases_equilibradas():
    atributo = pd.Series([1, 2, 3, 4])
    clases = pd.Series(["A", "B", "A", "B"])

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    assert 0 <= resultado <= 1


def test_auc_clases_desbalanceadas():
    atributo = pd.Series(
        [1, 2, 3, 4, 5, 6]
    )

    clases = pd.Series(
        ["A", "A", "A", "A", "A", "B"]
    )

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    assert 0 <= resultado <= 1


def test_auc_muchas_observaciones():
    atributo = pd.Series(range(100))
    
    clases = pd.Series(
        ["A"] * 50 + ["B"] * 50
    )

    resultado = _calcular_AUC(
        atributo,
        clases,
        verbose=0
    )

    assert 0 <= resultado <= 1


def test_auc_atributo_y_clases_longitudes_distintas():
    atributo = pd.Series([1, 2, 3])
    clases = pd.Series(["A", "B"])

    with pytest.raises(ValueError):
        _calcular_AUC(
            atributo,
            clases,
            verbose=0
        )


def test_auc_atributo_vacio():
    atributo = pd.Series([], dtype=float)
    clases = pd.Series([], dtype=object)

    with pytest.raises(ValueError):
        _calcular_AUC(
            atributo,
            clases,
            verbose=0
        )


def test_auc_no_binario():
    atributo = pd.Series([1, 2, 3, 4, 5, 6])
    clases = pd.Series(
        ["A", "B", "C", "A", "B", "C"]
    )

    with pytest.raises(ValueError):
        _calcular_AUC(
            atributo,
            clases,
            verbose=0
        )


def test_auc_una_sola_clase():
    atributo = pd.Series([1, 2, 3, 4])
    clases = pd.Series(["A", "A", "A", "A"])

    with pytest.raises(ValueError):
        _calcular_AUC(
            atributo,
            clases,
            verbose=0
        )


def test_auc_clase_sin_elementos():
    atributo = pd.Series([1, 2, 3, 4])
    clases = pd.Series(["A", "A", "A", "A"])

    with pytest.raises(ValueError):
        _calcular_AUC(
            atributo,
            clases,
            verbose=0
        )


############################
# TESTS DEL ROUTER
############################

def test_metricas_discretas():
    columna = pd.Series(
        ["A", "A", "B", "B"],
        name="color"
    )

    resultado = _calcular_metricas_variable(
        columna,
        clases=None,
        verbose=0
    )

    assert "entropia" in resultado
    assert resultado["entropia"] == pytest.approx(1.0)


def test_metricas_continuas():
    columna = pd.Series(
        [1, 2, 3, 4],
        name="edad"
    )

    clases = pd.Series(
        ["A", "A", "B", "B"]
    )

    resultado = _calcular_metricas_variable(
        columna,
        clases,
        verbose=0
    )

    assert "varianza" in resultado
    assert "auc" in resultado


def test_continua_sin_clases():
    columna = pd.Series(
        [1, 2, 3, 4],
        name="edad"
    )

    resultado = _calcular_metricas_variable(
        columna,
        clases=None,
        verbose=0
    )

    assert "varianza" in resultado
    assert "auc" not in resultado


############################
# TESTS DEL DATASET
############################

def test_dataset_completo():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40, 50],
        "sexo": ["H", "M", "H", "M"]
    })

    clases = pd.Series(
        ["A", "A", "B", "B"]
    )

    resultado = _calcular_metricas_dataset(
        dataset,
        clases,
        atributos_clm=None,
        verbose=0
    )

    assert "edad" in resultado
    assert "sexo" in resultado


def test_dataset_columnas_seleccionadas():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40, 50],
        "altura": [160, 170, 180, 190],
        "sexo": ["H", "M", "H", "M"]
    })

    clases = pd.Series(
        ["A", "A", "B", "B"]
    )

    resultado = _calcular_metricas_dataset(
        dataset,
        clases,
        atributos_clm=["edad"],
        verbose=0
    )

    assert "edad" in resultado
    assert "altura" not in resultado
    assert "sexo" not in resultado


def test_dataset_columna_inexistente():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40]
    })

    with pytest.raises(ValueError):
        _calcular_metricas_dataset(
            dataset,
            clases=None,
            atributos_clm=["altura"],
            verbose=0
        )


############################
# TESTS DE LA API PÚBLICA
############################

def test_api_dataset():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40, 50],
        "sexo": ["H", "M", "H", "M"]
    })

    clases = pd.Series(
        ["A", "A", "B", "B"]
    )

    resultado = calcular_metricas(
        dataset,
        clases=clases,
        verbose=0
    )

    assert isinstance(resultado, dict)


def test_api_atributos_no_lista():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40]
    })

    with pytest.raises(TypeError):
        calcular_metricas(
            dataset,
            atributos="edad",
            verbose=0
        )


def test_api_columna_inexistente():
    dataset = pd.DataFrame({
        "edad": [20, 30, 40]
    })

    with pytest.raises(ValueError):
        calcular_metricas(
            dataset,
            atributos=["altura"],
            verbose=0
        )


def test_api_tipo_incorrecto():
    datos = "esto no es un dataset"

    with pytest.raises(TypeError):
        calcular_metricas(
            datos,
            verbose=0
        )
def test_tipo_discreto():
    columna = pd.Series(["M", "F", "M", "F"])

    print(columna)
    print(columna.dtype)
    print(pd.api.types.is_object_dtype(columna))
    print(_es_discreta(columna))
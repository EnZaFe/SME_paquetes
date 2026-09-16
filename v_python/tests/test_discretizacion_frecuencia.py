"""
Tests de la discretización por igual frecuencia.

Objetivos:
- Probar frecuencia indicando n_intervalos.
- Probar frecuencia indicando frecuencia.
- Comprobar el reparto de sobrantes.
- Comprobar los intervalos obtenidos.
- Comprobar que los intervalos cubren todos los datos.
- Comprobar errores relacionados con frecuencia.
"""

import pandas as pd
import pytest

from python.src.SME_python.discretizacion import discretizar


# ============================================================
# 1. FRECUENCIA CON n_intervalos
# ============================================================

def test_frecuencia_n_intervalos():

    columna = pd.Series([
        10, 20, 30,
        40, 50, 60,
        70, 80, 90,
        100, 110, 120
    ])

    n_intervalos = 4

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        n_intervalos=n_intervalos,
        verbose=0
    )

    assert isinstance(resultado, list)
    assert len(resultado) == len(columna)

    # Obtener intervalos únicos manteniendo el orden.
    intervalos = list(dict.fromkeys(resultado))

    # 12 elementos / 4 intervalos = 3 elementos por intervalo.
    assert len(intervalos) == 4

    frecuencias = [
        resultado.count(intervalo)
        for intervalo in intervalos
    ]

    assert frecuencias == [3, 3, 3, 3]


# ============================================================
# 2. FRECUENCIA INDICADA DIRECTAMENTE
# ============================================================

def test_frecuencia_directa():

    columna = pd.Series([
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    ])

    frecuencia = 3

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        frecuencia=frecuencia,
        verbose=0
    )

    assert isinstance(resultado, list)
    assert len(resultado) == len(columna)

    intervalos = list(dict.fromkeys(resultado))

    # 9 elementos / frecuencia 3 = 3 intervalos.
    assert len(intervalos) == 3

    frecuencias = [
        resultado.count(intervalo)
        for intervalo in intervalos
    ]

    assert frecuencias == [3, 3, 3]


# ============================================================
# 3. FRECUENCIA CON SOBRANTES
# ============================================================

def test_frecuencia_sobrantes():

    columna = pd.Series([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    frecuencia = 3

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        frecuencia=frecuencia,
        verbose=0
    )

    assert isinstance(resultado, list)
    assert len(resultado) == len(columna)

    intervalos = list(dict.fromkeys(resultado))

    # 10 // 3 = 3 intervalos.
    # No se crea un cuarto intervalo.
    assert len(intervalos) == 3

    frecuencias = [
        resultado.count(intervalo)
        for intervalo in intervalos
    ]

    # 10 % 3 = 1 sobrante.
    #
    # Se reparte como:
    # 4, 3, 3
    assert frecuencias == [4, 3, 3]


# ============================================================
# 4. VARIOS SOBRANTES
# ============================================================

def test_frecuencia_varios_sobrantes():

    columna = pd.Series([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140
    ])

    frecuencia = 4

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        frecuencia=frecuencia,
        verbose=0
    )

    assert isinstance(resultado, list)
    assert len(resultado) == len(columna)

    intervalos = list(dict.fromkeys(resultado))

    # 14 // 4 = 3 intervalos.
    assert len(intervalos) == 3

    frecuencias = [
        resultado.count(intervalo)
        for intervalo in intervalos
    ]

    # 14 % 4 = 2 sobrantes.
    #
    # Se reparte como:
    # 5, 5, 4
    assert frecuencias == [5, 5, 4]


# ============================================================
# 5. FORMATO DE LOS INTERVALOS
# ============================================================

def test_frecuencia_intervalos_numericos():

    columna = pd.Series([
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    ])

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        frecuencia=3,
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


# ============================================================
# 6. LOS INTERVALOS CUBREN TODOS LOS DATOS
# ============================================================

def test_frecuencia_cubre_todos_los_datos():

    columna = pd.Series([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    resultado = discretizar(
        datos=columna,
        metodo="frecuencia",
        frecuencia=3,
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


# ============================================================
# 7. ERROR: FRECUENCIA = 0
# ============================================================

def test_error_frecuencia_cero():

    datos = pd.Series([10, 20, 30])

    with pytest.raises(
        ValueError,
        match="frecuencia debe ser mayor que 0"
    ):
        discretizar(
            datos=datos,
            metodo="frecuencia",
            frecuencia=0
        )


# ============================================================
# 8. ERROR: FRECUENCIA NEGATIVA
# ============================================================

def test_error_frecuencia_negativa():

    datos = pd.Series([10, 20, 30])

    with pytest.raises(
        ValueError,
        match="frecuencia debe ser mayor que 0"
    ):
        discretizar(
            datos=datos,
            metodo="frecuencia",
            frecuencia=-2
        )


# ============================================================
# 9. ERROR: n_intervalos > NÚMERO DE CASOS
# ============================================================

def test_error_n_intervalos_demasiados():

    datos = pd.Series([10, 20, 30])

    with pytest.raises(
        ValueError,
        match=r"n_intervalos \d+ no puede ser mayor que el número de casos \d+"
    ):
        discretizar(
            datos=datos,
            metodo="frecuencia",
            n_intervalos=5
        )


# ============================================================
# 10. ERROR: FRECUENCIA > NÚMERO DE CASOS
# ============================================================

def test_error_frecuencia_demasiado_grande():

    datos = pd.Series([10, 20, 30])

    with pytest.raises(
        ValueError,
        match=r"frecuencia \d+ no puede ser mayor que el número de casos \d+"
    ):
        discretizar(
            datos=datos,
            metodo="frecuencia",
            frecuencia=10
        )


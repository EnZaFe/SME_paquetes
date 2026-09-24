'''
Enunciado:
Algoritmos de discretización para un solo atributo y para un dataset completo (ambas opciones): Igual frecuencia e igual anchura


discretización = transformar una variable continua/numérica en categorías o intervalos


'''
from auxiliar.verbose import _verbose
from auxiliar.auxiliar_es import _es_dataset, _es_continua
import pandas as pd

def discretizar(datos, metodo="anchura", n_intervalos=None,
                frecuencia=None, columnas=None, verbose=1):
    """
    Discretiza una variable numérica o un dataset completo en intervalos de `(min, max)`.

        discretizar(series) -> una sola columna (pd.Series)
        discretizar(df, columnas=[...]) -> varias columnas a la vez
        discretizar(df) -> todas las columnas numéricas del DataFrame
    """

    _verbose('Iniciando discretizacion...', verbose)
    _verbose(f'''Parametros elegidos: 
    metodo: {metodo}, 
    n_intervalos: {n_intervalos}, 
    frecuenca: {frecuencia}, 
    columnas: {columnas}
             ''', verbose, nivel=2)

    # RAISE ERRORS para uso adecuado de las funciones
    if metodo not in ("anchura", "frecuencia"):
        raise ValueError(
            "Método no válido. Debe ser 'anchura' o 'frecuencia'."
        )
    if columnas is not None and not isinstance(columnas, list):
        raise TypeError("columnas debe ser una lista o None")

    
    # Si es una variable NUMERICA
    if _es_continua(datos):#if true
        return _discretizar_variable(
            datos, metodo, n_intervalos, frecuencia, verbose
        )

    # Si es un dataset
    elif _es_dataset(datos): #if true
        return _discretizar_dataset(
            datos, metodo, n_intervalos, frecuencia, columnas, verbose
        )

    else:
        raise TypeError("datos debe ser una Series numérica o un DataFrame.")


def _discretizar_dataset(dataset, metodo,
                         n_intervalos, frecuencia,
                         columnas, verbose=1):
    """Discretiza un dataset entero aplicando `_discretizar_variable()` a cada columna.

    Si `columnas` se especifica, solo se procesan esas columnas (todas deben ser numéricas).
    Si `columnas` es `None`, se discretizan automáticamente todas las columnas numéricas del DataFrame.
    """
    import pandas as pd

    if columnas is not None: # Cuando los columnas estan señaladas por el usuario
        df = dataset.copy()
        for col in columnas:
            if col not in dataset.columns:
                raise ValueError(
                    f"La columna '{col}' no existe en el dataset."
                )

            if not _es_continua(df[col]):
                raise TypeError(
                    f"La columna '{col}' no es numérica."
                )
            
            df[col] = _discretizar_variable(df[col], metodo, n_intervalos, frecuencia, verbose)
        return df # FINISH

    # TODO: En caso de dataset entero columnas NONE? Con definir arriba las columnas TODAS las numericas en caso de NONE suficiente  no?
    df = dataset.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = _discretizar_variable(df[col], metodo, n_intervalos, frecuencia, verbose)

    return df #FINISH

def _discretizar_variable(columna_variable, metodo,
                          n_intervalos, frecuencia, verbose=1):

    if metodo == "anchura":
        return _discretizar_anchura(
            columna_variable, n_intervalos, verbose
        )

    elif metodo == "frecuencia":
        return _discretizar_frecuencia(
            columna_variable, n_intervalos, frecuencia, verbose
        )

    else:
        raise ValueError("Método no válido. USAGE:   a....")
    
def _discretizar_anchura(columna_variable, n_intervalos, verbose=1):
    """Discretización por igual anchura (intervalos de tamaño fijo)."""
    if n_intervalos is None: 
        n_intervalos = round(len(columna_variable) * 0.50)
        
        n_intervalos = max(1, n_intervalos) ## Como mínimo habrá 1 intervalo 
        _verbose( f"No se ha especificado n_intervalos. " 
                 f"Se utilizarán {n_intervalos} intervalos "
                 f"(50% del número de casos).", 
                 verbose, nivel=1, tipo="warning" )

    if n_intervalos <= 0:
        raise ValueError("n_intervalos debe ser mayor que 0")
    num_casos = len(columna_variable)
    _verbose(
            f"Número de casos: {num_casos}",
            verbose,
            nivel=2
        )

    if n_intervalos > num_casos:
        raise ValueError(
            f"n_intervalos {n_intervalos} no puede ser mayor que el número de casos {num_casos}."
    )
    # algoritmo de discretización por igual anchura.
    minimo = columna_variable.min()
    maximo = columna_variable.max()

    if minimo == maximo:
        raise ValueError(
            "No se puede discretizar una variable donde todos los valores son iguales."
        )
    _verbose(f"Mínimo: {round(float(minimo), 2)}", verbose, nivel=2)
    _verbose(f"Máximo: {round(float(maximo), 2)}", verbose, nivel=2)

    anchura = (maximo - minimo) / n_intervalos # Aqui sacamos el tamaño que necesita cada intervalo
    _verbose(
        f"Anchura de los intervalos: {round(float(anchura), 2)}",
        verbose,
        nivel=2
    )
    intervalos = []

    for i in range(n_intervalos + 1): # Creamos [min, min+anchura*1, min+anchura*2,..., max]
        intervalo = minimo + i * anchura
        intervalos.append(intervalo)

    intervalos_mostrados = [ # para que no salga np.float
        round(float(intervalo), 2)
        for intervalo in intervalos
    ]

    _verbose(
        f"Límites de los intervalos: {intervalos_mostrados}",
        verbose,
        nivel=2
    )
    resultado = []

    for valor in columna_variable:
        for i in range(n_intervalos):
            if valor <= intervalos[i + 1]: # Si el valor real es mayor que el rango superior del intervalo siguiente
                resultado.append((
                    round(float(intervalos[i]), 2), # Por comodidad round y float
                    round(float(intervalos[i + 1]), 2) # Por comodidad round y float
                ))
                break
    _verbose(
        "Discretización finalizada correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )
    return resultado

def _discretizar_frecuencia(columna_variable, n_intervalos, frecuencia, verbose=1):
    
    if n_intervalos is not None and frecuencia is not None:
        raise ValueError(
            "Debes especificar n_intervalos o frecuencia, pero no ambos."
        )

    if n_intervalos is None and frecuencia is None:
        raise ValueError(
            "Debes especificar n_intervalos o frecuencia."
        )

    num_casos = len(columna_variable)
    _verbose(
            f"Número de casos: {num_casos}",
            verbose,
            nivel=2
        )

    if n_intervalos is not None:
        if n_intervalos <= 0:
            raise ValueError("n_intervalos debe ser mayor que 0.")
        if n_intervalos > num_casos:
            raise ValueError(
                f"n_intervalos {n_intervalos} no puede ser mayor que el número de casos {num_casos}."
        )
        # calcular frecuencia
        frecuencia = num_casos // n_intervalos
        sobrantes = num_casos % n_intervalos

        _verbose(
            f"Frecuencia base calculada: {frecuencia}",
            verbose,
            nivel=2
        )




    elif frecuencia is not None:
        if frecuencia <= 0:
            raise ValueError("frecuencia debe ser mayor que 0.")
        if frecuencia > num_casos:
            raise ValueError(
                f"frecuencia {frecuencia} no puede ser mayor que el número de casos {num_casos}."
        )
        #calcular intervalos
        n_intervalos= num_casos // frecuencia
        sobrantes=num_casos % frecuencia 
        # Si hay sobrantes es que no es divisible y no se puede hacer CON esa frecuencia exacta.

        if sobrantes > 0:
            _verbose(
                f"{sobrantes} elementos sobrantes. "
                "Se repartirán entre los primeros intervalos.",
                verbose,
                nivel=1,
                tipo="warning"
            )





    _verbose( #Esto hay que gestionarlo despues
        f"Elementos sobrantes: {sobrantes}",
        verbose,
        nivel=2
    )

    valores_ordenados= sorted(columna_variable)

    _verbose(
        f"Valores ordenados: {valores_ordenados[:10]}",
        verbose,
        nivel=2
    )


    intervalos=[]
    grupos = []
    posicion=0
    for i in range(n_intervalos): 
        meter_sobrante=0
        if i < sobrantes: #Vamos añadiendo a cada grupo del primero al ultimo un sobrante hasta que deje de haberlos
            meter_sobrante = 1

        grupo = valores_ordenados[posicion:posicion+frecuencia+meter_sobrante]

        grupos.append(grupo) #Aqui el vector de valores del intervalo # Esto a lo m ejor incluso sobra directamente

        intervalos.append(( #Aqui nos quedamos con min_max de intervalo
            float(grupo[0]),
            float(grupo[-1])
        ))

        posicion += frecuencia + meter_sobrante

    resultado = []

    for valor in columna_variable: # MUy marronero ponerlo por separado?

        for i in range(n_intervalos):

            if valor in grupos[i]:

                resultado.append((
                    round(float(intervalos[i][0]), 2),
                    round(float(intervalos[i][1]), 2)
                ))

                break
    _verbose(
        "Discretización finalizada correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )
    return resultado
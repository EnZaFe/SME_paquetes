'''
enunciado:
Cálculo de métricas para los atributos de un dataset: varianza y AUC para las variables contínuas y entropía para las discretas. La función deberá reconocer el tipo de atributo y actuar en consecuencia. Notese que en el caso del AUC, el dataset debe ser supervisado, es decir, es necesario especificar una variable clase binaria con la que evaluar el AUC de los atributos numéricos.
'''
from auxiliar.verbose import _verbose 
from auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta,_es_continua
import pandas as pd
from math import log2

def calcular_metricas(dataset, clases=None, atributos=None, verbose=1):
    '''API PUBLICA
    
    dataset para tener la info
    clases para AUC
    atributos, por si se quieren calcular solo algunos atributos
    '''
    #Sirve como router continuo/discreto

    #TODO: RAISE ERRORS
    #TODO. VERBOSES


    _verbose( "Iniciando cálculo de métricas...", verbose ) 
    _verbose( f"""Parámetros elegidos: \n
                atributos: {atributos} \n
                clases: {clases}
                """, 
             verbose, nivel=2 )


    if atributos is not None and not isinstance(atributos, list): 
        raise TypeError("columnas debe ser una lista o None.")

    # Si es una variable 
    if _es_variable(dataset):  #Esto iual sobra?
        return _calcular_metricas_variable( dataset, clases, verbose ) 
    # Si es un dataset 
    elif _es_dataset(dataset): 
        return _calcular_metricas_dataset( dataset, clases, atributos, verbose ) 

    else: 
        raise TypeError( "dataset debe ser una Series o un DataFrame." )


def _calcular_metricas_dataset(dataset, clases, atributos_clm, verbose=1):
    """Calcular métricas para las columnas de un dataset entero."""
    
    metricas = {} 
    # Si el usuario especifica columnas 
    if atributos_clm is not None: 
        for col in atributos_clm: 
            if col not in dataset.columns: 
                raise ValueError( f"La columna '{col}' no existe en el dataset." ) 

            metricas[col] = _calcular_metricas_variable(dataset[col], clases, verbose) 
            return metricas 
    # Si no se especifican columnas, calculamos para todas las columnas 
    for col in dataset.columns: 
        metricas[col] = _calcular_metricas_variable(dataset[col], clases, verbose) 
    return metricas



def _calcular_metricas_variable(columna, clases, verbose=1): 
    """Router para calcular las métricas de una variable.""" 
    if _es_discreta(columna): 
        _verbose( f"Se ha detectado que '{columna.name}' es discreta."
                  "Calculando entropía...",
                    verbose, nivel=1 ) 
        return _calcular_metricas_discreta( columna, verbose ) 

    elif _es_continua(columna): 
        _verbose( f"Se ha detectado que '{columna.name}' es continua. " 
                 "Calculando varianza y AUC...", 
                 verbose, nivel=1 ) 
        return _calcular_metricas_continua( columna, clases, verbose ) 

    else: 
        raise TypeError( f"No se puede determinar el tipo de la columna " 
                        f"'{columna.name}'." )

def _calcular_metricas_discreta(columna, verbose=1): 
    """Calcula las métricas de una variable discreta.""" 
    entropia = _calcular_entropia(columna, verbose) 
    _verbose( f"Entropía calculada: {round(entropia, 4)}", verbose, nivel=2 ) 

    return { "entropia": entropia }

def _calcular_metricas_continua(columna, clases, verbose=1): #Iual demasiado def dentro de def ya se podria hacer afuera
    """Calcula las métricas de una variable continua.""" 
    varianza = _calcular_varianza(columna, verbose) 
    _verbose( f"Varianza calculada: {round(varianza, 4)}", 
             verbose, nivel=2 ) 
    
    if clases is None: 
        raise ValueError( "Se necesitan las clases para calcular el AUC." ) 
    auc = _calcular_AUC(columna, clases, verbose)

    _verbose( f"AUC calculado: {round(auc, 4)}", verbose, nivel=2 ) 

    return { "varianza": varianza, "auc": auc }



#######################
# Variables Discretas #
#######################
def _calcular_entropia(atributo_clm, verbose):
    ''' Explicacion para saber exacto que estamos haciendo:
    La entropía mide cuánta incertidumbre o diversidad hay respecto a las clases de una variable objetivo. Si todos los ejemplos pertenecen a la misma clase, la entropía es 0 porque no existe incertidumbre. Cuanto más repartidos estén los ejemplos entre las distintas clases, mayor será la entropía.

    Para calcular la entropía esperada de un atributo A:

    E(D,A) = sumatorio(v pertenece a Valores(A))
            (|Dv| / |D|) * E(Dv)

    Valores(A) son los distintos valores que puede tomar el atributo A.

    Dv es el subconjunto de datos formado por los ejemplos cuyo atributo A
    toma el valor v.

    (|Dv| / |D|) es el peso proporcional de ese subconjunto respecto al
    dataset completo.

    E(Dv) es la entropía de las clases dentro del subconjunto Dv.


    La entropía del dataset D se calcula como:

    E(D) = - sumatorio(c pertenece a Clases)
        (|Dc| / |D|) * log2(|Dc| / |D|)

    Clases son los distintos valores que puede tomar la variable objetivo.

    Dc es el subconjunto de datos que pertenece a la clase c.

    (|Dc| / |D|) es la proporción de ejemplos que pertenecen a la clase c.

    El signo negativo se utiliza porque el logaritmo de una probabilidad
    entre 0 y 1 es negativo, y queremos obtener una entropía positiva.
    '''


    if pd.isna(atributo_clm).any():

        num_nulos = pd.isna(atributo_clm).sum()

        _verbose(
            f"Se han detectado {num_nulos} valores nulos. "
            "Se eliminarán antes de calcular la entropía.",
            verbose,
            nivel=1,
            tipo="warning"
        )
        atributo_clm = atributo_clm.dropna()

    if len(atributo_clm) == 0:
        raise ValueError(
            "No se puede calcular la entropía: "
            "el atributo no contiene valores válidos."
        )
    
    atributo_clm = atributo_clm.dropna()

    valores_unicos = set(atributo_clm) # Sacamos los subconjuntos
    _verbose(
        f"""Número de casos: {len(atributo_clm)} \n
            Valores unicos: {valores_unicos} """,
        verbose,
        nivel=2
    )

    entropia = 0

    for valor in valores_unicos:

        num_rep = 0
        for elemento in atributo_clm:
            if elemento == valor:
                num_rep += 1 # Contar cuántas veces aparece el valor

        # Probabilidad de ese valor
        probabilidad = num_rep / len(atributo_clm)
        _verbose(
            f"Valor '{valor}': {num_rep} apariciones "
            f"({probabilidad:.2%})",
            verbose,
            nivel=2
        )

        # Añadir su contribución a la entropía
        entropia -= probabilidad * log2(probabilidad) # El - es clave porque log2 da negativos
        _verbose(
            f"Entropía calculada: {entropia:.4f}",
            verbose
        )
    return entropia

#######################
# Variables Continuas #
#######################

def _calcular_varianza(atributo_clm, verbose):
    '''
    Mide cuánto se dispersan los valores respecto a su media.
    Formula:

        Var(X) = (1/n) * sumatorio((xi - media)^2)

        donde:

        n es el número de valores del atributo.
        xi es cada uno de los valores del atributo.
        media es la media de todos los valores.

        Una varianza pequeña indica que los valores están cerca de la media.
        Una varianza grande indica que los valores están más dispersos.
    '''

    if pd.isna(atributo_clm).any():

        num_nulos = pd.isna(atributo_clm).sum()

        _verbose(
            f"Se han detectado {num_nulos} valores nulos. "
            "Se eliminarán antes de calcular la varianza.",
            verbose,
            nivel=1,
            tipo="warning"
        )

        atributo_clm = atributo_clm.dropna()

    if len(atributo_clm) == 0:
        raise ValueError(
            "No se puede calcular la varianza: "
            "el atributo no contiene valores válidos."
        )

    # 1. Calcular la media
    suma = 0

    for valor in atributo_clm:
        suma += valor

    media = suma / len(atributo_clm)

    _verbose(
        f"Media: {media:.4f}",
        verbose,
        nivel=2
    )

    # 2. Calcular las diferencias respecto a la media,
    #    elevarlas al cuadrado y acumularlas
    suma_diferencias = 0

    for valor in atributo_clm:

        diferencia = valor - media
        diferencia_cuadrado = diferencia ** 2

        suma_diferencias += diferencia_cuadrado

    # 3. Dividir entre el número de valores
    varianza = suma_diferencias / len(atributo_clm)

    _verbose(
        f"Varianza: {varianza:.4f}",
        verbose,
        nivel=2
    )

    return varianza

def _calcular_AUC(atributo_clm, clases, verbose): #Dataset debe ser supervisado
    '''
  AUC (Area Under the ROC Curve) mide la capacidad que tiene una
    variable numérica para distinguir entre las diferentes clases
    de una variable binaria.

    El AUC toma valores entre 0 y 1:

    - AUC cercano a 1: el atributo distingue muy bien las clases.
    - AUC cercano a 0.5: el atributo no permite distinguir las clases.
    - AUC cercano a 0: el atributo distingue las clases en sentido contrario.

    Para calcularlo se comparan los valores del atributo pertenecientes
    a cada una de las dos clases.

    Formula:

    AUC = (comparaciones_correctas + 0.5 * empates)
          / (numero_valores_clase_1 * numero_valores_clase_2)
    '''

    _verbose("Iniciando cálculo de AUC...", verbose)


    if len(atributo_clm) != len(clases):
        raise ValueError(
            "El atributo y las clases deben tener el mismo número de elementos."
        )

    if len(atributo_clm) == 0:
        raise ValueError(
            "No se puede calcular el AUC de un atributo vacío."
        )

    if not _es_continua(atributo_clm):
        raise TypeError(
            "El atributo debe ser numérico para calcular el AUC."
        )

    valores_clases = set(clases)

    if len(valores_clases) != 2: # Para ser binario
        raise ValueError(
            "El atributo clases debe contener exactamente dos clases."
        )


    _verbose(
        f"Clases detectadas: {valores_clases}",
        verbose,
        nivel=2
    )

    # 1. Separamos los valores del atributo según su clase

    clase_1 = list(valores_clases)[0]
    clase_2 = list(valores_clases)[1]

    valores_clase_1 = []
    valores_clase_2 = []

    for i in range(len(atributo_clm)):

        if clases.iloc[i] == clase_1:
            valores_clase_1.append(atributo_clm.iloc[i])

        elif clases.iloc[i] == clase_2:
            valores_clase_2.append(atributo_clm.iloc[i])

    # 2. Comparamos todos los valores de una clase
    #    con todos los valores de la otra

    comparaciones_correctas = 0
    empates = 0

    for valor_1 in valores_clase_1:

        for valor_2 in valores_clase_2:

            if valor_1 > valor_2:
                comparaciones_correctas += 1

            elif valor_1 == valor_2:
                empates += 1

    # 3. Calculamos el número total de comparaciones

    total_comparaciones = (
        len(valores_clase_1) * len(valores_clase_2)
    )

    if total_comparaciones == 0:
        raise ValueError(
            "No se puede calcular el AUC porque una de las clases no tiene valores."
        )

    # 4. Calculamos el AUC

    auc = (
        comparaciones_correctas + 0.5 * empates
    ) / total_comparaciones
    auc = max(auc, 1 - auc) # Por si hemos calculado alreves


    _verbose(
        f"Comparaciones correctas: {comparaciones_correctas}",
        verbose,
        nivel=2
    )

    _verbose(
        f"Empates: {empates}",
        verbose,
        nivel=2
    )

    _verbose(
        f"AUC: {auc:.4f}",
        verbose,
        nivel=2
    )

    return auc

source("../auxiliar/verbose.R")
source("../auxiliar/auxiliar_es.r")

#' Algoritmos de discretización para un solo atributo y para un dataset completo.
#'
#' La discretización transforma una variable continua/númerica en categorías o
#' intervalos. Se contemplan dos métodos:
#'
#' \itemize{
#'   \item "anchura": igual anchura e igual número de casos por intervalo.
#'   \item "frecuencia": igual frecuencia (número de casos) e igual anchura variable.
#' }
#'
#' @section Uso:
#' ```
#' discretizar(series) -> una sola columna
#' discretizar(df, columnas = c(...)) -> varias columnas a la vez
#' discretizar(df) -> todas las columnas numéricas del DataFrame
#' ```
#'
#' @param datos Serie numérica o DataFrame sobre el que realizar la discretización.
#' @param metodo `"anchura"` (igual anchura) o `"frecuencia"` (igual frecuencia).
#' @param n_intervalos Número de intervalos a crear. Si es NULL, se calcula automáticamente como el 50% del número de casos.
#' @param frecuencia Vector de frecuencias objetivo por intervalo (método "frecuencia").
#' @param columnas Lista de nombres de columnas a discretizar, o NULL para usar todas las columnas numéricas del dataset.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return La variable discretizada (un vector/factor) si `datos` es una serie, o un DataFrame con las columnas discretizadas si `datos` es un dataset.
#' @export
discretizar <- function(datos, metodo = "anchura", n_intervalos = NULL,
                        frecuencia = NULL, columnas = NULL, verbose = 1) {
    # Objetivo: discretizar una variable o un dataset.
    # Uso: discretizar(df, metodo = "anchura", n_intervalos = 5, columnas = c("Edad", "Dinero"), verbose = 0)

    .verbose("Iniciando discretizacion...", verbose)
    .verbose(paste0(
        "Parametros elegidos: \n",
        "metodo: ", metodo, ", \n",
        "n_intervalos: ", n_intervalos, ", \n",
        "frecuenca: ", frecuencia, ", \n",
        "columnas: ", columnas
    ), verbose, nivel = 2)

    # RAISE ERRORS para uso adecuado de las funciones
    if (!metodo %in% c("anchura", "frecuencia")) {
        .verbose("Método no válido. Debe ser 'anchura' o 'frecuencia'.", verbose, nivel = 1, tipo = "error")
    }
    if (!is.null(columnas) && !is.list(columnas)) {
        .verbose("columnas debe ser una lista o None", verbose, nivel = 1, tipo = "error")
    }

    # Si es una variable NUMÉRICA
    if (.es_variable(datos)) {
        return(discretizar_variable(
            datos, metodo, n_intervalos, frecuencia, verbose
        ))
    }

    # Si es un dataset
    else if (.es_dataset(datos)) {
        return(discretizar_dataset(
            datos, metodo, n_intervalos, frecuencia, columnas, verbose
        ))
    }

    else {
        .verbose("datos debe ser una variable numérica o un data.frame.", verbose, nivel = 1, tipo = "error")
    }
}


#' Discretiza un dataset entero aplicando discretizar_variable() a cada columna.
#' Selecciona las columnas a discretizar y aplica el método elegido (anchura o frecuencia) a cada una.
#' @param dataset Datos de entrada. Puede ser un vector numérico o un data.frame.
#' @param metodo Método de discretización: "anchura" (igual anchura) o "frecuencia" (igual frecuencia).
#' @param n_intervalos Número de intervalos deseados.
#' @param frecuencia Número de casos por intervalo.
#' @param columnas Columnas a discretizar. Si es NULL, se discretizan todas las columnas numéricas/continuas.
#' @param verbose Nivel de verbosidad.
#' @return El dataset con las columnas discretizadas.
discretizar_dataset <- function(dataset, metodo,
                                n_intervalos, frecuencia,
                                columnas, verbose = 1) {
    # Discretizar un dataset entero.
    # Selecciona las columnas a discretizar y aplica discretizar_variable() a cada una.

    if (!is.null(columnas)) { # Cuando las columnas están señaladas por el usuario
        df <- dataset
        for (col in columnas) {
            if (!(col %in% names(df))) {
                .verbose(paste0("La columna '", col, "' no existe en el dataset."), verbose, nivel = 1, tipo = "error")
            }

            if (!.es_continua(df[[col]])) {
                .verbose(paste0("La columna '", col, "' no es numérica."), verbose, nivel = 1, tipo = "error")
            }

            df[[col]] <- discretizar_variable(df[[col]], metodo, n_intervalos, frecuencia, verbose)
        }
        return(df) # FIN
    }

    # Si no se indican columnas, se discretizan todas las columnas numéricas/continuas.
    df <- dataset
    for (col in names(df)) {
        if (.es_continua(df[[col]])) {
            df[[col]] <- discretizar_variable(df[[col]], metodo, n_intervalos, frecuencia, verbose)
        }
    }
    return(df) # FIN
}


#' Decide qué método de discretización utilizar según el parámetro "metodo".
#' @param columna_variable Variable numérica a discretizar.
#' @param metodo Método a aplicar: "anchura" o "frecuencia".
#' @param n_intervalos Número de intervalos deseados.
#' @param frecuencia Número de casos por intervalo.
#' @param verbose Nivel de verbosidad.
#' @return El resultado de la discretización según el método indicado.
discretizar_variable <- function(columna_variable, metodo,
                                 n_intervalos, frecuencia, verbose = 1) {

    if (metodo == "anchura") {
        return(discretizar_anchura(
            columna_variable, n_intervalos, verbose
        ))

    } else if (metodo == "frecuencia") {
        return(discretizar_frecuencia(
            columna_variable, n_intervalos, frecuencia, verbose
        ))

    } else {
        .verbose("Método no válido. USAGE:   a....", verbose, nivel = 1, tipo = "error")
    }
}


#' Discretiza una variable numérica en intervalos de igual anchura.
#' @param columna_variable Variable numérica a discretizar.
#' @param n_intervalos Número de intervalos deseados. Si es NULL, se calcula como el 50% del número de casos (mínimo 1).
#' @param verbose Nivel de verbosidad.
#' @return Una lista con los pares de límites de intervalo [limite_inferior, limite_superior].
discretizar_anchura <- function(columna_variable, n_intervalos, verbose = 1) {
    if (is.null(n_intervalos)) {
        n_intervalos <- round(length(columna_variable) * 0.50)
        n_intervalos <- max(1, n_intervalos) # Como mínimo habrá 1 intervalo
        .verbose(paste0("No se ha especificado n_intervalos. ",
                        "Se utilizarán ", n_intervalos, " intervalos ",
                        "(50% del número de casos)."),
                 verbose, nivel = 1, tipo = "warning")
    }

    if (n_intervalos <= 0) {
        .verbose("n_intervalos debe ser mayor que 0", verbose, nivel = 1, tipo = "error")
    }

    num_casos <- length(columna_variable)
    .verbose(paste0("Número de casos: ", num_casos), verbose, nivel = 2)

    if (n_intervalos > num_casos) {
        .verbose(paste0("n_intervalos ", n_intervalos, " no puede ser mayor que el número de casos ", num_casos, "."),
                 verbose, nivel = 1, tipo = "error")
    }

    # Algoritmo de discretización por igual anchura.
    minimo <- min(columna_variable)
    maximo <- max(columna_variable)

    if (minimo == maximo) {
        .verbose("No se puede discretizar una variable donde todos los valores son iguales.", verbose, nivel = 1, tipo = "error")
    }

    .verbose(paste0("Mínimo: ", round(minimo, 2)), verbose, nivel = 2)
    .verbose(paste0("Máximo: ", round(maximo, 2)), verbose, nivel = 2)

    anchura <- (maximo - minimo) / n_intervalos # Aquí sacamos el tamaño que necesita cada intervalo
    .verbose(paste0("Anchura de los intervalos: ", round(anchura, 2)), verbose, nivel = 2)

    intervalos <- numeric(n_intervalos + 1) # Creamos [min, min+anchura*1, min+anchura*2,..., max]
    for (i in seq_len(n_intervalos + 1)) {
        intervalo <- minimo + i * anchura
        intervalos[i] <- intervalo
    }

    intervalos_mostrados <- round(intervalos, 2) # Para que no salga numérico con decimales extraños
    .verbose(paste0("Límites de los intervalos: ", paste(intervalos_mostrados, collapse = ", ")), verbose, nivel = 2)

    resultado <- list()

    for (valor in columna_variable) {
        for (i in seq_len(n_intervalos)) {
            if (valor <= intervalos[i + 1]) { # Si el valor real es menor que el límite superior del intervalo
                resultado[[length(resultado) + 1]] <- list(round(intervalos[i], 2), round(intervalos[i + 1], 2))
                break
            }
        }
    }

    return(resultado)
}


#' Discretiza una variable numérica en intervalos de igual frecuencia.
#' @param columna_variable Variable numérica a discretizar.
#' @param n_intervalos Número de intervalos deseados. Es exclusivo con "frecuencia".
#' @param frecuencia Número de casos por intervalo. Es exclusivo con "n_intervalos".
#' @param verbose Nivel de verbosidad.
#' @return Una lista con los pares de límites de intervalo [limite_inferior, limite_superior].
discretizar_frecuencia <- function(columna_variable, n_intervalos, frecuencia, verbose = 1) {

    # Algoritmo de discretización por igual frecuencia.
    if (!is.null(n_intervalos) && !is.null(frecuencia)) {
        .verbose("Debes especificar n_intervalos o frecuencia, pero no ambos.", verbose, nivel = 1, tipo = "error")
    }

    if (is.null(n_intervalos) && is.null(frecuencia)) {
        .verbose("Debes especificar n_intervalos o frecuencia.", verbose, nivel = 1, tipo = "error")
    }

    num_casos <- length(columna_variable)
    .verbose(paste0("Número de casos: ", num_casos), verbose, nivel = 2)

    if (!is.null(n_intervalos)) {
        if (n_intervalos <= 0) {
            .verbose("n_intervalos debe ser mayor que 0.", verbose, nivel = 1, tipo = "error")
        }
        if (n_intervalos > num_casos) {
            .verbose(paste0("n_intervalos ", n_intervalos, " no puede ser mayor que el número de casos ", num_casos, "."),
                     verbose, nivel = 1, tipo = "error")
        }
        # Calcular frecuencia
        frecuencia <- num_casos %/% n_intervalos
        sobrantes <- num_casos %% n_intervalos

        .verbose(paste0("Frecuencia base calculada: ", frecuencia), verbose, nivel = 2)

    } else if (!is.null(frecuencia)) {
        if (frecuencia <= 0) {
            .verbose("frecuencia debe ser mayor que 0.", verbose, nivel = 1, tipo = "error")
        }
        if (frecuencia > num_casos) {
            .verbose(paste0("frecuencia ", frecuencia, " no puede ser mayor que el número de casos ", num_casos, "."),
                     verbose, nivel = 1, tipo = "error")
        }
        # Calcular intervalos
        n_intervalos <- num_casos %/% frecuencia
        sobrantes <- num_casos %% frecuencia
        # Si hay sobrantes es que no es divisible y no se puede hacer CON esa frecuencia exacta.

        if (sobrantes > 0) {
            .verbose(paste0(sobrantes, " elementos sobrantes. ",
                            "Se repartirán entre los primeros intervalos."),
                     verbose, nivel = 1, tipo = "warning")
        }
    }

    .verbose(paste0("Elementos sobrantes: ", sobrantes), verbose, nivel = 2)

    valores_ordenados <- sort(columna_variable)

    .verbose(paste0("Valores ordenados: ", paste(head(valores_ordenados, 10), collapse = ", ")), verbose, nivel = 2)

    intervalos <- list()
    grupos <- list()
    posicion <- 0
    for (i in seq_len(n_intervalos)) {
        meter_sobrante <- 0
        if ((i - 1) < sobrantes) { # Vamos añadiendo a cada grupo del primero al último un sobrante hasta que deje de haberlos
            meter_sobrante <- 1
        }

        fin <- posicion + frecuencia + meter_sobrante
        grupo <- valores_ordenados[(posicion + 1):fin]

        grupos[[length(grupos) + 1]] <- grupo # Aquí el vector de valores del intervalo

        intervalos[[length(intervalos) + 1]] <- list(as.numeric(grupo[1]), as.numeric(grupo[length(grupo)]))

        posicion <- posicion + frecuencia + meter_sobrante
    }

    resultado <- list()

    for (valor in columna_variable) { # Muy marronero ponerlo por separado?

        for (i in seq_len(n_intervalos)) {

            if (valor %in% grupos[[i]]) {

                resultado[[length(resultado) + 1]] <- list(round(intervalos[[i]][[1]], 2), round(intervalos[[i]][[2]], 2))

                break
            }
        }
    }

    return(resultado)
} 

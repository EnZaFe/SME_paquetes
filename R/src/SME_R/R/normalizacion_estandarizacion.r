# Sourcear funciones auxiliares (verbose y helpers de tipo/verificación)
source("../auxiliar/verbose.R")
source("../auxiliar/auxiliar_es.r")


#' Normalización (min-max) y estandarización (z-score) de variables o datasets.
#'
#' Transforman las variables numéricas de un dataset a una escala
#' comparable: la normalización las lleva a \eqn{[0, 1]} (min-max) y la
#' estandarización las lleva a media \eqn{0} y desviación estándar \eqn{1}.
#' Las variables discretas quedan sin cambios.
#'
#' Casos contemplados:
#'
#' \itemize{
#'   \item una variable continua -> se transforma en escala [0, 1] o z-score
#'   \item un dataset con atributos indicados -> solo esas columnas se transforman
#'   \item un dataset sin atributos -> todas las columnas continuas se transforman
#' }
#'
#' @param dataset Datos de entrada. Puede ser un vector (variable) o un data.frame.
#' @param atributos Columnas específicas a transformar. Si es NULL, se transforman todas las columnas continuas.
#' @param verbose Nivel de verbosidad.
#' @return Para una variable: el vector transformado. Para un dataset: el mismo data.frame con las columnas continuas transformadas.
normalizar <- function(dataset, atributos = NULL, verbose = 1) {
    # Objetivo: normalizar una variable o un dataset a escala [0, 1] (min-max).
    # Uso: normalizar(df, atributos = list("Edad"), verbose = 0)

    .verbose("Iniciando normalización...", verbose)

    if (!is.null(atributos) && !is.list(atributos)) {
        .verbose("atributos debe ser una lista o None", verbose, nivel = 1, tipo = "error")
    }

    return(.transformar_dataset(
        dataset,
        atributos,
        .normalizar_variable,
        verbose
    ))
}


#' Estandarización (puntuación z) de variables o datasets.
#'
#' Transforma las variables numéricas a una escala con media \eqn{0} y
#' desviación estándar \eqn{1}. Las variables discretas quedan sin cambios.
#'
#' Casos contemplados:
#'
#' \itemize{
#'   \item una variable continua -> se estandariza con z-score
#'   \item un dataset con atributos indicados -> solo esas columnas se transforman
#'   \item un dataset sin atributos -> todas las columnas continuas se transforman
#' }
#'
#' @param dataset Datos de entrada. Puede ser un vector (variable) o un data.frame.
#' @param atributos Columnas específicas a transformar. Si es NULL, se transforman todas las columnas continuas.
#' @param verbose Nivel de verbosidad.
#' @return Para una variable: el vector transformado. Para un dataset: el mismo data.frame con las columnas continuas transformadas.
estandarizar <- function(dataset, atributos = NULL, verbose = 1) {
    # Objetivo: estandarizar una variable o un dataset con puntuación z (media 0, desviación 1).
    # Uso: estandarizar(df, atributos = list("Edad"), verbose = 0)

    .verbose("Iniciando estandarización...", verbose)

    if (!is.null(atributos) && !is.list(atributos)) {
        .verbose("atributos debe ser una lista o None", verbose, nivel = 1, tipo = "error")
    }

    return(.transformar_dataset(
        dataset,
        atributos,
        .estandarizar_variable,
        verbose
    ))
}


#' Aplica una función de transformación a cada variable continua de un dataset.
#'
#' Para una serie se aplica directamente; para un data.frame se transforman
#' solo las columnas continuas (las discretas quedan sin cambios).
#'
#' @param dataset Datos de entrada. Puede ser un vector (variable) o un data.frame.
#' @param atributos Columnas específicas a transformar. Si es NULL, se transforman todas las columnas continuas.
#' @param funcion Función que transforma una columna continua. Recibe la columna y el nivel de verbosidad, y devuelve la columna transformada.
#' @param verbose Nivel de verbosidad.
#' @return El dataset (o vector) con las columnas continuas transformadas por \code{funcion}.
.transformar_dataset <- function(dataset, atributos, funcion, verbose = 1) {
    # Aplica `funcion` a cada variable continua de un dataset.
    # Para una serie se aplica directamente; para un data.frame se transforma
    # solo las columnas continuas (las discretas quedan sin cambios).

    if (.es_variable(dataset)) {
        return(funcion(dataset, verbose))

    } else if (.es_dataset(dataset)) {
        resultado <- dataset

        if (!is.null(atributos)) {
            columnas <- atributos
        } else {
            columnas <- names(dataset)
        }

        for (col in columnas) {

            if (!(col %in% names(dataset))) {
                .verbose(paste0("La columna '", col, "' no existe en el dataset."), verbose, nivel = 1, tipo = "error")
            }

            if (.es_continua(dataset[[col]])) {
                resultado[[col]] <- funcion(dataset[[col]], verbose)
            }

        }

        return(resultado)

    } else {
        .verbose("dataset debe ser una Series o un DataFrame.", verbose, nivel = 1, tipo = "error")
    }
}


#' Normaliza una variable numérica a escala [0, 1] (min-max).
#'
#' @param columna Variable continua a normalizar.
#' @param verbose Nivel de verbosidad.
#' @return La variable transformada en el rango [0, 1].
.normalizar_variable <- function(columna, verbose = 1) {
    # Normaliza una variable numérica a escala [0, 1] (min-max).

    if (!.es_continua(columna)) {
        .verbose(paste0("La columna no es numérica."), verbose, nivel = 1, tipo = "error")
    }

    minimo <- min(columna)
    maximo <- max(columna)
    .verbose(
        paste0("Se han detectado: mínimo=", minimo, ", máximo=", maximo),
        verbose,
        nivel = 2
    )

    if (maximo == minimo) {
        .verbose(paste0("No se puede normalizar: todos sus valores son iguales."), verbose, nivel = 1, tipo = "error")
    }

    return((columna - minimo) / (maximo - minimo))
}


#' Estandariza una variable numérica con puntuación z (media 0, desviación 1).
#'
#' @param columna Variable continua a estandarizar.
#' @param verbose Nivel de verbosidad.
#' @return La variable transformada con media 0 y desviación estándar 1.
.estandarizar_variable <- function(columna, verbose = 1) {
    # Estandariza una variable numérica con puntuación z (media 0, desviación 1).

    if (!.es_continua(columna)) {
        .verbose(paste0("La columna no es numérica."), verbose, nivel = 1, tipo = "error")
    }

    media <- mean(columna)
    desviacion <- sd(columna)
    .verbose(
        paste0("Se han detectado: media=", media, ", desviacion=", desviacion),
        verbose,
        nivel = 2
    )

    if (desviacion == 0) {
        .verbose(paste0("No se puede estandarizar: todos sus valores son iguales."), verbose, nivel = 1, tipo = "error")
    }

    return((columna - media) / desviacion)
}

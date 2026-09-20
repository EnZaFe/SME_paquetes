source("../auxiliar/verbose.R")
source("../auxiliar/auxiliar_es.r")


#' Filtra las variables de un dataset según sus métricas.
#'
#' Se pueden combinar varios criterios (entropía, varianza y/o AUC). Una
#' variable se conserva solo si cumple todas las condiciones especificadas.
#' Si `metricas` no se pasa, se calculan automáticamente.
#'
#' @section Uso:
#' ```
#' filtrar_variables(df, entropia = (0.5, 1)) -> df con variables que cumplen
#' filtrar_variables(df, varianza = (5, 7), modo = "dentro") -> df filtrado
#' ```
#'
#' @param dataset DataFrame que queremos filtrar.
#' @param entropia `(min, max)` para filtrar variables discretas. Los extremos pueden ser `NULL` para dejarlos abiertos.
#' @param varianza `(min, max)` para filtrar variables continuas. Los extremos pueden ser `NULL` para dejarlos abiertos.
#' @param auc `(min, max)` para filtrar variables continuas. Los extremos pueden ser `NULL` para dejarlos abiertos.
#' @param modo `"dentro"` (el valor está dentro del intervalo) o `"fuera"` (el valor está fuera del intervalo).
#' @param clases Vector binario con las etiquetas de clase, necesario para calcular el AUC.
#' @param atributos Lista opcional de variables a evaluar. Si es `NULL`, todas.
#' @param metricas Diccionario de métricas precalculadas `{columna: {...}}`. Si es `NULL`, se calculan automáticamente.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un data.frame con las variables que cumplen los requisitos especificados.
#' @export
filtrar_variables <- function(dataset, entropia = NULL, varianza = NULL,
                              auc = NULL, modo = "dentro", clases = NULL,
                              atributos = NULL, metricas = NULL, verbose = 1) {

    # Objetivo: filtrar las variables de un dataset según sus métricas.
    # Uso: filtrar_variables(df, varianza = (5, 7), modo = "dentro", verbose = 0)

    .verbose("Iniciando filtrado de variables...", verbose)

    # RAISE ERRORS para uso adecuado de las funciones
    if (!.es_dataset(dataset)) {
        .verbose("El objeto recibido no es un dataset.", verbose, tipo = "warning")
        stop("dataset debe ser un data.frame.")
    }

    if (is.null(entropia) && is.null(varianza) && is.null(auc)) {
        .verbose("No se ha especificado ningún criterio de filtrado.", verbose, tipo = "warning")
    }

    if (is.null(metricas)) {
        .verbose("No se han recibido métricas. Se calcularán automáticamente.", verbose, nivel = 2)

        # El filtro AUC no puede evaluarse sin clases.
        if (is.null(clases) && !is.null(auc)) {
            .verbose("No se han proporcionado clases. El filtro AUC no será evaluado.", verbose, tipo = "warning")
        }

        # Si no se especifican atributos, se evalúan todas las variables.
        if (is.null(atributos)) {
            .verbose("No se han especificado atributos. Se evaluarán todas las variables.", verbose, nivel = 2)
        }

        metricas <- calcular_metricas(
            dataset,
            clases = clases,
            atributos = atributos,
            verbose = verbose
        )

    } else if (!is.list(metricas)) {
        .verbose("El objeto recibido no contiene métricas válidas.", verbose, tipo = "warning")
        stop("metricas debe ser un diccionario.")
    }

    columnas_validas <- character(0)

    for (columna in names(metricas)) {

        if (.es_discreta(dataset[[columna]])) {

            # Variables discretas: se filtran por entropía.
            if (.cumple_filtro(metricas[[columna]][["entropia"]], entropia, modo)) {
                columnas_validas <- c(columnas_validas, columna)

            }

        } else if (.es_continua(dataset[[columna]])) {

            # Variables continuas: se filtran por varianza y AUC.
            if (.cumple_filtro(metricas[[columna]][["varianza"]], varianza, modo) &&
                .cumple_filtro(metricas[[columna]][["auc"]], auc, modo)) {
                columnas_validas <- c(columnas_validas, columna)

            }

        }

    }

    return(dataset[columnas_validas])

}

#' Comprueba si un valor cumple un intervalo según el modo.
#'
#' Determina si `valor` se encuentra dentro o fuera del intervalo
#' `(minimo, maximo)` según el `modo` indicado.
#'
#' @section Uso:
#' ```
#' .cumple_filtro(5, c(NULL, 7), "dentro")  # TRUE
#' .cumple_filtro(8, c(NULL, 7), "dentro")  # FALSE
#' .cumple_filtro(8, c(NULL, 7), "fuera")   # TRUE
#' ```
#'
#' @param valor Valor numérico a comprobar.
#' @param intervalo Par `(minimo, maximo)` que el valor debe cumplir. Si es `NULL`, siempre se cumple.
#' @param modo `"dentro"` o `"fuera"`.
#' @return `TRUE` si el valor cumple el filtro, `FALSE` en caso contrario.
.cumple_filtro <- function(valor, intervalo, modo) {

    # Si no hay intervalo especificado, siempre se cumple.
    if (is.null(intervalo)) {
        return(TRUE)
    }

    # Si el valor no está calculado, se considera que cumple.
    if (is.null(valor)) {
        return(TRUE)
    }

    minimo <- intervalo[1]
    maximo <- intervalo[2]

    # Modo "dentro": el valor debe estar dentro del intervalo.
    if (modo == "dentro") {
        if (!is.null(minimo) && valor < minimo) {
            return(FALSE)
        }
        if (!is.null(maximo) && valor > maximo) {
            return(FALSE)
        }
        return(TRUE)

    }

    # Modo "fuera": el valor debe estar fuera del intervalo.
    else if (modo == "fuera") {
        if (!is.null(minimo) && !is.null(maximo)) {
            return(valor < minimo || valor > maximo)
        }
        if (!is.null(minimo)) {
            return(valor < minimo)
        }
        if (!is.null(maximo)) {
            return(valor > maximo)
        }
        return(TRUE)

    }

    # Modo no válido.
    stop("modo debe ser 'dentro' o 'fuera'.")

}



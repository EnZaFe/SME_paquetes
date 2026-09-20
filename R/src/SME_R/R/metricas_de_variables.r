source("../auxiliar/verbose.R")
source("../auxiliar/auxiliar_es.r")
#' Cálculo de métricas para los atributos de un dataset: entropía para las discretas, varianza y AUC para las continuas.
#' La función reconoce el tipo de atributo y actúa en consecuencia. En el caso del AUC, el dataset debe ser supervisado,
#' es decir, es necesario especificar una variable clase binaria con la que evaluar el AUC de los atributos numéricos.
#'
#' @param dataset Datos de entrada. Puede ser un vector (variable) o un data.frame.
#' @param clases Columna binaria opcional con la que evaluar el AUC de las variables continuas.
#' @param atributos Columnas específicas a analizar. Si es NULL, se analizan todas las columnas.
#' @param verbose Nivel de verbosidad.
#' @return Para una variable discreta: lista con `entropia`. Para una continua sin clases: lista con `varianza`. Para una continua con clases: lista con `varianza` y `auc`. Para un dataset, el mismo dataset con las columnas anotadas con sus métricas.
calcular_metricas <- function(dataset, clases = NULL, atributos = NULL, verbose = 1) {
    # Objetivo: calcular las métricas de una o varias variables de un dataset.
    # Uso: calcular_metricas(df, clases = "Clase", atributos = list("Edad"), verbose = 0)

    .verbose("Iniciando cálculo de métricas...", verbose)

    # RAISE ERRORS para uso adecuado de las funciones
    if (!is.list(atributos) && !is.null(atributos)) {
        .verbose("atributos debe ser una lista o None", verbose, nivel = 1, tipo = "error")
    }

    # Si es un dataset
    if (.es_dataset(dataset)) {
        return(.calcular_metricas_dataset(
            dataset, clases, atributos, verbose
        ))
    }

    # Si es una variable
    else if (.es_variable(dataset)) {
        return(.calcular_metricas_variable(
            dataset, clases, verbose
        ))
    }

    else {
        .verbose("dataset debe ser un data.frame.", verbose, nivel = 1, tipo = "error")
    }
}


#' Calcula las métricas de un dataset entero.
#' Se calculan las métricas de cada columna indicada (o de todas las columnas).
#'
#' @param dataset Datos de entrada. Un data.frame.
#' @param clases Columna binaria opcional con la que evaluar el AUC de las variables continuas.
#' @param atributos_clm Columnas específicas a analizar. Si es NULL, se analizan todas las columnas numéricas.
#' @param verbose Nivel de verbosidad.
#' @return El dataset con cada columna anotada con su métrica (entropía o varianza/auc).
.calcular_metricas_dataset <- function(dataset, clases, atributos_clm, verbose = 1) {
    # Calcular las métricas de un dataset entero.
    # Se calculan las métricas de cada columna indicada (o de todas las columnas).

    if (!is.null(clases)) {
        df <- dataset
        for (col in atributos_clm) {
            if (!(col %in% names(df))) {
                .verbose(paste0("La columna '", col, "' no existe en el dataset."), verbose, nivel = 1, tipo = "error")
            }

            if (!.es_variable(df[[col]])) {
                .verbose(paste0("La columna '", col, "' no es numérica."), verbose, nivel = 1, tipo = "error")
            }

            df[[col]] <- .calcular_metricas_variable(df[[col]], clases, verbose)
        }
        return(df) # FIN
    }

    # Si no se indican las clases, se calculan las métricas de todas las columnas.
    df <- dataset
    for (col in names(df)) {
        if (.es_variable(df[[col]])) {
            df[[col]] <- .calcular_metricas_variable(df[[col]], NULL, verbose)
        }
    }
    return(df) # FIN
}


#' Decide qué métrica calcular según el tipo de la variable.
#' Discreta -> entropía. Continua -> varianza y AUC (si se proporcionan clases).
#'
#' @param columna Variable a analizar.
#' @param clases Columna binaria opcional con la que evaluar el AUC de las variables continuas.
#' @param verbose Nivel de verbosidad.
#' @return El resultado de la métrica según el tipo de variable (entropía, varianza o varianza+auc).
.calcular_metricas_variable <- function(columna, clases, verbose = 1) {
    # Calcular las métricas de una sola variable.
    # Discreta -> entropía. Continua -> varianza y AUC (si se proporcionan clases).

    if (.es_discreta(columna)) {
        return(.calcular_metricas_discreta(
            columna,
            verbose
        ))

    } else if (.es_continua(columna)) {

        .verbose(
            paste0(
                "Se ha detectado que '", columna.name, "' es continua. ",
                "Calculando varianza y AUC..."
            ),
            verbose,
            nivel = 2
        )

        return(.calcular_metricas_continua(
            columna,
            clases,
            verbose
        ))

    } else {

        .verbose(
            paste0(
                "No se puede determinar el tipo de la columna ",
                "'", columna.name, "'."
            ),
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            paste0(
                "No se puede determinar el tipo de la columna ",
                "'", columna.name, "'."
            )
        )
    }
}


#' Calcula las métricas de una variable discreta.
#'
#' @param columna Variable discreta a analizar.
#' @param verbose Nivel de verbosidad.
#' @return Lista con la clave `entropia` y su valor.
.calcular_metricas_discreta <- function(columna, verbose = 1) {
    # Calcular las métricas de una variable discreta.

    entropia <- .calcular_entropia(
        columna,
        verbose
    )

    return(list(
        entropia = entropia
    ))
}


#' Calcula las métricas de una variable continua.
#'
#' @param columna Variable continua a analizar.
#' @param clases Columna binaria opcional con la que evaluar el AUC de las variables continuas.
#' @param verbose Nivel de verbosidad.
#' @return Lista con `varianza`. Si se proporcionan clases, también incluye `auc`.
.calcular_metricas_continua <- function(columna, clases, verbose = 1) {
    # Calcular las métricas de una variable continua.

    varianza <- .calcular_varianza(
        columna,
        verbose
    )

    .verbose(
        paste0("Varianza calculada: ", formatC(round(varianza, 4), format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    if (is.null(clases)) {

        .verbose(
            "No se han proporcionado clases. No se calculará el AUC.",
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        return(list(
            varianza = varianza
        ))

    }

    auc <- .calcular_AUC(
        columna,
        clases,
        verbose
    )

    .verbose(
        paste0("AUC calculado: ", formatC(round(auc, 4), format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    return(list(
        varianza = varianza,
        auc = auc
    ))
}


#' Calcula la entropía de una variable discreta respecto a las clases.
#' La entropía mide cuánta incertidumbre o diversidad hay respecto a las clases de una variable objetivo. Si todos los
#' ejemplos pertenecen a la misma clase, la entropía es 0 porque no existe incertidumbre. Cuanto más repartidos estén los
#' ejemplos entre las distintas clases, mayor será la entropía.
#'
#' @param atributo_clm Vector de valores discretos de la variable objetivo.
#' @param verbose Nivel de verbosidad.
#' @return El valor de la entropía calculada.
.calcular_entropia <- function(atributo_clm, verbose) {
    # La entropía mide cuánta incertidumbre o diversidad hay respecto a las clases de una variable objetivo.
    # Si todos los ejemplos pertenecen a la misma clase, la entropía es 0 porque no existe incertidumbre.
    # Cuanto más repartidos estén los ejemplos entre las distintas clases, mayor será la entropía.

    if (any(is.na(atributo_clm))) {

        num_nulos <- sum(is.na(atributo_clm))

        .verbose(
            paste0(
                "Se han detectado ", num_nulos,
                " valores nulos. ",
                "Se eliminarán antes de calcular la entropía."
            ),
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        atributo_clm <- atributo_clm[!is.na(atributo_clm)]

    }

    if (length(atributo_clm) == 0) {

        .verbose(
            "El atributo no contiene valores válidos.",
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "No se puede calcular la entropía: ",
            "el atributo no contiene valores válidos."
        )

    }

    atributo_clm <- atributo_clm[!is.na(atributo_clm)]

    valores_unicos <- sort(unique(atributo_clm))

    .verbose(
        paste0(
            "Datos preparados para la entropía:\n",
            "Número de casos: ", length(atributo_clm), "\n",
            "Valores únicos: ", paste(valores_unicos, collapse = ", ")
        ),
        verbose,
        nivel = 3
    )

    entropia <- 0

    for (valor in valores_unicos) {

        num_rep <- 0

        for (elemento in atributo_clm) {

            if (elemento == valor) {
                num_rep <- num_rep + 1
            }

        }

        probabilidad <- num_rep / length(atributo_clm)

        .verbose(
            paste0(
                "Valor '", valor, "': ", num_rep, " apariciones ",
                formatC(probabilidad, format = "f", digits = 2), "%"
            ),
            verbose,
            nivel = 3
        )

        entropia <- entropia - probabilidad * log2(probabilidad)

    }

    .verbose(
        paste0("Entropía final: ", formatC(entropia, format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    return(entropia)
}


#######################
#' Calcula la varianza de una variable continua.
#' Mide cuánto se dispersan los valores respecto a su media. Formula: Var(X) = (1/n) * sumatorio((xi - media)^2).
#'
#' @param atributo_clm Vector de valores continuos de la variable objetivo.
#' @param verbose Nivel de verbosidad.
#' @return El valor de la varianza calculada.
.calcular_varianza <- function(atributo_clm, verbose) {
    # Mide cuánto se dispersan los valores respecto a su media.
    # Formula:
    #     Var(X) = (1/n) * sumatorio((xi - media)^2)

    if (any(is.na(atributo_clm))) {

        num_nulos <- sum(is.na(atributo_clm))

        .verbose(
            paste0(
                "Se han detectado ", num_nulos,
                " valores nulos. ",
                "Se eliminarán antes de calcular la varianza."
            ),
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        atributo_clm <- atributo_clm[!is.na(atributo_clm)]

    }

    if (length(atributo_clm) == 0) {

        .verbose(
            "El atributo no contiene valores válidos.",
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "No se puede calcular la varianza: ",
            "el atributo no contiene valores válidos."
        )

    }

    suma <- 0

    for (valor in atributo_clm) {
        suma <- suma + valor
    }

    media <- suma / length(atributo_clm)

    .verbose(
        paste0("Media: ", formatC(media, format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    suma_diferencias <- 0

    for (valor in atributo_clm) {

        diferencia <- valor - media
        diferencia_cuadrado <- diferencia ^ 2
        suma_diferencias <- suma_diferencias + diferencia_cuadrado

    }

    varianza <- suma_diferencias / length(atributo_clm)

    .verbose(
        paste0("Varianza: ", formatC(varianza, format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    return(varianza)
}


#######################
#' Calcula el AUC (Área bajo la curva ROC) por comparación de pares entre las dos clases.
#' Cuenta cuántas veces un valor de la clase 1 supera a uno de la clase 2, promediando los empates como 0.5. Al ser binario,
#' ``auc`` y ``1 - auc`` describen el mismo separador, por lo que se devuelve ``max(auc, 1 - auc)``.
#'
#' @param atributo_clm Vector de valores continuos de la variable objetivo.
#' @param clases Vector binario con las etiquetas de clase (debe tener el mismo tamaño que `atributo_clm` y exactamente dos clases).
#' @param verbose Nivel de verbosidad.
#' @return El valor del AUC calculado.
.calcular_AUC <- function(atributo_clm, clases, verbose) {

    if (length(atributo_clm) != length(clases)) {

        .verbose(
            paste0(
                "El atributo y las clases tienen diferente número ",
                "de elementos ",
                length(atributo_clm),
                " /= ",
                length(clases)
            ),
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "El atributo y las clases deben tener el mismo número de elementos."
        )

    }

    if (length(atributo_clm) == 0) {

        .verbose(
            "No se puede calcular el AUC de un atributo vacío.",
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "No se puede calcular el AUC de un atributo vacío."
        )

    }

    if (!.es_continua(atributo_clm)) {

        .verbose(
            paste0(
                "El atributo no es numérico y no puede utilizarse ",
                "para calcular el AUC."
            ),
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "El atributo debe ser numérico para calcular el AUC."
        )

    }

    valores_clases <- sort(unique(clases))

    if (length(valores_clases) != 2) {

      .verbose(
        paste0(
          "Se han detectado ",
          length(valores_clases),
          " clases. ",
          "El AUC requiere exactamente dos."
        ),
        verbose,
        nivel = 1,
        tipo = "warning"
      )

        stop(
            "El atributo clases debe contener exactamente dos clases."
        )

    }

    .verbose(
        paste0("Clases detectadas: ", paste(valores_clases, collapse = ", ")),
        verbose,
        nivel = 2
    )

    clase_1 <- valores_clases[1]
    clase_2 <- valores_clases[2]

    valores_clase_1 <- numeric(0)
    valores_clase_2 <- numeric(0)

    for (i in seq_len(length(atributo_clm))) {

        if (clases[i] == clase_1) {
            valores_clase_1 <- c(valores_clase_1, atributo_clm[i])

        } else if (clases[i] == clase_2) {
            valores_clase_2 <- c(valores_clase_2, atributo_clm[i])
        }

    }

    .verbose(
        paste0("Elementos de la clase '", clase_1, "': ", length(valores_clase_1)),
        verbose,
        nivel = 3
    )

    .verbose(
        paste0("Elementos de la clase '", clase_2, "': ", length(valores_clase_2)),
        verbose,
        nivel = 3
    )

    comparaciones_correctas <- 0
    empates <- 0

    for (valor_1 in valores_clase_1) {

        for (valor_2 in valores_clase_2) {

            if (valor_1 > valor_2) {
                comparaciones_correctas <- comparaciones_correctas + 1

            } else if (valor_1 == valor_2) {
                empates <- empates + 1
            }

        }

    }

    total_comparaciones <- length(valores_clase_1) * length(valores_clase_2)

    if (total_comparaciones == 0) {

        .verbose(
            "Una de las clases no tiene valores para comparar.",
            verbose,
            nivel = 1,
            tipo = "warning"
        )

        stop(
            "No se puede calcular el AUC porque una de las clases no tiene valores."
        )

    }

    auc <- (
        comparaciones_correctas + 0.5 * empates
    ) / total_comparaciones

    auc <- max(auc, 1 - auc)

    .verbose(
        paste0("Comparaciones correctas: ", comparaciones_correctas),
        verbose,
        nivel = 3
    )

    .verbose(
        paste0("Empates: ", empates),
        verbose,
        nivel = 3
    )

    .verbose(
        paste0("AUC: ", formatC(auc, format = "f", digits = 4)),
        verbose,
        nivel = 3
    )

    return(auc)
}

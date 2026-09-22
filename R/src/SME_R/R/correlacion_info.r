
#' Cálculo de correlación/asociación entre atributos.
#'
#' Calcula una matriz de correlación/asociación entre los atributos
#' seleccionados del dataset.
#'
#' Casos contemplados:
#'
#' \itemize{
#'   \item numérica + numérica -> Correlación de Pearson
#'   \item categórica + categórica -> Información Mutua (Mutual Information)
#'   \item categórica + numérica -> ANOVA (Welch)
#' }
#'
#' @param dataset DataFrame sobre el que realizar el cálculo.
#' @param atributos Lista de nombres de columnas a analizar, o NULL para usar todas las columnas del dataset.
#' @param normalizar Si es TRUE, se normalizan los resultados a una escala 0-1 (actualmente no implementado).
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un list con dos elementos: `matriz` (DataFrame numérico de correlaciones/asociaciones entre los atributos) y `detalles` (diccionario con toda la información adicional de cada par de atributos, indexado por la tupla (atributo1, atributo2)).
#' @export
calcular_correlacion <- function(dataset, atributos = NULL, normalizar = FALSE, verbose = 1) {
    .verbose("Iniciando cálculo de correlaciones...", verbose)

    # RAISE ERRORS para uso adecuado de las funciones
    if (!.es_dataset(dataset)) {
        stop("dataset debe ser un DataFrame.", call. = FALSE)
    }

    if (!is.null(atributos) && !is.list(atributos)) {
        stop("atributos debe ser una lista o None.", call. = FALSE)
    }

    # Si no se especifican atributos, utilizamos todas las columnas.
    if (is.null(atributos)) {
        atributos <- as.list(names(dataset))
        .verbose(
            paste0("Warning: no se han indicado atributos se usaran todas las columnas del dataset."),
            verbose
        )
    }

    for (atributo in atributos) {

        if (!(atributo %in% names(dataset))) {
            stop(
                paste0("La columna '", atributo, "' no existe en el dataset.")
            )
        }

    }

    tipos <- list()

    for (atributo in atributos) {

        columna <- dataset[[atributo]]

        if (.es_continua(columna)) {

            tipos[[atributo]] <- "continua"

        } else if (.es_discreta(columna)) {

            tipos[[atributo]] <- "discreta"

        } else {

            .verbose(
                paste0("Warning: no se utilizará '", atributo, "' porque "),
                "no se detecta como continuo ni discreto.",
                verbose
            )

        }

    }

    atributos_validos <- unlist(lapply(tipos, names))

    .verbose(
        paste0("Se analizarán ", length(atributos_validos), " atributos."),
        verbose
    )

    # Matriz numérica de correlaciones.
    matriz <- data.frame(
        rownames = atributos_validos,
        check.names = FALSE,
        stringsAsFactors = FALSE
    )

    detalles <- list()

    # Una variable comparada consigo misma tiene asociación máxima 1.
    for (atributo in atributos_validos) {
        matriz[[atributo]] <- 1.0
    }

    # Las medidas utilizadas son simétricas:
    #
    #     Pearson(A, B) = Pearson(B, A)
    #     MI(A, B)      = MI(B, A)
    #     Welch(A, B)   = Welch(B, A)
    #
    for (i in seq_len(length(atributos_validos))) {

        atributo1 <- atributos_validos[i]

        for (atributo2 in atributos_validos[(i + 1):length(atributos_validos)]) {

            resultado <- .calc_atributo(
                dataset[[atributo1]],
                dataset[[atributo2]],
                tipos[[atributo1]],
                tipos[[atributo2]],
                normalizar,
                verbose
            )

            # En la matriz numérica solo guardamos el escalar.
            matriz[[atributo1]][[atributo2]] <- resultado[["valor"]]

            # En detalles guardamos TODO lo calculado, indexado en
            # ambos sentidos para que sea cómodo de consultar.
            detalles[[paste(atributo1, atributo2, sep = "|")]] <- resultado
            detalles[[paste(atributo2, atributo1, sep = "|")]] <- resultado

        }

    }

    .verbose(
        "Cálculo de correlaciones finalizado.",
        verbose
    )

    return(list(matriz = matriz, detalles = detalles))
}


# =============================================================================
# Decisión del método según los tipos de atributos
# =============================================================================

#' Decide qué método utilizar para calcular la correlación entre dos atributos.
#'
#' Función interna que, dado un par de atributos, decide qué función de cálculo
#' aplicar en base a los tipos de las columnas correspondientes.
#'
#' @param atributo1 Nombre del primer atributo.
#' @param atributo2 Nombre del segundo atributo.
#' @param tipo1 Tipo detectado del primer atributo (`"continua"` o `"discreta"`).
#' @param tipo2 Tipo detectado del segundo atributo (`"continua"` o `"discreta"`).
#' @param normalizar Si es TRUE, se normalizan los resultados a una escala 0-1.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un list con, como mínimo, las claves `valor` (el escalar de la matriz) y `tipo` (qué método se usó).
.calc_atributo <- function(atributo1, atributo2, tipo1, tipo2, normalizar = FALSE, verbose = 1) {
    # Decide qué método utilizar en función del tipo de los atributos.
    # Siempre devuelve un diccionario con, como mínimo, la clave
    # "valor" (el escalar de la matriz) y "tipo" (qué método se usó).

    if (tipo1 == "discreta" && tipo2 == "discreta") {

        return(.calc_corr_catg(
            atributo1,
            atributo2,
            normalizar,
            verbose
        ))

    } else if (tipo1 == "continua" && tipo2 == "continua") {

        return(.calc_corr_num(
            atributo1,
            atributo2,
            normalizar,
            verbose
        ))

    } else if (
        (tipo1 == "discreta" && tipo2 == "continua") ||
        (tipo1 == "continua" && tipo2 == "discreta")
    ) {

        return(.calc_corr_catg_num(
            atributo1,
            atributo2,
            normalizar,
            verbose
        ))

    } else {

        stop("Los atributos no son ni continuos ni discretos.", call. = FALSE)

    }
}


# =============================================================================
# Correlación numérica (Pearson)
# =============================================================================

#' Correlación de Pearson entre dos atributos numéricos.
#'
#' Mide la intensidad y dirección de la relación LINEAL entre dos variables
#' numéricas. El resultado está entre -1 y 1:
#'
#' \itemize{
#'   \item +1 -> relación lineal positiva perfecta
#'   \item 0 -> ausencia de relación lineal
#'   \item -1 -> relación lineal negativa perfecta
#' }
#'
#' @param atributo1 Nombre o columna del primer atributo numérico.
#' @param atributo2 Nombre o columna del segundo atributo numérico.
#' @param normalizar Si es TRUE, se normalizan los resultados a una escala 0-1.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un list con el coeficiente de correlación `r`, el coeficiente de determinación `r2`, el número de observaciones `n`, y los vectores `x` e `y` necesarios para pintar el gráfico de dispersión + recta de regresión.
.calc_corr_num <- function(atributo1, atributo2, normalizar = FALSE, verbose = 1) {
    # Correlación de Pearson.
    # Pearson mide la intensidad y dirección de la relación LINEAL
    # entre dos variables numéricas.
    #
    # El resultado está entre -1 y 1:
    #
    #     +1 -> relación lineal positiva perfecta
    #      0 -> ausencia de relación lineal
    #     -1 -> relación lineal negativa perfecta
    #
    # Fórmula:
    #     r = (n * Σ(xy) - Σx * Σy) /
    #         √[(n * Σ(x²) - (Σx)²) * (n * Σ(y²) - (Σy)²)]
    #
    # Devuelve un diccionario con el coeficiente y todo lo necesario
    # para pintar el gráfico de dispersión + recta de regresión sin
    # tener que recalcular nada (pendiente, intercepto, nombres...).

    nombre1 <- attr(atributo1, "name") %||% "Atributo 1"
    nombre2 <- attr(atributo2, "name") %||% "Atributo 2"

    datos <- data.frame(
        x = as.numeric(unlist(atributo1)),
        y = as.numeric(unlist(atributo2)),
        stringsAsFactors = FALSE
    )

    datos <- datos[!is.na(datos$x) & !is.na(datos$y), ]

    if (nrow(datos) == 0) {
        stop("No hay suficientes datos para calcular la correlación.", call. = FALSE)
    }

    x <- datos[["x"]]
    y <- datos[["y"]]

    n <- length(x)

    suma_x <- sum(x)
    suma_y <- sum(y)

    suma_x2 <- sum(x^2)
    suma_y2 <- sum(y^2)

    suma_xy <- sum(x * y)

    numerador <- (
        n * suma_xy - suma_x * suma_y
    )

    parte_x <- (
        n * suma_x2 - suma_x^2
    )
    parte_y <- (
        n * suma_y2 - suma_y^2
    )

    denominador <- sqrt(parte_x * parte_y)

    if (denominador == 0) {
        stop(
            "No se puede calcular la correlación de Pearson: ",
            "alguna de las variables es constante."
        )
    }

    r <- numerador / denominador

    if (normalizar) {
        # TODO:
        # Normalizar el resultado de Pearson a [0, 1].
    }

    return(list(
        tipo = "pearson",
        valor = r,
        r2 = r^2,
        n = n,
        nombre1 = nombre1,
        nombre2 = nombre2,
        x = x,
        y = y
    ))
}


# =============================================================================
# Correlación categórica (Información Mutua)
# =============================================================================

#' Información Mutua (MI) entre dos atributos categóricos.
#'
#' La información mutua mide cuánta información aporta conocer una variable
#' sobre la otra. Si MI(X, Y) = 0, entonces X e Y son independientes; cuanto
#' mayor sea la MI, mayor es la dependencia entre ambas variables.
#'
#' \deqn{MI(X,Y) = \sum_y \sum_x p(x,y) \log(p(x,y)/(p(x)p(y)))}
#'
#' @param atributo1 Nombre o columna del primer atributo categórico.
#' @param atributo2 Nombre o columna del segundo atributo categórico.
#' @param normalizar Si es TRUE, se normalizan los resultados a una escala 0-1.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un list con la información mutua `mi`, el número de observaciones `n`, y la tabla de contingencia ya calculada para poder pintar el heatmap sin recalcularla.
.calc_corr_catg <- function(atributo1, atributo2, normalizar = FALSE, verbose = 1) {
    # Mutual Information (Información Mutua).
    # La información mutua mide cuánta información aporta conocer
    # una variable sobre la otra.
    #
    # Si MI(X, Y) = 0, entonces X e Y son independientes. Cuanto mayor
    # sea la MI, mayor es la dependencia entre ambas variables.
    #
    # MI(X,Y) = sumatorio_y sumatorio_x [ p(x,y) * log(p(x,y)/(p(x)*p(y))) ]
    #
    # Devuelve un diccionario con la MI y la tabla de contingencia ya
    # calculada, para poder pintar el heatmap sin recalcularla.

    nombre1 <- attr(atributo1, "name") %||% "Atributo 1"
    nombre2 <- attr(atributo2, "name") %||% "Atributo 2"

    datos <- data.frame(
        x = as.numeric(unlist(atributo1)),
        y = as.numeric(unlist(atributo2)),
        stringsAsFactors = FALSE
    )

    datos <- datos[!is.na(datos$x) & !is.na(datos$y), ]

    if (nrow(datos) == 0) {
        stop("No hay suficientes datos para calcular la información mutua.", call. = FALSE)
    }

    x <- datos[["x"]]
    y <- datos[["y"]]

    n <- length(x)

    contando_x <- table(factor(x, levels = unique(x)))
    contando_y <- table(factor(y, levels = unique(y)))

    # Tabla de contingencia.
    tabla <- as.matrix(table(x, y))

    resultado <- 0

    for (valor_x in names(contando_x)) {
        for (valor_y in names(contando_y)) {

            pareja_count <- tabla[valor_x, valor_y]

            if (is.na(pareja_count) || pareja_count == 0) {
                continue
            }

            p_xy <- pareja_count / n
            p_x <- contando_x[valor_x] / n
            p_y <- contando_y[valor_y] / n

            resultado <- resultado + p_xy * log(p_xy / (p_x * p_y))

        }

    }

    if (normalizar) {
        # TODO:
        # Normalizar MI a [0, 1].
    }

    return(list(
        tipo = "informacion_mutua",
        valor = resultado,
        n = n,
        tabla = tabla,
        nombre1 = nombre1,
        nombre2 = nombre2
    ))
}


# =============================================================================
# Correlación categórica + numérica (Welch ANOVA)
# =============================================================================

# =============================================================================
# Correlación categórica + numérica (Welch ANOVA)
# =============================================================================

#' Welch ANOVA de un factor entre un atributo categórico y uno numérico.
#'
#' Comprueba si existen diferencias entre las medias de una variable continua
#' para los distintos grupos de una variable discreta. Welch ANOVA es una
#' variante del ANOVA de un factor que no requiere que las varianzas de los
#' grupos sean iguales.
#'
#' @param atributo1 Nombre o columna del primer atributo (categórico/discreto).
#' @param atributo2 Nombre o columna del segundo atributo (numérico/continuo).
#' @param normalizar Si es TRUE, se normalizan los resultados a una escala 0-1.
#' @param verbose Nivel de información que se muestra durante la ejecución.
#' @return Un list con el estadístico `f`, el valor-p `p_value`, el número de observaciones `n`, y toda la información por grupo (medias, tamaños, varianzas) para poder pintar el boxplot sin volver a tocar los datos originales.
.calc_corr_catg_num <- function(atributo1, atributo2, normalizar = FALSE, verbose = 1) {
    # Welch ANOVA de un factor.
    # Comprueba si existen diferencias entre las medias de una
    # variable continua para los distintos grupos de una variable
    # discreta. Welch ANOVA es una variante del ANOVA de un factor que
    # no requiere que las varianzas de los grupos sean iguales.
    #
    # Devuelve un diccionario con el estadístico F y toda la info por
    # grupo (medias, tamaños, varianzas) para poder pintar el boxplot
    # sin volver a tocar los datos originales.

    datos <- data.frame(
        x = as.numeric(unlist(atributo1)),
        y = as.numeric(unlist(atributo2)),
        stringsAsFactors = FALSE
    )

    datos <- datos[!is.na(datos$x) & !is.na(datos$y), ]

    x <- datos[["x"]]
    y <- datos[["y"]]

    nombre_categorico <- attr(atributo1, "name") %||% "Grupo"
    nombre_numerico <- attr(atributo2, "name") %||% "Valor"

    # Nos aseguramos de que x sea la discreta e y la continua.
    if (!(.es_discreta(x) && .es_continua(y))) {
        x <- y
        y <- x
        nombre_categorico <- nombre_numerico
        nombre_numerico <- nombre_categorico
    }

    grupos <- unique(x)

    if (length(grupos) < 2) {
        stop("Welch ANOVA necesita al menos dos grupos.", call. = FALSE)
    }

    valores_grupos <- list()
    n_grupos <- list()
    medias <- list()
    varianzas <- list()
    grupos_validos <- list()

    for (grupo in grupos) {

        valores <- y[x == grupo]

        if (length(valores) < 2) {
            .verbose(
                paste0("El grupo '", grupo, "' no tiene suficientes observaciones "),
                "y será excluido del cálculo.",
                verbose,
                nivel = 1,
                tipo = "warning"
            )

            next
        }

        valores_grupos[[as.character(grupo)]] <- valores
        n_grupos[[as.character(grupo)]] <- length(valores)
        medias[[as.character(grupo)]] <- mean(valores)
        varianzas[[as.character(grupo)]] <- var(valores)

        if (varianzas[[as.character(grupo)]] == 0) {
            .verbose(
                paste0("El grupo '", grupo, "' tiene varianza 0 "),
                "y será excluido del cálculo.",
                verbose,
                nivel = 1,
                tipo = "warning"
            )

            valores_grupos[[as.character(grupo)]] <- NULL
            n_grupos[[as.character(grupo)]] <- NULL
            medias[[as.character(grupo)]] <- NULL
            varianzas[[as.character(grupo)]] <- NULL
            next
        }

        grupos_validos[[as.character(grupo)]] <- as.character(grupo)

    }

    grupos <- unlist(grupos_validos)

    if (length(grupos) < 2) {
        stop("No hay suficientes grupos válidos para calcular Welch ANOVA.", call. = FALSE)
    }

    pesos <- list()

    for (grupo in grupos) {
        pesos[[grupo]] <- n_grupos[[grupo]] / varianzas[[grupo]]
    }

    suma_pesos <- sum(unlist(pesos))
    suma_pesos_media <- sum(
        unlist(lapply(grupos, function(g) pesos[[g]] * medias[[g]])))

    media_ponderada <- suma_pesos_media / suma_pesos

    k <- length(grupos)

    suma_entre <- sum(
        lapply(grupos, function(g) pesos[[g]] * (medias[[g]] - media_ponderada)^2))

    numerador <- suma_entre / (k - 1)

    correccion <- 0

    for (grupo in grupos) {
        parte <- 1 - pesos[[grupo]] / suma_pesos
        correccion <- correccion + parte^2 / (n_grupos[[grupo]] - 1)
    }

    correccion <- 1 + (2 * (k - 2) / (k^2 - 1)) * correccion

    f_welch <- numerador / correccion

    gl_entre <- k - 1

    # TODO:
    # Calcular los grados de libertad del denominador y el p-value
    # a partir de la distribución F.

    if (normalizar) {
        # TODO:
        # Normalizar el resultado a [0, 1].
    }

    return(list(
        tipo = "welch",
        valor = f_welch,
        gl_entre = gl_entre,
        grupos = grupos,
        n_grupos = n_grupos,
        medias = medias,
        varianzas = varianzas,
        media_general = mean(y),
        valores_grupos = valores_grupos,
        nombre_categorico = nombre_categorico,
        nombre_numerico = nombre_numerico
    ))
}


# =============================================================================
# Helpers internos
# =============================================================================

`%||%` <- function(a, b) {
    # Devuelve `a` si no es NULL/NA, de lo contrario `b`.
    if (is.null(a) || is.na(a)) {
        return(b)
    }

    a
}

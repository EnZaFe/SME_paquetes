# Sourcear funciones auxiliares (verbose y helpers de tipo/verificación)
source("../auxiliar/verbose.R")
source("../auxiliar/auxiliar_es.r")


#' Tema visual Gestalt: paleta semántica y configuración global.
#'
#' Tres colores fijos, cada uno con un único significado en todo el módulo:
#' `COLOR_DATO` para lo medido directamente, `COLOR_ESTIMADO` para estimaciones
#' (regresión, medias, densidades) y `COLOR_REFERENCIA` para referencias teóricas
#' (azar, media general).
COLOR_DATO <- "#2c6e91"      # datos medidos
COLOR_ESTIMADO <- "#e8734d"  # estimaciones (regresión, medias, densidades)
COLOR_REFERENCIA <- "#9a9a9a" # referencias teóricas (azar, media general)


#' Gráfico de la curva ROC y la distribución del atributo por clase.
#'
#' Curva ROC + distribución del atributo numérico por clase. Se puede llamar
#' de dos formas: con el AUC calculado automáticamente o reutilizando un AUC ya
#' calculado (evitando recalcularlo).
#'
#' @section Uso:
#' ```
#' graficar_auc(score, clases) -> fig con la curva ROC y la densidad por clase
#' graficar_auc(score, clases, resultado = info) -> reutiliza un AUC ya calculado
#' ```
#'
#' @param atributo_clm Serie numérica (score/probabilidad del modelo).
#' @param clases Vector de etiquetas de clase, exactamente dos. Deben tener el
#'   mismo número de elementos que `atributo_clm`.
#' @param resultado Lista opcional con el AUC ya calculado. Si se pasa, no se recalcula.
#' @param pregunta Pregunta que responde el gráfico. Si es `NULL`, se genera una automáticamente.
#' @param verbose Nivel de información mostrado durante la ejecución.
#' @param path Carpeta base donde se guarda la imagen.
#' @param nombre_carpeta Nombre de la subcarpeta (por defecto `"figures"`).
#' @return Un objeto `ggplot` con la curva ROC y la distribución por clase, guardado como `demo_auc.png`.
#' @export
graficar_auc <- function(atributo_clm, clases, resultado = NULL, pregunta = NULL,
                         verbose = 1, path = ".", nombre_carpeta = "figures") {

    # Objetivo: curva ROC + distribución del atributo por clase.
    # Uso: graficar_auc(score, clases)

    .verbose("Iniciando gráfico de AUC...", verbose)

    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE, showWarnings = FALSE)

    # RAISE ERRORS para uso adecuado de las funciones
    if (length(atributo_clm) != length(clases)) {
        .verbose("El atributo y las clases deben tener el mismo número de elementos.", verbose, nivel = 1, tipo = "error")
        stop("El atributo y las clases deben tener el mismo número de elementos.")
    }

    if (length(atributo_clm) == 0) {
        .verbose("No se puede calcular el AUC con datos vacíos.", verbose, nivel = 1, tipo = "error")
        stop("No se puede calcular el AUC con datos vacíos.")
    }

    if (!.es_continua(atributo_clm)) {
        .verbose("El atributo debe ser numérico para calcular el AUC.", verbose, nivel = 1, tipo = "error")
        stop("El atributo debe ser numérico para calcular el AUC.")
    }

    valores_clases <- unique(clases)
    if (length(valores_clases) != 2) {
        .verbose("Las clases deben contener exactamente dos clases.", verbose, nivel = 1, tipo = "error")
        stop("Las clases deben contener exactamente dos clases.")
    }

    # Calcular o reutilizar el AUC.
    if (!is.null(resultado) && !is.null(resultado[["valor"]])) {
        auc <- resultado[["valor"]]
    } else {
        .verbose("No se ha proporcionado el resultado de AUC. Se calculará automáticamente.", verbose, nivel = 2)
        auc <- .calcular_AUC(atributo_clm, clases, verbose = 0)
    }

    # Preparar los datos.
    datos <- data.frame(
        clase = as.character(clases),
        atributo = as.numeric(atributo_clm),
        stringsAsFactors = FALSE
    )
    datos <- datos[order(-datos$atributo), ]
    rownames(datos) <- NULL

    clase_1 <- valores_clases[1]
    clase_2 <- valores_clases[2]

    positivos <- sum(datos$clase == clase_1)
    negativos <- sum(datos$clase == clase_2)

    if (positivos == 0 || negativos == 0) {
        .verbose("Cada clase debe tener al menos una observación.", verbose, nivel = 1, tipo = "error")
        stop("Cada clase debe tener al menos una observación.")
    }

    # Construir curva ROC.
    tpr <- c(0)
    fpr <- c(0)
    vp <- 0
    fp <- 0

    for (clase in datos$clase) {
        if (clase == clase_1) vp <- vp + 1 else fp <- fp + 1
        tpr <- c(tpr, vp / positivos)
        fpr <- c(fpr, fp / negativos)
    }
    tpr <- c(tpr, 1)
    fpr <- c(fpr, 1)

    # Tema y título-pregunta.
    nombre_attr <- if (!is.null(attr(atributo_clm, "name"))) attr(atributo_clm, "name") else "el atributo"
    if (is.null(pregunta)) {
        pregunta <- paste0("¿Qué tan bien distingue ", nombre_attr, " entre las dos clases?")
    }

    paleta_clases <- c(COLOR_DATO, "#7a3b8c")

    ggplot(datos, aes(x = .(.), y = .)) +
        # Curva ROC: la curva se calcula a partir de los datos reales -> COLOR_DATO.
        stat_roc(data = data.frame(fpr = fpr, tpr = tpr), aes(x = fpr, y = tpr), color = COLOR_DATO, fill = COLOR_DATO, alpha = 0.15) +
        # La diagonal es una referencia teórica (clasificador al azar) -> COLOR_REFERENCIA.
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = COLOR_REFERENCIA, linewidth = 0.5) +
        annotate("text", x = 0.62, y = 0.57, label = "azar", color = COLOR_REFERENCIA,
                 fontsize = 11, fontface = "italic", rotate = 33) +
        # El AUC se ancla sobre la propia curva en vez de vivir lejos en una leyenda.
        annotate("text", x = fpr[ceiling(length(fpr) / 2)], y = tpr[ceiling(length(tpr) / 2)],
                 label = paste0("AUC = ", format(auc, digits = 3)), color = COLOR_DATO,
                 fontsize = 13, fontface = "bold",
                 hjust = inf(fpr[ceiling(length(fpr) / 2)] + 0.16), vjust = -inf(tpr[ceiling(length(tpr) / 2)] - 0.20)) +
        # Distribución del atributo por clase (densidad estimada).
        geom_density(data = datos, aes(x = atributo, fill = clase), alpha = 0.35, linewidth = 1.8) +
        annotate("text", x = rep(NA, 2), y = NA, label = "", hidden = TRUE) +
        # Etiquetas de cada clase ancladas junto a su propia curva.
        geom_text(data = data.frame(atributo = sapply(valores_clases, function(c) mean(datos$atributo[datos$clase == c])),
                                    clase = valores_clases),
                  aes(x = atributo, y = 0.94, label = paste0("clase ", .)), color = paleta_clases,
                  fontface = "bold", fontsize = 11) +
        labs(
            title = pregunta, subtitle = "",
            x = "Tasa de falsos positivos", y = "Tasa de verdaderos positivos",
            fill = NULL
        ) +
        theme_minimal(base_size = 15) +
        theme(
            plot.title = element_text(size = 17, face = "bold", hjust = 1),
            legend.position = "none",
            panel.grid.major.x = element_blank(),
            axis.text.x = element_blank(), axis.text.y = element_blank(),
            axis.ticks.x = element_blank(), axis.ticks.y = element_blank()
        )
}


#' Gráfico de dispersión con regresión lineal y coeficiente de Pearson.
#'
#' Dispersión + regresión lineal + distribuciones marginales entre dos variables
#' numéricas, junto con el coeficiente de Pearson. Se puede llamar calculando todo
#' internamente o reutilizando un resultado ya calculado.
#'
#' @section Uso:
#' ```
#' graficar_pearson(x, y) -> fig con dispersión + recta ajustada
#' graficar_pearson(x, y, resultado = info) -> reutiliza un resultado de Pearson
#' ```
#'
#' @param atributo1 Primera variable numérica.
#' @param atributo2 Segunda variable numérica. Debe tener el mismo número de elementos.
#' @param resultado Lista opcional con el resultado de Pearson ya calculado.
#' @param pregunta Pregunta que responde el gráfico. Si es `NULL`, se genera una automáticamente.
#' @param verbose Nivel de información mostrado durante la ejecución.
#' @param path Carpeta base donde se guarda la imagen.
#' @param nombre_carpeta Nombre de la subcarpeta (por defecto `"figures"`).
#' @return Un objeto `ggplot` con el gráfico de dispersión y la recta ajustada, guardado como `pearson.png`.
#' @export
graficar_pearson <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                             verbose = 1, path = ".", nombre_carpeta = "figures") {

    # Objetivo: dispersión + regresión lineal + coeficiente de Pearson.
    # Uso: graficar_pearson(x, y)

    .verbose("Iniciando gráfico de Pearson...", verbose)

    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE, showWarnings = FALSE)

    # RAISE ERRORS para uso adecuado de las funciones
    if (length(atributo1) != length(atributo2)) {
        .verbose("Los atributos deben tener el mismo número de elementos.", verbose, nivel = 1, tipo = "error")
        stop("Los atributos deben tener el mismo número de elementos.")
    }

    if (!.es_continua(atributo1)) {
        .verbose("El primer atributo debe ser numérico.", verbose, nivel = 1, tipo = "error")
        stop("El primer atributo debe ser numérico.")
    }

    if (!.es_continua(atributo2)) {
        .verbose("El segundo atributo debe ser numérico.", verbose, nivel = 1, tipo = "error")
        stop("El segundo atributo debe ser numérico.")
    }

    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de Pearson. Se calculará automáticamente.", verbose, nivel = 2)
        resultado <- .calc_corr_num(atributo1, atributo2, verbose = 0)
    }

    x <- as.numeric(atributo1)
    y <- as.numeric(atributo2)
    r <- resultado[["valor"]]
    r2 <- resultado[["r2"]]
    nombre1 <- if (!is.null(attr(atributo1, "name"))) attr(atributo1, "name") else "x"
    nombre2 <- if (!is.null(attr(atributo2, "name"))) attr(atributo2, "name") else "y"

    # Recta de regresión.
    pendiente <- coef(lm(y ~ x))[2]
    intercepto <- coef(lm(y ~ x))[1]
    y_linea <- pendiente * x + intercepto
    error_estandar <- sd(y - y_linea)

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Existe una relación lineal entre ", nombre1, " y ", nombre2, "?")
    }

    ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
        # Los puntos son datos reales -> COLOR_DATO.
        geom_point(color = COLOR_DATO, size = 2) +
        # La recta es una estimación -> COLOR_ESTIMADO, encerrada en banda sombreada.
        geom_smooth(method = "lm", formula = y ~ x, se = TRUE, fill = COLOR_ESTIMADO, alpha = 0.15, color = COLOR_ESTIMADO, linewidth = 2.5) +
        # Ecuación y coeficiente anclados junto al extremo de la recta.
        annotate("text", x = max(x), y = max(y_linea),
                 label = paste0("y = ", format(pendiente, digits = 2), "x + ", format(intercepto, digits = 2), "\n",
                               "r = ", format(r, digits = 3), "  (r² = ", format(r2, digits = 3), ")"),
                 color = COLOR_ESTIMADO, fontsize = 11, fontface = "bold", hjust = 0, vjust = 0) +
        labs(
            title = pregunta, subtitle = "",
            x = nombre1, y = nombre2
        ) +
        theme_minimal(base_size = 15) +
        theme(
            plot.title = element_text(size = 17, face = "bold", hjust = 1),
            legend.position = "none",
            panel.grid.major = element_line(color = "#e8e8e8"),
            axis.text = element_text(color = "#333333")
        )
}


#' Gráfico de la tabla de contingencia y la información mutua entre dos categóricas.
#'
#' Heatmap de la tabla de contingencia entre dos variables categóricas, con
#' frecuencias absolutas y relativas, y la información mutua entre ambas. Se puede
#' llamar calculando todo internamente o reutilizando un resultado ya calculado.
#'
#' @section Uso:
#' ```
#' graficar_informacion_mutua(a1, a2) -> fig con los dos heatmaps y el valor de MI
#' graficar_informacion_mutua(a1, a2, resultado = info) -> reutiliza un resultado de MI
#' ```
#'
#' @param atributo1 Primera variable categórica/discreta.
#' @param atributo2 Segunda variable categórica/discreta. Debe tener el mismo número de elementos.
#' @param resultado Lista opcional con la información mutua ya calculada.
#' @param pregunta Pregunta que responde el gráfico. Si es `NULL`, se genera una automáticamente.
#' @param verbose Nivel de información mostrado durante la ejecución.
#' @param path Carpeta base donde se guarda la imagen.
#' @param nombre_carpeta Nombre de la subcarpeta (por defecto `"figures"`).
#' @return Un objeto `ggplot` con los dos heatmaps y el valor de MI, guardado como `MI.png`.
#' @export
graficar_informacion_mutua <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                                       verbose = 1, path = ".", nombre_carpeta = "figures") {

    # Objetivo: heatmap de la tabla de contingencia + información mutua.
    # Uso: graficar_informacion_mutua(a1, a2)

    .verbose("Iniciando gráfico de información mutua...", verbose)

    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE, showWarnings = FALSE)

    # RAISE ERRORS para uso adecuado de las funciones
    if (length(atributo1) != length(atributo2)) {
        .verbose("Los atributos deben tener el mismo número de elementos.", verbose, nivel = 1, tipo = "error")
        stop("Los atributos deben tener el mismo número de elementos.")
    }

    if (!.es_discreta(atributo1)) {
        .verbose("El primer atributo debe ser categórico/discreto.", verbose, nivel = 1, tipo = "error")
        stop("El primer atributo debe ser categórico/discreto.")
    }

    if (!.es_discreta(atributo2)) {
        .verbose("El segundo atributo debe ser categórico/discreto.", verbose, nivel = 1, tipo = "error")
        stop("El segundo atributo debe ser categórico/discreto.")
    }

    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de información mutua. Se calculará automáticamente.", verbose, nivel = 2)
        resultado <- .calc_corr_catg(atributo1, atributo2, verbose = 0)
    }

    tabla <- resultado[["tabla"]]
    mi <- resultado[["valor"]]
    nombre1 <- if (!is.null(attr(atributo1, "name"))) attr(atributo1, "name") else "a1"
    nombre2 <- if (!is.null(attr(atributo2, "name"))) attr(atributo2, "name") else "a2"

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Están relacionadas las categorías de ", nombre1, " y ", nombre2, "?")
    }

    # Ordenar filas y columnas de mayor a menor frecuencia total.
    orden_filas <- names(sort(rowSums(tabla), decreasing = TRUE))
    orden_columnas <- names(sort(colSums(tabla), decreasing = TRUE))
    tabla <- tabla[orden_filas, orden_columnas]
    tabla_pct <- tabla / sum(tabla) * 100

    ggplot() +
        # Frecuencias absolutas (dato) -> codificación de intensidad viridis.
        geom_tile(data = as.data.frame(tabla), aes(x = rownames(tabla), y = colnames(tabla), fill = tabla),
                  fill = "viridis", color = "white") +
        annotate("text", data = as.data.frame(tabla), x = rownames(tabla), y = colnames(tabla), label = tabla, size = 3) +
        # Frecuencias relativas (%) -> derivado, mismo esquema de colores.
        geom_tile(data = as.data.frame(tabla_pct), aes(x = rownames(tabla), y = colnames(tabla), fill = tabla_pct),
                  fill = "viridis", color = "white") +
        annotate("text", data = as.data.frame(tabla_pct), x = rownames(tabla), y = colnames(tabla),
                 label = format(tabla_pct, digits = 1), size = 3) +
        # Información mutua colocada pegada a las dos tablas.
        annotate("text", x = 0.5, y = 0.88, label = paste0("Información mutua = ", format(mi, digits = 4), " bits"),
                 hjust = 0.5, color = COLOR_ESTIMADO, fontface = "bold", fontsize = 13) +
        labs(
            title = pregunta, subtitle = "",
            x = nombre2, y = nombre1
        ) +
        scale_fill_gradientn(colors = c("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"), name = "Frecuencia") +
        theme_minimal(base_size = 15) +
        theme(
            plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
            legend.position = "bottom",
            axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid = element_blank()
        )
}


#' Gráfico de Welch ANOVA: boxplot + puntos + medias por grupo.
#'
#' Boxplot + puntos individuales por grupo, con la media de cada grupo, la media
#' general y el estadístico F de Welch ANOVA. Se puede llamar calculando todo
#' internamente o reutilizando un resultado ya calculado.
#'
#' @section Uso:
#' ```
#' graficar_welch(grupo, valor) -> fig con boxplot + medias
#' graficar_welch(grupo, valor, orden_categorias = c(...)) -> usa una secuencia real para conectar las medias
#' ```
#'
#' @param atributo1 Variable categórica/discreta que define los grupos.
#' @param atributo2 Variable numérica medida dentro de cada grupo. Debe tener el mismo número de elementos.
#' @param resultado Lista opcional con el resultado de Welch ANOVA ya calculado.
#' @param pregunta Pregunta que responde el gráfico. Si es `NULL`, se genera una automáticamente.
#' @param orden_categorias Vector opcional de categorías con una secuencia real (años, trimestres...). Si se pasa, las medias se conectan con línea.
#' @param verbose Nivel de información mostrado durante la ejecución.
#' @param path Carpeta base donde se guarda la imagen.
#' @param nombre_carpeta Nombre de la subcarpeta (por defecto `"figures"`).
#' @return Un objeto `ggplot` con el boxplot, las medias y el estadístico F, guardado como `welch.png`.
#' @export
graficar_welch <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                           orden_categorias = NULL, verbose = 1, path = ".", nombre_carpeta = "figures") {

    # Objetivo: boxplot + puntos + medias por grupo + Welch ANOVA.
    # Uso: graficar_welch(grupo, valor)

    .verbose("Iniciando gráfico de Welch...", verbose)

    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE, showWarnings = FALSE)

    # RAISE ERRORS para uso adecuado de las funciones
    if (length(atributo1) != length(atributo2)) {
        .verbose("Los atributos deben tener el mismo número de elementos.", verbose, nivel = 1, tipo = "error")
        stop("Los atributos deben tener el mismo número de elementos.")
    }

    if (!((.es_discreta(atributo1) && .es_continua(atributo2)) || (.es_continua(atributo1) && .es_discreta(atributo2)))) {
        .verbose("Welch necesita un atributo categórico y otro numérico.", verbose, nivel = 1, tipo = "error")
        stop("Welch necesita un atributo categórico y otro numérico.")
    }

    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de Welch ANOVA. Se calculará automáticamente.", verbose, nivel = 2)
        resultado <- .calc_corr_catg_num(atributo1, atributo2, verbose = 0)
    }

    grupos <- resultado[["grupos"]]
    valores_grupos <- resultado[["valores_grupos"]]
    medias <- resultado[["medias"]]
    n_grupos <- resultado[["n_grupos"]]
    media_general <- resultado[["media_general"]]
    f_welch <- resultado[["valor"]]
    nombre_categorico <- if (!is.null(attr(atributo1, "name"))) attr(atributo1, "name") else "grupo"
    nombre_numerico <- if (!is.null(attr(atributo2, "name"))) attr(atributo2, "name") else "valor"

    # Ordenar grupos: si hay secuencia real, usarla; si no, por media descendente.
    es_secuencia_real <- !is.null(orden_categorias)
    if (es_secuencia_real) {
        grupos_ordenados <- orden_categorias[orden_categorias %in% grupos]
    } else {
        grupos_ordenados <- sort(names(medias), decreasing = TRUE)
    }

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Difiere ", nombre_numerico, " según ", nombre_categorico, "?")
    }

    datos <- data.frame(
        grupo = factor(grupos, levels = grupos_ordenados),
        valor = as.numeric(atributo2)
    )

    ggplot(datos, aes(x = grupo, y = valor)) +
        # Los puntos y las cajas son datos reales -> COLOR_DATO.
        geom_boxplot(color = COLOR_DATO, fill = COLOR_DATO, alpha = 0.30, outlier.shape = NA) +
        geom_jitter(color = COLOR_DATO, alpha = 0.45, size = 2, width = 0.2) +
        # Medias de grupo (estimadas) -> COLOR_ESTIMADO.
        geom_point(data = data.frame(grupo = grupos_ordenados, media = sapply(grupos_ordenados, function(g) medias[[g]])),
                   aes(x = grupo, y = media), shape = 17, size = 4, color = COLOR_ESTIMADO, fill = "white") +
        # n y la media ancladas junto al marcador de cada grupo.
        geom_text(data = data.frame(grupo = grupos_ordenados, n = sapply(grupos_ordenados, function(g) n_grupos[[g]]),
                                    media = sapply(grupos_ordenados, function(g) medias[[g]])),
                  aes(x = grupo, y = .media, label = paste0("n=", .n, "\nμ=", format(.media, digits = 2))),
                  color = COLOR_ESTIMADO, fontface = "bold", hjust = 1.2, vjust = 0) +
        # La media general es una referencia teórica neutra -> COLOR_REFERENCIA.
        geom_hline(yintercept = media_general, linetype = "dashed", color = COLOR_REFERENCIA, linewidth = 1.5) +
        annotate("text", x = length(grupos_ordenados) - 0.55, y = media_general,
                 label = paste0(" media general = ", format(media_general, digits = 2)),
                 color = COLOR_REFERENCIA, fontsize = 10, hjust = 0) +
        # Estadístico F de Welch ANOVA en el título.
        labs(
            title = paste0("Welch ANOVA · F = ", format(f_welch, digits = 3), " (", length(grupos_ordenados), " grupos)", "\n", pregunta),
            subtitle = "",
            x = nombre_categorico, y = nombre_numerico
        ) +
        theme_minimal(base_size = 15) +
        theme(
            plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
            legend.position = "none",
            panel.grid.major.y = element_line(color = "#e8e8e8"),
            axis.text.x = element_text(angle = 45, hjust = 1)
        )
}

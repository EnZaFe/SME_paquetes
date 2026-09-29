library(ggplot2)
library(patchwork)

COLOR_DATO <- "#2c6e91"       # datos medidos
COLOR_ESTIMADO <- "#e8734d"   # estimaciones (regresión, medias, densidades)
COLOR_REFERENCIA <- "#9a9a9a" # referencias teóricas (azar, media general)
COLOR_CLASE_2 <- "#7a3b8c"    # segunda clase en AUC / densidades


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
#' @return Un objeto `patchwork`/`ggplot` con la curva ROC y la distribución por clase,
#'   guardado como `demo_auc.png`.
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
 
    valores_clases <- unique(as.character(clases))
    if (length(valores_clases) != 2) {
        .verbose("Las clases deben contener exactamente dos clases.", verbose, nivel = 1, tipo = "error")
        stop("Las clases deben contener exactamente dos clases.")
    }

    nombre_variable <- deparse(substitute(atributo_clm))
    nombre_variable <- sub(".*\\$", "", nombre_variable)
    nombre_variable <- gsub("_", " ", nombre_variable)

    # Orden determinista: la clase "positiva" es la de mayor valor (igual que en Python).
    valores_clases <- sort(valores_clases)
    clase_neg <- valores_clases[1]
    clase_pos <- valores_clases[2]
 
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
 
    positivos <- sum(datos$clase == clase_pos)
    negativos <- sum(datos$clase == clase_neg)
 
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
        if (clase == clase_pos) vp <- vp + 1 else fp <- fp + 1
        tpr <- c(tpr, vp / positivos)
        fpr <- c(fpr, fp / negativos)
    }
    tpr <- c(tpr, 1)
    fpr <- c(fpr, 1)
    roc_df <- data.frame(fpr = fpr, tpr = tpr)
 
    # Tema y título-pregunta.
    nombre_attr <- if (!is.null(nombre_variable)) nombre_variable else "x"
    if (is.null(pregunta)) {
        pregunta <- paste0("¿Qué tan bien distingue ", nombre_attr, " entre las dos clases?")
    }
 
    paleta_clases <- stats::setNames(c(COLOR_CLASE_2, COLOR_DATO), c(clase_neg, clase_pos))
 
    # --- Panel 1: curva ROC ---
    panel_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
      geom_ribbon(aes(ymin = 0, ymax = tpr), fill = COLOR_DATO, alpha = 0.15) +
      geom_line(color = COLOR_DATO, linewidth = 1.2) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = COLOR_REFERENCIA, linewidth = 0.5) +
      annotate("text", x = 0.55, y = 0.50, label = "azar", color = COLOR_REFERENCIA,
               size = 4, fontface = "italic", angle = 33) +
      annotate("text", x = 0.60, y = 0.20,
               label = paste0("AUC = ", format(auc, digits = 3)),
               color = COLOR_DATO, size = 5, fontface = "bold", hjust = 0) +
      coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(
        title = "Curva ROC",
        x = paste0("Falsos positivos\n(% de «", clase_neg, "» marcados como «", clase_pos, "»)"),
        y = paste0("Verdaderos positivos\n(% de «", clase_pos, "» detectados)")
      ) +
      theme_minimal(base_size = 13) +
      theme(panel.grid.minor = element_blank())
    
    # --- Panel 2: distribución del atributo por clase ---
    rango_attr <- range(datos$atributo, na.rm = TRUE)
    margen <- diff(rango_attr) * 0.08
    
    panel_dist <- ggplot(datos, aes(x = atributo, fill = clase, color = clase)) +
      geom_density(alpha = 0.35, linewidth = 1) +
      scale_fill_manual(values = paleta_clases, breaks = c(clase_neg, clase_pos)) +
      scale_color_manual(values = paleta_clases, breaks = c(clase_neg, clase_pos)) +
      coord_cartesian(
        xlim = c(rango_attr[1] - margen, rango_attr[2] + margen)
      ) +
      labs(
        title = "Distribución por clase",
        x = nombre_attr, y = "Densidad (estimada)", fill = "clase", color = "clase"
      ) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom")
    
    # Añadimos la pregunta como subtítulo a cada panel (ya que no usamos patchwork::plot_annotation)
    panel_roc <- panel_roc +
      labs(subtitle = pregunta) +
      theme(plot.subtitle = element_text(size = 12, face = "italic", hjust = 0.5))
    
    panel_dist <- panel_dist +
      labs(subtitle = pregunta) +
      theme(plot.subtitle = element_text(size = 12, face = "italic", hjust = 0.5))
    
    .verbose(
      paste0(
        "ROC · Cómo leerla: al marcar como positivas las observaciones de mayor a menor ",
        nombre_attr, ", la curva sube con cada acierto. Cuanto más cerca de la esquina ",
        "superior izquierda, mejor; la diagonal es el azar."
      ),
      verbose, nivel = 1
    )
    
    # Guardar cada gráfico por separado
    ggsave(
      filename = file.path(carpeta_salida, "demo_auc_roc.png"),
      plot = panel_roc, width = 6, height = 6, dpi = 110, bg = "white"
    )
    ggsave(
      filename = file.path(carpeta_salida, "demo_auc_dist.png"),
      plot = panel_dist, width = 6, height = 6, dpi = 110, bg = "white"
    )
    
    # Imprimir cada gráfico por separado
    print(panel_roc)
    print(panel_dist)
    
    # Devolver los gráficos
    invisible(list(roc = panel_roc, dist = panel_dist))
}
 



#' Gráfico de dispersión con regresión lineal y coeficiente de Pearson.
#'
#' Dispersión + regresión lineal entre dos variables numéricas, junto con el
#' coeficiente de Pearson. Se puede llamar calculando todo internamente o
#' reutilizando un resultado ya calculado.
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

    nombre_variable_1 <- deparse(substitute(atributo1))
    nombre_variable_1 <- sub(".*\\$", "", nombre_variable_1)
    nombre_variable_1 <- gsub("_", " ", nombre_variable_1)

    nombre_variable_2 <- deparse(substitute(atributo2))
    nombre_variable_2 <- sub(".*\\$", "", nombre_variable_2)
    nombre_variable_2 <- gsub("_", " ", nombre_variable_2)


    x <- as.numeric(atributo1)
    y <- as.numeric(atributo2)
    r <- resultado[["valor"]]
    r2 <- resultado[["r2"]]
    nombre1 <- if (!is.null(nombre_variable_1)) nombre_variable_1 else "x"
    nombre2 <- if (!is.null(nombre_variable_2)) nombre_variable_2 else "y"

    # Recta de regresión.
    modelo <- stats::lm(y ~ x)
    pendiente <- stats::coef(modelo)[2]
    intercepto <- stats::coef(modelo)[1]
    y_linea <- pendiente * x + intercepto

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Existe una relación lineal entre ", nombre1, " y ", nombre2, "?")
    }

    signo <- if (intercepto >= 0) "+" else "\u2212"
    etiqueta <- paste0(
        "y = ", format(pendiente, digits = 2), "x ", signo, " ", format(abs(intercepto), digits = 2), "\n",
        "r = ", format(r, digits = 3), "   (r\u00b2 = ", format(r2, digits = 3), ")"
    )

    grafico <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
        # Los puntos son datos reales -> COLOR_DATO.
        geom_point(color = COLOR_DATO, size = 2, alpha = 0.8) +
        # La recta es una estimación -> COLOR_ESTIMADO, encerrada en banda sombreada.
        geom_smooth(
            method = "lm", formula = y ~ x, se = TRUE,
            fill = COLOR_ESTIMADO, alpha = 0.15, color = COLOR_ESTIMADO, linewidth = 1.5
        ) +
        annotate(
            "label", x = -Inf, y = Inf, label = etiqueta,
            color = COLOR_ESTIMADO, size = 4, fontface = "bold",
            hjust = -0.05, vjust = 1.2, label.size = 0, fill = scales::alpha("white", 0.8)
        ) +
        labs(title = pregunta, x = nombre1, y = nombre2) +
        theme_minimal(base_size = 15) +
        theme(
            plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
            legend.position = "none",
            panel.grid.major = element_line(color = "#e8e8e8"),
            axis.text = element_text(color = "#333333")
        )

    .verbose(
        paste0(
            "Pearson · Cómo leerlo: cada punto es una observación; la recta es el ajuste ",
            "lineal y la banda, el intervalo de confianza. r = ", format(r, digits = 3), "."
        ),
        verbose, nivel = 1
    )

    ggsave(
        filename = file.path(carpeta_salida, "pearson.png"),
        plot = grafico, width = 9, height = 8, dpi = 110, bg = "white"
    )

    grafico
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
#' @return Un objeto `patchwork`/`ggplot` con los dos heatmaps y el valor de MI, guardado como `MI.png`.
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

    nombre_variable_1 <- deparse(substitute(atributo1))
    nombre_variable_1 <- sub(".*\\$", "", nombre_variable_1)
    nombre_variable_1 <- gsub("_", " ", nombre_variable_1)

    nombre_variable_2 <- deparse(substitute(atributo2))
    nombre_variable_2 <- sub(".*\\$", "", nombre_variable_2)
    nombre_variable_2 <- gsub("_", " ", nombre_variable_2)


    tabla <- resultado[["tabla"]]
    mi <- resultado[["valor"]]
    nombre1 <- if (!is.null(nombre_variable_1)) nombre_variable_1 else "a1"
    nombre2 <- if (!is.null(nombre_variable_2)) nombre_variable_2 else "a2"

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Están relacionadas las categorías de ", nombre1, " y ", nombre2, "?")
    }

    tabla <- as.matrix(tabla)

    # Ordenar filas y columnas de mayor a menor frecuencia total.
    orden_filas <- names(sort(rowSums(tabla), decreasing = TRUE))
    orden_columnas <- names(sort(colSums(tabla), decreasing = TRUE))
    tabla <- tabla[orden_filas, orden_columnas, drop = FALSE]
    tabla_pct <- tabla / sum(tabla) * 100

    # Formato largo (Var1/Var2/Freq) que necesita geom_tile, en vez del
    # formato ancho que devolvía as.data.frame() sobre la matriz directamente.
    long_abs <- as.data.frame(as.table(tabla))
    names(long_abs) <- c("fila", "columna", "valor")
    long_abs$fila <- factor(long_abs$fila, levels = orden_filas)
    long_abs$columna <- factor(long_abs$columna, levels = orden_columnas)

    long_pct <- as.data.frame(as.table(tabla_pct))
    names(long_pct) <- c("fila", "columna", "valor")
    long_pct$fila <- factor(long_pct$fila, levels = orden_filas)
    long_pct$columna <- factor(long_pct$columna, levels = orden_columnas)

    tema_heatmap <- theme_minimal(base_size = 13) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            panel.grid = element_blank(),
            legend.position = "bottom"
        )

    # --- Panel 1: frecuencias absolutas ---
    panel_abs <- ggplot(long_abs, aes(x = columna, y = fila, fill = valor)) +
        geom_tile(color = "white") +
        geom_text(aes(label = valor), size = 3.5, color = "white", fontface = "bold") +
        scale_fill_viridis_c(name = "Frecuencia\nabsoluta") +
        labs(title = "Frecuencias absolutas", x = nombre2, y = nombre1) +
        tema_heatmap

    # --- Panel 2: frecuencias relativas ---
    panel_pct <- ggplot(long_pct, aes(x = columna, y = fila, fill = valor)) +
        geom_tile(color = "white") +
        geom_text(aes(label = sprintf("%.1f%%", valor)), size = 3.5, color = "white", fontface = "bold") +
        scale_fill_viridis_c(name = "% del\ntotal") +
        labs(title = "Frecuencias relativas (%)", x = nombre2, y = "") +
        tema_heatmap

    grafico <- (panel_abs + panel_pct) +
        patchwork::plot_annotation(
            title = pregunta,
            subtitle = paste0("Información mutua = ", format(mi, digits = 4), " bits"),
            theme = theme(
                plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
                plot.subtitle = element_text(size = 12, color = COLOR_ESTIMADO, face = "bold", hjust = 0.5)
            )
        )

    .verbose(
        paste0(
            "Información mutua · Cómo leerla: cada celda cuenta las observaciones de esa ",
            "combinación; si las filas se reparten de forma muy distinta, las variables ",
            "están relacionadas. MI = ", format(mi, digits = 4), " bits."
        ),
        verbose, nivel = 1
    )

    ggsave(
        filename = file.path(carpeta_salida, "MI.png"),
        plot = grafico, width = 12, height = 6.5, dpi = 110, bg = "white"
    )

    grafico
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
#' @param atributo1 Variable categórica/discreta que define los grupos, o la numérica (se detecta el tipo).
#' @param atributo2 La otra variable. Debe tener el mismo número de elementos.
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

    nombre_variable_1 <- deparse(substitute(atributo1))
    nombre_variable_1 <- sub(".*\\$", "", nombre_variable_1)
    nombre_variable_1 <- gsub("_", " ", nombre_variable_1)

    nombre_variable_2 <- deparse(substitute(atributo2))
    nombre_variable_2 <- sub(".*\\$", "", nombre_variable_2)
    nombre_variable_2 <- gsub("_", " ", nombre_variable_2)


    grupos <- resultado[["grupos"]]
    valores_grupos <- resultado[["valores_grupos"]]
    medias <- resultado[["medias"]]
    n_grupos <- resultado[["n_grupos"]]
    media_general <- resultado[["media_general"]]
    f_welch <- resultado[["valor"]]

    # Detectar cuál de los dos atributos es el categórico y cuál el numérico,
    # en vez de asumir siempre que atributo1 es el grupo (el orden puede venir
    # invertido, igual que en la versión Python).
    if (.es_discreta(atributo1)) {
        grupo_vec <- as.character(atributo1)
        valor_vec <- as.numeric(atributo2)
        nombre_categorico <- if (!is.null(nombre_variable_1)) nombre_variable_1 else "grupo"
        nombre_numerico <- if (!is.null(nombre_variable_2)) nombre_variable_2 else "valor"
    } else {
        grupo_vec <- as.character(atributo2)
        valor_vec <- as.numeric(atributo1)
        nombre_categorico <- if (!is.null(nombre_variable_2)) nombre_variable_2 else "grupo"
        nombre_numerico <- if (!is.null(nombre_variable_1)) nombre_variable_1 else "valor"
    }

    medias_vec <- unlist(medias)

    # Ordenar grupos: si hay secuencia real, usarla; si no, por media
    # descendente (no alfabéticamente, que es lo que hacía la versión rota).
    es_secuencia_real <- !is.null(orden_categorias)
    if (es_secuencia_real) {
        grupos_ordenados <- orden_categorias[orden_categorias %in% grupos]
    } else {
        grupos_ordenados <- names(sort(medias_vec, decreasing = TRUE))
    }

    if (is.null(pregunta)) {
        pregunta <- paste0("¿Difiere ", nombre_numerico, " según ", nombre_categorico, "?")
    }

    datos <- data.frame(grupo = grupo_vec, valor = valor_vec, stringsAsFactors = FALSE)
    datos$grupo <- factor(datos$grupo, levels = grupos_ordenados)

    medias_df <- data.frame(
        grupo = factor(grupos_ordenados, levels = grupos_ordenados),
        n = sapply(grupos_ordenados, function(g) n_grupos[[g]]),
        media = sapply(grupos_ordenados, function(g) medias[[g]])
    )

    grafico <- ggplot(datos, aes(x = grupo, y = valor)) +
        # Los puntos y las cajas son datos reales -> COLOR_DATO.
        geom_boxplot(color = COLOR_DATO, fill = COLOR_DATO, alpha = 0.30, outlier.shape = NA) +
        geom_jitter(color = COLOR_DATO, alpha = 0.45, size = 2, width = 0.2) +
        # La media general es una referencia teórica neutra -> COLOR_REFERENCIA.
        geom_hline(yintercept = media_general, linetype = "dashed", color = COLOR_REFERENCIA, linewidth = 1) +
        annotate(
            "text", x = length(grupos_ordenados), y = media_general,
            label = paste0(" media general = ", format(media_general, digits = 2)),
            color = COLOR_REFERENCIA, size = 3.5, hjust = 0, vjust = -0.5
        )

    # Solo se conectan las medias con una línea si hay una secuencia real
    # (años, trimestres...); conectar grupos sin orden real afirmaría una
    # relación que no existe.
    if (es_secuencia_real) {
        grafico <- grafico +
            geom_line(data = medias_df, aes(x = grupo, y = media, group = 1),
                      color = COLOR_ESTIMADO, linewidth = 1, alpha = 0.85)
    }

    grafico <- grafico +
        # Medias de grupo (estimadas) -> COLOR_ESTIMADO.
        geom_point(data = medias_df, aes(x = grupo, y = media),
                   shape = 17, size = 4, color = COLOR_ESTIMADO) +
        # n y la media ancladas junto al marcador de cada grupo.
        geom_text(
            data = medias_df,
            aes(x = grupo, y = media, label = paste0("n=", n, "\n\u03bc=", format(media, digits = 2))),
            color = COLOR_ESTIMADO, fontface = "bold", size = 3.2, hjust = -0.15, vjust = 0
        ) +
        labs(
            title = paste0("Welch ANOVA \u00b7 F = ", format(f_welch, digits = 3),
                            " (", length(grupos_ordenados), " grupos)\n", pregunta),
            x = nombre_categorico, y = nombre_numerico
        ) +
        theme_minimal(base_size = 14) +
        theme(
            plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
            legend.position = "none",
            panel.grid.major.y = element_line(color = "#e8e8e8"),
            panel.grid.major.x = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1)
        )

    .verbose(
        paste0(
            "Welch ANOVA \u00b7 C\u00f3mo leerlo: caja = 50% central del grupo, puntos = ",
            "observaciones, tri\u00e1ngulo = media del grupo, l\u00ednea gris = media general. F = ",
            format(f_welch, digits = 3), "."
        ),
        verbose, nivel = 1
    )

    ggsave(
        filename = file.path(carpeta_salida, "welch.png"),
        plot = grafico,
        width = max(9, 2.2 * length(grupos_ordenados) + 3), height = 7, dpi = 110, bg = "white"
    )

    grafico
}
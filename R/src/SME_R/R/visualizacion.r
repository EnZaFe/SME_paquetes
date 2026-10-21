#' @import ggplot2
#' @importFrom stats var sd cov
#' @importFrom utils head

UMBRAL_R <- 0.3
UMBRAL_AUC <- 0.7

COLOR_DATOS <- "#2c6e8f"
COLOR_AJUSTE <- "#e4572e"

# Tema común y etiquetas numéricas legibles (sin notación científica).
.tema <- function() {
    theme_minimal(base_size = 12) +
        theme(plot.title = element_text(hjust = 0.5, face = "bold"),
              plot.subtitle = element_text(hjust = 0.5, colour = "grey40"),
              panel.grid.minor = element_blank())
}
.etiq <- function(v) scales::label_number(big.mark = ".", decimal.mark = ",")(v)

#' Gráfico de la curva ROC.
#' @export
graficar_auc <- function(atributo_clm, clases, resultado = NULL, pregunta = NULL,
                         verbose = 1, path = ".", nombre_carpeta = "figures") {
    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE)
    if (length(atributo_clm) != length(clases)) stop("El atributo y las clases deben tener el mismo número de elementos.")
    if (length(atributo_clm) == 0) stop("No se puede calcular el AUC con datos vacíos.")
    if (!.es_continua(atributo_clm)) stop("El atributo debe ser numérico para calcular el AUC.")
    clases_unicas <- sort(unique(as.character(clases)))
    if (length(clases_unicas) != 2) stop("Las clases deben contener exactamente dos clases.")
    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado del AUC. Se calculará automáticamente.", verbose)
        resultado <- .calcular_AUC(atributo_clm, clases, verbose = 0)
    }
    auc <- if (is.list(resultado)) resultado[["valor"]] else resultado
    neg <- clases_unicas[1]
    pos <- clases_unicas[2]
    datos <- data.frame(clase = as.character(clases), atributo = as.numeric(atributo_clm))
    datos <- datos[order(-datos$atributo), ]
    positivos <- sum(datos$clase == pos)
    negativos <- sum(datos$clase == neg)
    vp <- fp <- 0
    tpr <- fpr <- 0
    for (clase in datos$clase) {
        if (clase == pos) vp <- vp + 1 else fp <- fp + 1
        tpr <- c(tpr, vp / positivos)
        fpr <- c(fpr, fp / negativos)
    }
    roc <- data.frame(fpr = c(fpr, 1), tpr = c(tpr, 1))
    area <- rbind(roc, data.frame(fpr = c(1, 0), tpr = c(0, 0)))
    grafico <- ggplot(roc, aes(x = fpr, y = tpr)) +
        geom_polygon(data = area, fill = COLOR_DATOS, alpha = 0.15) +
        geom_abline(linetype = "dashed", colour = "grey60") +
        annotate("text", x = 0.55, y = 0.5, label = "azar", angle = 45,
                 colour = "grey50", fontface = "italic") +
        geom_path(colour = COLOR_DATOS, linewidth = 1.2) +
        annotate("label", x = 0.75, y = 0.15, label = paste0("AUC = ", round(auc, 3), "\n(área sombreada)"),
                 colour = COLOR_DATOS, fontface = "bold") +
        coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
        scale_x_continuous(labels = scales::label_percent()) +
        scale_y_continuous(labels = scales::label_percent()) +
        labs(title = "Curva ROC",
             x = paste0("Falsos positivos\n(% de «", neg, "» marcados como «", pos, "»)"),
             y = paste0("Verdaderos positivos\n(% de «", pos, "» detectados)")) +
        .tema()
    .verbose(paste0("Curva ROC · Cómo leerla: cuanto más se acerca la curva a la esquina superior izquierda, ",
                    "mejor separa el atributo las dos clases; la diagonal es lo que haría el azar. ",
                    "Respondes SÍ si AUC ≥ ", UMBRAL_AUC, "."), verbose)
    ggsave(file.path(carpeta_salida, "demo_auc.png"), grafico, width = 6, height = 6, dpi = 110, bg = "white")
    .verbose("Gráfico de la curva ROC creado correctamente.", verbose)
    grafico
}

#' Gráfico de dispersión y recta de Pearson.
#' @export
graficar_pearson <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                             verbose = 1, path = ".", nombre_carpeta = "figures") {
    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE)
    if (length(atributo1) != length(atributo2)) stop("Los atributos deben tener el mismo número de elementos.")
    if (!.es_continua(atributo1)) stop("El primer atributo debe ser numérico.")
    if (!.es_continua(atributo2)) stop("El segundo atributo debe ser numérico.")
    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de Pearson. Se calculará automáticamente.", verbose)
        resultado <- .calc_corr_num(atributo1, atributo2, verbose = 0)
    }
    x <- as.numeric(resultado[["x"]])
    y <- as.numeric(resultado[["y"]])
    r <- resultado[["valor"]]
    b <- cov(x, y) / var(x)
    a <- mean(y) - b * mean(x)
    s <- sd(y - (a + b * x))
    datos <- data.frame(x = x, y = y)
    recta <- data.frame(x = range(x))
    recta$y <- a + b * recta$x
    grafico <- ggplot(datos, aes(x = x, y = y)) +
        geom_ribbon(data = recta, aes(ymin = y - s, ymax = y + s), fill = COLOR_AJUSTE, alpha = 0.15) +
        geom_point(colour = COLOR_DATOS, alpha = 0.5, size = 2) +
        geom_line(data = recta, colour = COLOR_AJUSTE, linewidth = 1) +
        annotate("label", x = Inf, y = Inf, hjust = 1.05, vjust = 1.2, colour = COLOR_AJUSTE, fontface = "bold",
                 label = sprintf("y = %.2fx + %.2f\nr = %.3f   (r² = %.3f)", b, a, r, r^2)) +
        scale_x_continuous(labels = .etiq) +
        scale_y_continuous(labels = .etiq) +
        labs(title = "Correlación de Pearson",
             x = resultado[["nombre1"]], y = resultado[["nombre2"]]) +
        .tema()
    .verbose(paste0("Pearson · Cómo leerlo: cada punto es una observación; la recta es el ajuste lineal y la banda, ",
                    "±1 desviación del error. Respondes SÍ si |r| ≥ ", UMBRAL_R, " y p < 0.05."), verbose)
    ggsave(file.path(carpeta_salida, "pearson.png"), grafico, width = 7, height = 5, dpi = 110, bg = "white")
    .verbose("Gráfico pearson creado correctamente.", verbose)
    grafico
}

#' Gráfico de la tabla de información mutua.
#' @export
graficar_informacion_mutua <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                                        verbose = 1, path = ".", nombre_carpeta = "figures") {
    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE)
    if (length(atributo1) != length(atributo2)) stop("Los atributos deben tener el mismo número de elementos.")
    if (!.es_discreta(atributo1)) stop("El primer atributo debe ser categórico/discreto.")
    if (!.es_discreta(atributo2)) stop("El segundo atributo debe ser categórico/discreto.")
    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de información mutua. Se calculará automáticamente.", verbose)
        resultado <- .calc_corr_catg(atributo1, atributo2, verbose = 0)
    }
    tabla_original <- resultado[["tabla"]]
    tabla <- as.data.frame(as.table(tabla_original), stringsAsFactors = FALSE)
    names(tabla) <- c("atributo1", "atributo2", "frecuencia")
    # Categorías ordenadas por frecuencia total: la más frecuente arriba y a la izquierda.
    tabla$atributo1 <- factor(tabla$atributo1,
                              levels = names(sort(tapply(tabla$frecuencia, tabla$atributo1, sum))))
    tabla$atributo2 <- factor(tabla$atributo2,
                              levels = names(sort(tapply(tabla$frecuencia, tabla$atributo2, sum), decreasing = TRUE)))
    mi <- resultado[["valor"]]
    grafico <- ggplot(tabla, aes(x = atributo2, y = atributo1, fill = frecuencia)) +
        geom_tile(colour = "white", linewidth = 1) +
        geom_text(aes(label = frecuencia, colour = frecuencia > max(frecuencia) / 2),
                  size = 5, show.legend = FALSE) +
        scale_colour_manual(values = c(`TRUE` = "black", `FALSE` = "white")) +
        scale_fill_viridis_c(name = "Frecuencia\nabsoluta") +
        labs(title = "Información mutua",
             subtitle = paste0("MI = ", round(mi, 4)),
             x = resultado[["nombre2"]], y = resultado[["nombre1"]]) +
        .tema() +
        theme(panel.grid = element_blank(),
              axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
    .verbose(paste0("Información mutua · Cómo leerla: cada celda cuenta las observaciones de esa combinación; ",
                    "si las filas se reparten de forma muy distinta, las variables están relacionadas. ",
                    "Respondes SÍ si la prueba G da p < 0.05 (MI mayor que la del azar; en bits no hay un umbral universal)."),
             verbose)
    esperados <- outer(rowSums(tabla_original), colSums(tabla_original)) / sum(tabla_original)
    if (any(esperados < 5)) {
        .verbose("Aviso: hay celdas con menos de 5 casos esperados; la prueba G es orientativa.", verbose)
    }
    ggsave(file.path(carpeta_salida, "MI.png"), grafico, width = 7, height = 5, dpi = 110, bg = "white")
    .verbose("Gráfico de información mutua creado correctamente.", verbose)
    grafico
}

#' Gráfico de Welch ANOVA.
#' @export
graficar_welch <- function(atributo1, atributo2, resultado = NULL, pregunta = NULL,
                            limite_y = NULL, orden_categorias = NULL, verbose = 1,
                            path = ".", nombre_carpeta = "figures") {
    carpeta_salida <- file.path(path, nombre_carpeta)
    if (!dir.exists(carpeta_salida)) dir.create(carpeta_salida, recursive = TRUE)
    if (length(atributo1) != length(atributo2)) stop("Los atributos deben tener el mismo número de elementos.")
    if (!((.es_discreta(atributo1) && .es_continua(atributo2)) ||
          (.es_continua(atributo1) && .es_discreta(atributo2)))) {
        stop("Welch necesita un atributo categórico y otro numérico.")
    }
    if (is.null(resultado)) {
        .verbose("No se ha proporcionado el resultado de Welch ANOVA. Se calculará automáticamente.", verbose)
        resultado <- .calc_corr_catg_num(atributo1, atributo2, verbose = 0)
    }
    datos <- data.frame(grupo = as.character(if (.es_discreta(atributo1)) atributo1 else atributo2),
                        valor = as.numeric(if (.es_continua(atributo1)) atributo1 else atributo2))
    datos <- datos[!is.na(datos$grupo) & datos$grupo != "NA" & !is.na(datos$valor), ]
    medias <- tapply(datos$valor, datos$grupo, mean)
    grupos <- if (is.null(orden_categorias)) names(sort(medias, decreasing = TRUE)) else
        orden_categorias[orden_categorias %in% datos$grupo]
    datos <- datos[datos$grupo %in% grupos, ]
    datos$grupo <- factor(datos$grupo, levels = grupos)
    resumen <- data.frame(grupo = factor(grupos, levels = grupos),
                          n = as.vector(table(datos$grupo)[grupos]),
                          media = as.vector(medias[grupos]))
    media_general <- mean(datos$valor)
    grafico <- ggplot(datos, aes(x = grupo, y = valor)) +
        geom_jitter(width = 0.25, height = 0, colour = COLOR_DATOS, alpha = 0.35, size = 1.2) +
        geom_boxplot(outlier.shape = NA, fill = COLOR_DATOS, alpha = 0.25, width = 0.6) +
        geom_hline(yintercept = media_general, linetype = "dashed", colour = "grey50") +
        geom_point(data = resumen, aes(y = media), shape = 23, size = 4, fill = COLOR_AJUSTE, colour = "white") +
        geom_label(data = resumen, aes(y = Inf, label = paste0("n = ", n, "\nμ = ", .etiq(media))),
                   vjust = 1.1, colour = COLOR_AJUSTE, size = 3.2) +
        scale_y_continuous(labels = .etiq, expand = expansion(mult = c(0.05, 0.2))) +
        labs(title = sprintf("Welch ANOVA · F = %.3f (%d grupos)", resultado[["valor"]], length(grupos)),
             subtitle = paste0("Línea discontinua: media general = ", .etiq(media_general)),
             x = resultado[["nombre_categorico"]], y = resultado[["nombre_numerico"]]) +
        .tema()
    if (!is.null(limite_y)) grafico <- grafico + coord_cartesian(ylim = c(NA, limite_y))
    .verbose(paste0("Welch ANOVA · Cómo leerlo: caja = 50% central del grupo, puntos = observaciones, ",
                    "rombo = media del grupo, línea gris = media general. Respondes SÍ si p < 0.05 ",
                    "(diferencias tan grandes serían raras si los grupos fueran iguales)."), verbose)
    ggsave(file.path(carpeta_salida, "welch.png"), grafico, width = 10, height = 5, dpi = 110, bg = "white")
    .verbose("Gráfico welch creado correctamente.", verbose)
    grafico
}
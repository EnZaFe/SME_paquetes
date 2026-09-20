# Cargar cualquier tipo de dataset y devolverlo como un data.frame.
#
# Este archivo replica en R la utilidad `router()` / `load_dataset()` del
# paquete Python `SME_python` (ver `v_python/src/auxiliar/load_dataset.py`).
#
# Formatos soportados:
#   - Datasets de scikit-learn por nombre: "iris", "wine", "breast_cancer"
#   - CSV  (.csv, .txt)  -> separador autodetectado (, ; tab |)
#   - TSV  (.tsv, .tab)
#   - Excel (.xlsx, .xlsm, .xls, .ods)
#
# Uso rápido:
#   df <- cargar("iris")
#   df <- cargar("notebooks/car data.csv")
#   df <- cargar("datos/ventas.xlsx", hoja = "Enero")

# ---------------------------------------------------------------------------
# Dependencias (se instalan si no están disponibles)
# ---------------------------------------------------------------------------
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}
if (!requireNamespace("readxl", quietly = TRUE)) {
  install.packages("readxl")
}

# ---------------------------------------------------------------------------
# Datasets de scikit-learn por nombre
# ---------------------------------------------------------------------------
# scikit-learn no está disponible en R, así que se cargan datasets pequeños
# equivalentes desde datasets() del paquete base.
_DATASETS_POR_NOMBRE <- list(
  iris = datasets::iris,
  wine = datasets::wine,
  breast_cancer = # aproximación: no hay un dataset idéntico en R; usar un pequeño placeholder
    NULL
)

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
.es_variable <- function(datos) {
  # Una variable es un vector (no un data.frame).
  is.vector(datos) || is.factor(datos)
}

.es_dataset <- function(datos) {
  is.data.frame(datos)
}

.es_continua <- function(columna) {
  # Continua si es numérica o convertible a numérica sin error.
  if (is.numeric(columna)) {
    return(TRUE)
  }
  if (is.character(columna)) {
    valores <- columna[!is.na(columna)]
    if (length(valores) == 0) {
      return(FALSE)
    }
    suppressWarnings(as.numeric(valores))
    !any(is.na(suppressWarnings(as.numeric(valores))))
  } else {
    FALSE
  }
}

# Mensajes informativos (equivalente a _verbose() de Python).
_verbose <- function(mensaje, verbose, nivel = 1, tipo = "info") {
  if (isTRUE(verbose < nivel)) {
    return(invisible(NULL))
  }

  if (identical(tipo, "warning")) {
    warning(mensaje, call. = FALSE)
  } else if (identical(tipo, "error")) {
    stop(mensaje, call. = FALSE)
  } else if (identical(tipo, "success")) {
    message(mensaje)
  } else if (nivel == 1) {
    message(mensaje)
  } else {
    message(mensaje)
  }
}

# ---------------------------------------------------------------------------
# Loaders por nombre
# ---------------------------------------------------------------------------
cargar_dataset <- function(dataset = "iris", verbose = 1) {
  # Carga un dataset de scikit-learn por nombre y lo devuelve como data.frame.
  _verbose(paste0("Cargando dataset '", dataset, "'..."), verbose)

  if (is.null(_DATASETS_POR_NOMBRE[[dataset]])) {
    soportados <- paste(names(_DATASETS_POR_NOMBRE), collapse = ", ")
    stop(sprintf(
      "Dataset '%s' no soportado. Disponibles: %s.",
      dataset, soportados
    ))
  }

  df <- as.data.frame(_DATASETS_POR_NOMBRE[[dataset]])
  _verbose(
    sprintf("Dataset '%s' cargado: %d filas x %d columnas.",
            dataset, nrow(df), ncol(df)),
    verbose
  )
  df
}

# ---------------------------------------------------------------------------
# Loaders de archivos
# ---------------------------------------------------------------------------
cargar_csv <- function(ruta, sep = NULL, encoding = NULL, verbose = 1, ...) {
  # Carga un dataset desde un archivo CSV.
  _verbose(sprintf("Cargando CSV desde '%s'...", ruta), verbose)

  if (!file.exists(ruta)) {
    stop(sprintf("No existe el archivo: '%s'", ruta))
  }

  codificaciones <- if (is.null(encoding)) {
    c("UTF-8", "UTF-8-BOM", "latin1")
  } else {
    encoding
  }

  ultimo_error <- NULL
  for (enc in codificaciones) {
    tryCatch({
      separador <- if (is.null(sep)) {
        # Autodetección del separador mirando el primer bloque.
        withCallingHandlers(
          readLines(ruta, n = 65536, encoding = enc, warn = FALSE),
          warning = function(w) {
            if (grepl("separator", conditionMessage(w), fixed = TRUE)) {
              stop(w, call. = FALSE)
            }
          }
        )
        "," # si no se puede detectar, se asume coma
      } else {
        sep
      }

      df <- readr::read_csv(
        ruta,
        delim = if (sep == "\t") "\t" else sep,
        show_col_types = FALSE,
        col_types = readr::cols(.default = readr::col_guess()),
        ...
      )
      _verbose(
        sprintf("Cargado: %d filas x %d columnas.", nrow(df), ncol(df)),
        verbose
      )
      return(df)
    }, error = function(e) {
      ultimo_error <- e
    })
  }

  stop(ultimo_error)
}

cargar_tsv <- function(ruta, encoding = NULL, verbose = 1, ...) {
  # Carga un dataset desde un archivo TSV (tab-separated values).
  _verbose(sprintf("Cargando TSV desde '%s'...", ruta), verbose)

  if (!file.exists(ruta)) {
    stop(sprintf("No existe el archivo: '%s'", ruta))
  }

  codificaciones <- if (is.null(encoding)) {
    c("UTF-8", "UTF-8-BOM", "latin1")
  } else {
    encoding
  }

  ultimo_error <- NULL
  for (enc in codificaciones) {
    tryCatch({
      df <- readr::read_tsv(
        ruta,
        show_col_types = FALSE,
        col_types = readr::cols(.default = readr::col_guess()),
        ...
      )
      _verbose(
        sprintf("Cargado: %d filas x %d columnas.", nrow(df), ncol(df)),
        verbose
      )
      return(df)
    }, error = function(e) {
      ultimo_error <- e
    })
  }

  stop(ultimo_error)
}

cargar_excel <- function(ruta, hoja = 0, verbose = 1, ...) {
  # Carga un dataset desde un archivo Excel.
  _verbose(sprintf("Cargando Excel desde '%s' (hoja %s)...", ruta, hoja), verbose)

  if (!file.exists(ruta)) {
    stop(sprintf("No existe el archivo: '%s'", ruta))
  }

  tryCatch({
    df <- readxl::read_excel(ruta, sheet = hoja, ...)
    _verbose(
      sprintf("Cargado: %d filas x %d columnas.", nrow(df), ncol(df)),
      verbose
    )
    df
  }, error = function(e) {
    stop(sprintf(
      "%s\nInstala el motor necesario: `install.packages('readxl')`",
      conditionMessage(e)
    ))
  })
}

# ---------------------------------------------------------------------------
# Enrutador
# ---------------------------------------------------------------------------
cargar <- function(dataset = "iris", verbose = 1, ...) {
  # Enrutador: elige el loader según lo que se le pase y devuelve un data.frame.
  #
  #   cargar("iris")                              -> dataset de scikit-learn
  #   cargar("notebooks/car data.csv")            -> CSV (sep autodetectado)
  #   cargar("datos/ventas.xlsx", hoja = "Enero") -> Excel por nombre de hoja
  #   cargar(df)                                  -> devuelve el df tal cual

  if (.es_dataset(dataset)) {
    return(dataset)
  }

  extension <- tolower(tools::file_ext(dataset))

  # 1) Archivo con extensión conocida -> loader por extensión
  loaders <- c(
    csv = cargar_csv,
    tsv = cargar_tsv,
    xlsx = cargar_excel,
    xlsm = cargar_excel,
    xls = cargar_excel,
    ods = cargar_excel
  )

  if (!is.null(extension) && extension %in% names(loaders)) {
    return(loaders[[extension]](dataset, verbose = verbose, ...))
  }

  # 2) Archivo existente con extensión no soportada
  if (file.exists(dataset)) {
    soportadas <- paste(names(loaders), collapse = ", ")
    stop(sprintf(
      "Extensión '%s' no soportada. Soportadas: %s.",
      extension, soportadas
    ))
  }

  # 3) Si no, se interpreta como nombre de dataset
  cargar_dataset(dataset, verbose)
}

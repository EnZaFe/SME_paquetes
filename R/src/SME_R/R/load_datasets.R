#
# Formatos soportados:
#   - Datasets por nombre: "iris", "wine", "breast_cancer"
#   - CSV / TXT  (.csv, .txt)  -> separador autodetectado (, ; tab |)
#   - TSV        (.tsv, .tab) -> separador tab por defecto
#   - Excel      (.xlsx, .xlsm, .xls, .ods)
#   - data.frame -> se devuelve tal cual
#
# Uso rapido:
#   df <- cargar_dataset("iris")
#   df <- cargar_dataset("notebooks/car data.csv")
#   df <- cargar_dataset("datos/ventas.xlsx", hoja = "Enero")

# ---------------------------------------------------------------------------
# Constantes internas (equivalentes a _EXCEL_EXT / _TEXTO_EXT en Python)
# ---------------------------------------------------------------------------
.EXCEL_EXT <- c("xlsx", "xlsm", "xls", "ods")
.TEXTO_EXT <- c("csv", "txt", "tsv", "tab")

# ---------------------------------------------------------------------------
# Loaders de datasets por nombre (equivalente a _SKLEARN_DATASETS)
# ---------------------------------------------------------------------------
# Se definen como funciones (no como valores ya evaluados) para que instalar
# o cargar un paquete solo ocurra si realmente se pide ese dataset.
.cargar_iris <- function() {
  datasets::iris
}

.cargar_wine <- function() {
  # No hay un dataset "wine" en R base; el equivalente habitual es
  # rattle.data::wine (mismos datos que sklearn.datasets.load_wine).
  if (!requireNamespace("rattle.data", quietly = TRUE)) {
    install.packages("rattle.data")
  }
  as.data.frame(rattle.data::wine)
}

.cargar_breast_cancer <- function() {
  # Aproximacion: no existe un equivalente identico a
  # sklearn.datasets.load_breast_cancer en R. Se usa mlbench::BreastCancer,
  # que es el dataset de cancer de mama de referencia en R (columnas y
  # codificacion distintas a la version de sklearn).
  if (!requireNamespace("mlbench", quietly = TRUE)) {
    install.packages("mlbench")
  }
  env <- new.env()
  utils::data("BreastCancer", package = "mlbench", envir = env)
  as.data.frame(env[["BreastCancer"]])
}

.DATASETS_POR_NOMBRE <- list(
  iris = .cargar_iris,
  wine = .cargar_wine,
  breast_cancer = .cargar_breast_cancer
)

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

# Mensajes informativos (equivalente a _verbose() de Python).
.verbose_msg <- function(mensaje, verbose) {
  if (isTRUE(verbose >= 1)) {
    message(mensaje)
  }
  invisible(NULL)
}

# Autodeteccion de separador para texto delimitado (equivalente aproximado
# a csv.Sniffer().sniff(muestra, delimiters=",;\t|") en Python).
.detectar_separador <- function(muestra) {
  candidatos <- c(",", ";", "\t", "|")
  lineas <- strsplit(muestra, "\r\n|\n")[[1]]
  lineas <- lineas[nzchar(lineas)]
  if (length(lineas) == 0) {
    return(",")
  }

  puntuar <- function(sep) {
    conteos <- lengths(regmatches(lineas, gregexpr(sep, lineas, fixed = TRUE)))
    # Buen separador: aparece de forma consistente (misma cantidad) en todas
    # las lineas de muestra, y al menos una vez.
    if (all(conteos == conteos[1]) && conteos[1] > 0) {
      conteos[1]
    } else {
      -1
    }
  }

  puntuaciones <- vapply(candidatos, puntuar, numeric(1))
  if (all(puntuaciones < 0)) {
    return(",")  # si no se detecta nada consistente, se asume coma
  }
  candidatos[[which.max(puntuaciones)]]
}

# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
cargar_dataset <- function(dataset = "iris", sep = NULL, encoding = NULL,
                            hoja = 0, verbose = 1, ...) {
  # Cargar un dataset (por nombre, ruta o data.frame) y devolverlo como
  # data.frame.
  #
  # Parametros
  # ----------
  # dataset  : character | data.frame, por defecto "iris"
  #     - Nombre de un dataset "de juguete" ("iris", "wine", "breast_cancer").
  #     - Ruta a un archivo .csv/.txt, .tsv/.tab o .xlsx/.xlsm/.xls/.ods.
  #     - Un data.frame (se devuelve tal cual).
  # sep      : character opcional. Separador para archivos de texto. Si es
  #     NULL se autodetecta entre `,` `;` tab y `|` (tab por defecto para
  #     .tsv/.tab).
  # encoding : character opcional. Si es NULL se prueba "UTF-8" y luego
  #     "latin1", que es lo habitual en CSV exportados desde Excel/Windows.
  # hoja     : integer | character, por defecto 0 (primera hoja). Solo para
  #     Excel; puede ser el indice (0 = primera) o el nombre de la hoja.
  # verbose  : integer. 0 = nada, 1 = mensajes basicos.
  # ...      : argumentos extra pasados a readr::read_csv/read_tsv o a
  #     readxl::read_excel.
  #
  # Devuelve
  # -------
  # data.frame con el dataset cargado.

  # 0) Ya es un data.frame: se devuelve tal cual.
  if (is.data.frame(dataset)) {
    .verbose_msg("Ya es un data.frame, se devuelve tal cual.", verbose)
    return(dataset)
  }

  ruta <- dataset
  ext <- tolower(tools::file_ext(ruta))

  # 1) Sin extension -> nombre de dataset "de juguete".
  if (!nzchar(ext)) {
    .verbose_msg(sprintf("Cargando dataset '%s'...", ruta), verbose)

    loader <- .DATASETS_POR_NOMBRE[[ruta]]
    if (is.null(loader)) {
      soportados <- paste(names(.DATASETS_POR_NOMBRE), collapse = ", ")
      stop(sprintf(
        "Dataset '%s' no soportado. Disponibles: %s.", ruta, soportados
      ))
    }
    df <- as.data.frame(loader())

  } else {
    # 2) Con extension -> archivo. Tiene que existir.
    if (!file.exists(ruta)) {
      stop(sprintf("No existe el archivo: '%s'", ruta))
    }

    # 2a) Excel (autodetectado por extension).
    if (ext %in% .EXCEL_EXT) {
      if (!requireNamespace("readxl", quietly = TRUE)) {
        install.packages("readxl")
      }
      .verbose_msg(
        sprintf("Cargando Excel desde '%s' (hoja %s)...", ruta, hoja), verbose
      )
      hoja_arg <- if (identical(hoja, 0)) 1L else hoja  # readxl usa 1-indexado
      df <- tryCatch(
        as.data.frame(readxl::read_excel(ruta, sheet = hoja_arg, ...)),
        error = function(e) {
          stop(sprintf(
            "%s\nInstala el motor necesario: `install.packages('readxl')`",
            conditionMessage(e)
          ))
        }
      )

    # 2b) Texto delimitado (CSV/TXT/TSV/TAB).
    } else if (ext %in% .TEXTO_EXT) {
      if (!requireNamespace("readr", quietly = TRUE)) {
        install.packages("readr")
      }
      .verbose_msg(sprintf("Cargando texto delimitado desde '%s'...", ruta), verbose)

      separador <- sep
      if (is.null(separador) && ext %in% c("tsv", "tab")) {
        separador <- "\t"
      }

      codificaciones <- if (is.null(encoding)) c("UTF-8", "latin1") else encoding

      df <- NULL
      ultimo_error <- NULL
      for (i in seq_along(codificaciones)) {
        enc <- codificaciones[[i]]
        resultado <- tryCatch({
          sep_usado <- separador
          if (is.null(sep_usado)) {
            muestra <- tryCatch(
              paste(readLines(ruta, n = 200, encoding = enc, warn = FALSE),
                    collapse = "\n"),
              error = function(e) stop(e)
            )
            sep_usado <- .detectar_separador(muestra)
            .verbose_msg(sprintf("Separador detectado: '%s'", sep_usado), verbose)
          }
          readr::read_delim(
            ruta,
            delim = sep_usado,
            locale = readr::locale(encoding = enc),
            show_col_types = FALSE,
            ...
          )
        }, error = function(e) e)

        if (!inherits(resultado, "error")) {
          df <- resultado
          break
        }
        ultimo_error <- resultado
        .verbose_msg(
          sprintf("Codificacion '%s' no valida, probando otra...", enc), verbose
        )
      }

      if (is.null(df)) {
        stop(ultimo_error)
      }
      df <- as.data.frame(df)

    # 2c) Extension desconocida.
    } else {
      soportadas <- paste(c(.TEXTO_EXT, .EXCEL_EXT), collapse = ", ")
      stop(sprintf(
        "Extension '%s' no soportada. Soportadas: %s.", ext, soportadas
      ))
    }
  }

  .verbose_msg(
    sprintf("Cargado: %d filas x %d columnas.", nrow(df), ncol(df)), verbose
  )
  df
}
.verbose <- function(mensaje, verbose, nivel = 1, tipo = "info") {

    if (verbose < nivel) {
        return(invisible(NULL))
    }

    if (tipo == "warning") {
        warning(mensaje, call. = FALSE)

    } else if (tipo == "error") {
        stop(mensaje, call. = FALSE)

    } else if (tipo == "success") {
        message(mensaje)

    } else if (nivel == 1) {
        message(mensaje)

    } else if (nivel == 2) {
        message(mensaje)

    } else if (nivel == 3) {
        message(mensaje)

    } else {
        message(mensaje)
    }
}
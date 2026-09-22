#Código auxiliar con funciones que se usan en varias funcionalidades.



.es_variable <- function(datos) {
    # Devuelve TRUE si `datos` es una variable numérica (vector numérico).

    inherits(datos, "numeric")
}


.es_dataset <- function(datos) {
    # Devuelve TRUE si `datos` es un dataset (data.frame).

    inherits(datos, "data.frame")
}


.es_continua <- function(columna) {
    # Devuelve TRUE si una columna contiene valores numéricos.

    # Si ya es numérica, incluyendo integer y double.
    if (is.numeric(columna)) {
        return(TRUE)
    }
    
    # Si es carácter, intentamos convertir sus valores a números.
    if (inherits(columna, "character")) {

        valores <- columna[!is.na(columna)]

        if (length(valores) == 0) {
            return(FALSE)
        }

        numeros <- suppressWarnings(as.numeric(valores))

        if (!any(is.na(numeros))) {
            return(TRUE)
        }

    }

    return(FALSE)
}


.es_discreta <- function(columna) {
    # Devuelve TRUE si una columna contiene valores discretos/categóricos.

    # Booleanos.
    if (is.logical(columna)) {
        return(TRUE)
    }

    # Categorical de pandas.
    if (inherits(columna, "factor")) {
        return(TRUE)
    }

    # Si ya sabemos que es continua, no puede ser discreta.
    if (.es_continua(columna)) {
        return(FALSE)
    }

    # Para object tenemos que mirar los valores reales.
    if (inherits(columna, "character")) {

        valores <- columna[!is.na(columna)]

        if (length(valores) == 0) {
            return(FALSE)
        }

        for (valor in valores) {

            # Estos tipos no son valores categóricos simples.
            if (is.list(valor) || is.null(valor)) {
                return(FALSE)
            }

        }

        # Si son valores simples, los consideramos categóricos.
        return(TRUE)

    }

    return(FALSE)
}

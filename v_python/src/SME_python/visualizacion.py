import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta, _es_continua
from SME_python.metricas_de_variables import _calcular_AUC
from SME_python.correlacion_info import _calc_corr_num, _calc_corr_catg, _calc_corr_catg_num
from auxiliar.verbose import _verbose


COLOR_DATO = "#2c6e91"         # azul 
COLOR_ESTIMADO = "#e8734d"     # naranja  
COLOR_REFERENCIA = "#9a9a9a"   # gris 
COLOR_CLASE_2 = "#7a3b8c"      # morado  

COLOR_TEXTO = "#333333"
COLOR_TEXTO_SUAVE = "#555555"

# Paleta de mapa de calor para magnitudes continuas (frecuencias)
PALETA = "viridis"

FUENTE_TITULO = dict(
    fontsize=17,
    fontweight="bold",
)



def _bonito(texto, cap=True):
    """
    Convierte un nombre técnico en un texto legible:
    'horas_de_estudio' -> 'Horas de estudio'.

    Cambia los "_" (y espacios repetidos) por un único espacio.
    Con ``cap=False`` no fuerza la mayúscula inicial (útil cuando
    el nombre va dentro de una frase).
    """
    if texto is None:
        return ""
    s = re.sub(r"[_\s]+", " ", str(texto)).strip()
    if cap and s:
        s = s[0].upper() + s[1:]
    return s


def _nombre(serie, defecto):
    """Nombre legible de una Series (o `defecto` si no tiene nombre)."""
    nombre = getattr(serie, "name", None)
    return _bonito(nombre, cap=False) if nombre is not None else defecto


def _aplicar_tema_base():
    """
    02 COMPLECIÓN: se retira todo el marco visual que no aporta
    lectura de dato (rejilla de fondo, bordes de más). El detalle
    mínimo necesario se añade gráfico a gráfico con `_despejar_ejes`.
    """
    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.6,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        },
    )


def _despejar_ejes(ax, grid_eje="y"):
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(axis=grid_eje, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _titulo_pregunta(fig, pregunta):
    """
    El título es siempre la pregunta a la que responde el gráfico.

    Si es larga se parte en varias líneas para que no se salga de la
    figura. Devuelve la fracción de altura (0-1) donde debe terminar
    el contenido por arriba, para que nada se solape con el título.
    """

    ancho, alto = fig.get_size_inches()
    lineas = textwrap.wrap(pregunta, width=max(30, int(ancho * 6))) or [pregunta]
    fig.suptitle(
        "\n".join(lineas),
        fontsize=FUENTE_TITULO["fontsize"],
        fontweight=FUENTE_TITULO["fontweight"],
        y=0.98,
    )
    alto_titulo_in = len(lineas) * FUENTE_TITULO["fontsize"] * 1.4 / 72
    return 0.98 - (alto_titulo_in + 0.15) / alto


def _resumen_respuesta(fig, tope, dato, veredicto, porque):
    """
    Escribe, justo debajo del título-pregunta, la respuesta en dos líneas:

        <dato> -> SÍ / NO
        Porque: <motivo>

    Devuelve la nueva fracción de altura (0-1) donde debe terminar el
    contenido por arriba, para que nada se solape con este texto.
    """
    ancho, alto = fig.get_size_inches()
    marca = "no evaluable" if veredicto is None else ("SÍ" if bool(veredicto) else "NO")

    fig.text(
        0.5, tope, f"{dato}  →  {marca}",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#111111",
    )

    alto_l1 = 13 * 1.35 / 72 + 0.05
    lineas = textwrap.wrap(f"Porque: {porque}", width=max(40, int(ancho * 9))) or [""]
    fig.text(
        0.5, tope - alto_l1 / alto, "\n".join(lineas),
        ha="center", va="top", fontsize=11, color=COLOR_TEXTO_SUAVE, linespacing=1.25,
    )
    alto_l2 = len(lineas) * 11 * 1.3 / 72

    return tope - (alto_l1 + alto_l2 + 0.20) / alto


def _fmt_p(p):
    """Formato legible para un valor p."""
    if p is None or not np.isfinite(p):
        return "n/d"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

## AUC

def graficar_auc(
    atributo_clm,
    clases,
    resultado=None,
    pregunta=None,
    umbral_auc=0.70,
    verbose=1,
    path=".",
    nombre_carpeta="figures"
):
    """
    Curva ROC + distribución del atributo numérico por clase.

    Se puede llamar de dos formas:

        graficar_auc(atributo_clm, clases)
            -> calcula el AUC automáticamente

        graficar_auc(atributo_clm, clases, resultado=info)
            -> reutiliza un AUC ya calculado (evita recalcularlo)

    La clase "positiva" es la de mayor valor (p. ej. 1 frente a 0,
    "Sí" frente a "No"), de modo que un score alto indica positivo.

    Parameters
    ----------
    atributo_clm : pandas.Series
        Atributo numérico (score/probabilidad del modelo).

    clases : list o pandas.Series
        Etiquetas de clase, exactamente dos. Deben tener el mismo
        número de elementos que ``atributo_clm``.

    resultado : dict, opcional
        Diccionario con el AUC ya calculado (por ejemplo devuelto por
        ``_calcular_AUC``). Si se pasa, no se recalcula.

    pregunta : str, opcional
        Pregunta que responde el gráfico. Si no se indica, se genera
        una automáticamente a partir del nombre del atributo.

    umbral_auc : float
        AUC mínimo para responder SÍ (por defecto 0.70, el límite
        habitual de "discriminación aceptable").

    verbose : int
        Nivel de información mostrado durante la ejecución.

    path : str
        Carpeta base donde se guarda la imagen.

    nombre_carpeta : str
        Nombre de la subcarpeta (por defecto ``figures``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura con la curva ROC y la distribución por clase, guardada
        como ``demo_auc.png`` en la carpeta de salida.
    """

    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if len(atributo_clm) != len(clases):
        raise ValueError(
            "El atributo y las clases deben tener el mismo número "
            "de elementos."
        )

    if len(atributo_clm) == 0:
        raise ValueError(
            "No se puede calcular el AUC con datos vacíos."
        )

    if not _es_continua(atributo_clm):
        raise TypeError(
            "El atributo debe ser numérico para calcular el AUC."
        )

    valores_clases = set(clases)

    if len(valores_clases) != 2:
        raise ValueError(
            "Las clases deben contener exactamente dos clases."
        )

    nombre_original = getattr(atributo_clm, "name", None)
    atributo_clm = pd.Series(atributo_clm).reset_index(drop=True)
    atributo_clm.name = nombre_original
    clases = pd.Series(clases).reset_index(drop=True)

    if resultado is not None and "valor" in resultado:
        auc = resultado["valor"]
    else:
        _verbose(
            "No se ha proporcionado el resultado de AUC. "
            "Se calculará automáticamente.",
            verbose,
            nivel=2,
        )
        auc = _calcular_AUC(atributo_clm, clases, verbose=0)


    datos = pd.DataFrame({"clase": clases, "atributo": atributo_clm})
    datos = datos.sort_values("atributo", ascending=False, kind="mergesort")

    # Orden determinista: la clase "positiva" es la de mayor valor.
    try:
        ordenadas = sorted(valores_clases)
    except TypeError:
        ordenadas = sorted(valores_clases, key=str)
    clase_neg, clase_pos = ordenadas[0], ordenadas[1]

    positivos = int((datos["clase"] == clase_pos).sum())
    negativos = int((datos["clase"] == clase_neg).sum())

    if positivos == 0 or negativos == 0:
        raise ValueError(
            "Cada clase debe tener al menos una observación."
        )


    #La curva

    tpr = [0]
    fpr = [0]
    vp = 0
    fp = 0

    for clase in datos["clase"]:
        if clase == clase_pos:
            vp += 1
        else:
            fp += 1
        tpr.append(vp / positivos)
        fpr.append(fp / negativos)

    tpr.append(1)
    fpr.append(1)

    # título

    _aplicar_tema_base()

    nombre_attr = _nombre(atributo_clm, "el atributo")
    pos = _bonito(clase_pos, cap=False)
    neg = _bonito(clase_neg, cap=False)

    if pregunta is None:
        pregunta = f"¿Qué tan bien distingue {nombre_attr} entre las dos clases?"

    fig, ejes = plt.subplots(1, 2, figsize=(14, 7.4))


    ax = ejes[0]


    ax.plot(fpr, tpr, color=COLOR_DATO, lw=3)
    ax.fill_between(fpr, tpr, color=COLOR_DATO, alpha=0.15)


    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCIA, lw=1.5)

    
    ax.text(0.50, 0.47, "azar", color=COLOR_REFERENCIA,
            fontsize=10.5, rotation=45, rotation_mode="anchor",
            ha="center", va="top", style="italic")


    ax.text(0.03, 0.97, "curva perfecta", color=COLOR_REFERENCIA,
            fontsize=10.5, ha="left", va="top", style="italic")


    ax.text(
        0.96, 0.06, f"AUC = {auc:.3f}\n(área sombreada)",
        transform=ax.transAxes, ha="right", va="bottom",
        color=COLOR_DATO, fontsize=13, fontweight="bold",
    )

    ax.set_xlabel(f"Falsos positivos\n(% de «{neg}» marcados como «{pos}»)")
    ax.set_ylabel(f"Verdaderos positivos\n(% de «{pos}» detectados)")
    ax.set_title("Curva ROC", fontsize=13, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    _despejar_ejes(ax, grid_eje="both")



    ax2 = ejes[1]

    # Dos clases = dos identidades categóricas
    paleta_clases = {clase_pos: COLOR_DATO, clase_neg: COLOR_CLASE_2}
    orden_hue = [clase_neg, clase_pos]

    sns.kdeplot(
        data=datos, x="atributo", hue="clase", hue_order=orden_hue,
        fill=True, common_norm=False, alpha=0.35, palette=paleta_clases,
        linewidth=1.8, ax=ax2, legend=False,
    )
    sns.rugplot(
        data=datos, x="atributo", hue="clase", hue_order=orden_hue,
        palette=paleta_clases, ax=ax2, legend=False, alpha=0.5,
    )

    # Margen extra arriba
    x_min, x_max = ax2.get_xlim()
    rango_x = x_max - x_min
    ax2.set_xlim(x_min - 0.08 * rango_x, x_max + 0.08 * rango_x)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.22)
    y_texto = ax2.get_ylim()[1] * 0.97
    separacion = 0.012 * (ax2.get_xlim()[1] - ax2.get_xlim()[0])

    medianas = {
        c: datos.loc[datos["clase"] == c, "atributo"].median()
        for c in (clase_neg, clase_pos)
    }
    ordenadas_med = sorted(medianas, key=lambda c: medianas[c])
    for posicion, clase_valor in enumerate(ordenadas_med):
        mediana = medianas[clase_valor]
        color = paleta_clases[clase_valor]
        a_la_izquierda = posicion == 0
        ax2.axvline(mediana, ymax=0.90, color=color, lw=1.2, linestyle=":")
        ax2.text(
            mediana + (-separacion if a_la_izquierda else separacion),
            y_texto,
            f"clase «{_bonito(clase_valor, cap=False)}»\nmediana {mediana:.2f}",
            color=color, fontsize=11, fontweight="bold",
            ha="right" if a_la_izquierda else "left", va="top",
        )

    ax2.set_xlabel(_bonito(nombre_attr))
    ax2.set_ylabel("Densidad (estimada)")
    ax2.set_title("Distribución por clase", fontsize=13, color="#444444")
    ax2.set_box_aspect(1)
    _despejar_ejes(ax2, grid_eje="y")



    cumple = auc >= umbral_auc
    _verbose(
        f"ROC · Cómo leerla: al marcar como «{pos}» las observaciones de mayor a menor "
        f"{nombre_attr}, la curva sube con cada acierto y avanza con cada error. Cuanto más "
        f"cerca de la esquina superior izquierda, mejor; la diagonal es el azar. "
        f"Respondes SÍ si AUC ≥ {umbral_auc:.2f} (0.5 azar · 0.8 bueno · 0.9 excelente).",
        verbose, nivel=1,
    )
    tope = _titulo_pregunta(fig, pregunta)
    tope = _resumen_respuesta(    # Respuesta
        fig, tope, f"AUC = {auc:.3f} (mín. {umbral_auc:.2f})", cumple,
        f"la «{pos}» puntúa más alto que la «{neg}» en el {auc * 100:.0f}% de los pares.",
    )
    fig.tight_layout(rect=[0, 0, 1, tope])
    fig.savefig(
        carpeta_salida / "demo_auc.png",
        dpi=110,
        bbox_inches="tight",
    )

    _verbose(
        "Grafico de AUC y distribucion creados correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )
    return fig


# Pearson

def graficar_pearson(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
    umbral_r=0.5,
    alfa=0.05,
    verbose=1,
    path=".",
    nombre_carpeta="figures"
):
    """
    Dispersión + regresión lineal + distribuciones marginales entre
    dos variables numéricas, junto con el coeficiente de Pearson.

        graficar_pearson(x, y)
            -> calcula todo internamente

        graficar_pearson(x, y, resultado=info)
            -> reutiliza un resultado ya calculado con
               ``_calc_corr_num`` / ``calcular_correlacion``

    Parameters
    ----------
    atributo1 : pandas.Series
        Primera variable numérica.

    atributo2 : pandas.Series
        Segunda variable numérica. Debe tener el mismo número de
        elementos que ``atributo1``.

    resultado : dict, opcional
        Diccionario con el resultado de Pearson ya calculado (por
        ejemplo devuelto por ``_calc_corr_num``). Si se pasa, no se
        recalcula.

    pregunta : str, opcional
        Pregunta que responde el gráfico. Si no se indica, se genera
        una automáticamente a partir de los nombres de las variables.

    umbral_r : float
        |r| mínimo para responder SÍ (por defecto 0.5: relación al
        menos moderada).

    alfa : float
        Nivel de significación: además del tamaño de r, se exige
        p < alfa para descartar que sea casualidad (por defecto 0.05).

    verbose : int
        Nivel de información mostrado durante la ejecución.

    path : str
        Carpeta base donde se guarda la imagen.

    nombre_carpeta : str
        Nombre de la subcarpeta (por defecto ``figures``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura del jointplot (dispersión + marginales), guardada como
        ``pearson.png`` en la carpeta de salida.
    """

    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if len(atributo1) != len(atributo2):
        raise ValueError(
            "Los atributos deben tener el mismo número de elementos."
        )

    if not _es_continua(atributo1):
        raise TypeError("El primer atributo debe ser numérico.")

    if not _es_continua(atributo2):
        raise TypeError("El segundo atributo debe ser numérico.")

    if resultado is None:
        _verbose(
            "No se ha proporcionado el resultado de Pearson. "
            "Se calculará automáticamente.",
            verbose,
            nivel=2,
        )
        resultado = _calc_corr_num(atributo1, atributo2, verbose=0)

    x = np.asarray(resultado["x"])
    y = np.asarray(resultado["y"])
    r = resultado["valor"]
    r2 = resultado["r2"]
    nombre1 = _bonito(resultado["nombre1"], cap=False)
    nombre2 = _bonito(resultado["nombre2"], cap=False)

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Existe una relación lineal entre {nombre1} y {nombre2}?"

    # Recta de regresión
    pendiente, intercepto = np.polyfit(x, y, 1)
    orden = np.argsort(x)
    x_ordenado = x[orden]
    y_linea = pendiente * x_ordenado + intercepto
    error_estandar = np.std(y - (pendiente * x + intercepto))

    try:
        p_valor = stats.pearsonr(x, y)[1]
    except Exception:
        p_valor = np.nan

    grafico = sns.jointplot(
        x=x, y=y, kind="scatter", height=8, color=COLOR_DATO,
        marginal_kws=dict(fill=True, color=COLOR_DATO),
    )
    ax = grafico.ax_joint


    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.20 * (y_max - y_min))


    ax.fill_between(
        x_ordenado, y_linea - error_estandar, y_linea + error_estandar,
        color=COLOR_ESTIMADO, alpha=0.15, zorder=1,
    )
    ax.plot(x_ordenado, y_linea, color=COLOR_ESTIMADO, lw=2.5, zorder=2)

    sube = pendiente >= 0
    signo = "+" if intercepto >= 0 else "−"
    ax.text(
        0.03 if sube else 0.97, 0.97,
        f"y = {pendiente:.2f}x {signo} {abs(intercepto):.2f}\n"
        f"r = {r:.3f}   (r² = {r2:.3f})",
        transform=ax.transAxes, ha="left" if sube else "right", va="top",
        color=COLOR_ESTIMADO, fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLOR_ESTIMADO,
                  lw=0.9, alpha=0.95),
        zorder=6,
    )

    grafico.set_axis_labels(_bonito(nombre1), _bonito(nombre2))
    _despejar_ejes(ax, grid_eje="both")

    # Respuesta corta bajo el título

    fuerza_r = abs(r)
    if fuerza_r < 0.3:
        fuerza = "débil"
    elif fuerza_r < 0.5:
        fuerza = "moderada"
    elif fuerza_r < 0.7:
        fuerza = "fuerte"
    else:
        fuerza = "muy fuerte"
    sentido = "positiva" if r >= 0 else "negativa"

    p_ok = (p_valor < alfa) if np.isfinite(p_valor) else True
    cumple = (abs(r) >= umbral_r) and p_ok

    _verbose(
        f"Pearson · Cómo leerlo: cada punto es una observación; la recta es el ajuste lineal "
        f"y la banda, ±1 desviación del error. Respondes SÍ si |r| ≥ {umbral_r:.1f} y "
        f"p < {alfa} (|r|: 0.3 débil · 0.5 moderada · 0.7 fuerte).",
        verbose, nivel=1,
    )

    if cumple:
        porque = (f"relación {fuerza} y {sentido}; la recta explica el "
                  f"{r2 * 100:.0f}% de la variación de {nombre2}.")
    elif abs(r) >= umbral_r:
        porque = (f"r es alto, pero p {_fmt_p(p_valor)} (máx. {alfa}): "
                  f"podría ser casualidad.")
    else:
        porque = (f"relación {fuerza} ({sentido}); la recta explica solo el "
                  f"{r2 * 100:.0f}% de la variación de {nombre2}.")

    tope = _titulo_pregunta(grafico.figure, pregunta)
    tope = _resumen_respuesta(
        grafico.figure, tope, f"r = {r:.3f} (mín. {umbral_r:.1f})", cumple, porque,
    )
    grafico.figure.subplots_adjust(top=tope, bottom=0.08)

    grafico.figure.savefig(
        carpeta_salida / "pearson.png",
        dpi=110,
        bbox_inches="tight",
    )

    plt.close(grafico.figure)
    _verbose(
        "Grafico pearson creado correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )

    return grafico.figure


# Información mutua

def _info_mutua_desde_tabla(tabla, alfa=0.05):
    """
    Calcula, a partir de la tabla de contingencia:
      - la información mutua en bits,
      - el % de incertidumbre de la 2ª variable que reduce conocer la 1ª,
      - el valor p de la prueba G (contrasta si la información mutua
        es mayor de lo que daría el azar entre variables independientes),
      - si alguna celda tiene menos de 5 casos esperados (prueba poco fiable).
    """
    t = np.asarray(tabla, dtype=float)
    n = t.sum()
    p_xy = t / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        termino = np.where(p_xy > 0, p_xy * np.log2(p_xy / (p_x * p_y)), 0.0)
    mi = float(termino.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        h_y = float(-np.nansum(np.where(p_y > 0, p_y * np.log2(p_y), 0.0)))
    reduccion = mi / h_y if h_y > 0 else 0.0

    gl = (t.shape[0] - 1) * (t.shape[1] - 1)
    g = 2 * n * np.log(2) * mi           # estadístico G (razón de verosimilitud)
    p_valor = float(stats.chi2.sf(g, gl)) if gl > 0 else 1.0

    esperadas = p_x * p_y * n
    pocas_muestras = bool((esperadas < 5).any())

    return mi, reduccion, g, gl, p_valor, pocas_muestras


def graficar_informacion_mutua(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
    alfa=0.05,
    verbose=1,
    path=".",
    nombre_carpeta="figures"
):
    """
    Heatmap de la tabla de contingencia entre dos variables categóricas,
    con frecuencias absolutas y relativas, y la información mutua entre
    ambas.

        graficar_informacion_mutua(a1, a2)
            -> calcula todo internamente

        graficar_informacion_mutua(a1, a2, resultado=info)
            -> reutiliza un resultado ya calculado con ``_calc_corr_catg``

    Parameters
    ----------
    atributo1 : pandas.Series
        Primera variable categórica/discreta.

    atributo2 : pandas.Series
        Segunda variable categórica/discreta. Debe tener el mismo
        número de elementos que ``atributo1``.

    resultado : dict, opcional
        Diccionario con la información mutua ya calculada (por ejemplo
        devuelto por ``_calc_corr_catg``). Si se pasa, no se recalcula.

    pregunta : str, opcional
        Pregunta que responde el gráfico. Si no se indica, se genera
        una automáticamente a partir de los nombres de las variables.

    alfa : float
        Nivel de significación de la prueba G (por defecto 0.05).
        Se responde SÍ si p < alfa.

    verbose : int
        Nivel de información mostrado durante la ejecución.

    path : str
        Carpeta base donde se guarda la imagen.

    nombre_carpeta : str
        Nombre de la subcarpeta (por defecto ``figures``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura con los dos heatmaps y el valor de MI, guardada como
        ``MI.png`` en la carpeta de salida.
    """

    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if len(atributo1) != len(atributo2):
        raise ValueError(
            "Los atributos deben tener el mismo número de elementos."
        )

    if not _es_discreta(atributo1):
        raise TypeError("El primer atributo debe ser categórico/discreto.")

    if not _es_discreta(atributo2):
        raise TypeError("El segundo atributo debe ser categórico/discreto.")

    if resultado is None:
        _verbose(
            "No se ha proporcionado el resultado de información mutua. "
            "Se calculará automáticamente.",
            verbose,
            nivel=2,
        )
        resultado = _calc_corr_catg(atributo1, atributo2, verbose=0)

    tabla = resultado["tabla"]
    mi = resultado["valor"]
    nombre1 = _bonito(resultado["nombre1"], cap=False)
    nombre2 = _bonito(resultado["nombre2"], cap=False)

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Están relacionadas las categorías de {nombre1} y {nombre2}?"

    orden_filas = tabla.sum(axis=1).sort_values(ascending=False).index
    orden_columnas = tabla.sum(axis=0).sort_values(ascending=False).index
    tabla = tabla.loc[orden_filas, orden_columnas].copy()

    # Nombres de categorías legibles (sin "_").
    tabla.index = [_bonito(v) for v in tabla.index]
    tabla.columns = [_bonito(v) for v in tabla.columns]

    tabla_pct = tabla / tabla.values.sum() * 100

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6.6))

    sns.heatmap(
        tabla, annot=True, fmt="d", cmap=PALETA,
        cbar_kws={"label": "Frecuencia absoluta (dato)"}, ax=ejes[0],
        linewidths=0.5, linecolor="white",
    )
    ejes[0].set_title("Frecuencias absolutas", fontsize=13, color="#444444")
    ejes[0].set_xlabel(_bonito(nombre2))
    ejes[0].set_ylabel(_bonito(nombre1))

    sns.heatmap(
        tabla_pct, annot=True, fmt=".1f", cmap=PALETA,
        cbar_kws={"label": "% del total (derivado)"}, ax=ejes[1],
        linewidths=0.5, linecolor="white",
    )
    ejes[1].set_title("Frecuencias relativas (%)", fontsize=13, color="#444444")
    ejes[1].set_xlabel(_bonito(nombre2))
    ejes[1].set_ylabel("")

    for ax in ejes:
        ax.tick_params(length=0)
        ax.tick_params(axis="y", labelrotation=0)


    mi_t, reduccion, g, gl, p_valor, pocas = _info_mutua_desde_tabla(tabla.values)  # Estadística para el criterio
    cumple = p_valor < alfa

    _verbose(
        "Información mutua · Cómo leerla: cada celda cuenta las observaciones de esa "
        "combinación; si las filas se reparten de forma muy distinta, las variables están "
        f"relacionadas. Respondes SÍ si la prueba G da p < {alfa} (MI mayor que la del azar; "
        "en bits no hay un umbral universal).",
        verbose, nivel=1,
    )
    if pocas:
        _verbose(
            "Aviso: hay celdas con menos de 5 casos esperados; la prueba G es orientativa.",
            verbose, nivel=1,
        )

    if cumple:
        porque = (f"conocer «{nombre1}» reduce un {reduccion * 100:.1f}% la incertidumbre "
                  f"sobre «{nombre2}» y no se explica por azar.")
    else:
        porque = (f"conocer «{nombre1}» solo reduce un {reduccion * 100:.1f}% la incertidumbre "
                  f"sobre «{nombre2}», y eso es compatible con el azar.")

    tope = _titulo_pregunta(fig, pregunta)
    tope = _resumen_respuesta(
        fig, tope,
        f"Información mutua = {mi:.4f} bits, p {_fmt_p(p_valor)} (máx. {alfa})",
        cumple, porque,
    )
    fig.tight_layout(rect=[0, 0, 1, tope])
    fig.savefig(
        carpeta_salida / "MI.png",
        dpi=110,
        bbox_inches="tight",
    )
    _verbose(
        "Grafico de información mutua creado correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )

    return fig


    # Welch ANOVA


def _welch_anova(valores_grupos):
    """
    Welch ANOVA a partir de los valores de cada grupo.

    Devuelve (F, gl1, gl2, p) o None si no es calculable (algún grupo
    con menos de 2 observaciones o sin variabilidad).
    """
    muestras = [np.asarray(v, dtype=float) for v in valores_grupos]
    k = len(muestras)
    if k < 2:
        return None

    n = np.array([len(m) for m in muestras], dtype=float)
    medias = np.array([m.mean() for m in muestras])
    varianzas = np.array([m.var(ddof=1) if len(m) > 1 else np.nan for m in muestras])

    if (n < 2).any() or not np.all(np.isfinite(varianzas)) or (varianzas <= 0).any():
        return None

    w = n / varianzas
    w_total = w.sum()
    media_w = (w * medias).sum() / w_total
    numerador = (w * (medias - media_w) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w_total) ** 2 / (n - 1)).sum()
    denominador = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f = numerador / denominador
    gl1 = k - 1
    gl2 = (k ** 2 - 1) / (3 * tmp)
    return float(f), gl1, float(gl2), float(stats.f.sf(f, gl1, gl2))

def graficar_welch(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
    orden_categorias=None,
    alfa=0.05,
    verbose=1,
    path=".",
    nombre_carpeta="figures",
    limite_y=None
):
    """
    Boxplot + puntos individuales por grupo, con la media de cada
    grupo, la media general y el estadístico F de Welch ANOVA.

        graficar_welch(grupo, valor)
            -> calcula todo internamente

        graficar_welch(grupo, valor, orden_categorias=[...])
            -> usa una secuencia real para conectar las medias

    Parameters
    ----------
    atributo1 : pandas.Series
        Variable categórica/discreta que define los grupos.

    atributo2 : pandas.Series
        Variable numérica medida dentro de cada grupo. Debe tener el
        mismo número de elementos que ``atributo1``.

    resultado : dict, opcional
        Diccionario con el resultado de Welch ANOVA ya calculado (por
        ejemplo devuelto por ``_calc_corr_catg_num``). Si se pasa, no
        se recalcula.

    pregunta : str, opcional
        Pregunta que responde el gráfico. Si no se indica, se genera
        una automáticamente a partir de los nombres de las variables.

    orden_categorias : list, opcional
        Úsalo SOLO si las categorías comparten una secuencia real
        (años, trimestres, fases de un proceso...). En ese caso las
        medias de los grupos se conectan con una línea, porque la
        conexión afirma correctamente una relación de orden que de
        verdad existe (05 CONECTIVIDAD).
        Si no se indica, los grupos se ordenan de mayor a menor media
        (06 CONTINUIDAD) y NO se conectan entre sí, porque no hay
        ninguna relación de orden real que la línea pueda afirmar sin
        engañar.

    alfa : float
        Nivel de significación (por defecto 0.05). Se responde SÍ si
        p < alfa.

    verbose : int
        Nivel de información mostrado durante la ejecución.

    path : str
        Carpeta base donde se guarda la imagen.

    nombre_carpeta : str
        Nombre de la subcarpeta (por defecto ``figures``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura con el boxplot, las medias y el estadístico F, guardada
        como ``welch.png`` en la carpeta de salida.
    """

    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if len(atributo1) != len(atributo2):
        raise ValueError(
            "Los atributos deben tener el mismo número de elementos."
        )

    if not (
        (_es_discreta(atributo1) and _es_continua(atributo2))
        or (_es_continua(atributo1) and _es_discreta(atributo2))
    ):
        raise TypeError("Welch necesita un atributo categórico y otro numérico.")

    if resultado is None:
        _verbose(
            "No se ha proporcionado el resultado de Welch ANOVA. "
            "Se calculará automáticamente.",
            verbose,
            nivel=2,
        )
        resultado = _calc_corr_catg_num(atributo1, atributo2, verbose=0)

    grupos = resultado["grupos"]
    valores_grupos = resultado["valores_grupos"]
    medias = resultado["medias"]
    n_grupos = resultado["n_grupos"]
    media_general = resultado["media_general"]
    f_welch = resultado["valor"]
    nombre_categorico = _bonito(resultado["nombre_categorico"], cap=False)
    nombre_numerico = _bonito(resultado["nombre_numerico"], cap=False)

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Difiere {nombre_numerico} según {nombre_categorico}?"

    es_secuencia_real = orden_categorias is not None

    if es_secuencia_real:
        grupos_ordenados = [g for g in orden_categorias if g in grupos]
    else:
        grupos_ordenados = sorted(grupos, key=lambda g: medias[g], reverse=True)

    etiquetas = [_bonito(g) for g in grupos_ordenados]

    datos = pd.concat([
        pd.DataFrame({"grupo": _bonito(g), "valor": valores_grupos[g]})
        for g in grupos_ordenados
    ], ignore_index=True)
    datos["grupo"] = pd.Categorical(datos["grupo"], categories=etiquetas, ordered=True)

    n_g = len(grupos_ordenados)
    fig, ax = plt.subplots(figsize=(max(11, 2.6 * n_g + 4), 7.4))


    sns.boxplot(
        data=datos, x="grupo", y="valor", color=COLOR_DATO,
        showfliers=False, width=0.5, ax=ax, boxprops=dict(alpha=0.30),
    )
    sns.stripplot(
        data=datos, x="grupo", y="valor", color=COLOR_DATO,
        alpha=0.45, size=4.5, jitter=0.2, ax=ax,
    )

    ys_medias = [medias[g] for g in grupos_ordenados]

    if es_secuencia_real:

        ax.plot(
            range(n_g), ys_medias,
            color=COLOR_ESTIMADO, lw=1.8, zorder=4, alpha=0.85,
        )
    # Si no hay secuencia real, NO se dibuja línea entre las
    # medias: conectarlas afirmaría un orden que no existe.

    y_min, y_max = ax.get_ylim()

    if limite_y is not None:
        y_max = limite_y

    ax.set_ylim(y_min, y_max)

    for i, g in enumerate(grupos_ordenados):

        ax.scatter(
            i, medias[g], marker="D", s=120,
            color=COLOR_ESTIMADO, edgecolor="white", linewidth=1, zorder=5,
        )

        ax.text(
            i, 0.985, f"n = {n_grupos[g]}\nμ = {medias[g]:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=10.5, color=COLOR_ESTIMADO, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLOR_ESTIMADO, lw=0.8, alpha=0.95),
        )


    ax.axhline(media_general, linestyle="--", color=COLOR_REFERENCIA, lw=1.5)

    ax.text(
        1.01, media_general, f"media general\n= {media_general:.2f}",
        transform=ax.get_yaxis_transform(), color=COLOR_REFERENCIA,
        fontsize=10.5, va="center", ha="left",
    )

    # Estadística para el criterio
    welch = _welch_anova([valores_grupos[g] for g in grupos_ordenados])
    if welch is not None:
        f_welch, gl1, gl2, p_valor = welch
        cumple = p_valor < alfa
    else:
        gl1 = gl2 = p_valor = None
        cumple = None

    ax.set_xlabel(_bonito(nombre_categorico))
    ax.set_ylabel(_bonito(nombre_numerico))
    ax.set_title(
        f"Welch ANOVA · F = {f_welch:.3f} ({n_g} grupos)",
        fontsize=13, color="#444444",
    )
    _despejar_ejes(ax, grid_eje="y")


    g_max = max(grupos_ordenados, key=lambda g: medias[g])
    g_min = min(grupos_ordenados, key=lambda g: medias[g])

    _verbose(
        "Welch ANOVA · Cómo leerlo: caja = 50% central del grupo, puntos = observaciones, "
        "rombo = media del grupo, línea gris = media general. "
        f"Respondes SÍ si p < {alfa} (diferencias tan grandes serían raras si los grupos "
        "fueran iguales).",
        verbose, nivel=1,
    )

    rango_medias = (f"las medias van de {medias[g_min]:.2f} («{_bonito(g_min, cap=False)}») "
                    f"a {medias[g_max]:.2f} («{_bonito(g_max, cap=False)}»)")
    if cumple is True:
        porque = f"{rango_medias} y esa diferencia no parece casualidad."
        dato = f"F = {f_welch:.2f}, p {_fmt_p(p_valor)} (máx. {alfa})"
    elif cumple is False:
        porque = f"{rango_medias}, pero esa diferencia es compatible con el azar."
        dato = f"F = {f_welch:.2f}, p {_fmt_p(p_valor)} (máx. {alfa})"
    else:
        porque = "algún grupo tiene menos de 2 datos o no varía."
        dato = f"F = {f_welch:.2f}"

    tope = _titulo_pregunta(fig, pregunta)
    tope = _resumen_respuesta(fig, tope, dato, cumple, porque)
    fig.tight_layout(rect=[0, 0, 1, tope])
    fig.savefig(
        carpeta_salida / "welch.png",
        dpi=110,
        bbox_inches="tight",
    )
    _verbose(
        "Grafico welch creado correctamente.",
        verbose,
        nivel=1,
        tipo="success"
    )

    return fig
"""
graficos_gestalt.py
====================

Tema visual personalizado para gráficos estadísticos, construido
aplicando seis principios de la psicología de la Gestalt a la
percepción de datos. Cada decisión de diseño está comentada en el
código con el número de principio que la justifica.

Título de cada gráfico
-----------------------
El título ya no es una etiqueta técnica ("Curva ROC", "Welch ANOVA")
sino la PREGUNTA a la que responde el gráfico. Cada función acepta
un parámetro `pregunta`; si no se indica, se genera una por defecto
a partir de los nombres de las variables.

01 · PROXIMIDAD
    La distancia entre grupos pesa más que una leyenda lejana. En
    vez de leyendas en una esquina, las etiquetas y los valores
    clave (AUC, r, media, información mutua...) se anclan justo
    encima o al lado del elemento gráfico al que describen.

02 · COMPLECIÓN
    La mente rellena lo que falta: cuanto más ruido visual
    (bordes, rejillas, marcos innecesarios) hay que descartar, más
    esfuerzo cuesta leer el dato. Se retira todo lo que no aporte
    lectura: `sns.despine`, rejillas mínimas, sin marco superior/
    derecho, ticks sin marca donde no hacen falta.

03 · SEMEJANZA
    Mismo significado, mismo código visual, en TODO el módulo:
      - COLOR_DATO      -> algo que se midió / recabó directamente.
      - COLOR_ESTIMADO  -> algo derivado, ajustado o resumido
                            (regresión, medias de grupo, densidad
                            estimada, información mutua...).
      - COLOR_REFERENCIA-> una referencia teórica o neutra (azar,
                            media general).
    El coral no se usa nunca "porque queda bien": se usa siempre y
    solo para decir "esto es una estimación".

04 · CIERRE
    La parte del gráfico que es una estimación —y no un dato
    recabado— se encierra visualmente (recuadro, banda sombreada)
    para que el ojo la agrupe como "una cosa aparte" de los datos
    crudos.

05 · CONECTIVIDAD
    Una línea afirma relación y orden. Solo se conectan con línea
    los puntos que comparten de verdad una secuencia (p. ej. el
    umbral de una curva ROC, o categorías temporales explícitas).
    No se conectan categorías nominales sin orden real.

06 · CONTINUIDAD
    La mirada sigue la trayectoria más fluida. Donde no hay una
    secuencia real que respetar, los datos se ordenan de mayor a
    menor para que la lectura fluya en una sola dirección.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from v_python.src.auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta, _es_continua
from v_python.src.SME_python.metricas_de_variables import _calcular_AUC
from v_python.src.SME_python.correlacion_info import _calc_corr_num, _calc_corr_catg, _calc_corr_catg_num
from v_python.src.auxiliar.verbose import _verbose


# ============================================================
# TEMA GESTALT — paleta semántica y configuración global
# ============================================================

# 03 SEMEJANZA: paleta fija de tres colores, cada uno con un único
# significado en todo el módulo. No cambian de un gráfico a otro.
COLOR_DATO = "#2c6e91"         # azul acero -> "esto se midió"
COLOR_ESTIMADO = "#e8734d"     # coral      -> "esto es una estimación"
COLOR_REFERENCIA = "#9a9a9a"   # gris       -> "esto es una referencia teórica"

# Paleta de mapa de calor para magnitudes continuas (frecuencias):
# esto es una codificación de intensidad, no de categoría, así que
# no entra en conflicto con la semántica de SEMEJANZA de arriba.
PALETA = "viridis"

FUENTE_TITULO = dict(
    fontsize=17,
    fontweight="bold",
)

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
        },
    )


def _despejar_ejes(ax, grid_eje="y"):
    """02 COMPLECIÓN: sin bordes superfluos, solo la rejilla útil para leer valores."""
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(axis=grid_eje, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _titulo_pregunta(fig, pregunta):
    """El título es siempre la pregunta a la que responde el gráfico."""
    fig.suptitle(
        pregunta,
        fontsize=FUENTE_TITULO["fontsize"],
        fontweight=FUENTE_TITULO["fontweight"],
        y=0.98,
    )

# ============================================================
# 1) AUC / Curva ROC
# ============================================================

def graficar_auc(
    atributo_clm,
    clases,
    resultado=None,
    pregunta=None,
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

    atributo_clm = pd.Series(atributo_clm).reset_index(drop=True)
    clases = pd.Series(clases).reset_index(drop=True)

    # --------------------------------
    # Calcular o reutilizar el AUC
    # --------------------------------

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

    # --------------------------------
    # Preparar los datos
    # --------------------------------

    datos = pd.DataFrame({"clase": clases, "atributo": atributo_clm})
    datos = datos.sort_values("atributo", ascending=False)

    clase_1 = list(valores_clases)[0]
    clase_2 = list(valores_clases)[1]

    positivos = int((datos["clase"] == clase_1).sum())
    negativos = int((datos["clase"] == clase_2).sum())

    if positivos == 0 or negativos == 0:
        raise ValueError(
            "Cada clase debe tener al menos una observación."
        )

    # --------------------------------
    # Construir curva ROC
    # --------------------------------

    tpr = [0]
    fpr = [0]
    vp = 0
    fp = 0

    for clase in datos["clase"]:
        if clase == clase_1:
            vp += 1
        else:
            fp += 1
        tpr.append(vp / positivos)
        fpr.append(fp / negativos)

    tpr.append(1)
    fpr.append(1)

    # --------------------------------
    # Tema y título-pregunta
    # --------------------------------

    _aplicar_tema_base()

    nombre_attr = atributo_clm.name if atributo_clm.name is not None else "el atributo"
    if pregunta is None:
        pregunta = f"¿Qué tan bien distingue {nombre_attr} entre las dos clases?"

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))

    # --------------------------------
    # Curva ROC
    # --------------------------------

    ax = ejes[0]

    # 03 SEMEJANZA: la curva ROC se calcula a partir de los datos
    # reales -> COLOR_DATO. La diagonal es una referencia teórica
    # (clasificador al azar) -> COLOR_REFERENCIA, igual que en el
    # resto de gráficos del módulo.
    ax.plot(fpr, tpr, color=COLOR_DATO, lw=3)
    ax.fill_between(fpr, tpr, color=COLOR_DATO, alpha=0.15)

    # 05 CONECTIVIDAD: esta línea diagonal SÍ es correcta porque
    # conecta dos puntos que comparten una relación real (el
    # comportamiento esperado de un clasificador sin capacidad
    # discriminativa), no una categoría arbitraria.
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCIA, lw=1.5)
    ax.text(0.62, 0.57, "azar", color=COLOR_REFERENCIA, fontsize=11,
            rotation=33, style="italic")

    # 01 PROXIMIDAD: el AUC se ancla sobre la propia curva en vez
    # de vivir, lejos, dentro de una leyenda en una esquina.
    idx_medio = len(fpr) // 2
    ax.annotate(
        f"AUC = {auc:.3f}",
        xy=(fpr[idx_medio], tpr[idx_medio]),
        xytext=(fpr[idx_medio] + 0.16, tpr[idx_medio] - 0.20),
        color=COLOR_DATO, fontsize=13, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=COLOR_DATO, lw=1),
    )

    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC", fontsize=13, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _despejar_ejes(ax, grid_eje="both")

    # --------------------------------
    # Distribución del atributo
    # --------------------------------

    ax2 = ejes[1]

    # Dos clases = dos identidades categóricas (ambas son datos
    # reales), no una relación dato/estimación: usan una paleta
    # categórica propia, distinta de COLOR_DATO/COLOR_ESTIMADO,
    # para no mezclar dos sistemas de significado distintos.
    paleta_clases = [COLOR_DATO, "#7a3b8c"]

    sns.kdeplot(
        data=datos, x="atributo", hue="clase", fill=True,
        common_norm=False, alpha=0.35, palette=paleta_clases,
        linewidth=1.8, ax=ax2, legend=False,
    )
    sns.rugplot(
        data=datos, x="atributo", hue="clase",
        palette=paleta_clases, ax=ax2, legend=False, alpha=0.5,
    )

    # 01 PROXIMIDAD: la etiqueta de cada clase se ancla junto a su
    # propia curva (sobre su mediana), no en una leyenda aparte.
    y_max = ax2.get_ylim()[1]
    for color, clase_valor in zip(paleta_clases, [clase_1, clase_2]):
        mediana = datos.loc[datos["clase"] == clase_valor, "atributo"].median()
        ax2.text(
            mediana, y_max * 0.94, f"clase {clase_valor}",
            color=color, fontsize=11, fontweight="bold", ha="center",
        )

    ax2.set_xlabel(nombre_attr)
    ax2.set_ylabel("Densidad (estimada)")
    ax2.set_title("Distribución por clase", fontsize=13, color="#444444")
    _despejar_ejes(ax2, grid_eje="y")

    _titulo_pregunta(fig, pregunta)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(
        carpeta_salida / "demo_auc.png",
        dpi=110,
        bbox_inches="tight",
    )
    return fig


# ============================================================
# 2) Correlación de Pearson
# ============================================================

def graficar_pearson(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
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
            1,
            nivel=2,
        )
        resultado = _calc_corr_num(atributo1, atributo2, verbose=0)

    x = np.asarray(resultado["x"])
    y = np.asarray(resultado["y"])
    r = resultado["valor"]
    r2 = resultado["r2"]
    nombre1 = resultado["nombre1"]
    nombre2 = resultado["nombre2"]

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Existe una relación lineal entre {nombre1} y {nombre2}?"

    # Recta de regresión
    pendiente, intercepto = np.polyfit(x, y, 1)
    orden = np.argsort(x)
    x_ordenado = x[orden]
    y_linea = pendiente * x_ordenado + intercepto
    error_estandar = np.std(y - (pendiente * x + intercepto))

    # 03 SEMEJANZA: los puntos son datos reales -> COLOR_DATO.
    grafico = sns.jointplot(
        x=x, y=y, kind="scatter", height=8, color=COLOR_DATO,
        marginal_kws=dict(fill=True, color=COLOR_DATO),
    )
    ax = grafico.ax_joint

    # 04 CIERRE: la recta es una ESTIMACIÓN, no un dato recabado,
    # así que se encierra en una banda sombreada que delimita
    # claramente "esto es un modelo ajustado, no una medición".
    ax.fill_between(
        x_ordenado, y_linea - error_estandar, y_linea + error_estandar,
        color=COLOR_ESTIMADO, alpha=0.15, zorder=1,
    )
    ax.plot(x_ordenado, y_linea, color=COLOR_ESTIMADO, lw=2.5, zorder=2)

    # 01 PROXIMIDAD: la ecuación y el coeficiente se anclan junto
    # al extremo de la propia recta, no en un cuadro de leyenda
    # alejado del dato.
    ax.annotate(
        f"y = {pendiente:.2f}x + {intercepto:.2f}\nr = {r:.3f}  (r² = {r2:.3f})",
        xy=(x_ordenado[-1], y_linea[-1]),
        xytext=(10, 0), textcoords="offset points",
        color=COLOR_ESTIMADO, fontsize=11, fontweight="bold", va="center",
    )

    grafico.set_axis_labels(nombre1, nombre2)
    _despejar_ejes(ax, grid_eje="both")

    _titulo_pregunta(grafico.figure, pregunta)
    grafico.figure.subplots_adjust(top=0.90)

    grafico.figure.savefig(
        carpeta_salida / "pearson.png",
        dpi=110,
        bbox_inches="tight",
    )

    plt.close(grafico.figure)
    return grafico.figure


# ============================================================
# 3) Información mutua entre dos categóricas
# ============================================================

def graficar_informacion_mutua(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
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
            1,
            nivel=2,
        )
        resultado = _calc_corr_catg(atributo1, atributo2, verbose=0)

    tabla = resultado["tabla"]
    mi = resultado["valor"]
    nombre1 = resultado["nombre1"]
    nombre2 = resultado["nombre2"]

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Están relacionadas las categorías de {nombre1} y {nombre2}?"

    # 06 CONTINUIDAD: sin una secuencia real que respetar (no hay
    # un orden natural entre categorías), se ordenan filas y
    # columnas de mayor a menor frecuencia total. Así la mirada
    # recorre la tabla en una trayectoria descendente y fluida, en
    # vez de un orden alfabético arbitrario.
    orden_filas = tabla.sum(axis=1).sort_values(ascending=False).index
    orden_columnas = tabla.sum(axis=0).sort_values(ascending=False).index
    tabla = tabla.loc[orden_filas, orden_columnas]

    tabla_pct = tabla / tabla.values.sum() * 100

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        tabla, annot=True, fmt="d", cmap=PALETA,
        cbar_kws={"label": "Frecuencia absoluta (dato)"}, ax=ejes[0],
        linewidths=0.5, linecolor="white",
    )
    ejes[0].set_title("Frecuencias absolutas", fontsize=13, color="#444444")
    ejes[0].set_xlabel(nombre2)
    ejes[0].set_ylabel(nombre1)

    sns.heatmap(
        tabla_pct, annot=True, fmt=".1f", cmap=PALETA,
        cbar_kws={"label": "% del total (derivado)"}, ax=ejes[1],
        linewidths=0.5, linecolor="white",
    )
    ejes[1].set_title("Frecuencias relativas (%)", fontsize=13, color="#444444")
    ejes[1].set_xlabel(nombre2)
    ejes[1].set_ylabel("")

    for ax in ejes:
        ax.tick_params(length=0)

    # 01 PROXIMIDAD: la información mutua se coloca pegada a las
    # dos tablas que la sustentan, no en un cuadro de texto suelto.
    fig.text(
        0.5, 0.88, f"Información mutua = {mi:.4f} bits",
        ha="center", fontsize=13, color=COLOR_ESTIMADO, fontweight="bold",
    )

    _titulo_pregunta(fig, pregunta)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(
        carpeta_salida / "MI.png",
        dpi=110,
        bbox_inches="tight",
    )
    return fig


# ============================================================
# 4) Welch ANOVA
# ============================================================

def graficar_welch(
    atributo1,
    atributo2,
    resultado=None,
    pregunta=None,
    orden_categorias=None,
    verbose=1,
    path=".",
    nombre_carpeta="figures"
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
            1,
            nivel=2,
        )
        resultado = _calc_corr_catg_num(atributo1, atributo2, verbose=0)

    grupos = resultado["grupos"]
    valores_grupos = resultado["valores_grupos"]
    medias = resultado["medias"]
    n_grupos = resultado["n_grupos"]
    media_general = resultado["media_general"]
    f_welch = resultado["valor"]
    nombre_categorico = resultado["nombre_categorico"]
    nombre_numerico = resultado["nombre_numerico"]

    _aplicar_tema_base()

    if pregunta is None:
        pregunta = f"¿Difiere {nombre_numerico} según {nombre_categorico}?"

    es_secuencia_real = orden_categorias is not None

    if es_secuencia_real:
        grupos_ordenados = [g for g in orden_categorias if g in grupos]
    else:
        grupos_ordenados = sorted(grupos, key=lambda g: medias[g], reverse=True)

    etiquetas = [str(g) for g in grupos_ordenados]

    datos = pd.concat([
        pd.DataFrame({"grupo": str(g), "valor": valores_grupos[g]})
        for g in grupos_ordenados
    ], ignore_index=True)
    datos["grupo"] = pd.Categorical(datos["grupo"], categories=etiquetas, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    # 03 SEMEJANZA: los puntos y las cajas son datos reales
    # recabados -> COLOR_DATO en todo momento.
    sns.boxplot(
        data=datos, x="grupo", y="valor", color=COLOR_DATO,
        showfliers=False, width=0.5, ax=ax, boxprops=dict(alpha=0.30),
    )
    sns.stripplot(
        data=datos, x="grupo", y="valor", color=COLOR_DATO,
        alpha=0.45, size=4.5, jitter=0.2, ax=ax,
    )

    xs_medias = [medias[g] for g in grupos_ordenados]

    if es_secuencia_real:
        # 05 CONECTIVIDAD: aquí SÍ se conecta, porque los grupos
        # comparten una secuencia real y la línea afirma
        # correctamente esa relación de orden.
        ax.plot(
            range(len(grupos_ordenados)), xs_medias,
            color=COLOR_ESTIMADO, lw=1.8, zorder=4, alpha=0.85,
        )
    # Si no hay secuencia real, NO se dibuja línea entre las
    # medias: conectarlas afirmaría un orden que no existe.

    for i, g in enumerate(grupos_ordenados):
        # 03 SEMEJANZA: la media de grupo es un valor ESTIMADO
        # (resumen), no un dato individual -> COLOR_ESTIMADO.
        ax.scatter(
            i, medias[g], marker="D", s=120,
            color=COLOR_ESTIMADO, edgecolor="white", linewidth=1, zorder=5,
        )
        # 01 PROXIMIDAD + 04 CIERRE: n y la media se anclan junto
        # al marcador y se encierran en una cajita, separándolos
        # visualmente de los puntos de dato individuales.
        ax.annotate(
            f"n={n_grupos[g]}\nμ={medias[g]:.2f}",
            (i, medias[g]), textcoords="offset points", xytext=(16, 0),
            fontsize=9.5, color=COLOR_ESTIMADO,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLOR_ESTIMADO, lw=0.8, alpha=0.9),
        )

    # 03 SEMEJANZA: la media general es una referencia teórica
    # neutra -> COLOR_REFERENCIA, igual código que la diagonal de
    # azar en la curva ROC.
    ax.axhline(media_general, linestyle="--", color=COLOR_REFERENCIA, lw=1.5)
    ax.text(
        len(grupos_ordenados) - 0.55, media_general,
        f" media general = {media_general:.2f}",
        color=COLOR_REFERENCIA, fontsize=10, va="bottom",
    )

    ax.set_xlabel(nombre_categorico)
    ax.set_ylabel(nombre_numerico)
    ax.set_title(
        f"Welch ANOVA · F = {f_welch:.3f} ({len(grupos_ordenados)} grupos)",
        fontsize=13, color="#444444",
    )
    _despejar_ejes(ax, grid_eje="y")

    _titulo_pregunta(fig, pregunta)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(
        carpeta_salida / "welch.png",
        dpi=110,
        bbox_inches="tight",
    )
    return fig


# ============================================================
# Demostración
# ============================================================

def main():
    """
    Demostración de los cuatro tipos de gráfico del módulo.

    Genera un ejemplo para cada técnica (AUC/ROC, Pearson,
    información mutua y Welch ANOVA) con datos sintéticos, y guarda
    las figuras en la carpeta ``figures``.

    No se conecta a ningún dataset real: sirve como referencia visual
    de cómo se comporta cada gráfico y de cómo aplicar el tema Gestalt.
    """

    # Nota: se corrige el orden de los argumentos respecto al
    # ejemplo original (atributo numérico primero, clases después).
    clases = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="Compra")
    probabilidades = pd.Series(
        [0.10, 0.35, 0.40, 0.60, 0.70, 0.30, 0.50, 0.65, 0.80, 0.90],
        name="Score del modelo",
    )
    graficar_auc(probabilidades, clases)
    plt.close("all")

    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="Horas de estudio")
    y = pd.Series([2, 5, 4, 8, 7, 11, 10, 15, 13, 17], name="Nota")
    graficar_pearson(x, y)
    plt.close("all")

    sexo = pd.Series(
        ["Hombre", "Mujer", "Mujer", "Hombre", "Mujer",
         "Hombre", "Mujer", "Mujer", "Hombre", "Hombre"],
        name="Sexo",
    )
    compra = pd.Series(
        ["Sí", "No", "Sí", "Sí", "No",
         "Sí", "No", "Sí", "Sí", "No"],
        name="Compra",
    )
    graficar_informacion_mutua(sexo, compra)
    plt.close("all")

    # Ejemplo SIN secuencia real: los grupos se ordenan por media,
    # sin conectarlos con línea.
    grupos = pd.Series(
        ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        name="Grupo",
    )
    valores = pd.Series(
        [10, 12, 11, 13, 18, 17, 20, 19, 14, 15, 13, 16],
        name="Puntuación",
    )
    graficar_welch(grupos, valores)
    plt.close("all")

    # Ejemplo CON secuencia real (años): aquí sí tiene sentido
    # conectar las medias con una línea.
    anios = pd.Series(
        ["2021", "2021", "2021", "2022", "2022", "2022",
         "2023", "2023", "2023", "2024", "2024", "2024"],
        name="Año",
    )
    ventas = pd.Series(
        [10, 12, 11, 14, 15, 13, 19, 18, 20, 22, 24, 23],
        name="Ventas (miles €)",
    )
    graficar_welch(anios, ventas, orden_categorias=["2021", "2022", "2023", "2024"])
    plt.close("all")


if __name__ == "__main__":
    main()
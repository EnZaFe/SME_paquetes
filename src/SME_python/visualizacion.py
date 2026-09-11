import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from auxiliar.auxiliar_es import _es_dataset, _es_variable, _es_discreta, _es_continua
from SME_python.metricas_de_variables import _calcular_AUC
from SME_python.correlacion_info import _calc_corr_num, _calc_corr_catg, _calc_corr_catg_num

sns.set_theme(style="whitegrid", context="talk")
PALETA = "viridis"


def graficar_auc(y_real, y_probabilidad, resultado=None):
    """
    Curva ROC + distribución de probabilidades por clase.

    Se puede llamar de dos formas:

        graficar_auc(y_real, y_probabilidad)
            -> calcula todo internamente

        graficar_auc(y_real, y_probabilidad, resultado=info)
            -> reutiliza el AUC ya calculado (p.ej. desde
               calcular_metricas), evitando recalcularlo
    """

    if len(y_real) != len(y_probabilidad):
        raise ValueError(
            "y_real e y_probabilidad deben tener el mismo número de elementos."
        )

    if len(y_real) == 0:
        raise ValueError(
            "No se puede calcular la curva ROC con datos vacíos."
        )

    if not _es_continua(y_probabilidad):
        raise TypeError(
            "y_probabilidad debe ser una variable numérica."
        )

    clases = set(y_real)

    if len(clases) != 2:
        raise ValueError(
            "y_real debe contener exactamente dos clases."
        )

    y_real = pd.Series(y_real).reset_index(drop=True)
    y_probabilidad = pd.Series(y_probabilidad).reset_index(drop=True)

    # Si no nos pasan el AUC ya calculado, lo calculamos con la misma
    # función que usan las métricas, para que el número sea siempre
    # coherente en toda la librería.
    if resultado is not None and "valor" in resultado:
        auc = resultado["valor"]
    else:
        auc = _calcular_AUC(y_probabilidad, y_real, verbose=0)

    datos = pd.DataFrame({
        "clase": y_real,
        "probabilidad": y_probabilidad
    }).sort_values("probabilidad", ascending=False)

    clase_1, clase_2 = list(clases)

    positivos = int((datos["clase"] == clase_1).sum())
    negativos = int((datos["clase"] == clase_2).sum())

    if positivos == 0 or negativos == 0:
        raise ValueError(
            "Cada clase debe tener al menos una observación."
        )

    tpr, fpr = [0], [0]
    vp, fp = 0, 0

    for clase in datos["clase"]:
        if clase == clase_1:
            vp += 1
        else:
            fp += 1
        tpr.append(vp / positivos)
        fpr.append(fp / negativos)

    tpr.append(1)
    fpr.append(1)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Curva ROC ---
    ax = ejes[0]
    ax.plot(fpr, tpr, color="#2c7fb8", lw=3, label=f"ROC (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.2, color="#2c7fb8")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5,
            label="Clasificador aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curva ROC · AUC = {auc:.3f}", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Distribución de la probabilidad por clase ---
    ax2 = ejes[1]
    sns.kdeplot(
        data=datos, x="probabilidad", hue="clase",
        fill=True, common_norm=False, alpha=0.4, palette=PALETA, ax=ax2
    )
    sns.rugplot(
        data=datos, x="probabilidad", hue="clase",
        palette=PALETA, ax=ax2, legend=False
    )
    ax2.set_title("Distribución de probabilidades por clase", fontweight="bold")
    ax2.set_xlabel("Probabilidad estimada")

    fig.suptitle("Análisis del clasificador", fontsize=18, fontweight="bold")
    fig.tight_layout()

    return fig


def graficar_pearson(atributo1, atributo2, resultado=None):
    """
    Dispersión + regresión lineal + distribuciones marginales entre
    dos variables numéricas, junto con el coeficiente de Pearson.

        graficar_pearson(x, y)
            -> calcula todo internamente

        graficar_pearson(x, y, resultado=info)
            -> reutiliza un resultado ya calculado con
               _calc_corr_num / calcular_correlacion
    """

    if len(atributo1) != len(atributo2):
        raise ValueError(
            "Los atributos deben tener el mismo número de elementos."
        )

    if not _es_continua(atributo1):
        raise TypeError("El primer atributo debe ser numérico.")

    if not _es_continua(atributo2):
        raise TypeError("El segundo atributo debe ser numérico.")

    if resultado is None:
        resultado = _calc_corr_num(atributo1, atributo2, verbose=0)

    x = resultado["x"]
    y = resultado["y"]
    r = resultado["valor"]
    r2 = resultado["r2"]
    pendiente = resultado["pendiente"]
    intercepto = resultado["intercepto"]
    nombre1 = resultado["nombre1"]
    nombre2 = resultado["nombre2"]

    grafico = sns.jointplot(
        x=x, y=y, kind="scatter",
        height=8, color="#2c7fb8",
        marginal_kws=dict(fill=True)
    )

    grafico.ax_joint.plot(
        x, pendiente * x + intercepto,
        color="#d95f02", lw=2.5,
        label=f"y = {pendiente:.3f}x + {intercepto:.3f}"
    )

    grafico.set_axis_labels(nombre1, nombre2)
    grafico.ax_joint.legend(loc="best", frameon=True)

    grafico.figure.suptitle(
        f"Correlación de Pearson · r = {r:.3f}  (r² = {r2:.3f})",
        fontsize=16, fontweight="bold"
    )
    grafico.figure.subplots_adjust(top=0.92)

    return grafico.figure


def graficar_informacion_mutua(atributo1, atributo2, resultado=None):
    """
    Heatmap de la tabla de contingencia entre dos variables
    categóricas, con frecuencias absolutas y relativas, y la
    información mutua entre ambas.
    """

    if len(atributo1) != len(atributo2):
        raise ValueError(
            "Los atributos deben tener el mismo número de elementos."
        )

    if not _es_discreta(atributo1):
        raise TypeError("El primer atributo debe ser categórico/discreto.")

    if not _es_discreta(atributo2):
        raise TypeError("El segundo atributo debe ser categórico/discreto.")

    if resultado is None:
        resultado = _calc_corr_catg(atributo1, atributo2, verbose=0)

    tabla = resultado["tabla"]
    mi = resultado["valor"]
    nombre1 = resultado["nombre1"]
    nombre2 = resultado["nombre2"]

    tabla_pct = tabla / tabla.values.sum() * 100

    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        tabla, annot=True, fmt="d", cmap=PALETA,
        cbar_kws={"label": "Frecuencia absoluta"}, ax=ejes[0]
    )
    ejes[0].set_title("Frecuencias absolutas", fontweight="bold")
    ejes[0].set_xlabel(nombre2)
    ejes[0].set_ylabel(nombre1)

    sns.heatmap(
        tabla_pct, annot=True, fmt=".1f", cmap=PALETA,
        cbar_kws={"label": "% del total"}, ax=ejes[1]
    )
    ejes[1].set_title("Frecuencias relativas (%)", fontweight="bold")
    ejes[1].set_xlabel(nombre2)
    ejes[1].set_ylabel("")

    fig.suptitle(
        f"Información Mutua · MI = {mi:.4f}",
        fontsize=18, fontweight="bold"
    )
    fig.tight_layout()

    return fig


def graficar_welch(atributo1, atributo2, resultado=None):
    """
    Boxplot + puntos individuales por grupo, con la media de cada
    grupo, la media general y el estadístico F de Welch.
    """

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
        resultado = _calc_corr_catg_num(atributo1, atributo2, verbose=0)

    grupos = resultado["grupos"]
    valores_grupos = resultado["valores_grupos"]
    medias = resultado["medias"]
    n_grupos = resultado["n_grupos"]
    media_general = resultado["media_general"]
    f_welch = resultado["valor"]
    nombre_categorico = resultado["nombre_categorico"]
    nombre_numerico = resultado["nombre_numerico"]

    datos = pd.concat([
        pd.DataFrame({
            "grupo": str(grupo),
            "valor": valores_grupos[grupo]
        })
        for grupo in grupos
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.boxplot(
        data=datos, x="grupo", y="valor",
        palette=PALETA, showfliers=False, width=0.55, ax=ax
    )
    sns.stripplot(
        data=datos, x="grupo", y="valor",
        color="black", alpha=0.35, size=4, jitter=0.2, ax=ax
    )

    for i, grupo in enumerate(grupos):
        ax.scatter(
            i, medias[grupo], marker="D", s=110,
            color="#d95f02", zorder=5,
            label="Media del grupo" if i == 0 else None
        )
        ax.annotate(
            f"n={n_grupos[grupo]}\nμ={medias[grupo]:.2f}",
            (i, medias[grupo]),
            textcoords="offset points", xytext=(15, 0),
            fontsize=10
        )

    ax.axhline(
        media_general, linestyle="--", color="gray", lw=1.5,
        label=f"Media general = {media_general:.2f}"
    )

    ax.set_xlabel(nombre_categorico)
    ax.set_ylabel(nombre_numerico)
    ax.set_title(
        f"Welch ANOVA · F = {f_welch:.3f}  ({len(grupos)} grupos)",
        fontsize=16, fontweight="bold"
    )
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    return fig


def main():

    clases = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probabilidades = pd.Series(
        [0.10, 0.35, 0.40, 0.60, 0.70, 0.30, 0.50, 0.65, 0.80, 0.90]
    )
    graficar_auc(clases, probabilidades)
    plt.show()

    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="Horas de estudio")
    y = pd.Series([2, 5, 4, 8, 7, 11, 10, 15, 13, 17], name="Nota")
    graficar_pearson(x, y)
    plt.show()

    sexo = pd.Series(
        ["Hombre", "Mujer", "Mujer", "Hombre", "Mujer",
         "Hombre", "Mujer", "Mujer", "Hombre", "Hombre"],
        name="Sexo"
    )
    compra = pd.Series(
        ["Sí", "No", "Sí", "Sí", "No",
         "Sí", "No", "Sí", "Sí", "No"],
        name="Compra"
    )
    graficar_informacion_mutua(sexo, compra)
    plt.show()

    grupos = pd.Series(
        ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        name="Grupo"
    )
    valores = pd.Series(
        [10, 12, 11, 13, 18, 17, 20, 19, 14, 15, 13, 16],
        name="Puntuación"
    )
    graficar_welch(grupos, valores)
    plt.show()


if __name__ == "__main__":
    main()
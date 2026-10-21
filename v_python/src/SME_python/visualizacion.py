from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, PercentFormatter

from auxiliar.auxiliar_es import _es_discreta, _es_continua
from auxiliar.verbose import _verbose
from SME_python.correlacion_info import (
    _calc_corr_catg,
    _calc_corr_catg_num,
    _calc_corr_num,
)
from SME_python.metricas_de_variables import _calcular_AUC

COLOR_DATOS = "#2c6e8f"
COLOR_AJUSTE = "#e4572e"
CAJA_TEXTO = dict(boxstyle="round", facecolor="white", edgecolor=COLOR_AJUSTE)


def _num(v):
    """Número legible sin notación científica: 1.234.567,5"""
    return f"{v:,.10g}".replace(",", "X").replace(".", ",").replace("X", ".")


FORMATO_NUM = FuncFormatter(lambda v, _: _num(v))


def _estilo(ax, titulo, xlabel, ylabel, subtitulo=None, rejilla=True):
    """Estilo común: título en negrita, subtítulo gris, sin bordes superior/derecho."""
    ax.set_title(titulo, fontweight="bold", pad=20 if subtitulo else 6)
    if subtitulo:
        ax.text(0.5, 1.02, subtitulo, transform=ax.transAxes,
                ha="center", va="bottom", color="grey")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if rejilla:
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)


def graficar_auc(atributo_clm, clases, resultado=None, pregunta=None,
                 umbral_auc=0.70, verbose=1, path=".", nombre_carpeta="figures"):
    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    atributo_clm = pd.Series(atributo_clm)
    clases = pd.Series(clases)
    if len(atributo_clm) != len(clases):
        raise ValueError("El atributo y las clases deben tener el mismo número de elementos.")
    if len(atributo_clm) == 0:
        raise ValueError("No se puede calcular el AUC con datos vacíos.")
    if not _es_continua(atributo_clm):
        raise TypeError("El atributo debe ser numérico para calcular el AUC.")
    valores_clases = list(dict.fromkeys(clases))
    if len(valores_clases) != 2:
        raise ValueError("Las clases deben contener exactamente dos clases.")
    if resultado is None:
        _verbose("No se ha proporcionado el resultado de AUC. Se calculará automáticamente.",
                 verbose, nivel=2)
        resultado = _calcular_AUC(atributo_clm, clases, verbose=0)
    auc = resultado["valor"] if isinstance(resultado, dict) else resultado
    datos = pd.DataFrame({"clase": clases, "atributo": atributo_clm})
    clase_neg, clase_pos = sorted(valores_clases, key=str)
    positivos = (datos["clase"] == clase_pos).sum()
    negativos = len(datos) - positivos
    datos = datos.sort_values("atributo", ascending=False)
    tpr = [0]
    fpr = [0]
    vp = fp = 0
    for clase in datos["clase"]:
        if clase == clase_pos:
            vp += 1
        else:
            fp += 1
        tpr.append(vp / positivos)
        fpr.append(fp / negativos)
    fpr.append(1)
    tpr.append(1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(fpr, tpr, color=COLOR_DATOS, alpha=0.15)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.text(0.55, 0.5, "azar", rotation=45, color="grey", style="italic",
            ha="center", va="center")
    ax.plot(fpr, tpr, color=COLOR_DATOS, linewidth=2)
    ax.text(0.75, 0.15, f"AUC = {auc:.3f}\n(área sombreada)", ha="center",
            color=COLOR_DATOS, fontweight="bold",
            bbox=dict(CAJA_TEXTO, edgecolor=COLOR_DATOS))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _estilo(ax, "Curva ROC",
            f"Falsos positivos\n(% de «{clase_neg}» marcados como «{clase_pos}»)",
            f"Verdaderos positivos\n(% de «{clase_pos}» detectados)")
    _verbose("Curva ROC · Cómo leerla: cuanto más se acerca la curva a la esquina superior "
             "izquierda, mejor separa el atributo las dos clases; la diagonal es lo que haría "
             f"el azar. Respondes SÍ si AUC ≥ {umbral_auc}.", verbose, nivel=1)
    fig.savefig(carpeta_salida / "demo_auc.png", dpi=110, bbox_inches="tight")
    _verbose("Gráfico de la curva ROC creado correctamente.", verbose, nivel=1)
    return fig


def graficar_pearson(atributo1, atributo2, resultado=None, pregunta=None,
                     umbral_r=0.5, alfa=0.05, verbose=1, path=".",
                     nombre_carpeta="figures"):
    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    atributo1 = pd.Series(atributo1)
    atributo2 = pd.Series(atributo2)
    if len(atributo1) != len(atributo2):
        raise ValueError("Los atributos deben tener el mismo número de elementos.")
    if not _es_continua(atributo1):
        raise TypeError("El primer atributo debe ser numérico.")
    if not _es_continua(atributo2):
        raise TypeError("El segundo atributo debe ser numérico.")
    if resultado is None:
        _verbose("No se ha proporcionado el resultado de Pearson. Se calculará automáticamente.",
                 verbose, nivel=2)
        resultado = _calc_corr_num(atributo1, atributo2, verbose=0)
    x = np.asarray(resultado["x"], dtype=float)
    y = np.asarray(resultado["y"], dtype=float)
    r = resultado["valor"]
    pendiente, intercepto = np.polyfit(x, y, 1)
    s = np.std(y - (pendiente * x + intercepto), ddof=1)
    x_recta = np.array([x.min(), x.max()])
    y_recta = pendiente * x_recta + intercepto

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(x_recta, y_recta - s, y_recta + s, color=COLOR_AJUSTE, alpha=0.15)
    ax.scatter(x, y, color=COLOR_DATOS, alpha=0.5, s=20, edgecolors="none")
    ax.plot(x_recta, y_recta, color=COLOR_AJUSTE, linewidth=2)
    ax.text(0.98, 0.97,
            f"y = {pendiente:.2f}x + {intercepto:.2f}\nr = {r:.3f}   (r² = {r ** 2:.3f})",
            transform=ax.transAxes, ha="right", va="top",
            color=COLOR_AJUSTE, fontweight="bold", bbox=CAJA_TEXTO)
    ax.xaxis.set_major_formatter(FORMATO_NUM)
    ax.yaxis.set_major_formatter(FORMATO_NUM)
    _estilo(ax, "Correlación de Pearson", str(resultado["nombre1"]), str(resultado["nombre2"]))
    _verbose("Pearson · Cómo leerlo: cada punto es una observación; la recta es el ajuste "
             "lineal y la banda, ±1 desviación del error. "
             f"Respondes SÍ si |r| ≥ {umbral_r} y p < {alfa}.", verbose, nivel=1)
    fig.savefig(carpeta_salida / "pearson.png", dpi=110, bbox_inches="tight")
    _verbose("Gráfico pearson creado correctamente.", verbose, nivel=1)
    return fig


def graficar_informacion_mutua(atributo1, atributo2, resultado=None, pregunta=None,
                                alfa=0.05, verbose=1, path=".",
                                nombre_carpeta="figures"):
    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    atributo1 = pd.Series(atributo1)
    atributo2 = pd.Series(atributo2)
    if len(atributo1) != len(atributo2):
        raise ValueError("Los atributos deben tener el mismo número de elementos.")
    if not _es_discreta(atributo1):
        raise TypeError("El primer atributo debe ser categórico/discreto.")
    if not _es_discreta(atributo2):
        raise TypeError("El segundo atributo debe ser categórico/discreto.")
    if resultado is None:
        _verbose("No se ha proporcionado el resultado de información mutua. Se calculará automáticamente.",
                 verbose, nivel=2)
        resultado = _calc_corr_catg(atributo1, atributo2, verbose=0)
    tabla = pd.DataFrame(resultado["tabla"])
    # Categorías ordenadas por frecuencia total: la más frecuente arriba y a la izquierda.
    tabla = tabla.loc[tabla.sum(axis=1).sort_values(ascending=False).index,
                      tabla.sum(axis=0).sort_values(ascending=False).index]
    valores = tabla.values.astype(float)
    mi = resultado["valor"]

    fig, ax = plt.subplots(figsize=(7, 5))
    imagen = ax.imshow(valores, cmap="viridis", aspect="auto")
    fig.colorbar(imagen, ax=ax, label="Frecuencia absoluta")
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            color = "black" if valores[i, j] > valores.max() / 2 else "white"
            ax.text(j, i, f"{valores[i, j]:.0f}", ha="center", va="center",
                    fontsize=13, color=color)
    ax.set_xticks(range(len(tabla.columns)), [str(c) for c in tabla.columns], rotation=90)
    ax.set_yticks(range(len(tabla.index)), [str(f) for f in tabla.index])
    # Líneas blancas entre celdas.
    ax.set_xticks(np.arange(-0.5, valores.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, valores.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    _estilo(ax, "Información mutua", str(resultado["nombre2"]), str(resultado["nombre1"]),
            subtitulo=f"MI = {mi:.4f}", rejilla=False)
    ax.spines[:].set_visible(False)
    _verbose("Información mutua · Cómo leerla: cada celda cuenta las observaciones de esa "
             "combinación; si las filas se reparten de forma muy distinta, las variables están "
             f"relacionadas. Respondes SÍ si la prueba G da p < {alfa} (MI mayor que la del azar; "
             "en bits no hay un umbral universal).", verbose, nivel=1)
    esperados = np.outer(valores.sum(axis=1), valores.sum(axis=0)) / valores.sum()
    if (esperados < 5).any():
        _verbose("Aviso: hay celdas con menos de 5 casos esperados; la prueba G es orientativa.",
                 verbose, nivel=1)
    fig.savefig(carpeta_salida / "MI.png", dpi=110, bbox_inches="tight")
    _verbose("Gráfico de información mutua creado correctamente.", verbose, nivel=1)
    return fig


def _welch_anova(valores_grupos):
    muestras = [np.asarray(v, dtype=float) for v in valores_grupos]
    k = len(muestras)
    if k < 2:
        return None
    n = np.array([len(m) for m in muestras], dtype=float)
    medias = np.array([m.mean() for m in muestras])
    varianzas = np.array([m.var(ddof=1) for m in muestras])
    if (n < 2).any() or not np.all(np.isfinite(varianzas)) or (varianzas <= 0).any():
        return None
    w = n / varianzas
    media_w = (w * medias).sum() / w.sum()
    numerador = (w * (medias - media_w) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    denominador = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    return float(numerador / denominador)


def graficar_welch(atributo1, atributo2, resultado=None, pregunta=None,
                   orden_categorias=None, alfa=0.05, verbose=1, path=".",
                   nombre_carpeta="figures", limite_y=None):
    carpeta_salida = Path(path) / nombre_carpeta
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    atributo1 = pd.Series(atributo1)
    atributo2 = pd.Series(atributo2)
    if len(atributo1) != len(atributo2):
        raise ValueError("Los atributos deben tener el mismo número de elementos.")
    if not ((_es_discreta(atributo1) and _es_continua(atributo2)) or
            (_es_continua(atributo1) and _es_discreta(atributo2))):
        raise TypeError("Welch necesita un atributo categórico y otro numérico.")
    if resultado is None:
        _verbose("No se ha proporcionado el resultado de Welch ANOVA. Se calculará automáticamente.",
                 verbose, nivel=2)
        resultado = _calc_corr_catg_num(atributo1, atributo2, verbose=0)
    grupos = [g for g in resultado["grupos"] if pd.notna(g) and str(g) != "NA"]
    valores_grupos = resultado["valores_grupos"]
    medias = resultado["medias"]
    f_welch = resultado["valor"]
    if orden_categorias is None:
        grupos = sorted(grupos, key=lambda grupo: medias[grupo], reverse=True)
    else:
        grupos = [grupo for grupo in orden_categorias if grupo in grupos]
    valores = [np.asarray(valores_grupos[g], dtype=float) for g in grupos]
    posiciones = range(1, len(grupos) + 1)
    media_general = np.concatenate(valores).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(0)
    for pos, v in zip(posiciones, valores):
        ax.scatter(pos + rng.uniform(-0.25, 0.25, len(v)), v,
                   color=COLOR_DATOS, alpha=0.35, s=8, edgecolors="none")
    ax.boxplot(valores, tick_labels=[str(g) for g in grupos], widths=0.6,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=to_rgba(COLOR_DATOS, 0.25)),
               medianprops=dict(color="black"))
    ax.axhline(media_general, linestyle="--", color="grey")
    ax.scatter(posiciones, [v.mean() for v in valores], marker="D", s=80,
               color=COLOR_AJUSTE, edgecolors="white", zorder=3)
    # Hueco arriba para las etiquetas n / μ.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, limite_y if limite_y is not None else y_max + 0.2 * (y_max - y_min))
    for pos, v in zip(posiciones, valores):
        ax.text(pos, 0.98, f"n = {len(v)}\nμ = {_num(round(v.mean()))}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=COLOR_AJUSTE, fontsize=9, bbox=CAJA_TEXTO)
    ax.yaxis.set_major_formatter(FORMATO_NUM)
    _estilo(ax, f"Welch ANOVA · F = {f_welch:.3f} ({len(grupos)} grupos)",
            str(resultado["nombre_categorico"]), str(resultado["nombre_numerico"]),
            subtitulo=f"Línea discontinua: media general = {_num(round(media_general))}")
    _verbose("Welch ANOVA · Cómo leerlo: caja = 50% central del grupo, puntos = observaciones, "
             "rombo = media del grupo, línea gris = media general. "
             f"Respondes SÍ si p < {alfa} (diferencias tan grandes serían raras si los grupos "
             "fueran iguales).", verbose, nivel=1)
    fig.savefig(carpeta_salida / "welch.png", dpi=110, bbox_inches="tight")
    _verbose("Gráfico welch creado correctamente.", verbose, nivel=1)
    return fig
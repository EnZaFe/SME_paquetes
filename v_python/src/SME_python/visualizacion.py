from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auxiliar.auxiliar_es import _es_discreta, _es_continua
from auxiliar.verbose import _verbose
from SME_python.correlacion_info import (
    _calc_corr_catg,
    _calc_corr_catg_num,
    _calc_corr_num,
)
from SME_python.metricas_de_variables import _calcular_AUC


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
    clases_ordenadas = sorted(valores_clases, key=str)
    clase_pos = clases_ordenadas[1]
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
    fig, ax = plt.subplots()
    ax.plot(fpr + [1], tpr + [1])
    ax.set_title("Curva ROC")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.text(0.05, 0.95, f"AUC = {auc:.3f}", transform=ax.transAxes, va="top")
    _verbose(f"AUC = {auc:.3f}. Respuesta: {'Sí' if auc >= umbral_auc else 'No'}.",
             verbose, nivel=1)
    fig.savefig(Path(path) / "demo_auc.png", dpi=110, bbox_inches="tight")
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
    orden = np.argsort(x)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x[orden], pendiente * x[orden] + intercepto)
    ax.set_title("Correlación de Pearson")
    ax.set_xlabel(str(resultado["nombre1"]))
    ax.set_ylabel(str(resultado["nombre2"]))
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, va="top")
    _verbose(f"r = {r:.3f}. Respuesta: {'Sí' if abs(r) >= umbral_r else 'No'}.",
             verbose, nivel=1)
    fig.savefig(Path(path) / "pearson.png", dpi=110, bbox_inches="tight")
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
    tabla = resultado["tabla"]
    mi = resultado["valor"]
    fig, ax = plt.subplots()
    imagen = ax.imshow(tabla.values if hasattr(tabla, "values") else tabla)
    fig.colorbar(imagen, ax=ax)
    ax.set_title("Información mutua")
    ax.set_xlabel(str(resultado["nombre2"]))
    ax.set_ylabel(str(resultado["nombre1"]))
    ax.text(0.05, 0.95, f"MI = {mi:.4f}", transform=ax.transAxes, va="top")
    _verbose(f"Información mutua = {mi:.4f}.", verbose, nivel=1)
    fig.savefig(Path(path) / "MI.png", dpi=110, bbox_inches="tight")
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
    grupos = resultado["grupos"]
    valores_grupos = resultado["valores_grupos"]
    medias = resultado["medias"]
    f_welch = resultado["valor"]
    if orden_categorias is None:
        grupos = sorted(grupos, key=lambda grupo: medias[grupo], reverse=True)
    else:
        grupos = [grupo for grupo in orden_categorias if grupo in grupos]
    fig, ax = plt.subplots()
    ax.boxplot(
        [valores_grupos[grupo] for grupo in grupos],
        tick_labels=[str(g) for g in grupos]
    )
    ax.set_title(f"Welch ANOVA (F = {f_welch:.3f})")
    ax.set_xlabel(str(resultado["nombre_categorico"]))
    ax.set_ylabel(str(resultado["nombre_numerico"]))
    if limite_y is not None:
        ax.set_ylim(top=limite_y)
    _verbose(f"F = {f_welch:.3f}.", verbose, nivel=1)
    fig.savefig(Path(path) / "welch.png", dpi=110, bbox_inches="tight")
    return fig

"""
features.py
============
Calculo de features a partir de ventanas de telemetria.

Este modulo es el CONTRATO entre el entrenamiento (train.py) y la API
(api.py): ambos deben usar exactamente las mismas funciones para que las
features que ve el modelo en produccion sean identicas a las que vio durante
el entrenamiento. Si un lado calcula features distintas, el modelo recibe
datos con un significado distinto al que aprendio y falla en silencio.

No se importa nada de train.py ni de api.py aqui: este archivo solo sabe de
DataFrames de pandas con columnas de señales crudas, nunca de episodios,
estados ni HTTP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Señales crudas que llegan de la GPU, en el orden en que se documentan en el
# enunciado. NUNCA se agregan aqui 'estado', 'segundo' ni 'episodio_id': el
# primero es la etiqueta (fuga de información) y los otros dos son
# identificadores de la fila, no telemetria de la GPU.
SEÑALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]

# Tamaño de ventana usado por defecto al entrenar (segundos).
TAM_VENTANA_DEFAULT = 30

# Minimo de lecturas que se necesitan para poder calcular una desviación
# estandar razonablemente confiable. Se usa tanto para descartar ventanas
# incompletas al entrenar como para validar la entrada de la API (B.2).
MIN_LECTURAS = 10


def _stats_señal(valores: np.ndarray) -> dict[str, float]:
    """Estadisticas de una sola señal dentro de una ventana."""
    return {
        "mean": float(np.mean(valores)),
        "std": float(np.std(valores, ddof=0)),
        "min": float(np.min(valores)),
        "max": float(np.max(valores)),
        "range": float(np.max(valores) - np.min(valores)),
    }


def calcular_features(ventana: pd.DataFrame) -> dict[str, float]:
    """
    Convierte una ventana de telemetria cruda (una fila por segundo, columnas
    = SEÑALES) en un diccionario de features numericas para un clasificador.

    Por cada señal se calculan: media, desviación estandar, minimo, maximo y
    rango (max - min). La desviación estandar delata inestabilidad (p.ej.
    clock_mhz oscilando en falla_alimentacion); el rango de potencia delata
    lo mismo desde otro angulo. Ademas se agrega el total de ecc_errors de la
    ventana, que delata degradacion_memoria (errores que se acumulan).

    Parametros
    ----------
    ventana: DataFrame con al menos las columnas de SEÑALES, una fila por
        segundo de telemetria. Debe pertenecer a un unico episodio (nunca se
        mezclan segundos de dos episodios distintos en una misma ventana).

    Retorna
    -------
    dict con una entrada por feature, en un orden estable (ver
    `nombres_features()`).
    """
    faltantes = [c for c in SEÑALES if c not in ventana.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas de señal en la ventana: {faltantes}")

    features: dict[str, float] = {}
    for señal in SEÑALES:
        valores = ventana[señal].to_numpy(dtype=float)
        for nombre_stat, valor in _stats_señal(valores).items():
            features[f"{señal}_{nombre_stat}"] = valor

    # Total de errores ECC en la ventana (delata degradacion_memoria: no es
    # lo mismo un error suelto que una ventana entera acumulando errores).
    features["ecc_errors_total"] = float(ventana["ecc_errors"].sum())

    return features


def nombres_features() -> list[str]:
    """
    Orden canonico de las columnas de features. train.py y api.py DEBEN usar
    esta misma lista (nunca `dict.keys()` de un dict distinto) para que el
    DataFrame que entra al modelo tenga siempre las columnas en el mismo
    orden, tanto al entrenar como al predecir.
    """
    columnas = []
    for señal in SEÑALES:
        for stat in ("mean", "std", "min", "max", "range"):
            columnas.append(f"{señal}_{stat}")
    columnas.append("ecc_errors_total")
    return columnas


def features_a_dataframe(lista_de_features: list[dict[str, float]]) -> pd.DataFrame:
    """Empaqueta una lista de dicts de features en un DataFrame con columnas
    en el orden canonico (`nombres_features()`), listo para pipeline.predict.
    """
    return pd.DataFrame(lista_de_features, columns=nombres_features())


def crear_ventanas(
    df: pd.DataFrame,
    tam_ventana: int = TAM_VENTANA_DEFAULT,
) -> list[dict]:
    """
    Parte la telemetria cruda de varios episodios en ventanas de
    `tam_ventana` segundos, SIN mezclar segundos de episodios distintos.

    Se asume que `df` tiene columnas: episodio_id, segundo, SEÑALES..., y
    opcionalmente 'estado' (si esta presente, cada ventana hereda el estado
    de su episodio -- esa es la y del entrenamiento).

    Ventanas incompletas (el remanente al final de un episodio que no llega
    a `tam_ventana` segundos) se descartan: mezclarían menos información que
    el resto y no aportan una comparación justa.

    Retorna una lista de dicts: {"episodio_id", "ventana_idx", "datos"
    (DataFrame con las señales de esos `tam_ventana` segundos), "estado"
    (si estaba presente en la entrada, si no None)}.
    """
    ventanas = []
    tiene_estado = "estado" in df.columns

    for episodio_id, grupo in df.groupby("episodio_id", sort=True):
        grupo = grupo.sort_values("segundo").reset_index(drop=True)
        n_segundos = len(grupo)
        n_ventanas = n_segundos // tam_ventana

        for i in range(n_ventanas):
            inicio = i * tam_ventana
            fin = inicio + tam_ventana
            trozo = grupo.iloc[inicio:fin]

            estado = trozo["estado"].iloc[0] if tiene_estado else None
            if tiene_estado and trozo["estado"].nunique() != 1:
                # No deberia pasar nunca (un episodio = un solo estado), pero
                # se valida explicitamente para no entrenar con etiquetas
                # ambiguas si algun dia el dataset cambia.
                raise ValueError(
                    f"Episodio {episodio_id} tiene mas de un estado dentro "
                    "de una misma ventana; revisa el dataset."
                )

            ventanas.append(
                {
                    "episodio_id": episodio_id,
                    "ventana_idx": i,
                    "datos": trozo[SEÑALES].reset_index(drop=True),
                    "estado": estado,
                }
            )

    return ventanas

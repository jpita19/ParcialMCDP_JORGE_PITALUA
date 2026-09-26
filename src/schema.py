"""
schema.py
=========
Contrato de datos para la telemetria cruda, usando pandera.

Se valida ANTES de ventanear/entrenar: si la telemetria no cumple el
contrato, se rechaza aqui y nunca llega al pipeline de features. Esto evita
que, por ejemplo, un sensor dañado (temperatura negativa, potencia negativa,
un estado mal escrito) contamine el entrenamiento.

Los rangos fisicos estan pensados para una GPU NVIDIA L40 (TDP ~300W):
- temp_c: una GPU no opera por debajo de temperatura ambiente ni por encima
  de ~120 C (a esa temperatura ya se apagaria por proteccion termica).
- power_w: el consumo nunca es negativo; se deja margen sobre el TDP nominal
  para picos transitorios.
- util_pct: es un porcentaje, 0-100.
- clock_mhz: positivo, con margen amplio (el throttling puede bajarlo mucho
  y el boost puede subirlo bastante).
- ecc_errors: un conteo, nunca negativo.
- estado: debe ser uno de los cuatro estados validos del enunciado.

Nota para la sustentacion: si el proveedor empieza a mandar temp_c en
Fahrenheit, la mayoria de esos valores (p.ej. 100-200 F) caen FUERA del
rango fisico en Celsius que exige `TEMP_C_MAX` y el contrato los rechaza. El
contrato no "sabe" que la unidad cambio, pero si nota que el numero ya no es
una temperatura Celsius plausible para una GPU. El unico hueco es la franja
en la que un valor en Fahrenheit coincide por accidente con un rango
Celsius valido (p.ej. 32-120 F ~ 0-49 C); ahi el contrato NO lo atraparia
porque el numero sigue siendo fisicamente plausible como Celsius. Por eso en
un caso real conviene ademas validar la unidad en el contrato de la fuente
(metadata), no solo el rango del valor.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ESTADOS_VALIDOS = [
    "normal",
    "sobrecalentamiento",
    "degradacion_memoria",
    "falla_alimentacion",
]

# Rangos fisicos razonables para una GPU NVIDIA L40.
TEMP_C_MIN, TEMP_C_MAX = 0.0, 120.0
POWER_W_MIN, POWER_W_MAX = 0.0, 500.0
UTIL_PCT_MIN, UTIL_PCT_MAX = 0.0, 100.0
CLOCK_MHZ_MIN, CLOCK_MHZ_MAX = 0.0, 3500.0
ECC_ERRORS_MIN = 0


class TelemetriaSchema(pa.DataFrameModel):
    """Contrato de datos para una fila de telemetria cruda (un segundo)."""

    episodio_id: Series[int] = pa.Field(ge=0, coerce=True)
    segundo: Series[int] = pa.Field(ge=0, coerce=True)
    temp_c: Series[float] = pa.Field(
        ge=TEMP_C_MIN, le=TEMP_C_MAX, coerce=True, nullable=False
    )
    power_w: Series[float] = pa.Field(
        ge=POWER_W_MIN, le=POWER_W_MAX, coerce=True, nullable=False
    )
    util_pct: Series[float] = pa.Field(
        ge=UTIL_PCT_MIN, le=UTIL_PCT_MAX, coerce=True, nullable=False
    )
    clock_mhz: Series[float] = pa.Field(
        ge=CLOCK_MHZ_MIN, le=CLOCK_MHZ_MAX, coerce=True, nullable=False
    )
    ecc_errors: Series[int] = pa.Field(ge=ECC_ERRORS_MIN, coerce=True)
    estado: Series[str] = pa.Field(isin=ESTADOS_VALIDOS, coerce=True)

    class Config:
        strict = False  # permite columnas extra sin romper la validación
        coerce = True


def validar_telemetria(df: pd.DataFrame) -> pd.DataFrame:
    """Valida un DataFrame de telemetria cruda contra el contrato (estricto).

    Lanza `pandera.errors.SchemaErrors` (con el detalle de cada fila y
    columna que incumple) si algo no cumple el contrato. Se usa
    `lazy=True` para reportar TODOS los errores de una vez, no solo el
    primero, lo cual es mucho mas util para depurar un dataset grande.

    Este es el modo que usa la API: una ventana que no cumple el contrato se
    rechaza completa, sin intentar "arreglarla".
    """
    return TelemetriaSchema.validate(df, lazy=True)


def validar_y_limpiar_telemetria(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el DataFrame y, si hay filas que violan el contrato (p.ej. un
    sensor de potencia que por un instante reporta un valor negativo durante
    una falla de alimentación), las pone en cuarentena: las reporta por
    consola y las descarta, en vez de abortar TODO el entrenamiento por un
    puñado de lecturas dañadas.

    "Datos corruptos no deben entrar al pipeline" no siempre significa
    "revienta si hay una sola fila mala" -- en un dataset real de miles de
    filas, la respuesta operativa suele ser filtrar la fila dañada y seguir,
    dejando constancia de qué se descartó y por qué. Por eso este modo es el
    que usa `train.py` sobre el CSV crudo, mientras que la API usa el modo
    estricto `validar_telemetria` (una ventana que llega mal formada a la
    API se rechaza entera: no hay "más datos" con los que reemplazarla).

    Retorna el DataFrame limpio (ya validado en modo estricto tras quitar
    las filas problemáticas).
    """
    try:
        return TelemetriaSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        indices_malos = sorted(set(exc.failure_cases["index"].dropna().astype(int)))
        print(
            f"[schema] Se encontraron {len(indices_malos)} fila(s) que "
            "incumplen el contrato de datos; se descartan antes de entrenar:"
        )
        detalle = exc.failure_cases[["index", "column", "check", "failure_case"]]
        print(detalle.to_string(index=False))

        df_limpio = df.drop(index=indices_malos).reset_index(drop=True)
        # Se vuelve a validar en modo estricto: si sigue habiendo problemas
        # (es decir, si la cuarentena no fue suficiente), se deja que la
        # excepción se propague en vez de entrenar sobre datos sucios.
        return TelemetriaSchema.validate(df_limpio, lazy=True)

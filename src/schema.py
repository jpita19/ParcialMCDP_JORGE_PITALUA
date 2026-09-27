"""
schema.py
=========
Contrato de datos para la telemetria cruda, usando pandera.

Se valida ANTES de calcular features, tanto al entrenar (train.py) como en
cada ventana que llega a la API (api.py): una lectura que no cumple el
contrato se descarta aqui y nunca llega al pipeline de features. Esto evita
que, por ejemplo, un sensor dañado (temperatura negativa, potencia negativa,
un estado mal escrito) contamine el entrenamiento o una predicción.

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
rango fisico en Celsius que exige `TEMP_C_MAX` y el contrato los descarta
(si toda la ventana viene asi, la API la rechaza con 422 por quedarse sin
lecturas validas suficientes). El
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


class LecturaSchema(pa.DataFrameModel):
    """Contrato de datos para las señales de una lectura (un segundo), sin
    identificadores ni etiqueta. Es lo que valida la API sobre cada ventana
    que recibe, y la base del contrato del CSV de entrenamiento."""

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

    class Config:
        strict = False  # permite columnas extra sin romper la validación
        coerce = True


class TelemetriaSchema(LecturaSchema):
    """Contrato de datos para una fila de telemetria cruda del CSV de
    entrenamiento: las señales de `LecturaSchema` mas los identificadores
    del episodio y la etiqueta."""

    episodio_id: Series[int] = pa.Field(ge=0, coerce=True)
    segundo: Series[int] = pa.Field(ge=0, coerce=True)
    estado: Series[str] = pa.Field(isin=ESTADOS_VALIDOS, coerce=True)


def poner_en_cuarentena(
    df: pd.DataFrame, schema: type[pa.DataFrameModel] = TelemetriaSchema
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida `df` contra `schema` y descarta las filas que violan el
    contrato (p.ej. un sensor de potencia que por un instante reporta un
    valor negativo durante una falla de alimentación), en vez de rechazar
    TODO el DataFrame por un puñado de lecturas dañadas.

    "Datos corruptos no deben entrar al pipeline" no siempre significa
    "revienta si hay una sola fila mala": la respuesta operativa suele ser
    filtrar la fila dañada y seguir, dejando constancia de qué se descartó y
    por qué. Lo usan los dos lados, con el mismo criterio:
    - `train.py` (via `validar_y_limpiar_telemetria`) sobre el CSV crudo.
    - `api.py` con `LecturaSchema` sobre cada ventana que recibe: una
      ventana de `falla_alimentacion` con una lectura de potencia negativa
      sigue siendo una ventana de `falla_alimentacion`, no una petición
      inválida.

    Se usa `lazy=True` para recoger TODAS las violaciones de una vez, no
    solo la primera.

    Retorna (df_limpio, fallas): el DataFrame ya validado en modo estricto
    tras quitar las filas problemáticas, y el detalle de qué fila violó qué
    regla (vacío si no hubo ninguna).
    """
    try:
        return schema.validate(df, lazy=True), pd.DataFrame()
    except pa.errors.SchemaErrors as exc:
        fallas = exc.failure_cases[["index", "column", "check", "failure_case"]]
        indices_malos = sorted(set(fallas["index"].dropna().astype(int)))
        df_limpio = df.drop(index=indices_malos).reset_index(drop=True)
        # Se vuelve a validar en modo estricto: si sigue habiendo problemas
        # (es decir, si la cuarentena no fue suficiente, p.ej. falta una
        # columna entera), se deja que la excepción se propague en vez de
        # seguir con datos sucios.
        return schema.validate(df_limpio, lazy=True), fallas


def validar_y_limpiar_telemetria(df: pd.DataFrame) -> pd.DataFrame:
    """Cuarentena sobre el CSV crudo de entrenamiento (ver
    `poner_en_cuarentena`), reportando por consola qué filas se descartaron
    y por qué."""
    df_limpio, fallas = poner_en_cuarentena(df, TelemetriaSchema)
    if not fallas.empty:
        print(
            f"[schema] Se encontraron {len(df) - len(df_limpio)} fila(s) que "
            "incumplen el contrato de datos; se descartan antes de entrenar:"
        )
        print(fallas.to_string(index=False))
    return df_limpio

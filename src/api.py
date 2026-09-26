"""
api.py
======
API de FastAPI que expone el detector de fallas de GPU.

Flujo interno del endpoint principal (POST /predecir), tal como pide el
enunciado:

    ventana (JSON) -> validar (BaseModel) -> calcular features (features.py)
        -> pipeline.predict -> responder

`features.py` es el MISMO modulo que usa `train.py`: aqui nunca se
recalculan a mano las estadisticas de la ventana, para garantizar que el
modelo ve en producción exactamente el mismo tipo de features que vio al
entrenar.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Se agrega el directorio de este archivo a sys.path para poder hacer
# `import features` / `import schema` sin importar si la API se levanta como
# `uvicorn api:app --app-dir src` o de alguna otra forma.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat  # noqa: E402
from schema import (  # noqa: E402
    CLOCK_MHZ_MAX,
    CLOCK_MHZ_MIN,
    ECC_ERRORS_MIN,
    POWER_W_MAX,
    POWER_W_MIN,
    TEMP_C_MAX,
    TEMP_C_MIN,
    UTIL_PCT_MAX,
    UTIL_PCT_MIN,
)

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_MODELO = RUTA_BASE / "models" / "modelo.joblib"


# ---------------------------------------------------------------------------
# Esquemas de entrada/salida (Parte B.2: validación en la puerta)
# ---------------------------------------------------------------------------
class Lectura(BaseModel):
    """Una lectura de telemetria de un segundo. Los mismos rangos fisicos
    que exige el contrato de pandera (schema.py) se validan aqui, del lado
    de la API, para que una lectura mal formada nunca llegue al modelo."""

    temp_c: float = Field(ge=TEMP_C_MIN, le=TEMP_C_MAX, description="Temperatura en °C")
    power_w: float = Field(ge=POWER_W_MIN, le=POWER_W_MAX, description="Consumo en vatios")
    util_pct: float = Field(ge=UTIL_PCT_MIN, le=UTIL_PCT_MAX, description="Utilización en %")
    clock_mhz: float = Field(ge=CLOCK_MHZ_MIN, le=CLOCK_MHZ_MAX, description="Reloj en MHz")
    ecc_errors: int = Field(ge=ECC_ERRORS_MIN, description="Errores ECC en ese segundo")


class VentanaTelemetria(BaseModel):
    """Una ventana = varias lecturas consecutivas de UNA sola GPU/episodio.

    Se exige un minimo de `feat.MIN_LECTURAS` lecturas: con menos no se puede
    calcular una desviación estándar confiable (con 2 o 3 puntos, el ruido
    domina la estadística y el modelo recibiría features poco informativas).
    """

    lecturas: Annotated[list[Lectura], Field(min_length=feat.MIN_LECTURAS)]


class PrediccionResponse(BaseModel):
    estado_predicho: str
    confianza: float


# ---------------------------------------------------------------------------
# Carga del modelo (el pipeline YA entrenado; la API nunca entrena)
# ---------------------------------------------------------------------------
_artefacto = None  # se carga al arrancar la API, ver `lifespan` abajo


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artefacto
    if not RUTA_MODELO.exists():
        raise RuntimeError(
            f"No se encontró el modelo en {RUTA_MODELO}. Corre "
            "`python src/train.py` antes de levantar la API "
            "(el contenedor debe incluir el modelo ya entrenado)."
        )
    _artefacto = joblib.load(RUTA_MODELO)
    print(f"Modelo cargado desde {RUTA_MODELO}. Clases: {_artefacto['clases']}")
    yield
    _artefacto = None


app = FastAPI(
    title="Detector de fallas de GPU",
    description="Clasifica ventanas de telemetria de GPUs NVIDIA L40 en "
    "normal / sobrecalentamiento / degradacion_memoria / falla_alimentacion.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/salud")
def salud() -> dict:
    """Health check simple, util para saber si el contenedor ya cargó el modelo."""
    return {"status": "ok", "modelo_cargado": _artefacto is not None}


@app.post("/predecir", response_model=PrediccionResponse)
def predecir(ventana: VentanaTelemetria) -> PrediccionResponse:
    """
    Recibe una ventana de telemetria (lista de lecturas de un segundo cada
    una, todas de la MISMA GPU/episodio) y devuelve el estado predicho.

    La validación de forma y rangos ya la hizo Pydantic al parsear el
    request (B.2): si llega aquí, la ventana tiene todos los campos, del
    tipo correcto, en rango físico, y con al menos `MIN_LECTURAS` lecturas.
    """
    if _artefacto is None:
        raise HTTPException(status_code=503, detail="El modelo todavía no está cargado.")

    # Ventana -> DataFrame con las mismas columnas crudas que espera
    # features.py (el mismo módulo que usó train.py).
    df_ventana = pd.DataFrame([lectura.model_dump() for lectura in ventana.lecturas])

    features_dict = feat.calcular_features(df_ventana)
    X = feat.features_a_dataframe([features_dict])
    # Se reordena/filtra explícitamente a las columnas con las que se
    # entrenó el modelo, por si el artefacto viene de una versión de
    # features.py con un orden distinto.
    X = X[_artefacto["feature_names"]]

    pipeline = _artefacto["pipeline"]
    probabilidades = pipeline.predict_proba(X)[0]
    idx_max = probabilidades.argmax()
    estado_predicho = pipeline.classes_[idx_max]
    confianza = float(probabilidades[idx_max])

    return PrediccionResponse(estado_predicho=estado_predicho, confianza=round(confianza, 4))

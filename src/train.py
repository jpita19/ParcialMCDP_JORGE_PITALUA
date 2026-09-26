"""
train.py
========
Entrena el clasificador de estados de GPU y serializa el pipeline completo.

Flujo (el mismo que describe el enunciado):

    telemetria cruda -> validar (pandera) -> ventanas -> features -> entrenar -> serializar

Uso:
    python src/train.py
    python src/train.py --tam-ventana 30 --test-size 0.25

El modelo resultante (`models/modelo.joblib`) es exactamente lo que la API
carga en `api.py`: un pipeline de sklearn (escalado + clasificador) MAS los
metadatos necesarios para que la API sepa qué features calcular y en qué
orden, sin tener que adivinar nada.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features as feat
from schema import validar_y_limpiar_telemetria

RANDOM_STATE = 42

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS_DEFAULT = RUTA_BASE / "data" / "telemetria_publica.csv"
RUTA_MODELO_DEFAULT = RUTA_BASE / "models" / "modelo.joblib"


def cargar_y_validar(ruta_csv: Path) -> pd.DataFrame:
    """Carga la telemetria cruda y la valida contra el contrato de datos
    (Parte A.2). Las filas corruptas se ponen en cuarentena (ver
    schema.validar_y_limpiar_telemetria); nunca llegan al pipeline de
    features."""
    print(f"Cargando telemetria cruda de: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print(f"  {len(df)} filas, {df['episodio_id'].nunique()} episodios.")

    df_valido = validar_y_limpiar_telemetria(df)
    print(f"  {len(df_valido)} filas validas tras el contrato de datos.")
    return df_valido


def separar_episodios_train_test(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[list, list]:
    """Divide los EPISODIOS (no las filas ni las ventanas) en train/test,
    estratificando por estado. Esto evita fuga de información: dos ventanas
    del mismo episodio_id nunca quedan una en train y otra en test, porque
    son casi identicas entre si (telemetria consecutiva del mismo evento) y
    eso inflaria artificialmente la accuracy de validación.
    """
    estado_por_episodio = df.groupby("episodio_id")["estado"].first()
    ids_train, ids_test = train_test_split(
        estado_por_episodio.index.to_numpy(),
        test_size=test_size,
        stratify=estado_por_episodio.to_numpy(),
        random_state=random_state,
    )
    return sorted(ids_train), sorted(ids_test)


def construir_dataset(df: pd.DataFrame, tam_ventana: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Ventanea (Parte A.1) y calcula features (usando features.py, el mismo
    modulo que usara la API) para un DataFrame de telemetria cruda.

    Retorna (X, y, episodio_ids) donde X ya tiene las columnas en el orden
    canonico de `features.nombres_features()`.
    """
    ventanas = feat.crear_ventanas(df, tam_ventana=tam_ventana)
    filas_features = [feat.calcular_features(v["datos"]) for v in ventanas]
    X = feat.features_a_dataframe(filas_features)
    y = np.array([v["estado"] for v in ventanas])
    episodio_ids = np.array([v["episodio_id"] for v in ventanas])
    return X, y, episodio_ids


def construir_pipeline() -> Pipeline:
    """Pipeline completo: escalado + clasificador. Se serializa ENTERO (no
    solo el clasificador) para que el preprocesamiento viaje siempre junto
    con el modelo, tal como pide el enunciado (A.3)."""
    return Pipeline(
        steps=[
            ("escalador", StandardScaler()),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=RUTA_DATOS_DEFAULT)
    parser.add_argument("--modelo-salida", type=Path, default=RUTA_MODELO_DEFAULT)
    parser.add_argument("--tam-ventana", type=int, default=feat.TAM_VENTANA_DEFAULT)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    # 1) Cargar y validar (Parte A.2) ------------------------------------
    df = cargar_y_validar(args.datos)

    # 2) Separar episodios en train/test ANTES de ventanear, para que la
    #    validacion sea honesta (sin fuga entre ventanas del mismo episodio)
    ids_train, ids_test = separar_episodios_train_test(
        df, test_size=args.test_size, random_state=RANDOM_STATE
    )
    print(f"\nEpisodios -> train: {len(ids_train)}, test: {len(ids_test)}")

    df_train = df[df["episodio_id"].isin(ids_train)]
    df_test = df[df["episodio_id"].isin(ids_test)]

    # 3) Ventanear + features (Parte A.1), por separado en train y test ---
    X_train, y_train, _ = construir_dataset(df_train, args.tam_ventana)
    X_test, y_test, _ = construir_dataset(df_test, args.tam_ventana)
    print(f"Ventanas -> train: {len(X_train)}, test: {len(X_test)} (tam_ventana={args.tam_ventana}s)")

    # 4) Entrenar y evaluar (validacion honesta, episodios nunca vistos) --
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n=== Evaluación sobre episodios de test (nunca vistos en train) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    etiquetas = sorted(set(y_train) | set(y_test))
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=etiquetas),
        index=etiquetas,
        columns=etiquetas,
    ))

    # 5) Reentrenar el pipeline final sobre TODOS los episodios -----------
    #    Con solo 48 episodios en total, conviene que el modelo que se
    #    empaqueta en la imagen use el 100% de los datos disponibles: la
    #    evaluación honesta ya se hizo arriba con el split por episodio.
    print("\nReentrenando el modelo final sobre todos los episodios disponibles...")
    X_full, y_full, _ = construir_dataset(df, args.tam_ventana)
    pipeline_final = construir_pipeline()
    pipeline_final.fit(X_full, y_full)

    # 6) Serializar el pipeline completo + metadatos (Parte A.3) ----------
    args.modelo_salida.parent.mkdir(parents=True, exist_ok=True)
    artefacto = {
        "pipeline": pipeline_final,
        "feature_names": feat.nombres_features(),
        "tam_ventana_entrenamiento": args.tam_ventana,
        "clases": sorted(pipeline_final.classes_.tolist()),
    }
    joblib.dump(artefacto, args.modelo_salida)
    print(f"\nModelo serializado en: {args.modelo_salida}")


if __name__ == "__main__":
    main()

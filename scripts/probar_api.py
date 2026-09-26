"""
probar_api.py
=============
Prueba manual del endpoint /predecir contra una API ya levantada (con
Docker o con uvicorn local), usando ventanas REALES del dataset publico
-- una por cada uno de los 4 estados -- mas una ventana corta para
confirmar que la validacion de la API la rechaza.

Uso (con la API corriendo en otra terminal, ver README "Como correrlo"):

    python scripts/probar_api.py
    python scripts/probar_api.py --url http://localhost:8000

Solo usa la libreria estandar (urllib) para no necesitar `requests` como
dependencia extra del proyecto.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_BASE / "data" / "telemetria_publica.csv"

SEÑALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
ESTADOS = ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]


def post_predecir(url: str, lecturas: list[dict]) -> tuple[int, dict]:
    body = json.dumps({"lecturas": lecturas}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/predecir", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--tam-ventana", type=int, default=30)
    args = parser.parse_args()

    # Chequeo rapido de que la API este arriba.
    try:
        with urllib.request.urlopen(f"{args.url}/salud", timeout=5) as resp:
            salud = json.loads(resp.read())
            print(f"[salud] {salud}\n")
    except (urllib.error.URLError, ConnectionRefusedError) as exc:
        print(f"No se pudo conectar a {args.url}. ¿Está la API corriendo? ({exc})")
        sys.exit(1)

    df = pd.read_csv(RUTA_DATOS)

    print("=== Una ventana real por cada estado ===")
    aciertos = 0
    for estado in ESTADOS:
        ventana = df[df["estado"] == estado].iloc[: args.tam_ventana]
        lecturas = ventana[SEÑALES].to_dict(orient="records")

        status, respuesta = post_predecir(args.url, lecturas)
        if status == 200:
            ok = "OK  " if respuesta["estado_predicho"] == estado else "FALLO"
            if respuesta["estado_predicho"] == estado:
                aciertos += 1
            print(
                f"{ok} real={estado:22s} -> predicho={respuesta['estado_predicho']:22s} "
                f"confianza={respuesta['confianza']}"
            )
        else:
            print(f"FALLO real={estado:22s} -> HTTP {status}: {respuesta}")

    print(f"\n{aciertos}/{len(ESTADOS)} ventanas clasificadas correctamente.\n")

    print("=== Ventana corta (3 lecturas) -- debe rechazarse con 422 ===")
    ventana_corta = df.iloc[:3][SEÑALES].to_dict(orient="records")
    status, respuesta = post_predecir(args.url, ventana_corta)
    esperado = "OK  " if status == 422 else "FALLO"
    print(f"{esperado} HTTP {status}: {respuesta}")


if __name__ == "__main__":
    main()

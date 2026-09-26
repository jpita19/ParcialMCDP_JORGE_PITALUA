# Detector de fallas de GPU — del dato al contenedor

Servicio que clasifica el estado de una GPU NVIDIA L40 (`normal`,
`sobrecalentamiento`, `degradacion_memoria`, `falla_alimentacion`) a partir
de una ventana de telemetría, empaquetado como una API FastAPI dentro de un
contenedor Docker.

## Estructura del repositorio (objetivo)

```
src/
  features.py   # ventaneo + cálculo de features (compartido por train y api)
  schema.py     # contrato de datos con pandera
  train.py      # entrena y serializa el pipeline completo
  api.py        # API de FastAPI (POST /predecir)
data/
  telemetria_publica.csv   # dataset de entrenamiento
models/
  modelo.joblib  # pipeline entrenado y serializado (se genera con train.py)
Dockerfile
.dockerignore
requirements.txt
```

Este README se va llenando a medida que se construye cada pieza; por ahora
solo está listo el dato crudo y su análisis exploratorio.

## Análisis exploratorio de los datos (EDA)

Antes de escribir una sola línea de `schema.py` o `features.py` vale la pena
mirar qué hay realmente en `telemetria_publica.csv`, porque de ahí salen casi
todas las decisiones de las siguientes secciones.

**Tamaño y balance:**
- 14,400 filas = 48 episodios × 300 segundos cada uno, sin excepciones (todo
  episodio tiene exactamente 300 filas).
- Los 4 estados están perfectamente balanceados: 12 episodios de cada uno
  (`normal`, `sobrecalentamiento`, `degradacion_memoria`, `falla_alimentacion`).
  Esto importa porque significa que no hace falta lidiar con clases minoritarias
  raras — aun así se dejó `class_weight="balanced"` en el clasificador como
  buena práctica, por si el dataset cambia.
- Ninguna columna tiene valores nulos.

**Calidad de los datos — lo que encontramos y por qué importa para el contrato:**
- 3 filas (de 14,400) tienen `power_w` negativo (-13.5, -3.9, -18.1 W), las
  tres dentro de episodios `falla_alimentacion`. Es información realista (un
  sensor de potencia puede leer basura cuando la alimentación se pone
  inestable), pero viola cualquier contrato físico razonable ("la potencia
  es positiva"). Esto se convirtió directamente en la lógica de cuarentena de
  `schema.py` (ver más abajo): no tiene sentido descartar los otros 297
  segundos sanos del episodio por 1-2 lecturas dañadas.

**Separabilidad de las clases por señal** (media ± desviación estándar,
calculado por segundo, no por ventana):

| estado | temp_c | power_w | util_pct | clock_mhz | ecc_errors |
|---|---|---|---|---|---|
| normal | 54.9 ± 4.0 | 200.1 ± 20.0 | 69.8 ± 12.0 | 2400.7 ± 29.7 | 0.02 ± 0.16 |
| sobrecalentamiento | **88.0 ± 3.0** | 290.3 ± 10.1 | 94.9 ± 3.7 | 2098.6 ± 80.4 | 0.10 ± 0.31 |
| degradacion_memoria | 60.0 ± 5.0 | 209.1 ± 25.1 | 67.8 ± 13.9 | 2378.9 ± 39.8 | **5.01 ± 2.27** |
| falla_alimentacion | 50.0 ± 8.0 | 140.5 ± 45.1 | 54.5 ± 24.0 | 1798.7 ± **204.4** | 0.20 ± 0.46 |

De esta tabla salen directamente las intuiciones de qué feature delata qué
falla (las mismas que pide el enunciado en A.1):
- **`sobrecalentamiento`** salta por `temp_c` — ~88°C de media, sin solape
  con ninguna otra clase (todas las demás rondan 50-60°C).
- **`degradacion_memoria`** salta por `ecc_errors` — media de 5 errores por
  segundo contra ≤0.2 en el resto. Por eso se agregó explícitamente
  `ecc_errors_total` como feature (A.1 lo pide: "los errores ECC delatan la
  memoria").
- **`falla_alimentacion`** salta por la **desviación estándar** de
  `clock_mhz` (204.4, entre 2.5x y 7x más alta que las otras tres clases) y,
  en menor medida, de `power_w` (45.1). No es la media la que delata esta
  falla (de hecho tiene la temperatura y potencia promedio más *bajas* de
  las cuatro) sino la inestabilidad — de ahí que A.1 insista en calcular
  `std` de cada señal, no solo la media.
- **`normal`** no "salta" por ningún lado: todas sus estadísticas son
  moderadas y estables, lo cual es justamente la definición de operación
  sana.

Esta tabla es también la razón por la que, más adelante, el modelo entrenado
llega a 100% de accuracy en el test: las clases casi no se solapan en el
espacio de las features que se calculan a partir de estas señales.

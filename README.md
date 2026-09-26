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

## Validación de datos (`src/schema.py`)

Contrato de pandera sobre la telemetría cruda: rangos físicos para una L40
(`temp_c` 0–120°C, `power_w` 0–500W, `util_pct` 0–100%, `clock_mhz`
0–3500MHz, `ecc_errors` ≥ 0) y `estado` restringido a los cuatro valores
válidos del enunciado. El límite superior de `power_w` (500W) no sale de
ningún dato duro — es criterio de ingeniería: la L40 tiene un TDP nominal de
~300W, y se deja margen para picos transitorios sin abrir la puerta a
valores absurdos. `Config.strict = False` permite columnas extra en el
DataFrame sin romper la validación (para no ser más rígidos de lo necesario
con el resto del esquema).

**Dos modos de validación, a propósito:**
- `validar_telemetria()` (estricto) — revienta con `SchemaErrors` ante
  cualquier violación. Es el que usa la API: una ventana mal formada de un
  cliente se rechaza completa, porque no hay "más datos" con los que
  arreglarla.
- `validar_y_limpiar_telemetria()` (cuarentena) — pensado para el CSV de
  entrenamiento. El dataset público trae 3 filas (de 14,400, ver EDA arriba)
  con `power_w` negativo por un glitch de sensor durante
  `falla_alimentacion`. Abortar todo el entrenamiento por un 0.02% de filas
  dañadas no tiene sentido operativo: esta función reporta por consola
  exactamente qué filas violan qué regla, las descarta, y valida de nuevo el
  resto en modo estricto (si la cuarentena no alcanza, ahí sí se propaga el
  error).

## El pipeline de features (`src/features.py`, Parte A.1)

```
telemetria cruda (validada)  ->  ventanas de 30s  ->  features  ->  (train.py | api.py)
```

Este módulo lo importan **tanto `train.py` como `api.py`**: es el único lugar
donde se calculan features, para que el modelo nunca reciba algo distinto en
producción de lo que vio al entrenar (la trampa #2 del enunciado).

**Ventaneo (`crear_ventanas`):** cada `episodio_id` se corta en ventanas de
30 segundos **no solapadas** — segundos 0-29, 30-59, 60-89, etc. Cada segundo
del episodio cae en una sola ventana. Se prefirió esto sobre un *sliding
window* solapado (por ejemplo, avanzando de a 5 segundos) porque dos
ventanas solapadas comparten la mayoría de sus segundos y son casi idénticas
entre sí: "infla" el número de muestras sin agregar información real, y si
alguna vez se separara train/test por ventana en lugar de por episodio,
ventanas casi iguales podrían quedar una en cada lado e inflar el accuracy
de forma artificial. Con 48 episodios × 10 ventanas de 30s salen 480
ventanas en total — suficiente para este dataset (accuracy 100% en test, ver
más abajo). Las ventanas incompletas al final de un episodio se descartan.
Si un episodio terminara con más de un `estado` dentro de una misma ventana
(no debería pasar nunca), la función lanza un error en vez de asignar el
estado mayoritario en silencio.

**Features (`calcular_features`):** por cada señal (`temp_c`, `power_w`,
`util_pct`, `clock_mhz`, `ecc_errors`) se calculan `mean`, `std`, `min`,
`max` y `range` (max−min); más un feature extra, `ecc_errors_total` (la
suma, no solo el promedio), porque lo que delata `degradacion_memoria` es
que los errores ECC *se acumulan* en la ventana. 26 features en total.

**Sin fuga de información:** `estado`, `segundo` y `episodio_id` nunca son
features — están explícitamente fuera de la lista `SEÑALES`. El orden de las
26 columnas está fijo en `nombres_features()`, y tanto `train.py` como
`api.py` arman su DataFrame de entrada llamando a esa misma función (nunca
`dict.keys()` de un dict cualquiera), para que el orden de columnas que ve
el modelo sea siempre el mismo.

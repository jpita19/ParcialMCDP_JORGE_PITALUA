# Parcial — Detector de fallas de GPU: del dato al contenedor

## El contexto

Trabajas en el equipo de infraestructura de un centro de datos que entrena modelos de IA
sobre **GPUs NVIDIA L40**. Cada GPU emite **telemetría** cada segundo: temperatura,
potencia, utilización, frecuencia de reloj y errores de memoria. Cuando una GPU empieza a
degradarse, la telemetría cambia de forma característica *antes* del fallo catastrófico —
pero nadie puede vigilar miles de GPUs mirando gráficas.

Tu misión: construir un **servicio de detección de fallas** completo. No basta con entrenar
un modelo: hay que empaquetarlo en una API dentro de un contenedor Docker, tal como se
despliega en producción. Este parcial junta TODO lo del curso — validación, pipeline,
serialización, API y Docker — en un solo entregable.

## Los cuatro estados a clasificar

| Estado | Qué le pasa a la GPU |
|---|---|
| normal | operación sana |
| sobrecalentamiento | temperatura crítica, throttling del reloj |
| degradacion_memoria | errores ECC de memoria en aumento |
| falla_alimentacion | potencia y reloj inestables |

## Los datos

**telemetria_publica.csv** — tu dataset de trabajo. Telemetría cruda, **una fila por
segundo**:

| Columna | Significado |
|---|---|
| episodio_id | identifica un tramo continuo de telemetría de una GPU |
| segundo | el segundo dentro del episodio (0, 1, 2, ...) |
| temp_c | temperatura en grados Celsius |
| power_w | consumo en vatios |
| util_pct | utilización en % |
| clock_mhz | frecuencia del reloj en MHz |
| ecc_errors | errores de memoria ECC ese segundo |
| estado | **la etiqueta**: el estado de la GPU durante ese episodio |

Cada episodio_id es un tramo de 300 segundos completo en un solo estado. La columna
estado es tu verdad de terreno.

---

## Parte A — El modelo (lo que ya sabes hacer)

### A.1 — El pipeline de features (el orden importa)

No se entrena sobre la telemetría cruda. El flujo es siempre:

```
telemetria cruda  ->  ventanas  ->  features por ventana  ->  entrenar
```

- **Ventanear:** parte cada episodio_id en ventanas de N segundos (ej. 30). Nunca mezcles
  segundos de episodios distintos en una ventana.
- **Calcular features:** de cada ventana, saca estadísticas — media, desviación, min, max de
  cada señal; total de ecc_errors; rango de potencia... La **desviación estándar** delata la
  inestabilidad; los **errores ECC** delatan la memoria. Piensa qué feature revela cada falla.
- **Etiquetar:** cada ventana hereda el estado de su episodio. Esa es la y del entrenamiento.

### A.2 — Validación de datos (Semana 4)

Antes de entrenar, valida la telemetría con un **contrato de datos (pandera)**: temperaturas
en rango físico, potencia positiva, ecc_errors no negativos, estado dentro de los cuatro
válidos. Datos corruptos no deben entrar al pipeline.

### A.3 — Entrenar y serializar (Semanas 3 y 6)

Entrena un clasificador y **serializa el pipeline completo** en un .joblib (o el formato
que uses). Recuerda: se guarda el pipeline entero, no solo el modelo, para que el
preprocesamiento viaje con él.

---

## Parte B — El servicio (lo nuevo que integra el curso)

### B.1 — La API con FastAPI (Semana 7)

Construye una API que exponga tu modelo. El endpoint principal:

**POST /predecir** — recibe **una ventana de telemetría** (una lista de lecturas de un
segundo cada una) y devuelve el estado predicho.

Entrada (JSON):
```json
{
  "lecturas": [
    {"temp_c": 88.3, "power_w": 296.5, "util_pct": 95.8, "clock_mhz": 2038, "ecc_errors": 0},
    {"temp_c": 89.8, "power_w": 298.9, "util_pct": 95.4, "clock_mhz": 2091, "ecc_errors": 1}
  ]
}
```

Salida (JSON):
```json
{"estado_predicho": "sobrecalentamiento", "confianza": 0.98}
```

La API debe, internamente: recibir la ventana -> **calcular las MISMAS features que en el
entrenamiento** -> predecir -> responder. Usa el mismo features.py en el entrenamiento y en
la API (si calculas features distintas en cada lado, el modelo recibe basura).

### B.2 — Validación en la puerta (conexión Semana 4 + 7)

Con el BaseModel de FastAPI, valida la ventana que llega: que cada lectura tenga todos los
campos, del tipo correcto, y que la ventana tenga un tamaño mínimo razonable (ej. al menos 10
lecturas — no puedes calcular una desviación confiable con 2 datos). Si la petición no
cumple, la API responde con un error claro, sin tocar el modelo.

### B.3 — Empaquetar en Docker (Semana 9)

Escribe un **Dockerfile** que empaquete tu API: parte de python:3.12-slim, instala las
dependencias, copia tu código y el modelo, expone el puerto, y arranca uvicorn. Incluye un
.dockerignore. El contenedor debe construirse y correr con:

```bash
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

Y quedar respondiendo en http://localhost:8000/docs

---

## Entregable

Un repositorio que se pueda **construir y correr con Docker**, con:

```
detector-fallas-gpu/
|- src/
|  |- features.py      # calculo de features (usado por train Y api)
|  |- schema.py        # contrato de datos (pandera)
|  |- train.py         # entrena y serializa el modelo
|  |- api.py           # la API de FastAPI
|- models/
|  |- modelo.joblib    # el modelo serializado
|- Dockerfile
|- .dockerignore
|- requirements.txt
|- README.md            # como construir la imagen y correr el contenedor
```

Más un historial de commits que cuente el proceso.

## Cómo se evalúa (examen oculto)

El profesor tiene ventanas de telemetría de GPUs que nunca viste. **Levantará tu contenedor**
y le enviará esas ventanas al endpoint /predecir, comparando tus predicciones contra el
estado real.

| Criterio | Peso | Qué se evalúa |
|---|---|---|
| El contenedor construye y corre | 20% | docker build y docker run funcionan; la API responde. |
| Accuracy contra las ventanas ocultas | 25% | qué fracción de ventanas clasificaste bien. |
| Pipeline de features correcto | 20% | ventana -> features -> etiqueta, sin fugas; mismas features en train y api. |
| Validación (pandera + BaseModel) | 15% | contrato de datos al entrenar y en la puerta de la API. |
| Serialización + estructura + README | 20% | pipeline serializado completo; repo ordenado; se entiende y se corre. |

## Las trampas a evitar

- **No metas estado, segundo ni episodio_id como features.** El primero es la etiqueta
  (leakage); los otros dos son identificadores, no telemetría.
- **Features iguales en train y api.** Si entrenas con 8 features y la API calcula 6, el
  modelo falla en silencio. Reutiliza el mismo código.
- **No ventanees a través de episodios distintos.** Una ventana = un solo episodio_id.
- **El contenedor debe traer el modelo YA entrenado.** No entrenes dentro del contenedor al
  arrancar; entrena antes, serializa, y copia el .joblib a la imagen.

## Preguntas de sustentación (para la defensa)

- Muéstrame una petición a tu API con una ventana de sobrecalentamiento y explícame la respuesta.
- ¿Qué feature delata mejor la falla_alimentacion, y por qué?
- Si te mando una ventana de solo 3 lecturas, ¿qué hace tu API? ¿Por qué así?
- ¿Por qué el modelo va DENTRO de la imagen y no se entrena al arrancar el contenedor?
- Si el proveedor empieza a mandar la temperatura en Fahrenheit, ¿tu contrato lo atraparía?

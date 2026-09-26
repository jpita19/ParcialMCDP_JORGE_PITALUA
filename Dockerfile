# Imagen del detector de fallas de GPU.
#
# IMPORTANTE: el contenedor NUNCA entrena. El modelo ya viene entrenado y
# serializado en models/modelo.joblib; aqui solo se empaqueta el codigo de
# la API y ese artefacto, y se arranca uvicorn.
FROM python:3.12-slim

WORKDIR /app

# Dependencias primero, para aprovechar el cache de capas de Docker: si solo
# cambia el codigo (src/), no hace falta reinstalar todo de nuevo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la API/entrenamiento y el modelo YA entrenado.
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# --app-dir src hace que uvicorn agregue src/ a sys.path e importe
# directamente el modulo "api" (api.py), sin depender de que src/ sea un
# paquete de Python formal.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

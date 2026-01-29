# MLflow Workshop Dockerfile
# Responsabilidad: entorno reproducible para entrenamiento

FROM python:3.10-slim

# ¿Por qué slim? 
# - Menor tamaño de imagen
# - Menos superficie de ataque
# - Más rápido de descargar

WORKDIR /app

# ¿Por qué copiar requirements.txt primero?
# - Aprovecha cache de Docker layers
# - Solo reinstala dependencias si cambian
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ src/
COPY data/ data/

# Exponer puerto para MLflow UI
EXPOSE 5000

# Punto de entrada por defecto
CMD ["python", "src/train.py"]
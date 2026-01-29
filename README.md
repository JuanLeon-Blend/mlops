# 🧪 MLflow + Docker Workshop
## Del dato al modelo desplegado

### Objetivo
Este proyecto implementa el ciclo completo de un modelo de Machine Learning usando MLflow y Docker, desde la exploración de datos hasta el serving del modelo.

### Arquitectura del proyecto
```
project/
├── data/
│   └── dataset.csv
├── src/
│   ├── train.py
│   └── predict_test.py
├── docker/
│   └── Dockerfile
├── mlruns/          ← generado por MLflow
├── requirements.txt
└── README.md
```

### Reglas del taller
1. **Python local** = SOLO exploración
2. **Entrenamiento, tracking, serving** = SOLO Docker
3. **Todo lo que no quede documentado, no existe**

### Fases del workshop
- [ ] Fase 1: Exploración local del dataset
- [ ] Fase 2: Preparar entorno reproducible
- [ ] Fase 3: Entrenamiento + Tracking con MLflow
- [ ] Fase 4: Model Registry
- [ ] Fase 5: Serving del modelo
- [ ] Fase 6: Conexión con MLOps

### Cómo ejecutar
```bash
# 1. Exploración local
python src/explore_data.py

# 2. Build Docker image
docker build -f docker/Dockerfile -t mlflow-train .

# 3. Entrenar modelo
docker run -p 5000:5000 mlflow-train

# 4. Ver MLflow UI
mlflow ui

# 5. Servir modelo
mlflow models serve -m models:/<model_name>/Staging -p 1234
```

### Próximos pasos hacia producción
| Taller | Producción |
|--------|------------|
| Docker local | ECR |
| MLflow local | SageMaker |
| Serving local | API Gateway |
| Manual | Pipelines CI/CD |
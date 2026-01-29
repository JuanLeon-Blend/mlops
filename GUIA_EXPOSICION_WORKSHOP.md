# 🎤 Guía de Exposición: Workshop MLflow + Docker

## 📋 Estructura de la Presentación (45-60 minutos)

---

## 🎯 INTRODUCCIÓN (5 minutos)

### Slide 1: Título y Contexto
**"MLflow + Docker: Del Dato al Modelo Desplegado"**

**Qué vas a decir:**
> "Hoy vamos a ver cómo implementamos un pipeline completo de Machine Learning, desde la exploración de datos hasta tener un modelo sirviendo predicciones via API REST. No se trata de hacer el mejor modelo, sino de entender cómo los modelos pueden vivir en el mundo real."

### Slide 2: El Problema que Resolvemos
**Mostrar estos pain points:**
- "¿Qué modelo entrené la semana pasada?"
- "¿Con qué parámetros funcionó mejor?"
- "¿Cómo pongo este modelo en producción?"
- "¿Cómo sé si mi modelo sigue funcionando?"

**Qué vas a decir:**
> "Estos son problemas reales que enfrentamos cuando pasamos de notebooks a sistemas productivos. MLflow y Docker nos ayudan a resolverlos."

### Slide 3: Arquitectura Mental
```
┌────────────────────┐
│   Local Machine    │  ← Solo exploración
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│      Docker        │  ← Training, tracking, serving
│ ┌───────────────┐ │
│ │    MLflow     │ │
│ └───────────────┘ │
└────────────────────┘
```

**Qué vas a decir:**
> "La regla de oro: Python local solo para explorar datos. Todo lo importante ocurre en Docker. Esto nos da reproducibilidad desde el día uno."

---

## 🔍 FASE 1: EXPLORACIÓN DE DATOS (8 minutos)

### Slide 4: ¿Por qué Empezar con Datos?
**Qué vas a decir:**
> "Antes de entrenar cualquier modelo, necesitamos entender nuestros datos. Un mal dataset no se arregla con un buen modelo."

### Slide 5: Dataset Iris - Características
**Mostrar:**
- 150 muestras, 4 features
- 3 clases balanceadas (50 cada una)
- Sin valores faltantes
- Features numéricas ya normalizadas

**Código clave a mostrar:**
```python
# Cargar y explorar
iris = datasets.load_iris()
df = pd.DataFrame(X, columns=iris.feature_names)
print(f"Shape: {df.shape}")
print(f"Clases: {iris.target_names}")
```

**Qué vas a decir:**
> "Esto es exploración, no ML. Queremos entender: ¿qué tipo de problema es? ¿Los datos están limpios? ¿Hay balance en las clases? Esta fase conecta directamente con el data lifecycle que ya conocemos."

### Slide 6: Conexión con Data Lifecycle
**Mostrar tabla:**
| Data Lifecycle | Lo que hicimos |
|----------------|----------------|
| Ingestión | `datasets.load_iris()` |
| Calidad | Verificar missing values, tipos |
| Transformación | Mínima (datos ya limpios) |

**Qué vas a decir:**
> "Vean cómo los conceptos del data lifecycle se aplican aquí. La diferencia es que ahora el output no es un dashboard, sino un modelo."

---

## 🐳 FASE 2: ENTORNO REPRODUCIBLE (8 minutos)

### Slide 7: ¿Por qué Docker?
**Problemas que resuelve:**
- "En mi máquina funciona"
- Dependencias conflictivas
- Versiones diferentes de Python/librerías
- Dificultad para desplegar

**Qué vas a decir:**
> "Docker no es solo para DevOps. En ML, la reproducibilidad es crítica. Si no puedes reproducir tu experimento, no puedes confiar en él."

### Slide 8: Dockerfile Explicado
**Mostrar código línea por línea:**
```dockerfile
FROM python:3.10-slim          # ¿Por qué slim?
WORKDIR /app                   # Organización
COPY requirements.txt .        # ¿Por qué primero?
RUN pip install -r requirements.txt
COPY src/ src/                 # Código fuente
COPY data/ data/               # Dataset
EXPOSE 5000                    # Puerto MLflow
CMD ["python", "src/train.py"] # Punto de entrada
```

**Qué vas a decir para cada línea:**
- **slim**: "Imagen más pequeña, menos superficie de ataque, descarga más rápida"
- **requirements primero**: "Aprovecha el cache de Docker. Solo reinstala si cambian dependencias"
- **EXPOSE**: "Documentamos qué puerto usa MLflow UI"

### Slide 9: Principio de Dependencias Mínimas
**Mostrar requirements.txt:**
```txt
pandas>=1.5.0
scikit-learn>=1.3.0
mlflow>=2.8.0
```

**Qué vas a decir:**
> "Solo lo esencial. Menos dependencias = menos problemas. Cada librería adicional es un punto de falla potencial."

---

## 🎯 FASE 3: ENTRENAMIENTO + MLFLOW (12 minutos)

### Slide 10: ¿Qué es MLflow?
**Definición simple:**
> "MLflow = Git + Docker + Métricas para Modelos"

**Componentes:**
- **Tracking**: Experimentos, parámetros, métricas
- **Registry**: Versionado de modelos
- **Serving**: Modelos como APIs

### Slide 11: Anatomía de un MLflow Run
**Mostrar código estructura:**
```python
with mlflow.start_run() as run:
    # 1. Registrar parámetros
    mlflow.log_param("solver", "lbfgs")
    
    # 2. Entrenar modelo
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # 3. Registrar métricas
    mlflow.log_metric("test_accuracy", accuracy)
    
    # 4. Registrar modelo
    mlflow.sklearn.log_model(model, "iris_classifier")
```

**Qué vas a decir:**
> "Cada run es un experimento completo. MLflow registra automáticamente todo: cuándo corrió, qué parámetros usó, qué métricas obtuvo, y el modelo resultante."

### Slide 12: Conceptos Clave MLflow
**Definir cada uno:**
- **Run**: Experimento individual con ID único
- **Experiment**: Agrupación de runs relacionados
- **Parameters**: Hiperparámetros del modelo (input)
- **Metrics**: Métricas de evaluación (output)
- **Artifacts**: Archivos generados (modelo, gráficos, etc.)

### Slide 13: Resultados Obtenidos
**Mostrar métricas:**
```
✅ Test Accuracy: 96.67%
✅ Train Accuracy: 97.50%
✅ Overfitting: 0.83% (excelente)
✅ F1-Score: 96.66%
```

**Qué vas a decir:**
> "Excelentes resultados, pero lo importante no es el 96% de accuracy. Lo importante es que todo está registrado y es reproducible. Cualquiera puede tomar este run ID y obtener exactamente el mismo modelo."

### Slide 14: ¿Por qué Tracking es Crítico?
**Escenarios reales:**
- Comparar 50 experimentos diferentes
- Recordar qué funcionó hace 3 meses
- Reproducir resultados para un paper
- Explicar a tu jefe por qué el modelo cambió

**Qué vas a decir:**
> "Sin tracking, cada experimento es una caja negra. Con MLflow, tienes trazabilidad completa."

---

## 🏷️ FASE 4: MODEL REGISTRY (8 minutos)

### Slide 15: Modelo ≠ Archivo
**Comparación visual:**
```
❌ modelo.pkl
   - Solo pesos
   - Sin contexto
   - Sin versión
   - Sin métricas

✅ Modelo MLflow
   - Código + Pesos
   - Entorno completo
   - Versión automática
   - Métricas + Lineage
```

**Qué vas a decir:**
> "Esta es la diferencia fundamental. Un .pkl es solo un archivo. Un modelo registrado en MLflow es un artefacto de software completo."

### Slide 16: Model Registry - Governance
**Stages del modelo:**
- **None**: Recién registrado
- **Staging**: Listo para testing
- **Production**: Sirviendo usuarios reales
- **Archived**: Deprecado

**Qué vas a decir:**
> "Esto es governance. No cualquier modelo puede ir a producción. Debe pasar por staging, ser validado, y solo entonces promovido."

### Slide 17: MLflow UI - Demo
**Mostrar screenshots de:**
- Experiments tab con runs
- Models tab con versiones
- Comparación de métricas
- Transición de stages

**Qué vas a decir:**
> "La UI nos da visibilidad completa. Podemos comparar experimentos, ver la evolución de métricas, y gestionar el ciclo de vida de modelos."

---

## 🌐 FASE 5: MODEL SERVING (10 minutos)

### Slide 18: ¿Por qué Serving?
**Qué vas a decir:**
> "Un modelo que no se puede consumir es solo un experimento. El valor real está en generar predicciones para sistemas en producción."

### Slide 19: Anatomía de una API de ML
**Mostrar estructura:**
```python
@app.route('/invocations', methods=['POST'])
def predict():
    # Input: {"instances": [[5.1, 3.5, 1.4, 0.2]]}
    # Output: {"predictions": [0]}
```

**Endpoints implementados:**
- `GET /health`: Health check
- `POST /invocations`: Predicciones (formato MLflow)
- `POST /predict_names`: Con nombres de clases
- `GET /`: Info del servicio

### Slide 20: Test de Inferencia
**Mostrar ejemplo real:**
```python
# Datos de prueba
test_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [7.0, 3.2, 4.7, 1.4],  # Versicolor  
    [6.3, 3.3, 6.0, 2.5]   # Virginica
]

# Request
curl -X POST http://localhost:1235/invocations \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}'

# Response
{"predictions": [0]}
```

**Qué vas a decir:**
> "Así es como otros sistemas consumirían nuestro modelo. JSON in, JSON out. Estándar, simple, escalable."

---

## 🔄 CONEXIÓN DATA → MODEL LIFECYCLE (8 minutos)

### Slide 21: Mapeo Conceptual
**Tabla comparativa:**
| Data Lifecycle | Model Lifecycle | Qué se Mantiene | Qué se Amplía |
|----------------|-----------------|-----------------|---------------|
| Ingestión | Feature Extraction | Pipelines | Versionado |
| Transformación | Training | Reproducibilidad | Métricas |
| Consumo | Inference | APIs/Outputs | Serving/Monitoreo |

**Qué vas a decir:**
> "Vean cómo los conceptos se mapean. No estamos aprendiendo algo completamente nuevo, estamos extendiendo lo que ya sabemos sobre datos hacia modelos."

### Slide 22: Flujo Completo Implementado
**Diagrama de flujo:**
```
Datos → Exploración → Features → Entrenamiento → Modelo → Registry → Serving → Predicciones
  ↓         ↓           ↓           ↓           ↓         ↓         ↓          ↓
 CSV     Pandas      Arrays    Scikit-learn  MLflow   Versions   Flask    JSON API
```

**Qué vas a decir:**
> "Este es el pipeline completo que implementamos. Cada paso tiene su herramienta, cada herramienta tiene su propósito."

---

## 🚀 ESCALABILIDAD HACIA PRODUCCIÓN (6 minutos)

### Slide 23: Taller vs Producción
**Tabla de mapeo:**
| Taller | Producción | AWS Service |
|--------|------------|-------------|
| Docker local | Container Registry | ECR |
| MLflow local | Managed ML Platform | SageMaker |
| Flask serving | API Gateway + Compute | API Gateway + Lambda |
| Manual deploy | CI/CD Pipelines | CodePipeline |

**Qué vas a decir:**
> "Lo que hicimos hoy es la base. En producción, cada componente tiene su equivalente escalable y managed."

### Slide 24: Próximos Pasos Técnicos
**Roadmap:**
1. **CI/CD**: Automatizar entrenamiento y despliegue
2. **Monitoring**: Métricas de modelo en producción
3. **A/B Testing**: Comparar versiones de modelos
4. **Auto-scaling**: Manejar carga variable
5. **Security**: Autenticación y autorización

---

## 💡 MENSAJES CLAVE Y CIERRE (5 minutos)

### Slide 25: Frases Potentes
**Destacar estos mensajes:**

> **"Un modelo que no está trackeado, no existe"**
> Sin tracking, no hay reproducibilidad ni comparación.

> **"Un modelo que no se puede consumir, es solo un experimento"**
> El valor está en generar predicciones para sistemas reales.

> **"MLflow no es solo para ML, Docker no es solo DevOps"**
> Es software engineering aplicado a modelos.

### Slide 26: Lo que Logramos Hoy
**Checklist visual:**
- ✅ Pipeline completo: Datos → Modelo → API
- ✅ Reproducibilidad con Docker
- ✅ Tracking completo con MLflow
- ✅ Versionado de modelos
- ✅ Serving como servicio REST
- ✅ Base sólida para MLOps

### Slide 27: Pregunta Final
**"¿Ya no tienen miedo a poner un modelo a correr?"**

**Qué vas a decir:**
> "Esa era la meta. Que entiendan que los modelos son software, que se pueden versionar, trackear, y desplegar como cualquier aplicación. Ya tienen las herramientas y el conocimiento para hacerlo."

---

## 🎯 TIPS PARA LA EXPOSICIÓN

### Antes de Empezar
1. **Tener MLflow UI abierto** en http://localhost:5000
2. **Preparar terminal** con comandos listos
3. **Screenshots** de cada fase por si algo falla
4. **Código visible** en editor para mostrar

### Durante la Presentación
1. **Mostrar código real**, no solo slides
2. **Ejecutar comandos en vivo** cuando sea posible
3. **Conectar cada concepto** con problemas reales
4. **Usar analogías**: "MLflow es como Git para modelos"
5. **Hacer preguntas** para mantener engagement

### Manejo de Preguntas
**Preguntas frecuentes y respuestas:**

**"¿Por qué no usar Jupyter notebooks?"**
> "Notebooks son excelentes para exploración, pero no para producción. No son reproducibles, no versionan bien, y no escalan."

**"¿MLflow es mejor que Weights & Biases?"**
> "Son herramientas similares. MLflow es open source y se integra bien con cualquier stack. W&B tiene mejor UI pero es SaaS."

**"¿Esto funciona con deep learning?"**
> "Absolutamente. MLflow soporta TensorFlow, PyTorch, cualquier framework. Los conceptos son los mismos."

**"¿Cómo manejan datos grandes?"**
> "Para datos grandes, usarías S3 + Spark + SageMaker. Los principios son iguales, solo cambia la escala."

### Timing por Sección
- **Introducción**: 5 min (no te extiendas)
- **Fase 1-2**: 8 min cada una (mostrar código)
- **Fase 3**: 12 min (la más importante)
- **Fase 4-5**: 8-10 min cada una
- **Conexiones**: 8 min (conceptual)
- **Producción**: 6 min (futuro)
- **Cierre**: 5 min (mensajes clave)

### Backup Plans
1. **Si MLflow UI falla**: Usar screenshots
2. **Si Docker falla**: Mostrar código y explicar conceptos
3. **Si serving falla**: Usar curl examples y mostrar JSON
4. **Si todo falla**: Focus en conceptos y arquitectura

---

## 🎤 SCRIPT DE APERTURA

**"Buenos días/tardes. Hoy vamos a implementar juntos un pipeline completo de Machine Learning. No vamos a hacer el modelo más preciso del mundo, vamos a hacer algo más importante: un modelo que puede vivir en producción.**

**¿Cuántos han tenido un notebook que funciona perfecto, pero no saben cómo ponerlo en producción? ¿O han entrenado 20 modelos y no recuerdan cuál funcionó mejor?**

**Esos son problemas reales. Y hoy los vamos a resolver con MLflow y Docker. Al final de esta sesión, van a tener un modelo sirviendo predicciones via API REST, completamente trackeado y reproducible.**

**La regla de oro de hoy: Python local solo para explorar datos. Todo lo importante ocurre en Docker. ¿Listos? Empezamos."**

---

**🎯 ¡Con esta guía tienes todo lo necesario para una exposición exitosa del workshop!**
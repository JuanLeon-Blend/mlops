"""
FASE 5 - Model Serving (Alternative approach)
Objetivo: servir modelo via Flask API

Cuando MLflow registry tiene problemas, podemos servir directamente
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Cargar modelo desde el run más reciente
def load_latest_model():
    """
    Carga el modelo desde el experimento más reciente
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("MLflow Workshop - Iris Classification")
    
    # Si no encuentra el experimento, buscar el más reciente
    if experiment is None:
        experiments = client.search_experiments()
        if experiments:
            experiment = experiments[0]  # Tomar el más reciente
    
    if experiment is None:
        raise Exception("Experimento no encontrado")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise Exception("No se encontraron runs")
    
    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/iris_classifier"
    
    print(f"🔄 Cargando modelo desde: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    return model, latest_run.info.run_id

# Cargar modelo al iniciar
try:
    model, run_id = load_latest_model()
    print(f"✅ Modelo cargado exitosamente desde run: {run_id}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de salud
    """
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "run_id": run_id
    })

@app.route('/invocations', methods=['POST'])
def predict():
    """
    Endpoint de predicción compatible con MLflow
    
    Input format: {"instances": [[feature1, feature2, feature3, feature4], ...]}
    Output format: {"predictions": [class1, class2, ...]}
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Parsear request JSON
        data = request.get_json()
        
        if 'instances' not in data:
            return jsonify({"error": "Missing 'instances' in request"}), 400
        
        instances = data['instances']
        
        # Convertir a numpy array
        X = np.array(instances)
        
        # Validar shape
        if X.shape[1] != 4:
            return jsonify({"error": f"Expected 4 features, got {X.shape[1]}"}), 400
        
        # Hacer predicción
        predictions = model.predict(X)
        
        # Convertir a lista de Python (JSON serializable)
        predictions_list = predictions.tolist()
        
        return jsonify({"predictions": predictions_list})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_names', methods=['POST'])
def predict_names():
    """
    Endpoint adicional que devuelve nombres de clases
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        instances = data['instances']
        X = np.array(instances)
        
        predictions = model.predict(X)
        
        # Mapear a nombres de clases
        class_names = ['setosa', 'versicolor', 'virginica']
        prediction_names = [class_names[pred] for pred in predictions]
        
        return jsonify({
            "predictions": predictions.tolist(),
            "prediction_names": prediction_names
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def info():
    """
    Información del servicio
    """
    return jsonify({
        "service": "Iris Classification Model",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/invocations": "POST - MLflow compatible predictions",
            "/predict_names": "POST - Predictions with class names"
        },
        "example_request": {
            "instances": [[5.1, 3.5, 1.4, 0.2]]
        }
    })

if __name__ == '__main__':
    print("🚀 FASE 5: Model Serving")
    print("=" * 50)
    print("🌐 Iniciando servidor Flask...")
    print("📡 Endpoints disponibles:")
    print("  - GET  /health          - Health check")
    print("  - POST /invocations     - Predicciones (formato MLflow)")
    print("  - POST /predict_names   - Predicciones con nombres")
    print("  - GET  /               - Información del servicio")
    print()
    print("🧪 Test con curl:")
    print("curl -X POST http://localhost:1235/invocations \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"instances\": [[5.1, 3.5, 1.4, 0.2]]}'")
    print()
    
    app.run(host='0.0.0.0', port=1235, debug=False)
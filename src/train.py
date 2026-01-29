"""
FASE 3 - Entrenamiento + Tracking con MLflow
Objetivo: entrenar modelo Y registrar todo en MLflow

Regla: Entrenamiento = SOLO Docker
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def train_model():
    """
    Entrenamiento con tracking completo en MLflow
    
    Conceptos clave:
    - mlflow.start_run(): inicia un experimento
    - mlflow.log_param(): registra hiperparámetros
    - mlflow.log_metric(): registra métricas
    - mlflow.sklearn.log_model(): registra el modelo
    """
    
    print("🚀 FASE 3: Entrenamiento + MLflow Tracking")
    print("=" * 50)
    
    # Configurar experimento MLflow
    mlflow.set_experiment("MLflow Workshop - Iris Classification")
    
    # Cargar datos
    print("📂 Cargando dataset...")
    df = pd.read_csv('data/iris_dataset.csv')
    
    # Preparar features y target
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    X = df[feature_cols]
    y = df['target']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train shape: {X_train.shape}")
    print(f"📊 Test shape: {X_test.shape}")
    
    # Hiperparámetros del modelo
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "C": 1.0  # regularización
    }
    
    # 🔥 INICIO DEL RUN DE MLFLOW
    with mlflow.start_run() as run:
        print(f"🏃 MLflow Run ID: {run.info.run_id}")
        
        # 1. Registrar hiperparámetros
        print("📝 Registrando parámetros...")
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Registrar info del dataset
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", len(feature_cols))
        mlflow.log_param("classes", len(np.unique(y)))
        
        # 2. Entrenar modelo
        print("🎯 Entrenando modelo...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # 3. Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 4. Calcular métricas
        print("📊 Calculando métricas...")
        
        # Métricas de entrenamiento
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_precision = precision_score(y_train, y_pred_train, average='weighted')
        train_recall = recall_score(y_train, y_pred_train, average='weighted')
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        
        # Métricas de test
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # 5. Registrar métricas en MLflow
        print("📈 Registrando métricas...")
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_f1", train_f1)
        
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Calcular overfitting
        overfitting = train_accuracy - test_accuracy
        mlflow.log_metric("overfitting", overfitting)
        
        # 6. Registrar el modelo
        print("💾 Registrando modelo...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_classifier",
            registered_model_name="iris_logistic_regression"
        )
        
        # 7. Mostrar resultados
        print("\n🎉 RESULTADOS:")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Overfitting: {overfitting:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        print(f"\n📊 MLflow Run: {run.info.run_id}")
        print("🌐 Ver en MLflow UI: http://localhost:5000")
        
        # 8. Documentar conceptos clave
        print("\n💡 Conceptos MLflow aplicados:")
        print("- ✅ run: experimento individual con ID único")
        print("- ✅ params: hiperparámetros del modelo")
        print("- ✅ metrics: métricas de evaluación")
        print("- ✅ artifacts: modelo serializado")
        print("- ✅ experiment: agrupación de runs relacionados")
        
        return run.info.run_id

if __name__ == "__main__":
    run_id = train_model()
    print(f"\n🏁 Entrenamiento completado. Run ID: {run_id}")
    print("➡️  Siguiente: Model Registry (Fase 4)")
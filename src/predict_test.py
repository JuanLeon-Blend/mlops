"""
FASE 5 - Test de inferencia del modelo servido
Objetivo: probar el modelo via API REST

Regla: Serving = Docker + MLflow
"""

import requests
import json
import numpy as np
import pandas as pd

def test_model_serving(model_url="http://localhost:1235/invocations"):
    """
    Test del modelo servido via MLflow
    
    Conceptos clave:
    - endpoint REST: /invocations
    - formato JSON: {"instances": [...]}
    - response: {"predictions": [...]}
    """
    
    print("🧪 FASE 5: Test de inferencia del modelo")
    print("=" * 50)
    
    # Datos de prueba (flores de iris)
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [7.0, 3.2, 4.7, 1.4],  # Versicolor  
        [6.3, 3.3, 6.0, 2.5]   # Virginica
    ]
    
    expected_classes = ['setosa', 'versicolor', 'virginica']
    
    print("📊 Datos de prueba:")
    for i, sample in enumerate(test_samples):
        print(f"  Muestra {i+1}: {sample} (esperado: {expected_classes[i]})")
    
    # Preparar request
    data = {
        "instances": test_samples
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"\n🌐 Enviando request a: {model_url}")
        response = requests.post(
            model_url, 
            data=json.dumps(data), 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            predictions = response.json()
            print("✅ Predicciones exitosas!")
            print(f"📋 Response: {predictions}")
            
            # Mapear predicciones a nombres de clases
            class_names = ['setosa', 'versicolor', 'virginica']
            
            print("\n🎯 Resultados:")
            for i, pred in enumerate(predictions['predictions']):
                predicted_class = class_names[int(pred)]
                expected_class = expected_classes[i]
                status = "✅" if predicted_class == expected_class else "❌"
                
                print(f"  Muestra {i+1}: {predicted_class} {status} (esperado: {expected_class})")
            
            print("\n💡 Conceptos de serving aplicados:")
            print("- ✅ endpoint REST: /invocations")
            print("- ✅ formato input: JSON con 'instances'")
            print("- ✅ formato output: JSON con 'predictions'")
            print("- ✅ modelo como servicio: consumible por otros sistemas")
            
        else:
            print(f"❌ Error en request: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al modelo.")
        print("💡 Asegúrate de que el modelo esté corriendo:")
        print("   mlflow models serve -m models:/iris_logistic_regression/Staging -p 1234")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def test_with_curl_example():
    """
    Muestra ejemplo equivalente con curl
    """
    print("\n🔧 Ejemplo equivalente con curl:")
    print("curl -X POST http://localhost:1234/invocations \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"instances\": [[5.1, 3.5, 1.4, 0.2]]}'")

if __name__ == "__main__":
    test_model_serving()
    test_with_curl_example()
    
    print("\n🏁 Test de inferencia completado")
    print("➡️  Siguiente: Conexión con MLOps (Fase 6)")
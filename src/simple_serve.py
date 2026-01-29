"""
Serving simple para demostración
"""
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Entrenar modelo simple para serving
print("🔄 Entrenando modelo para serving...")
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Modelo entrenado. Accuracy: {accuracy:.4f}")

class_names = ['setosa', 'versicolor', 'virginica']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "accuracy": f"{accuracy:.4f}"})

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        instances = data['instances']
        X = np.array(instances)
        
        predictions = model.predict(X)
        return jsonify({"predictions": predictions.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_names', methods=['POST'])
def predict_names():
    try:
        data = request.get_json()
        instances = data['instances']
        X = np.array(instances)
        
        predictions = model.predict(X)
        prediction_names = [class_names[pred] for pred in predictions]
        
        return jsonify({
            "predictions": predictions.tolist(),
            "prediction_names": prediction_names
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def info():
    return jsonify({
        "service": "Iris Classification API",
        "accuracy": f"{accuracy:.4f}",
        "endpoints": ["/health", "/invocations", "/predict_names"],
        "example": {
            "url": "/invocations",
            "method": "POST",
            "body": {"instances": [[5.1, 3.5, 1.4, 0.2]]}
        }
    })

if __name__ == '__main__':
    print("🚀 Iniciando servidor en http://localhost:1235")
    app.run(host='0.0.0.0', port=1235, debug=False)
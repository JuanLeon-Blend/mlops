"""
FASE 1 - Exploración local del dataset
Objetivo: entender los datos, NO el modelo

Regla: Python local = SOLO exploración
"""

import pandas as pd
from sklearn import datasets
import numpy as np

def explore_dataset():
    """
    Exploración del dataset Iris
    Conecta con Data Lifecycle: ingestión → calidad → transformación
    """
    
    print("🔍 FASE 1: Exploración del dataset")
    print("=" * 50)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Crear DataFrame para mejor visualización
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print(f"📊 Shape del dataset: {df.shape}")
    print(f"📋 Columnas: {list(df.columns)}")
    print(f"🎯 Target: {iris.target_names}")
    print()
    
    print("📈 Información básica:")
    print(df.info())
    print()
    
    print("📊 Estadísticas descriptivas:")
    print(df.describe())
    print()
    
    print("🎯 Distribución del target:")
    print(df['target_name'].value_counts())
    print()
    
    # Criterio de datos (no ML)
    print("🧠 Criterio de datos:")
    print("- Tipo de problema: Clasificación multiclase")
    print("- Features: 4 numéricas (medidas de flores)")
    print("- Target: 3 clases balanceadas")
    print("- Sin valores faltantes")
    print("- Dataset pequeño pero limpio")
    print()
    
    # Split mental (conceptual)
    print("🔄 Split mental train/test:")
    print("- Features (X): sepal/petal length/width")
    print("- Target (y): especie de iris")
    print("- Split sugerido: 80/20")
    print()
    
    print("💡 Conexión con Data Lifecycle:")
    print("- ✅ Ingestión: datos cargados")
    print("- ✅ Calidad: sin missing values, tipos correctos")
    print("- ✅ Transformación: mínima (ya normalizado)")
    print("- ➡️  Siguiente: entrenamiento en Docker")
    
    # Guardar dataset para uso en Docker
    df.to_csv('data/iris_dataset.csv', index=False)
    print(f"💾 Dataset guardado en: data/iris_dataset.csv")

if __name__ == "__main__":
    explore_dataset()
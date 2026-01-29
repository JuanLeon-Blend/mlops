# 🏷️ Phase 4: Model Registry Instructions

## What you'll see in MLflow UI (http://localhost:5000):

### 1. Experiments Tab
- **Experiment**: "MLflow Workshop - Iris Classification"
- **Run**: Your training run with ID `57986499592a46f58d4ded4b54f06bc3`
- **Metrics**: accuracy, precision, recall, f1-score
- **Parameters**: solver, max_iter, C, etc.
- **Artifacts**: The trained model

### 2. Models Tab
- **Registered Model**: `iris_logistic_regression`
- **Version 1**: Currently in "None" stage
- **Model URI**: `models:/iris_logistic_regression/1`

## 🎯 Actions to perform:

### Step 1: Explore the Run
1. Click on **Experiments** tab
2. Click on your run ID
3. Explore:
   - Parameters logged
   - Metrics tracked
   - Artifacts (model files)

### Step 2: Model Registry
1. Click on **Models** tab
2. Click on `iris_logistic_regression`
3. You'll see Version 1 in "None" stage

### Step 3: Transition to Staging
1. Click on Version 1
2. Click **"Transition to"** button
3. Select **"Staging"**
4. Add description: "First model version ready for testing"
5. Click **"OK"**

## 💡 Key Concepts:

### Model ≠ File
- A `.pkl` file is just weights
- A **registered model** includes:
  - Code version
  - Environment
  - Metrics
  - Metadata
  - Lineage

### Model Stages
- **None**: Just registered
- **Staging**: Ready for testing
- **Production**: Serving real users
- **Archived**: Deprecated

### Why Model Registry?
- **Versioning**: Track model evolution
- **Governance**: Control what goes to production
- **Lineage**: Know how model was created
- **Collaboration**: Team can see all models

## ➡️ Next: Phase 5 - Model Serving

Once you transition to Staging, we'll serve the model via REST API!
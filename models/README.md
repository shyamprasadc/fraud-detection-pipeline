# Models Directory

This directory contains trained machine learning models and related files for the fraud detection pipeline.

## Contents

- `*.pkl` - Trained model files (Random Forest, XGBoost, etc.)
- `scaler.pkl` - Feature scaler for preprocessing
- `feature_names.json` - List of feature names used by the models
- `model_metadata.json` - Model metadata and performance metrics
- `*.json` - Model configuration and feature definitions

## Model Files

### Random Forest Model

- **File**: `random_forest_model.pkl`
- **Type**: Random Forest Classifier
- **Use Case**: Primary fraud detection model
- **Performance**: Typically 95%+ precision, 90%+ recall

### XGBoost Model

- **File**: `xgboost_model.pkl`
- **Type**: XGBoost Classifier
- **Use Case**: Alternative model for comparison
- **Performance**: Often slightly better than Random Forest

### Feature Scaler

- **File**: `scaler.pkl`
- **Type**: StandardScaler
- **Use Case**: Normalize features before model inference
- **Note**: Must be used with the same features used during training

## Usage

```python
import pickle
import json

# Load model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Make predictions
features_scaled = scaler.transform(features)
predictions = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)
```

## Training

Models are trained using the `scripts/train_model.py` script or the Jupyter notebooks in the `notebooks/` directory.

## Model Versioning

Each model version includes:

- Model file (.pkl)
- Scaler file (.pkl)
- Feature names (JSON)
- Metadata with performance metrics
- Training configuration

## Performance Targets

- **Precision**: >95%
- **Recall**: >90%
- **AUC**: >0.95
- **Latency**: <100ms per prediction
- **Throughput**: >1000 predictions/second

## Monitoring

Models should be monitored for:

- Prediction drift
- Performance degradation
- Feature distribution changes
- Model staleness

## Retraining

Models should be retrained:

- Monthly or when performance degrades
- When new fraud patterns emerge
- When feature engineering changes
- When data distribution shifts significantly

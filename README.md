# Ads Click-Through Rate Prediction 2025

A TensorFlow-based machine learning model for predicting Click-Through Rates (CTR) in digital advertising. This project includes feature engineering, model training, and comprehensive ablation studies to measure the contribution of different feature groups to model performance.

## Overview

This project develops a CTR prediction model trained on a simulated ads dataset with the following components:

- **Feature Engineering**: 
  - User Recency Features: Time since last user interaction
  - Semantic Similarity Features: Ad-user preference alignment
  - Historical Engagement Features: User click history and interaction patterns

- **Model**: Deep neural network implemented with TensorFlow/Keras
- **Evaluation Metric**: AUC-ROC on held-out test set
- **Ablation Studies**: Systematic analysis of each feature group's contribution to model performance

## Project Structure

```
.
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── data/                    # Data directory
│   └── ads_dataset.csv      # Generated simulated ads dataset
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_generator.py    # Simulated dataset generation
│   ├── features.py          # Feature engineering
│   ├── model.py             # TensorFlow model definition
│   └── utils.py             # Utility functions
├── models/                  # Trained model checkpoints
├── notebooks/               # Jupyter notebooks for analysis
├── train.py                 # Training script
├── ablation_study.py        # Feature ablation analysis
└── evaluate.py              # Model evaluation script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ads-CTR-Prediction.git
cd Ads-CTR-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Dataset
```bash
python -c "from src.data_generator import generate_ads_dataset; generate_ads_dataset('data/ads_dataset.csv', n_samples=50000)"
```

### 2. Train Model
```bash
python train.py --data data/ads_dataset.csv --output models/ctr_model --epochs 50
```

### 3. Run Ablation Studies
```bash
python ablation_study.py --data data/ads_dataset.csv --model models/ctr_model
```

### 4. Evaluate on Test Set
```bash
python evaluate.py --model models/ctr_model --data data/ads_dataset.csv
```

## Feature Groups

### 1. User Recency Features
- Days since last interaction
- Interaction frequency (30-day window)
- Session count

### 2. Semantic Similarity Features
- Ad category affinity score
- User interest vector similarity
- Topic relevance score

### 3. Historical Engagement Features
- Click-through history (normalized)
- Average engagement rate
- User lifetime value estimate
- Conversion probability estimate

## Model Architecture

- Input Layer: Normalized feature vectors
- Hidden Layers: Dense layers with ReLU activation and Dropout (0.3)
- Output Layer: Sigmoid activation for binary classification
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: AUC-ROC

## Results

The model evaluation includes:
- **AUC-ROC Score**: Performance on held-out test set
- **Feature Importance**: Ablation study showing contribution of each feature group
- **Cross-Validation**: k-fold cross-validation results
- **Confusion Matrix**: Classification performance breakdown

## Ablation Study Results

The ablation study systematically removes feature groups and measures the impact on model performance:
- Full Model (all features)
- Without Recency Features
- Without Semantic Similarity Features
- Without Engagement Features
- Individual feature group performance

## Usage Example

```python
from src.data_generator import generate_ads_dataset
from src.features import FeatureEngineer
from src.model import build_ctr_model
import tensorflow as tf

# Generate data
generate_ads_dataset('data/ads_dataset.csv', n_samples=50000)

# Engineer features
engineer = FeatureEngineer()
X, y = engineer.prepare_data('data/ads_dataset.csv')

# Build and train model
model = build_ctr_model(X.shape[1])
model.fit(X, y, epochs=50, validation_split=0.2)

# Evaluate
auc = model.evaluate(X_test, y_test)[1]
print(f"AUC-ROC: {auc:.4f}")
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- scikit-learn
- pandas
- numpy
- matplotlib

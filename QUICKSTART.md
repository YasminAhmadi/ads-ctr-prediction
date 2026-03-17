# Ads Click-Through Rate Prediction

## Quick Description

A machine learning project for predicting Click-Through Rates (CTR) in digital advertising using TensorFlow. The project includes:

- **TensorFlow Deep Learning Model**: Binary classification neural network
- **Feature Engineering**: Three feature groups with domain significance
  - User Recency Features
  - Semantic Similarity Features
  - Historical Engagement Features
- **Ablation Studies**: Systematic analysis of feature group contributions
- **Simulated Dataset**: Realistic ads dataset with 50k+ samples

## One-Line Setup

```bash
bash setup.sh && python3 train.py && python3 ablation_study.py && python3 evaluate.py
```

## File Structure

```
├── README.md                    # Full documentation
├── setup.sh                     # Automated setup script
├── requirements.txt             # Python dependencies
├── train.py                     # Training script
├── ablation_study.py           # Feature ablation analysis
├── evaluate.py                 # Model evaluation
├── src/
│   ├── data_generator.py       # Dataset generation
│   ├── features.py             # Feature engineering
│   ├── model.py                # TensorFlow models
│   └── utils.py                # Visualization utilities
├── data/                       # Data directory
├── models/                     # Model checkpoints
└── results/                    # Evaluation results
```

## Key Features

✅ Production-ready code with proper structure
✅ Comprehensive feature engineering pipeline
✅ Ablation studies for feature importance analysis
✅ Complete evaluation with multiple metrics
✅ Professional documentation and licensing
✅ Ready to push to GitHub

## Performance Metrics

The model evaluates on:
- **AUC-ROC**: Area Under Receiver Operating Characteristic
- **Precision & Recall**: Classification performance
- **Accuracy**: Overall correctness
- **Feature Importance**: Contribution of each feature group

See `results/ablation/ablation_results.csv` for detailed results.

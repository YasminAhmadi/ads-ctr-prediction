"""
Evaluation script for trained CTR model
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from src.features import FeatureEngineer
from src.utils import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    print_classification_report
)
import tensorflow as tf
import matplotlib.pyplot as plt


def main(args):
    """Main evaluation function"""
    
    # Load model
    print("="*60)
    print("LOADING ARTIFACTS")
    print("="*60)
    
    model = tf.keras.models.load_model(os.path.join(args.model, 'ctr_model.h5'))
    print(f"✓ Model loaded from {args.model}/ctr_model.h5")
    
    # Load scaler
    scaler_path = os.path.join(args.model, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded")
    
    # Load feature groups
    feature_groups_path = os.path.join(args.model, 'feature_groups.json')
    with open(feature_groups_path, 'r') as f:
        feature_groups = json.load(f)
    print(f"✓ Feature groups loaded: {list(feature_groups.keys())}")
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING TEST DATA")
    print("="*60)
    
    engineer = FeatureEngineer(test_size=0.2, random_state=42)
    engineer.scaler = scaler  # Use loaded scaler
    
    X_train, X_test, y_train, y_test = engineer.prepare_data(args.data)
    print(f"✓ Test set size: {X_test.shape}")
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Compute metrics
    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    
    metrics = {name: value for name, value in zip(metric_names, results)}
    
    print(f"\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'eval_metrics.json')
    os.makedirs(args.output, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Detailed classification report
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    print_classification_report(y_test, y_pred)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60)
    
    plots_dir = os.path.join(args.output, 'eval_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Confusion matrix
    fig = plot_confusion_matrix(y_test, y_pred, "Test Set Confusion Matrix")
    fig.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Confusion matrix plot saved")
    
    # ROC curve
    fig = plot_roc_curve(y_test, y_pred_proba, "ROC Curve")
    fig.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ ROC curve plot saved")
    
    # Precision-Recall curve
    fig = plot_precision_recall_curve(y_test, y_pred_proba, "Precision-Recall Curve")
    fig.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Precision-Recall curve plot saved")
    
    # Prediction distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='No Click')
    ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Click')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(plots_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Prediction distribution plot saved")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained CTR model')
    parser.add_argument('--model', type=str, default='models/ctr_model',
                       help='Directory containing trained model artifacts')
    parser.add_argument('--data', type=str, default='data/ads_dataset.csv',
                       help='Path to input dataset')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    main(args)

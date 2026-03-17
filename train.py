"""
Training script for CTR prediction model
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from src.data_generator import generate_ads_dataset
from src.features import FeatureEngineer
from src.model import build_ctr_model, CTRPredictor
from src.utils import plot_training_history, plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt


def main(args):
    """Main training function"""
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    
    # Generate data if needed
    if not os.path.exists(args.data):
        print("Dataset not found. Generating simulated dataset...")
        os.makedirs('data', exist_ok=True)
        generate_ads_dataset(args.data, n_samples=50000)
    
    # Prepare features
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    engineer = FeatureEngineer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = engineer.prepare_data(args.data)
    
    # Build model
    print("\n" + "="*60)
    print("MODEL BUILDING & TRAINING")
    print("="*60)
    model = build_ctr_model(
        input_dim=X_train.shape[1],
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout
    )
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # Train model
    predictor = CTRPredictor(model, verbose=1)
    
    # Split training data for validation
    n_train = len(X_train)
    split_idx = int(0.8 * n_train)
    
    X_train_fit = X_train.iloc[:split_idx]
    y_train_fit = y_train[:split_idx]
    
    X_val = X_train.iloc[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"\nTraining with:")
    print(f"  Training set: {X_train_fit.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    history = predictor.train(
        X_train_fit, y_train_fit,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    metrics, y_pred_proba, y_pred = predictor.evaluate(X_test, y_test)
    
    print(f"\nTest Set Performance:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save model
    model_path = os.path.join(args.output, 'ctr_model.h5')
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Save feature groups
    feature_groups = engineer.get_feature_groups()
    feature_groups_path = os.path.join(args.output, 'feature_groups.json')
    with open(feature_groups_path, 'w') as f:
        json.dump(feature_groups, f, indent=2)
    print(f"✓ Feature groups saved to {feature_groups_path}")
    
    # Save feature scaler
    import pickle
    scaler_path = os.path.join(args.output, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(engineer.scaler, f)
    print(f"✓ Feature scaler saved to {scaler_path}")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plots_dir = os.path.join(args.output, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training history
    fig = plot_training_history(history, "Model Training History")
    fig.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Training history plot saved")
    
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
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print(f"\nArtifacts saved to: {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CTR prediction model')
    parser.add_argument('--data', type=str, default='data/ads_dataset.csv',
                       help='Path to input dataset')
    parser.add_argument('--output', type=str, default='models/ctr_model',
                       help='Output directory for model and artifacts')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    args = parser.parse_args()
    main(args)

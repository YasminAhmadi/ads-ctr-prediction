"""
Ablation Study Script
Measures the contribution of each feature group to model performance
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import pickle
from src.data_generator import generate_ads_dataset
from src.features import FeatureEngineer, create_feature_subsets
from src.model import build_lightweight_model, CTRPredictor
from src.utils import plot_ablation_results
import matplotlib.pyplot as plt


def run_ablation_study(data_path, model_output_dir, n_runs=3):
    """
    Run ablation study to measure feature group contributions.
    
    Parameters:
    -----------
    data_path : str
        Path to dataset
    model_output_dir : str
        Directory containing trained model artifacts
    n_runs : int
        Number of runs for each configuration
    
    Returns:
    --------
    results : pd.DataFrame
        Ablation study results
    """
    
    # Generate data if needed
    if not os.path.exists(data_path):
        print("Dataset not found. Generating simulated dataset...")
        generate_ads_dataset(data_path, n_samples=50000)
    
    # Prepare features
    print("\n" + "="*60)
    print("PREPARING FEATURES")
    print("="*60)
    engineer = FeatureEngineer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = engineer.prepare_data(data_path)
    
    feature_groups = engineer.get_feature_groups()
    print(f"Feature groups: {list(feature_groups.keys())}")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features - {features}")
    
    # Create feature subsets for ablation
    feature_subsets = create_feature_subsets(X_train, X_test, feature_groups)
    
    print(f"\nFeature subsets for ablation:")
    for subset_name, features in feature_subsets.items():
        print(f"  {subset_name}: {len(features)} features")
    
    # Run ablation study
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)
    
    results_list = []
    
    for subset_name, feature_list in feature_subsets.items():
        print(f"\nTesting: {subset_name} ({len(feature_list)} features)")
        
        # Select features
        X_train_subset = X_train[feature_list]
        X_test_subset = X_test[feature_list]
        
        # Train multiple runs
        run_auc_scores = []
        run_accuracies = []
        
        for run in range(n_runs):
            if n_runs > 1:
                print(f"  Run {run + 1}/{n_runs}...", end=' ')
            
            # Build and train model
            model = build_lightweight_model(X_train_subset.shape[1], learning_rate=0.001)
            predictor = CTRPredictor(model, verbose=0)
            
            # Validation split
            split_idx = int(0.8 * len(X_train_subset))
            X_train_fit = X_train_subset.iloc[:split_idx]
            y_train_fit = y_train[:split_idx]
            X_val = X_train_subset.iloc[split_idx:]
            y_val = y_train[split_idx:]
            
            # Train
            history = predictor.train(
                X_train_fit, y_train_fit,
                X_val, y_val,
                epochs=30,
                batch_size=32
            )
            
            # Evaluate
            metrics, _, _ = predictor.evaluate(X_test_subset, y_test)
            run_auc_scores.append(metrics['auc_roc'])
            run_accuracies.append(metrics['accuracy'])
            
            if n_runs > 1:
                print(f"AUC: {metrics['auc_roc']:.4f}")
        
        # Average metrics
        avg_auc = np.mean(run_auc_scores)
        std_auc = np.std(run_auc_scores)
        avg_accuracy = np.mean(run_accuracies)
        
        results_list.append({
            'feature_set': subset_name,
            'n_features': len(feature_list),
            'auc_roc': avg_auc,
            'auc_roc_std': std_auc,
            'accuracy': avg_accuracy,
        })
        
        print(f"  Average AUC-ROC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Calculate impact
    baseline_auc = results_df[results_df['feature_set'] == 'all_features']['auc_roc'].values[0]
    results_df['auc_drop'] = baseline_auc - results_df['auc_roc']
    results_df['auc_drop_pct'] = (results_df['auc_drop'] / baseline_auc) * 100
    
    return results_df


def main(args):
    """Main ablation study function"""
    
    os.makedirs(args.output, exist_ok=True)
    
    # Run ablation study
    results_df = run_ablation_study(
        args.data,
        args.model,
        n_runs=args.runs
    )
    
    # Print results
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(args.output, 'ablation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline = results_df[results_df['feature_set'] == 'all_features']['auc_roc'].values[0]
    print(f"\nBaseline Model (all features): AUC-ROC = {baseline:.4f}")
    
    print("\nFeature Group Impact (sorted by importance):")
    without_results = results_df[results_df['feature_set'].str.startswith('without_')]
    without_sorted = without_results.sort_values('auc_drop', ascending=False)
    
    for _, row in without_sorted.iterrows():
        group_name = row['feature_set'].replace('without_', '')
        print(f"  {group_name}: {row['auc_drop']:.4f} AUC drop ({row['auc_drop_pct']:.2f}%)")
    
    # Plot results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plots_dir = os.path.join(args.output, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    fig = plot_ablation_results(results_df, metric='auc_roc', 
                                title="Feature Group Ablation: Impact on AUC-ROC")
    fig.savefig(os.path.join(plots_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Ablation results plot saved")
    
    print("\n" + "="*60)
    print("✓ ABLATION STUDY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation study on CTR model')
    parser.add_argument('--data', type=str, default='data/ads_dataset.csv',
                       help='Path to input dataset')
    parser.add_argument('--model', type=str, default='models/ctr_model',
                       help='Directory containing trained model artifacts')
    parser.add_argument('--output', type=str, default='results/ablation',
                       help='Output directory for ablation results')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for each configuration')
    
    args = parser.parse_args()
    main(args)

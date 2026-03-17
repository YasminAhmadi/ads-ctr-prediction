#!/bin/bash

# Quick Setup Script for Ads CTR Prediction Project

echo "====================================================="
echo "Ads Click-Through Rate Prediction - Setup Script"
echo "====================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "Creating data and model directories..."
mkdir -p data models results/ablation results/evaluation

# Generate dataset
echo ""
echo "Generating simulated ads dataset..."
python3 -c "from src.data_generator import generate_ads_dataset; generate_ads_dataset('data/ads_dataset.csv', n_samples=50000)"

echo ""
echo "====================================================="
echo "✓ SETUP COMPLETE"
echo "====================================================="
echo ""
echo "Next steps:"
echo "1. Train the model:"
echo "   python3 train.py --data data/ads_dataset.csv --output models/ctr_model"
echo ""
echo "2. Run ablation studies:"
echo "   python3 ablation_study.py --data data/ads_dataset.csv --output results/ablation"
echo ""
echo "3. Evaluate the model:"
echo "   python3 evaluate.py --model models/ctr_model --output results/evaluation"
echo ""

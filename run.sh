#!/bin/bash
# Run the full pipeline
# Usage: bash run.sh

set -e

echo "=========================================="
echo "  AEP Ohio Load Forecasting Pipeline"
echo "=========================================="

echo ""
echo "[1/5] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/5] Data preprocessing & feature engineering..."
python src/data_prep.py

echo ""
echo "[3/5] Exploratory Data Analysis..."
python src/eda.py

echo ""
echo "[4/5] Training models..."
echo "  --- Ridge / Lasso / ElasticNet ---"
python src/ridge_model.py

echo ""
echo "  --- XGBoost ---"
python src/xgboost_model.py

echo ""
echo "  --- PatchTST Transformer ---"
python src/transformer_model.py

echo ""
echo "[5/5] Residual analysis & model comparison..."
python src/residual_analysis.py

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  Results in output/"
echo "=========================================="

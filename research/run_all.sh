#!/bin/bash
# =============================================================================
# run_all.sh
# =========
# Complete reproducibility script for KAN-GAT AML research project.
#
# Usage:
#   cd /Users/blank/Desktop/AMLSim
#   source aml_env/bin/activate
#   cd research
#   bash run_all.sh
#
# This will:
#   1. Build the graph dataset from raw CSVs
#   2. Compute tabular features
#   3. Train XGBoost baseline
#   4. Train GCN baseline
#   5. Train GAT+MLP baseline
#   6. Train GAT+Linear ablation
#   7. Train KAN-GAT (OUR METHOD)
#   8. Evaluate all models and generate figures
#   9. Generate KAN spline visualizations
#
# Total estimated runtime: ~20-40 minutes on CPU
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     KAN-GAT: Interpretable AML Detection Research          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Working directory: $(pwd)"
echo ""

# Check Python environment
PYTHON=$(command -v python3 || command -v python)
echo "[CHECK] Using Python: $PYTHON"
$PYTHON -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Activate the aml_env:"
    echo "  source $BASE_DIR/aml_env/bin/activate"
    exit 1
}
$PYTHON -c "import torch_geometric" 2>/dev/null || {
    echo "ERROR: PyTorch Geometric not found."
    exit 1
}

START_TIME=$(date +%s)

# =============================================================================
# STEP 1: Data Preparation
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 1/9: Building graph dataset..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 01_data_prep/build_graph_dataset.py
echo "  DONE."

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 2/9: Computing tabular features..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 01_data_prep/feature_engineering.py
echo "  DONE."

# =============================================================================
# STEP 3-5: Baselines
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 3/9: Training XGBoost baseline..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 02_baselines/run_xgboost.py
echo "  DONE."

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 4/9: Training GCN baseline..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 02_baselines/run_gcn.py
echo "  DONE."

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 5/9: Training GAT+MLP baseline & GAT+Linear ablation..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 02_baselines/run_gat_mlp.py
echo "  DONE."

# =============================================================================
# STEP 6: KAN-GAT (OUR METHOD)
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 6/9: Training KAN-GAT (OUR METHOD)..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 03_kan_gat/train_kan_gat.py
echo "  DONE."

# =============================================================================
# STEP 7-8: Evaluation & Figures
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 7/9: Evaluating all models..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 04_eval/evaluate_all.py
echo "  DONE."

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 8/9: Generating paper figures..."
echo "────────────────────────────────────────────────────────────────"
$PYTHON 04_eval/plot_results.py
echo "  DONE."

# =============================================================================
# STEP 9: KAN Spline Visualization
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "STEP 9/9: Generating KAN spline visualization..."
echo "────────────────────────────────────────────────────────────────"
if [ -f "results/kan_gat_best_model.pt" ]; then
    $PYTHON 03_kan_gat/visualize_splines.py
    echo "  DONE."
else
    echo "  SKIPPED: KAN-GAT model not found at results/kan_gat_best_model.pt"
fi

# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE COMPLETE                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total time: $(($DURATION / 60))m $(($DURATION % 60))s"
echo ""
echo "  Output files:"
echo "    results/all_results.json     - All metrics"
echo "    results/aml_graph.pt         - PyG graph dataset"
echo "    results/tabular_features.csv - Tabular features"
echo "    results/kan_gat_best_model.pt - Best KAN-GAT weights"
echo ""
echo "  Figures:"
echo "    figures/pr_curves.png        - PR curves comparison"
echo "    figures/ablation_bar.png     - Ablation study bar chart"
echo "    figures/spline_plot.png      - KAN spline visualization"
echo "    figures/feature_importance.png - Feature importance"
echo ""
echo "  To view results table:"
echo "    cat results/all_results.json"
echo ""

# Print the results table
if [ -f "results/all_results.json" ]; then
    echo "Results Summary:"
    echo "────────────────────────────────────────────────────────────────"
    $PYTHON -c "
import json
with open('results/all_results.json') as f:
    data = json.load(f)
for model, metrics in data.items():
    print(f'  {model:15s}: AUC-PR={metrics[\"AUC-PR\"]}, F2={metrics[\"F2\"]}')
"
    echo ""
fi
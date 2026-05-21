"""
plot_results.py
===============
Standalone script to regenerate publication-quality figures from saved results.
Can run independently after all models have been evaluated.

Usage:
    python plot_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "04_eval"))
from evaluate_all import load_all_results, plot_pr_curves, plot_ablation_study, print_results_table

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_results_summary():
    """Generate all paper figures from saved results."""
    print("=" * 60)
    print("Paper Figure Generator")
    print("=" * 60)

    all_results = load_all_results()

    if not all_results:
        print("No results found. Run 'bash run_all.sh' first.")
        return

    print_results_table(all_results)

    # Figure 1: PR Curves
    print("\nGenerating Figure 1: PR Curves...")
    plot_pr_curves(all_results, os.path.join(FIGURES_DIR, "pr_curves.png"))

    # Figure 2: Ablation Study
    print("Generating Figure 2: Ablation Study...")
    plot_ablation_study(all_results, os.path.join(FIGURES_DIR, "ablation_bar.png"))

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    plot_results_summary()
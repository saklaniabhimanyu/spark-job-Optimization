import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import FEATURES, OUTPUT_DIR


def compute_shap_values(model, X, feature_names=None):
    """Compute SHAP values for a fitted tree-based model (XGBoost, Random
    Forest, Gradient Boosting all supported via TreeExplainer)."""
    import shap

    feature_names = feature_names or FEATURES
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_values.feature_names = feature_names
    return shap_values


def plot_shap_summary(shap_values, save_path=None, show=True):
    """Beeswarm summary plot: feature impact + direction across all
    predictions in one view."""
    import shap

    save_path = save_path or os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path


def plot_shap_dependence(shap_values, feature_name, save_path=None, show=True):
    """Dependence plot for a single feature: how does the model's predicted
    execution time change as this one feature varies?"""
    import shap

    save_path = save_path or os.path.join(
        OUTPUT_DIR, f"shap_dependence_{feature_name}.png")
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, feature_name], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path


def explain_single_prediction(model, x_row, feature_names=None):
    """Return a sorted list of (feature, shap_value) for one prediction —
    used to explain a single recommendation in human-readable terms, e.g.
    'partitions=200 added +3.1s vs. baseline; cache_enabled=1 saved -1.8s'."""
    import shap

    feature_names = feature_names or FEATURES
    explainer = shap.TreeExplainer(model)
    x_row = np.asarray(x_row).reshape(1, -1)
    sv = explainer(x_row)

    contributions = list(zip(feature_names, sv.values[0]))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    print(f"Base value (average predicted execution time): {sv.base_values[0]:.2f}s")
    print("Per-feature contribution to this prediction:")
    for feat, val in contributions:
        sign = "+" if val >= 0 else ""
        print(f"  {feat:<20} {sign}{val:.3f}s")

    return contributions

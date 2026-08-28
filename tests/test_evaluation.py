"""
Tests for src/evaluation.py — pure Python/sklearn, no Spark required.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.evaluation import ablation_study, bootstrap_ci, paired_model_significance


def _make_regression_data(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    true_coef = np.array([3.0, -1.5, 0.0, 0.5])  # feature 2 is pure noise
    y = X @ true_coef + rng.normal(scale=0.5, size=n)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y, cols


def test_paired_model_significance_returns_expected_shape():
    X_df, y, cols = _make_regression_data()
    models = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=20, random_state=0),
    }
    results_df, fold_scores = paired_model_significance(models, X_df.values, y, cv_folds=5)

    assert len(results_df) == 1  # C(2,2) pairs
    assert set(fold_scores.keys()) == {"linear", "rf"}
    assert len(fold_scores["linear"]) == 5
    row = results_df.iloc[0]
    assert 0.0 <= row["t_pvalue"] <= 1.0
    assert 0.0 <= row["wilcoxon_pvalue"] <= 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.1, size=100)

    result = bootstrap_ci(y_true, y_pred, r2_score, n_bootstrap=200, seed=1)

    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["ci_level"] == 0.95
    assert result["n_bootstrap"] == 200


def test_ablation_study_flags_the_noise_feature_as_least_important():
    X_df, y, cols = _make_regression_data(n=300, seed=2)

    def factory():
        return LinearRegression()

    result = ablation_study(factory, X_df, y, cols, cv_folds=5, seed=2)

    assert "removed_feature" in result.columns
    assert len(result) == len(cols) + 1  # +1 for the "full model" baseline row

    # f2 has a true coefficient of 0 — removing it should hurt performance
    # the least (smallest negative or largest delta) among the real features.
    real_feature_rows = result[result["removed_feature"] != "(none — full model)"]
    least_harmful = real_feature_rows.sort_values("delta_vs_full", ascending=False).iloc[0]
    assert least_harmful["removed_feature"] == "f2"

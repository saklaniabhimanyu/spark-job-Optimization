from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score


def paired_model_significance(models: dict, X, y, cv_folds=5, seed=42,
                               metric="r2", alpha=0.05):
    """For every pair of models, run the SAME KFold splits and compare their
    per-fold scores with a paired t-test AND the non-parametric Wilcoxon
    signed-rank test (reported together since t-test assumes normally
    distributed fold-to-fold differences, which is a strong assumption at
    n_folds=5).

    Returns a DataFrame: model_a, model_b, mean_diff, t_stat, t_pvalue,
    wilcoxon_stat, wilcoxon_pvalue, significant_at_alpha.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_scores = {}
    for name, model in models.items():
        fold_scores[name] = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)

    rows = []
    for name_a, name_b in combinations(fold_scores.keys(), 2):
        a, b = fold_scores[name_a], fold_scores[name_b]
        diff = a - b

        t_stat, t_p = stats.ttest_rel(a, b)

        # Wilcoxon requires at least one non-zero difference
        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = stats.wilcoxon(a, b)

        rows.append({
            "model_a": name_a, "model_b": name_b,
            "mean_diff": float(diff.mean()),
            "t_stat": float(t_stat), "t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
            "significant_at_alpha": bool(t_p < alpha),
        })

    return pd.DataFrame(rows), fold_scores


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a point metric computed on
    (y_true, y_pred) pairs, e.g. metric_fn=sklearn.metrics.r2_score.

    Resamples (with replacement) the *test-set predictions* n_bootstrap
    times and recomputes the metric each time, then reports the empirical
    percentile interval. This quantifies how much the reported metric would
    plausibly move around if you'd gotten a slightly different test split.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(y_true[idx], y_pred[idx])

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    lower, upper = np.percentile(boot_scores, [lower_pct, upper_pct])
    point_estimate = metric_fn(y_true, y_pred)

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def ablation_study(model_factory, X_df: pd.DataFrame, y, feature_cols,
                    cv_folds=5, seed=42, metric="r2"):
    """Leave-one-feature-out ablation: for each feature, retrain/re-evaluate
    with that feature removed and report how much CV performance drops.
    A large drop = that feature is doing real work; a near-zero (or
    positive) change = the model isn't relying on it.

    model_factory: a zero-arg callable returning a fresh unfitted estimator
    (e.g. `lambda: XGBRegressor(**best_params)`) — each feature subset needs
    its own untrained model instance.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    full_scores = cross_val_score(model_factory(), X_df[feature_cols].values, y,
                                   cv=cv, scoring=metric, n_jobs=-1)
    full_mean = full_scores.mean()

    rows = [{"removed_feature": "(none — full model)", "cv_mean": full_mean,
             "cv_std": full_scores.std(), "delta_vs_full": 0.0}]

    for feat in feature_cols:
        remaining = [f for f in feature_cols if f != feat]
        scores = cross_val_score(model_factory(), X_df[remaining].values, y,
                                  cv=cv, scoring=metric, n_jobs=-1)
        rows.append({
            "removed_feature": feat,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "delta_vs_full": scores.mean() - full_mean,
        })

    df = pd.DataFrame(rows).sort_values("delta_vs_full")
    return df.reset_index(drop=True)

"""
Trains and compares regression models for predicting Spark job execution
time, then tunes the best-performing model (XGBoost) via RandomizedSearchCV.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import FEATURES, MODELS_DIR, RANDOM_STATE, TARGET
from src.experiment_tracker import log_run


def get_baseline_models():
    """Candidate model zoo compared before tuning."""
    return {
        "Linear Regression": Pipeline([
            ("sc", StandardScaler()),
            ("m", LinearRegression()),
        ]),
        "Elastic Net": Pipeline([
            ("sc", StandardScaler()),
            ("m", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_and_compare(df_raw: pd.DataFrame):
    """Fit each candidate model, score it, and return (results_dict, X, y,
    X_train, X_test, y_train, y_test)."""
    X = df_raw[FEATURES].values
    y = df_raw[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = get_baseline_models()
    results_ml = {}

    print("\nTraining ML models …\n")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}  CV-R2 (+/-std)")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        results_ml[name] = {
            "model": model, "MAE": mae, "RMSE": rmse, "R2": r2,
            "CV_mean": cv_scores.mean(), "CV_std": cv_scores.std(),
        }
        print(f"{name:<25} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}"
              f"  {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

    best_name = max(results_ml, key=lambda k: results_ml[k]["CV_mean"])
    print(f"\nBest model (initial): {best_name}")
    print(f"    CV R2           : {results_ml[best_name]['CV_mean']:.4f}")

    return results_ml, X, y, X_train, X_test, y_train, y_test


def tune_xgboost(results_ml, X, y, X_train, X_test, y_train, y_test, n_iter=40):
    """RandomizedSearchCV over XGBoost's hyperparameter space; adds the tuned
    model into results_ml under 'XGBoost (tuned)' and returns (results_ml,
    best_name, best_model)."""
    print("\nTuning XGBoost hyperparameters …\n")

    xgb_base = XGBRegressor(objective="reg:squarederror",
                             random_state=RANDOM_STATE, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
        "reg_lambda": [0.5, 1, 1.5, 2, 5],
        "min_child_weight": [1, 3, 5, 7],
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        n_iter=n_iter,          # raise for a deeper search
        scoring="r2",
        cv=5,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_
    xgb_preds = best_xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_r2 = r2_score(y_test, xgb_preds)

    print("\nTuned XGBoost - Best Parameters:")
    print(xgb_search.best_params_)
    print("\nTuned XGBoost Performance")
    print(f"  MAE  : {xgb_mae:.2f}")
    print(f"  RMSE : {xgb_rmse:.2f}")
    print(f"  R2   : {xgb_r2:.4f}")
    print(f"  CV R2: {xgb_search.best_score_:.4f}")

    results_ml["XGBoost (tuned)"] = {
        "model": best_xgb, "MAE": xgb_mae, "RMSE": xgb_rmse, "R2": xgb_r2,
        "CV_mean": xgb_search.best_score_, "CV_std": 0.0,
    }

    best_name = max(results_ml, key=lambda k: results_ml[k]["R2"])
    best_model = results_ml[best_name]["model"]
    print(f"\nFinal best model : {best_name}")
    print(f"    R2            : {results_ml[best_name]['R2']:.4f}")

    log_run(
        run_name="xgboost_randomsearch_tuning",
        params=xgb_search.best_params_,
        metrics={"R2": xgb_r2, "MAE": xgb_mae, "RMSE": xgb_rmse,
                 "CV_R2": xgb_search.best_score_},
        tags={"model": "xgboost", "stage": "tuning", "search": "RandomizedSearchCV"},
    )

    return results_ml, best_name, best_model


def get_feature_importances(model, results_ml, best_name):
    """Extract feature importances from the best model; falls back to
    |coefficients| for linear models, or Random Forest if unavailable."""
    inner = model.named_steps["m"] if hasattr(model, "named_steps") else model

    if hasattr(inner, "feature_importances_"):
        return inner.feature_importances_, best_name
    if hasattr(inner, "coef_"):
        coefs = np.abs(inner.coef_)
        return coefs / coefs.sum(), f"{best_name} |coef|"

    rf_fallback = results_ml["Random Forest"]["model"]
    return rf_fallback.feature_importances_, "Random Forest (fallback)"


def save_model(model, filename="best_model.joblib"):
    """Persist the trained model to models/ for later inference."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(filename="best_model.joblib"):
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from src.config import RANDOM_STATE


def _objective(trial, X, y, cv_folds=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, **params
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    return float(scores.mean())


def tune_xgboost_bayesian(X, y, n_trials=50, cv_folds=5, seed=RANDOM_STATE, verbose=True):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(lambda t: _objective(t, X, y, cv_folds=cv_folds),
                    n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params
    best_model = XGBRegressor(
        objective="reg:squarederror", random_state=seed, n_jobs=-1, **best_params
    )
    best_model.fit(X, y)

    if verbose:
        print(f"Best trial: #{study.best_trial.number}  CV R2 = {study.best_value:.4f}")
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k:<20}: {v}")

    return best_model, best_params, study


def compare_to_random_search(bayes_best_r2, random_search_best_r2):
    """Small helper for reporting the delta between the two tuning strategies
    in the notebook / README."""
    delta = bayes_best_r2 - random_search_best_r2
    print(f"RandomizedSearchCV best CV R2 : {random_search_best_r2:.4f}")
    print(f"Optuna (TPE) best CV R2       : {bayes_best_r2:.4f}")
    print(f"Delta                         : {delta:+.4f}")
    return delta
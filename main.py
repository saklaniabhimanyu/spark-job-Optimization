import argparse
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from src.comparison import compare_default_vs_optimized
from src.config import DATA_DIR, FEATURES, HARDWARE_CONFIGS, RANDOM_STATE, TARGET
from src.eda import plot_eda
from src.model_training import save_model
from src.recommendation_engine import print_recommendation
from src.spark_utils import generate_and_merge_all_configs


def parse_args():
    parser = argparse.ArgumentParser(description="AI-based Spark job optimizer pipeline")
    parser.add_argument("--rows-per-config", type=int, default=60,
                         help="benchmark rows to collect PER hardware config "
                              "(total rows = this x len(HARDWARE_CONFIGS))")
    parser.add_argument("--search-iter", type=int, default=40,
                         help="RandomizedSearchCV iterations for XGBoost tuning")
    parser.add_argument("--skip-comparison", action="store_true",
                         help="skip the default-vs-optimized real Spark run at the end")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Generating benchmark dataset across {len(HARDWARE_CONFIGS)} hardware "
          f"configs ({args.rows_per_config} rows each) ...")
    df = generate_and_merge_all_configs(
        HARDWARE_CONFIGS, total_target_per_config=args.rows_per_config)
    merged_path = os.path.join(DATA_DIR, "Spark_realtime_metrices.csv")
    df.to_csv(merged_path, index=False)
    print(f"Merged dataset -> {merged_path} ({len(df)} rows)")

    print("\nGenerating EDA plots ...")
    plot_eda(df, show=False)

    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    print(f"\nTuning XGBoost ({args.search_iter} iterations) ...")
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_grid, n_iter=args.search_iter, cv=5,
        scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    preds = model.predict(X_test)
    print("\nHold-out test performance")
    print(f"  R2   : {r2_score(y_test, preds):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_test, preds):.2f}s")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, preds)):.2f}s")

    save_model(model)

    print("\nSample recommendation:")
    sample_workload = {
        "num_records": 1_000_000, "skewness": 0.3, "joins": 1,
        "group_cardinality": 1_000, "num_aggregations": 2,
        "has_orderby": 0, "filter_selectivity": 0.7,
    }
    print_recommendation(model, sample_workload)

    if not args.skip_comparison:
        compare_default_vs_optimized(model)

    print("\nDone.")


if __name__ == "__main__":
    main()

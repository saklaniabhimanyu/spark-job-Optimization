import pandas as pd

from src.config import CANDIDATE_CONFIGS, FEATURES
from src.spark_utils import compute_derived_features


def _build_feature_row(workload: dict, tunable: dict):
    """Assemble one full feature row in the exact FEATURES column order."""
    derived = compute_derived_features(
        num_records=workload["num_records"],
        cores=tunable["cores"],
        memory_gb=tunable["executor_memory_gb"],
        partitions=tunable["partitions"],
        joins=workload["joins"],
    )
    full = {**workload, **tunable, **derived}
    return [full[f] for f in FEATURES]


def recommend_configs(model, workload: dict, top_n=5, candidate_grid=None):
    """Score every candidate tunable configuration against a fixed workload
    profile and return the top_n with lowest predicted execution time.

    workload must contain: num_records, skewness, joins, group_cardinality,
    num_aggregations, has_orderby, filter_selectivity.
    """
    candidate_grid = candidate_grid or CANDIDATE_CONFIGS

    rows = []
    for tunable in candidate_grid:
        feature_row = _build_feature_row(workload, tunable)
        predicted_time = model.predict([feature_row])[0]
        derived = compute_derived_features(
            num_records=workload["num_records"], cores=tunable["cores"],
            memory_gb=tunable["executor_memory_gb"], partitions=tunable["partitions"],
            joins=workload["joins"],
        )
        rows.append({**tunable, "predicted_time_sec": predicted_time, **derived})

    df_ranked = pd.DataFrame(rows).sort_values("predicted_time_sec").reset_index(drop=True)
    return df_ranked.head(top_n)


def print_recommendation(model, workload: dict):
    """Convenience wrapper: pretty-print the single best recommendation."""
    top1 = recommend_configs(model, workload, top_n=1).iloc[0]
    print(f"Workload: {workload['num_records']:,} records, "
          f"skew={workload['skewness']:.2f}, joins={workload['joins']}, "
          f"group_cardinality={workload['group_cardinality']}, "
          f"aggregations={workload['num_aggregations']}, "
          f"orderby={bool(workload['has_orderby'])}, "
          f"filter_selectivity={workload['filter_selectivity']}")
    print("Recommended configuration:")
    print(f"  Cores           : {int(top1['cores'])}")
    print(f"  Executor memory : {int(top1['executor_memory_gb'])}g")
    print(f"  Partitions      : {int(top1['partitions'])}")
    print(f"  Cache enabled   : {bool(top1['cache_enabled'])}")
    print(f"  Predicted time  : {top1['predicted_time_sec']:.2f}s")
    return top1

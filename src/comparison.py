import pandas as pd

from src.config import DEFAULT_SPARK_CONFIG, TEST_CASES
from src.recommendation_engine import recommend_configs
from src.spark_utils import create_spark_session, run_job


def _get_session(session_cache: dict, cores: int, memory_gb: int):
    key = (cores, memory_gb)
    if key not in session_cache:
        session_cache[key] = create_spark_session(cores, memory_gb)
    return session_cache[key]


def _workload_dict(case):
    n_records, skew, joins, group_card, n_aggs, orderby, filter_sel, _label = case
    return {
        "num_records": n_records, "skewness": skew, "joins": joins,
        "group_cardinality": group_card, "num_aggregations": n_aggs,
        "has_orderby": orderby, "filter_selectivity": filter_sel,
    }


def compare_default_vs_optimized(model, test_cases=None, use_real_spark=True, session_cache=None, stop_sessions_after=True):
    test_cases = test_cases or TEST_CASES
    owns_cache = session_cache is None
    session_cache = session_cache if session_cache is not None else {}
    rows = []

    print("\nDefault vs. AI-optimized comparison\n")
    try:
        for case in test_cases:
            workload = _workload_dict(case)
            label = case[-1]
            top1 = recommend_configs(model, workload, top_n=1).iloc[0]

            if use_real_spark:
                default_spark = _get_session(session_cache, DEFAULT_SPARK_CONFIG["cores"],
                                              DEFAULT_SPARK_CONFIG["executor_memory_gb"])
                default_time = run_job(
                    default_spark, num_records=workload["num_records"],
                    partitions=DEFAULT_SPARK_CONFIG["partitions"], skew=workload["skewness"],
                    joins=workload["joins"], cache=DEFAULT_SPARK_CONFIG["cache_enabled"],
                    memory_gb=DEFAULT_SPARK_CONFIG["executor_memory_gb"],
                    group_cardinality=workload["group_cardinality"],
                    num_aggregations=workload["num_aggregations"],
                    has_orderby=workload["has_orderby"],
                    filter_selectivity=workload["filter_selectivity"],
                )["execution_time"]

                opt_spark = _get_session(session_cache, int(top1["cores"]),
                                          int(top1["executor_memory_gb"]))
                optimized_time = run_job(
                    opt_spark, num_records=workload["num_records"],
                    partitions=int(top1["partitions"]), skew=workload["skewness"],
                    joins=workload["joins"], cache=int(top1["cache_enabled"]),
                    memory_gb=int(top1["executor_memory_gb"]),
                    group_cardinality=workload["group_cardinality"],
                    num_aggregations=workload["num_aggregations"],
                    has_orderby=workload["has_orderby"],
                    filter_selectivity=workload["filter_selectivity"],
                )["execution_time"]
            else:
                from src.recommendation_engine import _build_feature_row
                default_row = _build_feature_row(workload, DEFAULT_SPARK_CONFIG)
                default_time = model.predict([default_row])[0]
                optimized_time = top1["predicted_time_sec"]

            improvement_pct = (default_time - optimized_time) / default_time * 100
            rows.append({
                "profile": label,
                "default_time_sec": round(default_time, 2),
                "optimized_time_sec": round(optimized_time, 2),
                "improvement_pct": round(improvement_pct, 1),
            })
            print(f"  {label:<45} default={default_time:.2f}s  "
                  f"optimized={optimized_time:.2f}s  improvement={improvement_pct:+.1f}%")

        df = pd.DataFrame(rows)
        print(f"\nAverage improvement: {df['improvement_pct'].mean():+.1f}%")
        return df

    finally:
        if owns_cache and stop_sessions_after:
            for spark in session_cache.values():
                spark.stop()
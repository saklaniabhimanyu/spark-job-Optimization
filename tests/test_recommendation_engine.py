"""
Unit tests that don't require a running SparkSession — fast checks on the
config grid and recommendation-engine logic. Run with: pytest tests/
"""

import numpy as np

from src.config import CANDIDATE_CONFIGS, FEATURES, TEST_CASES, build_candidate_configs
from src.recommendation_engine import recommend_configs
from src.spark_utils import compute_derived_features


class DummyModel:
    """Predicts execution time as a simple deterministic function of the
    16-feature vector (FEATURES order), so recommend_configs can be tested
    without training a real model or starting Spark."""

    def predict(self, X):
        X = np.array(X, dtype=float)
        idx = {f: i for i, f in enumerate(FEATURES)}
        num_records = X[:, idx["num_records"]]
        cores = X[:, idx["cores"]]
        partitions = X[:, idx["partitions"]]
        skewness = X[:, idx["skewness"]]
        cache_enabled = X[:, idx["cache_enabled"]]
        # more partitions/cores/cache -> faster; more skew -> slower
        return (num_records / (partitions * cores + 1)) * (1 + skewness) * \
            np.where(cache_enabled > 0, 0.8, 1.0)


SAMPLE_WORKLOAD = {
    "num_records": 1_000_000, "skewness": 0.4, "joins": 1,
    "group_cardinality": 1_000, "num_aggregations": 2,
    "has_orderby": 0, "filter_selectivity": 0.7,
}


def test_build_candidate_configs_matches_grid_size():
    configs = build_candidate_configs()
    assert len(configs) == len(CANDIDATE_CONFIGS)
    assert len(configs) > 0
    for cfg in configs[:5]:
        assert set(cfg.keys()) == {"cores", "executor_memory_gb", "partitions", "cache_enabled"}


def test_features_list_matches_real_csv_column_order():
    # Order matters: model training reads df[FEATURES].values, and
    # recommend_configs builds feature rows in this exact order.
    assert FEATURES == [
        "num_records", "cores", "executor_memory_gb", "partitions",
        "partition_efficiency", "skewness", "joins", "cache_enabled",
        "group_cardinality", "num_aggregations", "has_orderby",
        "filter_selectivity", "data_size_gb", "shuffle_size_mb",
        "shuffle_intensity", "memory_pressure",
    ]


def test_recommend_configs_returns_sorted_top_n():
    model = DummyModel()
    top5 = recommend_configs(model, SAMPLE_WORKLOAD, top_n=5)

    assert len(top5) == 5
    times = top5["predicted_time_sec"].tolist()
    assert times == sorted(times)


def test_recommend_configs_respects_top_n():
    model = DummyModel()
    for n in (1, 3, 10):
        result = recommend_configs(model, SAMPLE_WORKLOAD, top_n=n)
        assert len(result) == n


def test_recommend_configs_output_has_derived_features():
    model = DummyModel()
    top1 = recommend_configs(model, SAMPLE_WORKLOAD, top_n=1).iloc[0]
    for col in ("partition_efficiency", "data_size_gb", "shuffle_size_mb",
                "shuffle_intensity", "memory_pressure"):
        assert col in top1


def test_compute_derived_features_matches_real_formulas():
    result = compute_derived_features(
        num_records=1_000_000, cores=2, memory_gb=4, partitions=8, joins=1)

    expected_data_size_gb = round((1_000_000 * 100) / (1024 ** 3), 4)
    expected_partition_efficiency = round(8 / 2, 4)
    expected_shuffle_size_mb = round(1_000_000 * 1 * 0.00005, 4)
    expected_shuffle_intensity = round(1 * expected_data_size_gb, 4)
    expected_memory_pressure = round(expected_data_size_gb / 4, 4)

    assert result["data_size_gb"] == expected_data_size_gb
    assert result["partition_efficiency"] == expected_partition_efficiency
    assert result["shuffle_size_mb"] == expected_shuffle_size_mb
    assert result["shuffle_intensity"] == expected_shuffle_intensity
    assert result["memory_pressure"] == expected_memory_pressure


def test_test_cases_well_formed():
    for case in TEST_CASES:
        n_records, skew, joins, group_card, n_aggs, orderby, filter_sel, label = case
        assert n_records > 0
        assert 0 <= skew <= 1
        assert joins >= 0
        assert group_card > 0
        assert 1 <= n_aggs <= 4
        assert orderby in (0, 1)
        assert 0 <= filter_sel <= 1
        assert isinstance(label, str) and len(label) > 0

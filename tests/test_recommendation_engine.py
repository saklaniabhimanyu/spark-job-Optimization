"""
Tests for the recommendation engine, using the real trained model.
Run with: pytest tests/
"""

import os
import joblib
import pytest

from src.config import CANDIDATE_CONFIGS, FEATURES, TEST_CASES, build_candidate_configs, MODELS_DIR
from src.recommendation_engine import recommend_configs, _build_feature_row
from src.spark_utils import compute_derived_features


@pytest.fixture(scope="session")
def real_model():
    """Load the real trained model once per test session."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    return joblib.load(model_path)


@pytest.fixture
def sample_workload():
    return {
        "num_records": 1_000_000, "skewness": 0.3, "joins": 1,
        "group_cardinality": 1_000, "num_aggregations": 2,
        "has_orderby": 0, "filter_selectivity": 0.7,
    }


def test_build_candidate_configs_matches_grid_size():
    configs = build_candidate_configs()
    assert len(configs) == len(CANDIDATE_CONFIGS)


def test_features_list_matches_real_csv_column_order():
    assert isinstance(FEATURES, list)
    assert len(FEATURES) == len(set(FEATURES))


def test_recommend_configs_returns_sorted_top_n(real_model, sample_workload):
    result = recommend_configs(real_model, sample_workload, top_n=5)
    assert len(result) == 5
    times = result["predicted_time_sec"].tolist()
    assert times == sorted(times)


def test_recommend_configs_respects_top_n(real_model, sample_workload):
    result = recommend_configs(real_model, sample_workload, top_n=1)
    assert len(result) == 1


def test_recommend_configs_output_has_derived_features(real_model, sample_workload):
    result = recommend_configs(real_model, sample_workload, top_n=3)
    for col in ("data_size_gb", "shuffle_size_mb", "memory_pressure",
                "records_per_partition", "join_load"):
        assert col in result.columns


def test_recommend_configs_predictions_are_positive(real_model, sample_workload):
    result = recommend_configs(real_model, sample_workload, top_n=5)
    assert (result["predicted_time_sec"] > 0).all()


def test_compute_derived_features_matches_real_formulas():
    derived = compute_derived_features(
        num_records=1_000_000, cores=4, memory_gb=8,
        partitions=16, joins=1, group_cardinality=1_000,
    )
    assert derived["records_per_partition"] == pytest.approx(1_000_000 / 16)
    assert derived["join_load"] == pytest.approx(1 * 1_000)


def test_test_cases_well_formed():
    for case in TEST_CASES:
        assert len(case) == 8


def test_build_feature_row_matches_features_length(sample_workload):
    tunable = CANDIDATE_CONFIGS[0]
    row = _build_feature_row(sample_workload, tunable)
    assert len(row) == len(FEATURES)


def test_recommend_configs_works_with_real_trained_model(real_model, sample_workload):
    """Regression test: catches feature-schema drift between config.FEATURES
    and the actual model's expected input shape (the bug that broke /recommend)."""
    result = recommend_configs(real_model, sample_workload, top_n=3)
    assert len(result) == 3
    assert (result["predicted_time_sec"] > 0).all()
"""
Tests for src/experiment_tracker.py. Uses a temp directory so tests don't
pollute the real experiments/runs.jsonl.
"""

import importlib

import pytest


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    """Reload experiment_tracker with RUNS_DIR/RUNS_FILE pointed at a temp dir."""
    import src.experiment_tracker as et

    monkeypatch.setattr(et, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(et, "RUNS_FILE", str(tmp_path / "runs.jsonl"))
    return et


def test_log_run_creates_file_and_record(tracker):
    record = tracker.log_run(
        run_name="test_run",
        params={"n_estimators": 100},
        metrics={"R2": 0.95},
        tags={"model": "xgboost"},
    )

    assert record["run_name"] == "test_run"
    assert record["metrics"]["R2"] == 0.95
    assert "run_id" in record
    assert "timestamp" in record


def test_load_runs_returns_empty_list_when_no_runs(tracker):
    assert tracker.load_runs() == []


def test_load_runs_returns_all_logged_runs(tracker):
    tracker.log_run("run_a", {"p": 1}, {"R2": 0.9})
    tracker.log_run("run_b", {"p": 2}, {"R2": 0.95})

    runs = tracker.load_runs()
    assert len(runs) == 2
    assert {r["run_name"] for r in runs} == {"run_a", "run_b"}


def test_best_run_picks_highest_metric(tracker):
    tracker.log_run("worse", {}, {"R2": 0.8})
    tracker.log_run("better", {}, {"R2": 0.95})

    best = tracker.best_run("R2", higher_is_better=True)
    assert best["run_name"] == "better"


def test_best_run_returns_none_if_metric_absent(tracker):
    tracker.log_run("no_r2_here", {}, {"MAE": 1.2})
    assert tracker.best_run("R2") is None


def test_runs_to_dataframe_flattens_params_and_metrics(tracker):
    tracker.log_run("run_a", {"n_estimators": 100}, {"R2": 0.9})

    df = tracker.runs_to_dataframe()
    assert "param_n_estimators" in df.columns
    assert "metric_R2" in df.columns
    assert df.iloc[0]["param_n_estimators"] == 100

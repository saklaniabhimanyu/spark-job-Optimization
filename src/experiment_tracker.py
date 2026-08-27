"""
Minimal, dependency-free experiment tracking: every training run appends one
JSON record (params, metrics, timestamp, git commit if available) to
experiments/runs.jsonl. Not a replacement for MLflow/W&B in a real team
setting, but demonstrates the *concept* — reproducible runs, comparable
metrics over time — without needing a tracking server for a solo/portfolio
project.
"""

import json
import os
import subprocess
import time
import uuid

from src.config import PROJECT_ROOT

RUNS_DIR = os.path.join(PROJECT_ROOT, "experiments")
RUNS_FILE = os.path.join(RUNS_DIR, "runs.jsonl")


def _git_commit_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def log_run(run_name, params: dict, metrics: dict, tags: dict = None):
    """Append one experiment run to experiments/runs.jsonl.

    params  : hyperparameters / config used for this run
    metrics : evaluation results (R2, MAE, RMSE, CV scores, ...)
    tags    : free-form metadata (e.g. {"model": "xgboost", "stage": "tuning"})
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    record = {
        "run_id": uuid.uuid4().hex[:12],
        "run_name": run_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit_hash(),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
    }

    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Logged run '{run_name}' ({record['run_id']}) -> {RUNS_FILE}")
    return record


def load_runs():
    """Load all logged runs as a list of dicts (empty list if none logged yet)."""
    if not os.path.exists(RUNS_FILE):
        return []
    with open(RUNS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def best_run(metric_name, higher_is_better=True):
    """Return the logged run with the best value for a given metric key."""
    runs = load_runs()
    scored = [r for r in runs if metric_name in r.get("metrics", {})]
    if not scored:
        return None
    return max(scored, key=lambda r: r["metrics"][metric_name] * (1 if higher_is_better else -1))


def runs_to_dataframe():
    """Flatten logged runs into a pandas DataFrame for quick comparison
    (one row per run, params/metrics prefixed to avoid name collisions)."""
    import pandas as pd

    runs = load_runs()
    rows = []
    for r in runs:
        row = {"run_id": r["run_id"], "run_name": r["run_name"],
               "timestamp": r["timestamp"], "git_commit": r["git_commit"]}
        row.update({f"param_{k}": v for k, v in r.get("params", {}).items()})
        row.update({f"metric_{k}": v for k, v in r.get("metrics", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)

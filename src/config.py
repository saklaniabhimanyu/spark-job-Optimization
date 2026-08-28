import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DATA_PATH = os.path.join(DATA_DIR, "Spark_realtime_metrices.csv")

RANDOM_STATE = 42
BASE_SEED = 42  
WORKLOAD_FEATURES = [
    "num_records", "skewness", "joins", "group_cardinality",
    "num_aggregations", "has_orderby", "filter_selectivity",
]

TUNABLE_FEATURES = ["cores", "executor_memory_gb", "partitions", "cache_enabled"]

DERIVED_FEATURES = [
    "partition_efficiency", "data_size_gb", "shuffle_size_mb",
    "shuffle_intensity", "memory_pressure",
]

# Exact column order of the real benchmark CSV (minus the target).
FEATURES = [
    "num_records", "cores", "executor_memory_gb", "partitions",
    "partition_efficiency", "skewness", "joins", "cache_enabled",
    "group_cardinality", "num_aggregations", "has_orderby",
    "filter_selectivity", "data_size_gb", "shuffle_size_mb",
    "shuffle_intensity", "memory_pressure",
]
TARGET = "execution_time"

assert set(FEATURES) == set(WORKLOAD_FEATURES) | set(TUNABLE_FEATURES) | set(DERIVED_FEATURES)

DATASET_SIZES = [100_000, 500_000, 1_000_000, 3_000_000, 5_000_000, 10_000_000, 20_000_000]
GROUP_CARDINALITIES = [100, 1_000, 5_000, 20_000]
MAX_AGGREGATIONS = 4
FILTER_SELECTIVITY_RANGE = (0.3, 1.0)
SKEW_RANGE = (0.1, 0.9)
JOIN_OPTIONS = [1, 2, 3]
PARTITION_MULTIPLIER_RANGE = (2, 6)  
HARDWARE_CONFIGS = [
    {"cores": 1, "executor_memory_gb": 1},
    {"cores": 1, "executor_memory_gb": 2},
    {"cores": 1, "executor_memory_gb": 4},
    {"cores": 1, "executor_memory_gb": 6},
    {"cores": 1, "executor_memory_gb": 8},

    {"cores": 2, "executor_memory_gb": 1},
    {"cores": 2, "executor_memory_gb": 2},
    {"cores": 2, "executor_memory_gb": 4},
    {"cores": 2, "executor_memory_gb": 6},
    {"cores": 2, "executor_memory_gb": 8},

    {"cores": 4, "executor_memory_gb": 1},
    {"cores": 4, "executor_memory_gb": 2},
    {"cores": 4, "executor_memory_gb": 4},
    {"cores": 4, "executor_memory_gb": 6},
    {"cores": 4, "executor_memory_gb": 8},

    {"cores": 6, "executor_memory_gb": 1},
    {"cores": 6, "executor_memory_gb": 2},
    {"cores": 6, "executor_memory_gb": 4},
    {"cores": 6, "executor_memory_gb": 6},
    {"cores": 6, "executor_memory_gb": 8},
]

DEFAULT_SPARK_CONFIG = {
    "cores": 1,
    "executor_memory_gb": 1,
    "partitions": 2,  
    "cache_enabled": 0,
}

CORE_OPTIONS = [1, 2, 4, 6]
MEMORY_OPTIONS_GB = [1, 2, 4, 6, 8]
PARTITION_MULTIPLIERS = [2, 3, 4, 5, 6]
CACHE_OPTIONS = [0, 1]


def build_candidate_configs():
    """Cartesian product of the tunable search grid. `partitions` is derived
    as cores * multiplier, mirroring how the real generator samples it, so
    every candidate config is one the model could plausibly have seen a
    similar shape of during training."""
    configs = []
    for cores in CORE_OPTIONS:
        for mem in MEMORY_OPTIONS_GB:
            for mult in PARTITION_MULTIPLIERS:
                for cache in CACHE_OPTIONS:
                    configs.append({
                        "cores": cores,
                        "executor_memory_gb": mem,
                        "partitions": cores * mult,
                        "cache_enabled": cache,
                    })
    return configs


CANDIDATE_CONFIGS = build_candidate_configs()

TEST_CASES = [
    (500_000,    0.1, 0, 500,   1, 0, 0.9, "500K records, very low skew, minimal aggregation"),
    (1_000_000,  0.2, 1, 1_000, 2, 0, 0.7, "1M records, low skew, simple query"),
    (2_000_000,  0.4, 1, 2_000, 2, 0, 0.6, "2M records, moderate skew, single join"),
    (3_000_000,  0.3, 0, 3_000, 4, 0, 0.8, "3M records, low skew, aggregation-heavy query"),
    (5_000_000,  0.5, 2, 5_000, 3, 1, 0.5, "5M records, moderate skew, join + orderby"),
    (7_000_000,  0.2, 2, 7_000, 3, 0, 0.5, "7M records, low skew, multiple joins without orderby"),
    (8_000_000,  0.6, 2, 10_000, 4, 1, 0.4, "8M records, high aggregation, join + orderby"),
    (10_000_000, 0.8, 3, 20_000, 4, 1, 0.3, "10M records, high skew, heavy join workload"),
    (15_000_000, 0.9, 4, 30_000, 3, 1, 0.2, "15M records, extreme skew, complex join workload"),
    (20_000_000, 0.7, 3, 50_000, 4, 1, 0.1, "20M records, high skew, highly selective complex query"),
]

for _dir in (DATA_DIR, OUTPUT_DIR, MODELS_DIR):
    os.makedirs(_dir, exist_ok=True)

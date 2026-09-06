import gc
import logging
import os
import random
import sys
import time

from src.config import (
    BASE_SEED, DATASET_SIZES, DATA_DIR, FILTER_SELECTIVITY_RANGE,
    GROUP_CARDINALITIES, JOIN_OPTIONS, MAX_AGGREGATIONS, PROJECT_ROOT,
    PARTITION_MULTIPLIER_RANGE, SKEW_RANGE,
)

logger = logging.getLogger("spark_dataset_gen")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def configure_environment(java_home: str = None):
    """Point PySpark at the current Python interpreter. Pass java_home (or
    set the JAVA_HOME environment variable yourself before importing this
    module) if PySpark can't find a JDK automatically — required on some
    Windows installs. Not hardcoded here since it's machine-specific."""
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ["PATH"]

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def create_spark_session(cores: int, memory_gb: int, max_retries: int = 3,
                          retry_delay_sec: int = 15):
    """Create a local SparkSession with the given parallelism/memory.
    Plain retry loop, no manual clearing of SparkContext internals —
    forcibly resetting _gateway/_jvm/_active_spark_context left PySpark's
    state inconsistent in practice and caused getOrCreate() to hang with no
    error, rather than fixing anything."""
    from pyspark.sql import SparkSession

    configure_environment()

    for attempt in range(1, max_retries + 1):
        try:
            spark = (
                SparkSession.builder
                .appName("Spark_Perf_Dataset")
                .master(f"local[{cores}]")
                .config("spark.driver.memory", f"{memory_gb}g")
                .getOrCreate()
            )
            spark.sparkContext.setLogLevel("ERROR")
            return spark
        except Exception as e:
            logger.warning(f"Session creation failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            gc.collect()
            time.sleep(retry_delay_sec)
    raise RuntimeError("Unreachable")


def compute_derived_features(num_records, cores, memory_gb, partitions, joins, group_cardinality):
    data_size_gb = (num_records * 100) / (1024 ** 3)
    partition_efficiency = partitions / cores
    shuffle_size_mb = num_records * joins * 0.00005
    shuffle_intensity = joins * data_size_gb
    memory_pressure = data_size_gb / memory_gb
    records_per_partition = num_records / partitions
    join_load = joins * group_cardinality

    return {
        "data_size_gb": round(data_size_gb, 4),
        "partition_efficiency": round(partition_efficiency, 4),
        "shuffle_size_mb": round(shuffle_size_mb, 4),
        "shuffle_intensity": round(shuffle_intensity, 4),
        "memory_pressure": round(memory_pressure, 4),
        "records_per_partition": round(records_per_partition, 4),
        "join_load": round(join_load, 4),
    }


def build_agg_exprs(num_aggregations: int):
    """Aggregation expressions of varying complexity — num_aggregations
    controls how many of these are actually applied."""
    from pyspark.sql import functions as F

    all_aggs = [
        F.count("*").alias("cnt"),
        F.avg("value").alias("avg_val"),
        F.sum("value").alias("sum_val"),
        F.min("value").alias("min_val"),
    ]
    n = max(1, min(num_aggregations, len(all_aggs)))
    return all_aggs[:n]


def run_job(spark, num_records, partitions, skew, joins, cache, memory_gb,
            group_cardinality, num_aggregations, has_orderby, filter_selectivity):
    """Execute one real PySpark job (skewed key generation -> optional
    filter -> N self-joins -> groupBy aggregation -> optional orderBy) and
    return a dict matching every column of Spark_realtime_metrices.csv."""
    from pyspark.sql.functions import col, rand, when

    df = spark.range(num_records)
    data_size_gb = (num_records * 100) / (1024 ** 3)

    df = df.repartition(partitions)

    df = df.withColumn(
        "key",
        when(rand() < skew, 1)
        .otherwise((rand() * group_cardinality).cast("int"))
    )
    df = df.withColumn("value", rand() * 1000)

    if cache:
        df.persist()
        df.count()

    start = time.perf_counter()

    working = df
    if filter_selectivity < 0.999:
        working = working.filter(col("value") < filter_selectivity * 1000)

    result = working
    for i in range(joins):
        temp = working.select(col("key"), col("value").alias(f"value_{i}"))
        result = result.join(temp, "key", "left")

    agg_exprs = build_agg_exprs(num_aggregations)
    grouped = result.groupBy("key").agg(*agg_exprs)

    if has_orderby:
        grouped = grouped.orderBy("key")

    grouped.count()
    end = time.perf_counter()
    execution_time = round(end - start, 4)

    if cache:
        df.unpersist(blocking=True)

    actual_cores = spark.sparkContext.defaultParallelism
    derived = compute_derived_features(
        num_records, actual_cores, memory_gb, partitions, joins, group_cardinality
    )
    return {
        "num_records": num_records,
        "cores": actual_cores,
        "executor_memory_gb": memory_gb,
        "partitions": partitions,
        "partition_efficiency": derived["partition_efficiency"],
        "skewness": skew,
        "joins": joins,
        "cache_enabled": int(cache),
        "group_cardinality": group_cardinality,
        "num_aggregations": num_aggregations,
        "has_orderby": int(has_orderby),
        "filter_selectivity": filter_selectivity,
        "data_size_gb": derived["data_size_gb"],
        "shuffle_size_mb": derived["shuffle_size_mb"],
        "shuffle_intensity": derived["shuffle_intensity"],
        "memory_pressure": derived["memory_pressure"],
        "records_per_partition": derived["records_per_partition"],
        "join_load": derived["join_load"],
        "execution_time": execution_time,
    }


def sample_run_params(run_index: int, cores: int, base_seed: int = BASE_SEED):
    """Deterministic parameter sampling — seeded off (base_seed + run_index)
    so re-running after an interruption with the same run_index reproduces
    the exact same sequence rather than restarting from scratch."""
    rng = random.Random(base_seed + run_index)
    size = rng.choice(DATASET_SIZES)
    skew = rng.uniform(*SKEW_RANGE)
    joins = rng.choice(JOIN_OPTIONS)
    cache = rng.choice([0, 1])
    partitions = cores * rng.randint(*PARTITION_MULTIPLIER_RANGE)
    group_cardinality = rng.choice(GROUP_CARDINALITIES)
    num_aggregations = rng.randint(1, MAX_AGGREGATIONS)
    has_orderby = rng.choice([0, 1])
    filter_selectivity = round(rng.uniform(*FILTER_SELECTIVITY_RANGE), 2)
    return (size, skew, joins, cache, partitions, group_cardinality,
            num_aggregations, has_orderby, filter_selectivity)


def save_batch(data, output_file):
    import pandas as pd

    if not data:
        return
    df = pd.DataFrame(data)
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)


def count_existing_rows(output_file) -> int:
    if not os.path.exists(output_file):
        return 0
    with open(output_file, "r") as f:
        return sum(1 for _ in f) - 1


def generate_dataset_for_config(cores, memory_gb, total_target, batch_size=50,
                                 output_file=None, base_seed=BASE_SEED):
    """Run one hardware config's worth of benchmark jobs, checkpointing to
    CSV every batch_size rows, and resuming automatically if output_file
    already has rows from a previous (possibly interrupted) run.

    Returns the output_file path. Read it back with pandas when done, or
    use generate_and_merge_all_configs() below to do that for you across
    every entry in config.HARDWARE_CONFIGS.
    """
    output_file = output_file or os.path.join(
        DATA_DIR, f"config_{cores}_{memory_gb}.csv")

    data = []
    already_done = count_existing_rows(output_file)

    if already_done >= total_target:
        logger.info(f"Nothing to do -- {already_done} rows already exist "
                     f"(target={total_target}) in {output_file}.")
        return output_file

    completed = already_done
    run_index = already_done

    def flush():
        nonlocal data
        if data:
            save_batch(data, output_file)
            logger.info(f"Checkpoint written: {len(data)} rows -> {output_file} "
                        f"(total on disk: {completed})")
            data = []

    spark = None
    try:
        logger.info(f"Creating Spark session: cores={cores}, memory={memory_gb}g ...")
        spark = create_spark_session(cores, memory_gb)
        logger.info("Spark session created successfully.")

        while completed < total_target:
            (size, skew, joins, cache, partitions, group_cardinality,
             num_aggregations, has_orderby, filter_selectivity) = sample_run_params(
                run_index, cores, base_seed=base_seed)

            logger.info(
                f"[run {completed + 1}/{total_target}] cores={cores} mem={memory_gb}g "
                f"size={size} skew={skew:.2f} joins={joins} cache={cache} "
                f"partitions={partitions} group_card={group_cardinality} "
                f"n_aggs={num_aggregations} orderby={has_orderby} filter_sel={filter_selectivity}"
            )

            row = run_job(
                spark=spark, num_records=size, partitions=partitions, skew=skew,
                joins=joins, cache=cache, memory_gb=memory_gb,
                group_cardinality=group_cardinality, num_aggregations=num_aggregations,
                has_orderby=has_orderby, filter_selectivity=filter_selectivity,
            )
            data.append(row)
            completed += 1
            run_index += 1

            if len(data) >= batch_size:
                flush()

        flush()
        logger.info(f"ALL RUNS COMPLETE -> {completed}/{total_target} rows in {output_file}")

    except KeyboardInterrupt:
        logger.warning("Interrupted -- flushing buffered rows before exiting.")
        flush()
        logger.info(f"Safely stopped at {completed}/{total_target} rows.")

    except Exception as e:
        logger.exception(f"Dataset generation failed after {completed}/{total_target} "
                          f"completed runs: {e}")
        flush()
        raise

    finally:
        if spark is not None:
            logger.info(f"Stopping Spark session: cores={cores}, memory={memory_gb}g")
            spark.stop()
            logger.info("Spark session stopped.")

    return output_file


def generate_and_merge_all_configs(hardware_configs, total_target_per_config=250,
                                    batch_size=50, base_seed=BASE_SEED):
    """Run generate_dataset_for_config for every (cores, executor_memory_gb)
    pair in hardware_configs, then merge the resulting CSVs into one
    DataFrame — mirrors the original workflow of re-running the generator
    notebook cell once per hardware config and combining the files
    afterward for cross-config analysis."""
    import pandas as pd

    output_files = []
    for cfg in hardware_configs:
        path = generate_dataset_for_config(
            cores=cfg["cores"], memory_gb=cfg["executor_memory_gb"],
            total_target=total_target_per_config, batch_size=batch_size,
            base_seed=base_seed,
        )
        output_files.append(path)

    return merge_config_datasets(output_files)


def merge_config_datasets(file_paths):
    """Concatenate multiple per-hardware-config CSVs into one DataFrame."""
    import pandas as pd

    frames = [pd.read_csv(p) for p in file_paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(f"None of {file_paths} exist yet — run generation first.")
    return pd.concat(frames, ignore_index=True)


def correlation_by_cores(df):
    """Per-core-count correlation of every feature against execution_time,
    sorted descending — the exact analysis used to sanity-check which
    features actually drive runtime at each hardware config."""
    results = {}
    for core in sorted(df["cores"].unique()):
        subset = df[df["cores"] == core]
        corr = subset.corr(numeric_only=True)["execution_time"].sort_values(ascending=False)
        results[core] = corr
    return results

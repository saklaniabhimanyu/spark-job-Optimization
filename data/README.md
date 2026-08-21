# Data

This directory contains the Spark benchmark datasets used by the project.

### Full Dataset
`Spark_realtime_metrices.csv`

The full dataset is used for model training, evaluation, and configuration
recommendation. It is excluded from Git tracking via `.gitignore`.

### Sample Dataset
`sample/Spark_realtime_metrices.csv`

A smaller representative sample is included in the repository for testing and
demonstrating the pipeline.

### Columns

- `num_records` — number of records processed
- `cores` — number of CPU cores
- `executor_memory_gb` — configured memory in GB
- `partitions` — number of Spark partitions
- `partition_efficiency` — partition-to-core efficiency
- `skewness` — data skew factor
- `joins` — number of joins
- `cache_enabled` — caching enabled (`0`/`1`)
- `group_cardinality` — number of grouping keys
- `num_aggregations` — number of aggregations
- `has_orderby` — whether ordering is enabled (`0`/`1`)
- `filter_selectivity` — fraction of records retained
- `data_size_gb` — estimated data size
- `shuffle_size_mb` — estimated shuffle size
- `shuffle_intensity` — shuffle intensity
- `memory_pressure` — estimated memory pressure
- `execution_time` — measured execution time in seconds (target)

Each row represents a PySpark benchmark run.
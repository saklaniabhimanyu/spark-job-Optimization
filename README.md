# ⚡ AI-Based Spark Job Performance Optimization

> **A workload-aware ML system for predicting PySpark execution time and recommending optimized Spark configurations.**

This project uses real PySpark benchmark executions and machine learning to
learn the relationship between workload characteristics, Spark configurations,
and execution time.

The trained model predicts execution time for candidate configurations and
uses those predictions to recommend configurations expected to provide better
performance.

---

## 🎯 Objective

Given a Spark workload and a set of candidate configurations:

1. Predict the expected execution time.
2. Rank candidate Spark configurations.
3. Recommend the configuration with the lowest predicted runtime.
4. Validate the recommendation using actual PySpark execution.

The system considers parameters such as:

- CPU cores
- Executor memory
- Number of partitions
- Dataset size
- Number of records
- Data skew
- Caching
- Joins
- Aggregations
- Filtering
- Ordering

---

## 🔄 Pipeline

```text
Real PySpark Benchmarks
          ↓
     Data Validation
          ↓
          EDA
          ↓
  Feature Engineering
          ↓
    Model Comparison
          ↓
   5-Fold Cross-Validation
          ↓
 XGBoost Hyperparameter Tuning
          ↓
   Execution-Time Model
          ↓
 Configuration Recommendation
          ↓
 Predicted vs Actual Validation
          ↓
 Default vs ML Comparison
          ↓
 Generalization Experiments
````

---

# ✅ Completed

## 1. Real PySpark Benchmarking

Execution-time data has been collected from real PySpark benchmark runs across
different workload and Spark configuration combinations.

The benchmark varies:

* Cores
* Executor memory
* Partitions
* Dataset size
* Data skew
* Caching
* Joins
* Aggregations
* Filtering
* Ordering

Each benchmark records the measured Spark execution time.

---

## 2. Exploratory Data Analysis

Completed analysis includes:

* Dataset inspection
* Missing-value checks
* Duplicate checks
* Feature distributions
* Execution-time distribution
* Configuration coverage
* Correlation analysis
* Core-specific correlation analysis
* Data-skew analysis
* Partition/core analysis
* Configuration-performance relationships

---

## 3. Feature Engineering

Domain-specific Spark performance features have been created, including:

* `partition_efficiency`
* `data_size_gb`
* `shuffle_size_mb`
* `shuffle_intensity`
* `memory_pressure`

These features combine workload and Spark configuration information for
execution-time prediction.

---

## 4. Regression Model Comparison

The following regression models are evaluated:

| Model             | Purpose              |
| ----------------- | -------------------- |
| Linear Regression | Baseline             |
| ElasticNet        | Regularized baseline |
| Random Forest     | Non-linear ensemble  |
| Gradient Boosting | Boosting baseline    |
| XGBoost           | Primary model        |

Models are evaluated using:

* R²
* MAE
* RMSE
* 5-fold cross-validation

---

## 5. XGBoost Optimization

XGBoost is used as the primary execution-time prediction model.

Hyperparameters are tuned using cross-validation, including:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* Regularization parameters

---

## 6. Configuration Recommendation Engine

The trained model is used to evaluate candidate Spark configurations.

```text
Workload Profile
       ↓
Generate Candidate Configurations
       ↓
Predict Execution Time
       ↓
Rank Configurations
       ↓
Select Lowest Predicted Runtime
       ↓
Recommended Configuration
```

The recommendation engine searches over combinations of:

* Cores
* Executor memory
* Partitions

---

## 7. Predicted vs Actual Validation

Recommended configurations are executed using real PySpark workloads to
determine whether the model's predictions correspond to actual performance.

Validation includes:

* Predicted execution time
* Actual execution time
* Absolute prediction error
* Actual configuration ranking
* Recommendation regret

This provides an additional validation layer beyond standard ML metrics.

---

## 8. Default vs ML-Optimized Performance

The ML-selected configurations are compared against baseline/default Spark
configurations using actual benchmark execution.

The current experiments demonstrate measurable runtime improvements on the tested
workload profiles.

> Benchmark improvements are environment-dependent and should not be
> interpreted as universal Spark performance guarantees.

---

## 9. Generalization Analysis

The project also evaluates whether the trained model generalizes beyond a
random train/test split.

Experiments include:

### Hardware / Configuration Generalization

Testing on held-out core and memory configurations to measure performance on
configurations that were not represented during training.

### Workload Generalization

Testing across different workload characteristics to evaluate how well the
model transfers to previously unseen workload patterns.

---

# 📊 Current Status

The core ML and recommendation pipeline is complete.

Current results demonstrate:

* Strong execution-time prediction on the benchmark dataset
* XGBoost performing strongly among the evaluated models
* Clear relationships between Spark configuration and execution time
* Successful configuration ranking
* Real Spark validation of ML recommendations
* Measurable improvement over the tested baseline
* Separate evaluation of hardware and workload generalization

Exact benchmark results are reported in the notebook because execution time
depends on the local hardware and Spark environment.

---

# 🚧 Next Steps

The project is now moving from **model development** toward stronger
experimental validation.

## 1. Expand the Benchmark Dataset

Increase the number and diversity of real PySpark benchmark observations.

```text
More workloads
      +
More core/memory configurations
      +
More partition configurations
      +
Repeated executions
```

The goal is to reduce dependence on a single machine configuration and improve
generalization.

---

## 2. Stronger Recommendation Evaluation

Evaluate the recommendation engine on a larger set of unseen workloads.

For every workload:

```text
Generate Candidate Configurations
          ↓
ML Ranking
          ↓
Select Top-1 / Top-3 / Top-5
          ↓
Execute Selected Configurations
          ↓
Compare With Actual Best
```

Report:

* Top-1 accuracy
* Top-3 accuracy
* Top-5 accuracy
* Mean regret
* Median regret
* Worst-case regret
* Average runtime improvement

---

## 3. Compare Against Spark Heuristics

Add meaningful non-ML baselines:

```text
Default Spark
      ↓
Random Configuration
      ↓
2–4 Partitions/Core Heuristic
      ↓
ML Recommendation
```

This will determine whether the ML recommendation provides an advantage over
simple Spark configuration heuristics.

---

## 4. Repeated Runtime Validation

Execute each important configuration multiple times and report:

* Mean runtime
* Median runtime
* Standard deviation
* Confidence intervals

This will reduce the effect of machine load and runtime variability.

---

## 5. Improve Out-of-Distribution Testing

Increase testing on:

* Unseen hardware configurations
* Unseen workload types
* Different dataset sizes
* Different skew levels

The objective is to determine how reliably the model can recommend
configurations outside the training distribution.

---

## 6. Adaptive Optimization

The long-term goal is to evolve the current recommendation engine into a
closed-loop optimization system:

```text
              Spark Job
                  ↓
          Collect Runtime
                  ↓
         Predict Performance
                  ↓
       Recommend Configuration
                  ↓
            Execute Job
                  ↓
         Measure Performance
                  ↓
          Store New Result
                  ↓
           Update Model
                  ↓
      Improve Future Recommendations
```

This would allow the system to continuously learn from real Spark executions.

---

# ⚠️ Current Limitations

* Current benchmarks are controlled PySpark workloads rather than production
  Spark applications.
* Execution time depends on local hardware and system conditions.
* The dataset does not cover the complete Spark configuration space.
* Model performance depends on the benchmark distribution.
* Configuration recommendation currently uses candidate-grid search.
* Spark event-log and executor-level telemetry are not yet incorporated.
* More large-scale unseen-workload validation is required.
* Reported performance improvements are specific to the evaluated workloads.

---

# 📁 Repository Structure

```text
spark-job-Optimization/
│
├── notebooks/
│   └── spark_ai_optimizer.ipynb
│
├── data/
│   ├── Spark_realtime_metrices.csv
│   ├── sample_Spark_realtime_metrices.csv
│   └── README.md
│
├── outputs/
│   ├── eda_plots.png
│   ├── dashboard.png
│   └── feature_importance.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

The full benchmark dataset is excluded from Git using `.gitignore`.

A smaller sample dataset is included for testing and demonstrating the
pipeline.

---

# 🛠️ Technology Stack

### Big Data

* Apache Spark
* PySpark

### Machine Learning

* Python
* Scikit-learn
* XGBoost

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Experiment Tracking

* MLflow

### Development

* Jupyter Notebook
* Git
* GitHub

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/saklaniabhimanyu/spark-job-Optimization.git
cd spark-job-Optimization
```

## Create Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Place the full dataset at:

```text
data/Spark_realtime_metrices.csv
```

The full dataset is ignored by Git.

For a quick test, use the sample dataset:

```text
data/sample/Spark_realtime_metrices.csv
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/spark_ai_optimizer.ipynb
```

Run the notebook sequentially.

---

# 📦 Dataset

Each row represents one PySpark benchmark execution.

Important columns include:

* `num_records`
* `cores`
* `executor_memory_gb`
* `partitions`
* `partition_efficiency`
* `skewness`
* `joins`
* `cache_enabled`
* `group_cardinality`
* `num_aggregations`
* `has_orderby`
* `filter_selectivity`
* `data_size_gb`
* `shuffle_size_mb`
* `shuffle_intensity`
* `memory_pressure`
* `execution_time`

`execution_time` is the target variable used for ML prediction.

See [`data/README.md`](data/README.md) for dataset details.

---

# 🎯 Project Direction

The current system focuses on:

```text
Prediction
    +
Recommendation
    +
Real Spark Validation
```

The next stage focuses on:

```text
Generalization
    +
Stronger Baselines
    +
Larger Validation
    +
Repeated Experiments
    +
Adaptive Learning
```

The long-term objective is a workload-aware Spark optimization system that can
learn from previous executions and improve future configuration
recommendations.

---

# 👤 Author

**Abhimanyu Saklani**

GitHub: [https://github.com/saklaniabhimanyu](https://github.com/saklaniabhimanyu)

---

⭐ If you find this project useful, consider giving the repository a star.

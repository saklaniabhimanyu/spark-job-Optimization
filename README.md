# Spark Job Performance Optimization System

An ML-based system that predicts Apache Spark job execution time and recommends better Spark configurations for different workload profiles.

The system uses **XGBoost** to learn the relationship between workload characteristics, Spark resources, and execution time. It then evaluates candidate configurations and ranks them by predicted runtime.

---

## Overview

Spark performance depends heavily on configuration choices such as:

- Executor cores
- Executor memory
- Number of partitions
- Cache usage
- Workload size
- Data skew
- Number of joins and aggregations

Finding a good configuration manually can require repeated experimentation.

This project uses machine learning to reduce that trial-and-error process:

```text
Workload Profile
       │
       ▼
Feature Construction
       │
       ▼
Trained XGBoost Model
       │
       ▼
Evaluate Candidate Configurations
       │
       ▼
Rank by Predicted Execution Time
       │
       ▼
Recommended Spark Configuration
```

---

## Key Features

* **Execution-time prediction** using XGBoost regression
* **Spark-specific feature construction** for workload and resource characteristics
* **Hyperparameter optimization** using RandomizedSearchCV and Optuna/TPE
* **Cross-validation** and held-out test evaluation
* **SHAP-based explainability** for understanding model predictions
* **Configuration recommendation engine** for ranking candidate Spark configurations
* **Baseline comparison** between default and recommended configurations
* **Generalization experiments** across different workload conditions
* **Workload-type evaluation**
* **MLflow experiment tracking**
* **FastAPI REST API** for serving recommendations
* **Automated unit tests** for core recommendation and utility logic

---

## Model Performance

The tuned XGBoost model achieved:

| Metric |     Result |
| ------ | ---------: |
| R²     |  **0.941** |
| MAE    | **0.17 s** |

Evaluation was performed on a held-out test set.

The recommendation engine was additionally evaluated across **10 simulated workload profiles**, where the recommended configurations produced an average **42.1% observed runtime reduction** compared with the selected default configurations.

Because the workloads are simulated, these results should be interpreted as experimental validation of the optimization approach rather than production performance guarantees.

---

## Machine Learning Pipeline

### 1. Workload Generation

The project uses reproducible simulated Spark workloads with different combinations of:

* Number of records
* Data skew
* Joins
* Group cardinality
* Number of aggregations
* `ORDER BY`
* Filter selectivity
* Spark resource configurations

Execution time is used as the prediction target.

The simulation-based approach makes experiments reproducible without requiring a continuously running Spark cluster.

---

### 2. Feature Construction

The model uses workload and Spark configuration features such as:

```text
num_records
cores
executor_memory_gb
partitions
partition_efficiency
skewness
joins
cache_enabled
group_cardinality
num_aggregations
has_orderby
filter_selectivity
data_size_gb
shuffle_size_mb
shuffle_intensity
memory_pressure
```

Derived features capture relationships between workload characteristics and available resources.

---

### 3. Model Comparison

Multiple regression models are evaluated before selecting the final model:

* Linear Regression
* Elastic Net
* Random Forest
* Gradient Boosting
* XGBoost

Models are evaluated using:

* MAE
* RMSE
* R²
* Cross-validation

The tuned XGBoost model is used for the final recommendation pipeline.

---

### 4. Hyperparameter Optimization

XGBoost hyperparameters are optimized using:

* RandomizedSearchCV
* Optuna
* TPE-based optimization

The final model is selected based on validation performance and evaluated on a held-out test set.

---

## Explainability

SHAP is used to understand which features have the greatest influence on predicted Spark execution time.

The analysis helps identify important performance drivers such as:

* Partition count
* Data skew
* Number of records
* Cache enablement
* Resource allocation

This makes the recommendation system more interpretable than treating the ML model as a black box.

---

## Configuration Recommendation

For a given workload, the recommendation engine evaluates a predefined grid of Spark configurations.

Example:

```text
Workload
   │
   ├── 1M records
   ├── skew = 0.30
   ├── joins = 1
   └── aggregations = 2
          │
          ▼
   Candidate Configurations
          │
          ▼
   XGBoost Predictions
          │
          ▼
   Sort by predicted runtime
          │
          ▼
   Top-N configurations
```

The output includes the recommended:

* Executor cores
* Executor memory
* Number of partitions
* Cache setting
* Predicted execution time
* Derived resource/workload features

---

## Evaluation Experiments

The project includes several experiments to evaluate the recommendation system beyond model accuracy.

### Recommendation Validation

Tests whether recommended configurations consistently provide better predicted/observed performance than baseline configurations.

### Baseline Comparison

Compares default Spark configurations against ML-recommended configurations.

### Generalization

Evaluates how the model and recommendation approach behave across workload conditions different from the primary training scenarios.

### Workload-Type Evaluation

Tests recommendations across different workload profiles, including variations in:

* Data volume
* Skew
* Joins
* Aggregations
* Ordering
* Filtering

Experiment results are stored under:

```text
outputs/
```

---

## API

The trained model is exposed through a **FastAPI** service.

Start the API with:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Available endpoints

```text
GET  /
GET  /model/info
POST /recommend
```

The `/recommend` endpoint accepts a workload profile and returns ranked Spark configuration recommendations.

---

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/saklaniabhimanyu/spark-job-Optimization.git
cd spark-job-optimization
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://localhost:8000
```

---

## Running Tests

The project includes unit tests for the recommendation engine, feature construction, configuration generation, evaluation utilities, and experiment tracking.

Run:

```bash
pytest tests/ -v
```

Current test suite:

```text
16 tests
16 passed
```

The test suite validates:

* Candidate configuration generation
* Feature ordering
* Recommendation ranking
* `top_n` handling
* Derived feature calculations
* Evaluation utilities
* Experiment tracking

---

## 📁 Project Structure

```text
spark-job-optimization/
│
├── api/
│   └── main.py
│
├── src/
│   ├── config.py
│   ├── evaluation.py
│   ├── experiment_tracker.py
│   ├── recommendation_engine.py
│   └── spark_utils.py
│
├── notebooks/
│   └── ...
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── outputs/
│   ├── 09_ranking_evaluation.csv
│   ├── 10_recommendation_validation.csv
│   ├── 11_baseline_comparison.csv
│   ├── 12_generalization_experiment.csv
│   └── 13_workload_type_experiment.csv
│
├── tests/
│   ├── test_evaluation.py
│   ├── test_experiment_tracker.py
│   └── test_recommendation_engine.py
│
├── main.py
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-learn
* XGBoost
* Optuna
* SHAP

### Data & Big Data

* Apache Spark
* PySpark
* Pandas
* NumPy

### Experimentation

* MLflow
* Jupyter Notebook

### API & Deployment

* FastAPI
* Uvicorn
* Docker

### Testing & Development

* Pytest
* Git
* GitHub

---

## Results

The system demonstrates that a machine-learning model can learn relationships between Spark workload characteristics and execution performance and use those predictions to rank candidate configurations.

The tuned XGBoost model achieved:

**R² = 0.941**
**MAE = 0.17 seconds**

Across the evaluated simulated workloads, ML-based configuration recommendations produced an average:

**42.1% observed runtime reduction**

compared with the selected default configurations.

The project also uses SHAP analysis to provide insight into the factors influencing runtime predictions.

---
## Limitations

The current implementation has several limitations:

* Training data is primarily simulation-based.
* Simulated execution times may not capture all real Spark cluster behaviour.
* Hardware differences, network latency, JVM overhead, storage systems, and cluster contention are not fully represented.
* Recommendations are restricted to the candidate configuration grid.
* Runtime improvements depend on the workload and environment.

Therefore, recommendations should be validated against real Spark workloads before production use.

---

## Future Work

Potential improvements include:

* Training on real Spark History Server and execution logs
* Integration with real Spark clusters
* Cost-aware configuration optimization
* Kubernetes/YARN integration
* Online/adaptive configuration tuning
* Multi-objective optimization for runtime and resource cost
* Automated CI/CD deployment
* Larger and more diverse workload datasets

---

## Project Goal

The goal of this project is to explore how **machine learning can reduce manual Spark performance tuning** by predicting execution time and automatically identifying promising configurations for different workload conditions.

It combines:

**Apache Spark + Machine Learning + Explainable AI + Configuration Optimization + API Deployment**

into a reproducible end-to-end system.
---
## Author
- Abhimanyu Saklani

If you found this project useful or have suggestions for improvement, feel free to open an issue or connect with me.

GitHub: https://github.com/saklaniabhimanyu

⭐ If you like this project, consider giving it a star!
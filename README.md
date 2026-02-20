# UNSW Network Intrusion Detection — Production ML Pipeline (Spark, MLflow, Docker)

## Overview

This project implements a production-grade machine learning pipeline for network intrusion detection using the UNSW-NB15 dataset.

The system demonstrates a realistic ML workflow including:

* Distributed preprocessing with Apache Spark
* Feature engineering pipeline with categorical encoding and imputation
* RandomForest model training using Spark ML
* Experiment tracking and model management with MLflow
* Containerized training environment using Docker

This project reflects real-world ML engineering practices used in production systems.

---

## Architecture

```
Raw Data (Parquet)
        │
        ▼
Spark Preprocessing Pipeline
(StringIndexer, OneHotEncoder, Imputer, VectorAssembler)
        │
        ▼
RandomForest Training (Spark ML)
        │
        ▼
MLflow Tracking
(parameters, metrics, artifacts)
        │
        ▼
Saved Model
artifacts/rf_model/
```

---

## Technologies Used

* Python 3.10
* Apache Spark (PySpark)
* MLflow
* Docker
* RandomForest (Spark ML)
* Parquet data format

---

## Dataset

UNSW-NB15 network intrusion dataset.

Features include:

* Network flow statistics
* Protocol and service information
* Packet-level features
* Connection metadata

Target:

* attack_label (multi-class intrusion classification)

---

## Feature Engineering Pipeline

Implemented using Spark ML Pipeline.

### Categorical Encoding

* StringIndexer
* OneHotEncoder

Applied to:

* protocol
* state
* service

### Missing Value Handling

Imputer with median strategy applied to numeric features.

### Feature Vector Construction

VectorAssembler combines:

* Numeric features
* Encoded categorical features

Output column:

```
features
```

---

## Model Training

Model:

```
RandomForestClassifier
```

Configuration:

```
numTrees = 50
maxDepth = 10
```

Spark handles distributed training across partitions.

---

## Experiment Tracking (MLflow)

MLflow tracks:

* parameters
* metrics
* trained model artifacts

Example metric:

```
F1 score: 0.978
```

Model saved to:

```
artifacts/rf_model/
```

---

## Project Structure

```
unsw-intrusion-detection-mlops/

├── src/
│   ├── train.py
│   ├── spark_pipeline.py
│   ├── inference.py
│
├── artifacts/
│   └── rf_model/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## Running Training (Docker)

Build container:

```
docker compose build
```

Run training:

```
docker compose run training
```

This will:

* Load dataset
* Run Spark preprocessing pipeline
* Train RandomForest model
* Log metrics to MLflow
* Save model to artifacts/

---

## MLflow UI

Start MLflow UI:

```
mlflow ui
```

Open:

```
http://localhost:5000
```

You can view:

* experiment runs
* metrics
* parameters
* saved models

---

## Why Spark?

Spark enables:

* scalable preprocessing
* distributed model training
* handling large datasets
* production-grade data pipelines

---

## Production-Grade Design Principles Demonstrated

* Distributed feature engineering
* Pipeline-based preprocessing
* Model versioning
* Experiment tracking
* Containerized training
* Separation between training and inference

---

## Example Training Output

```
Training model...
Evaluating model...
F1 = 0.9783
Saving model...
Done.
```

---

## Future Improvements

Possible extensions:

* Model registry integration
* Automated retraining pipeline
* Batch inference pipeline
* Kubernetes deployment
* Model monitoring and drift detection

---

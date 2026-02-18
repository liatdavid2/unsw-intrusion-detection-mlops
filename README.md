# UNSW MLflow Training Pipeline

Production-grade Spark + MLflow training pipeline for intrusion detection.

## Features

- Spark training pipeline
- MLflow experiment tracking
- Model registry
- Parquet data support
- Production-ready structure

## Setup

Install dependencies:

    pip install -r requirements.txt

Add your dataset:

    data/raw/UNSW_Flow.parquet

## Run training

Start MLflow UI:

    mlflow ui

Run training:

    python src/train.py

Open MLflow UI:

    http://localhost:5000

## Project Structure

    data/raw/UNSW_Flow.parquet
    src/train.py
    src/spark_pipeline.py
    src/config.py

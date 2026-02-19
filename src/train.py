# src/train.py

import mlflow
import mlflow.spark

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.spark_pipeline import (
    create_spark_session,
    load_data,
    preprocess_data
)

DATA_PATH = "data/raw/UNSW_Flow.parquet"
MODEL_PATH = "artifacts/rf_model"


def main():

    # --------------------------------
    # Create Spark session
    # --------------------------------

    spark = create_spark_session()

    # --------------------------------
    # MLflow setup
    # --------------------------------

    mlflow.set_experiment("UNSW Intrusion Detection")

    with mlflow.start_run():

        # --------------------------------
        # Load data
        # --------------------------------

        df = load_data(spark, DATA_PATH)

        # --------------------------------
        # Preprocess
        # --------------------------------

        train_df, feature_cols = preprocess_data(df)

        # --------------------------------
        # Model definition
        # --------------------------------

        num_trees = 100
        max_depth = 10

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=42
        )

        # --------------------------------
        # Train model
        # --------------------------------

        model = rf.fit(train_df)

        # --------------------------------
        # Evaluate model
        # --------------------------------

        predictions = model.transform(train_df)

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        auc = evaluator.evaluate(predictions)

        print(f"AUC = {auc:.4f}")

        # --------------------------------
        # Log parameters to MLflow
        # --------------------------------

        mlflow.log_param("num_trees", num_trees)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("num_features", len(feature_cols))

        # --------------------------------
        # Log metrics to MLflow
        # --------------------------------

        mlflow.log_metric("auc", auc)

        # --------------------------------
        # Save model locally
        # --------------------------------

        model.write().overwrite().save(MODEL_PATH)

        print(f"Model saved to: {MODEL_PATH}")

        # --------------------------------
        # Log model to MLflow
        # --------------------------------

        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model"
        )

        print("Model logged to MLflow")

    spark.stop()


if __name__ == "__main__":
    main()
import os

import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.spark_pipeline import load_data, preprocess_data
from src.config import DATA_PATH, MODEL_PATH, MLFLOW_EXPERIMENT


def create_spark_session():

    return (
        SparkSession.builder
        .appName("UNSW Training")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main():

    os.makedirs("artifacts", exist_ok=True)

    spark = create_spark_session()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():

        df = load_data(spark, DATA_PATH)

        train_df, feature_cols = preprocess_data(df)

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=50,
            maxDepth=10,
            seed=42
        )

        print("Training model...")

        model = rf.fit(train_df)

        print("Evaluating model...")

        predictions = model.transform(train_df)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        f1 = evaluator.evaluate(predictions)

        print(f"F1 = {f1}")

        mlflow.log_metric("f1", f1)

        mlflow.log_param("numTrees", 100)
        mlflow.log_param("maxDepth", 10)

        print("Saving model...")

        mlflow.spark.save_model(
            spark_model=model,
            path=MODEL_PATH
        )

        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model"
        )

    spark.stop()


if __name__ == "__main__":
    main()
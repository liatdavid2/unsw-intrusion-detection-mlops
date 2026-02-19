import mlflow.spark
from pyspark.sql import SparkSession


def load_model():

    spark = SparkSession.builder.getOrCreate()

    model = mlflow.spark.load_model(
        "artifacts/rf_model"
    )

    return model, spark
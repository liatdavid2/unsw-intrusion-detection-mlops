import mlflow.spark
from spark_pipeline import create_spark_session, load_data
import config


def main():

    spark = create_spark_session()

    df = load_data(spark, config.DATA_PATH)

    model = mlflow.spark.load_model(
        f"models:/{config.MODEL_NAME}/latest"
    )

    predictions = model.transform(df)

    predictions.show(10)


if __name__ == "__main__":
    main()

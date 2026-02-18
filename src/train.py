import mlflow
import mlflow.spark

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from spark_pipeline import create_spark_session, load_data, preprocess_data
import config


def main():

    spark = create_spark_session()

    df = load_data(spark, config.DATA_PATH)

    df, features = preprocess_data(df)

    train_df, test_df = df.randomSplit(
        [config.TRAIN_SPLIT, 1 - config.TRAIN_SPLIT],
        seed=config.RANDOM_SEED
    )

    rf = RandomForestClassifier(
        numTrees=50,
        maxDepth=10,
        labelCol="label",
        featuresCol="features"
    )

    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run():

        model = rf.fit(train_df)

        predictions = model.transform(test_df)

        evaluator = BinaryClassificationEvaluator(
            labelCol="label"
        )

        auc = evaluator.evaluate(predictions)

        mlflow.log_metric("AUC", auc)
        mlflow.log_param("numTrees", 50)
        mlflow.log_param("maxDepth", 10)

        mlflow.spark.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME
        )

        print("AUC:", auc)


if __name__ == "__main__":
    main()

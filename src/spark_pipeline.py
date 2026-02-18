from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("UNSW-MLflow")
        .master("local[*]")   # uses all CPU cores
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    return spark


def load_data(spark, path):
    df = spark.read.parquet(path)
    return df


def preprocess_data(df):

    target_col = "binary_label"

    # columns to exclude from features
    exclude_cols = {
        "binary_label",
        "attack_label"
    }

    # select only numeric feature columns
    feature_cols = [
        f.name for f in df.schema.fields
        if f.name not in exclude_cols
        and f.dataType.simpleString() in ("int", "double", "float", "long")
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)

    # rename target to "label" for Spark convention
    df = df.withColumnRenamed(target_col, "label")

    df = df.select("features", "label")

    return df, feature_cols

from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler


def load_data(spark, path):

    print("Loading data...")

    df = spark.read.parquet(path)

    return df


def preprocess_data(df):

    print("Preprocessing...")

    label_col = "label"

    feature_cols = [
        c for c in df.columns
        if c != label_col
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)

    return df.select("features", "label"), feature_cols
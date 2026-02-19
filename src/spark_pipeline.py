# src/spark_pipeline.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml import Pipeline


# --------------------------------
# Load data
# --------------------------------
def load_data(spark: SparkSession, path: str):

    df = spark.read.parquet(path)

    return df


# --------------------------------
# Preprocess data
# --------------------------------
def preprocess_data(df):

    # Drop unsupported columns
    df = df.drop("source_ip", "destination_ip")

    categorical_cols = ["protocol", "state", "service"]

    label_col = "attack_label"

    # Numeric columns (exclude label columns)
    numeric_cols = [
        c for c, t in df.dtypes
        if t in ("int", "bigint", "double", "float")
        and c not in ("label", "binary_label")   # important fix
    ]

    # Fill null numeric values
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=numeric_cols,
        strategy="median"
    )

    # Label indexer
    label_indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep"
    )

    # Categorical indexers
    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep"
        )
        for c in categorical_cols
    ]

    # OneHotEncoder
    encoders = [
        OneHotEncoder(
            inputCol=f"{c}_idx",
            outputCol=f"{c}_vec"
        )
        for c in categorical_cols
    ]

    # Assemble features
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_vec" for c in categorical_cols],
        outputCol="features",
        handleInvalid="keep"   # important safety fix
    )

    # Full pipeline
    pipeline = Pipeline(
        stages=[label_indexer]
        + indexers
        + encoders
        + [imputer]
        + [assembler]
    )

    model = pipeline.fit(df)

    df = model.transform(df)

    feature_cols = numeric_cols + categorical_cols

    return df, feature_cols
"""
Real-time Fraud Detection Inference Pipeline V2

Training and inference MUST use the exact same feature schema.
"""

import logging
import os

import joblib
import yaml

from dotenv import load_dotenv

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    dayofmonth,
    dayofweek,
    from_json,
    hour,
    log1p,
    when,
)
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


class FraudDetectionInference:

    def __init__(
        self,
        config_path="/app/config.yaml",
    ):

        load_dotenv(
            dotenv_path="/app/.env"
        )

        self.config = self._load_config(
            config_path
        )

        self.spark = (
            self._init_spark_session()
        )

        model_package = self._load_model(
            self.config["model"]["path"]
        )

        self.model = model_package["model"]

        self.threshold = float(
            model_package["threshold"]
        )

        self.feature_version = (
            model_package.get(
                "feature_version",
                "unknown",
            )
        )

        logger.info(
            "Model loaded. feature_version=%s threshold=%.6f",
            self.feature_version,
            self.threshold,
        )

        self.broadcast_model = (
            self.spark.sparkContext.broadcast(
                self.model
            )
        )

        self.broadcast_threshold = (
            self.spark.sparkContext.broadcast(
                self.threshold
            )
        )

    # ========================================================
    # Config
    # ========================================================

    @staticmethod
    def _load_config(path):

        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ========================================================
    # Model
    # ========================================================

    @staticmethod
    def _load_model(path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        package = joblib.load(path)

        if not isinstance(package, dict):
            raise ValueError(
                "Model file is not V2 package."
            )

        if "model" not in package:
            raise ValueError(
                "Model package does not contain model."
            )

        if "threshold" not in package:
            raise ValueError(
                "Model package does not contain threshold."
            )

        return package

    # ========================================================
    # Spark
    # ========================================================

    def _init_spark_session(self):

        packages = (
            self.config
            .get("spark", {})
            .get("packages", "")
        )

        builder = (
            SparkSession.builder
            .appName(
                "FraudDetectionInferenceV2"
            )
        )

        if packages:
            builder = builder.config(
                "spark.jars.packages",
                packages,
            )

        return builder.getOrCreate()

    # ========================================================
    # Kafka
    # ========================================================

    def read_from_kafka(self):

        kafka_config = (
            self.config["kafka"]
        )

        bootstrap = kafka_config.get(
            "bootstrap_servers",
            "kafka:29092",
        )

        topic = kafka_config["topic"]

        reader = (
            self.spark.readStream
            .format("kafka")
            .option(
                "kafka.bootstrap.servers",
                bootstrap,
            )
            .option(
                "subscribe",
                topic,
            )
            .option(
                "startingOffsets",
                "latest",
            )
            .option(
                "failOnDataLoss",
                "false",
            )
        )

        df = reader.load()

        schema = StructType(
            [
                StructField(
                    "transaction_id",
                    StringType(),
                    True,
                ),
                StructField(
                    "user_id",
                    IntegerType(),
                    True,
                ),
                StructField(
                    "amount",
                    DoubleType(),
                    True,
                ),
                StructField(
                    "currency",
                    StringType(),
                    True,
                ),
                StructField(
                    "merchant",
                    StringType(),
                    True,
                ),
                StructField(
                    "timestamp",
                    TimestampType(),
                    True,
                ),
                StructField(
                    "location",
                    StringType(),
                    True,
                ),
                StructField(
                    "is_fraud",
                    IntegerType(),
                    True,
                ),
            ]
        )

        return (
            df.selectExpr(
                "CAST(value AS STRING) AS value"
            )
            .select(
                from_json(
                    col("value"),
                    schema,
                ).alias("data")
            )
            .select("data.*")
        )

    # ========================================================
    # Features
    # ========================================================

    def add_features(self, df):

        df = df.withColumn(
            "transaction_hour",
            hour(col("timestamp")),
        )

        df = df.withColumn(
            "is_night",
            when(
                (col("transaction_hour") >= 22)
                | (col("transaction_hour") < 5),
                1,
            ).otherwise(0),
        )

        df = df.withColumn(
            "is_weekend",
            when(
                (dayofweek(col("timestamp")) == 1)
                | (dayofweek(col("timestamp")) == 7),
                1,
            ).otherwise(0),
        )

        df = df.withColumn(
            "transaction_day",
            dayofmonth(col("timestamp")),
        )

        # Amount features
        df = df.withColumn(
            "amount_log",
            log1p(
                when(
                    col("amount") < 0,
                    0,
                ).otherwise(
                    col("amount")
                )
            ),
        )

        df = df.withColumn(
            "amount_over_500",
            when(
                col("amount") > 500,
                1,
            ).otherwise(0),
        )

        df = df.withColumn(
            "amount_over_1000",
            when(
                col("amount") > 1000,
                1,
            ).otherwise(0),
        )

        # User behavior signals
        df = df.withColumn(
            "user_id_mod_500",
            col("user_id") % 500,
        )

        df = df.withColumn(
            "user_id_mod_1000",
            col("user_id") % 1000,
        )

        df = df.withColumn(
            "user_id_divisible_500",
            when(
                (col("user_id") % 500) == 0,
                1,
            ).otherwise(0),
        )

        df = df.withColumn(
            "user_id_divisible_1000",
            when(
                (col("user_id") % 1000) == 0,
                1,
            ).otherwise(0),
        )

        high_risk_merchants = (
            self.config.get(
                "high_risk_merchants",
                [
                    "QuickCash",
                    "GlobalDigital",
                    "FastMoneyX",
                ],
            )
        )

        df = df.withColumn(
            "merchant_risk",
            col("merchant")
            .isin(high_risk_merchants)
            .cast("int"),
        )

        return df

    # ========================================================
    # Inference
    # ========================================================

    def run_inference(self):

        import pandas as pd

        df = self.read_from_kafka()

        df = self.add_features(df)

        model = self.broadcast_model
        threshold = self.broadcast_threshold

        @pandas_udf("double")
        def predict_probability(
            amount: pd.Series,
            amount_log: pd.Series,
            amount_over_500: pd.Series,
            amount_over_1000: pd.Series,
            transaction_hour: pd.Series,
            is_night: pd.Series,
            is_weekend: pd.Series,
            transaction_day: pd.Series,
            user_id_mod_500: pd.Series,
            user_id_mod_1000: pd.Series,
            user_id_divisible_500: pd.Series,
            user_id_divisible_1000: pd.Series,
            merchant_risk: pd.Series,
            location: pd.Series,
        ) -> pd.Series:

            input_df = pd.DataFrame(
                {
                    "amount": amount,
                    "amount_log": amount_log,
                    "amount_over_500": amount_over_500,
                    "amount_over_1000": amount_over_1000,
                    "transaction_hour": transaction_hour,
                    "is_night": is_night,
                    "is_weekend": is_weekend,
                    "transaction_day": transaction_day,
                    "user_id_mod_500": user_id_mod_500,
                    "user_id_mod_1000": user_id_mod_1000,
                    "user_id_divisible_500": user_id_divisible_500,
                    "user_id_divisible_1000": user_id_divisible_1000,
                    "merchant_risk": merchant_risk,
                    "location": location,
                }
            )

            probabilities = (
                model.value
                .predict_proba(input_df)[:, 1]
            )

            return pd.Series(
                probabilities
            )

        prediction_df = (
            df.withColumn(
                "fraud_probability",
                predict_probability(
                    col("amount"),
                    col("amount_log"),
                    col("amount_over_500"),
                    col("amount_over_1000"),
                    col("transaction_hour"),
                    col("is_night"),
                    col("is_weekend"),
                    col("transaction_day"),
                    col("user_id_mod_500"),
                    col("user_id_mod_1000"),
                    col("user_id_divisible_500"),
                    col("user_id_divisible_1000"),
                    col("merchant_risk"),
                    col("location"),
                ),
            )
        )

        prediction_df = prediction_df.withColumn(
            "prediction",
            when(
                col("fraud_probability")
                >= threshold.value,
                1,
            ).otherwise(0),
        )

        # Only fraud transactions go to output topic
        fraud_predictions = (
            prediction_df
            .filter(
                col("prediction") == 1
            )
        )

        writer = (
            fraud_predictions
            .selectExpr(
                "CAST(transaction_id AS STRING) AS key",
                "to_json(struct(*)) AS value",
            )
            .writeStream
            .format("kafka")
            .option(
                "kafka.bootstrap.servers",
                self.config["kafka"][
                    "bootstrap_servers"
                ],
            )
            .option(
                "topic",
                self.config["kafka"].get(
                    "output_topic",
                    "fraud_predictions",
                ),
            )
            .option(
                "checkpointLocation",
                "checkpoints/checkpoint_v2",
            )
            .outputMode("append")
            .start()
        )

        logger.info(
            "Inference started successfully."
        )

        writer.awaitTermination()


if __name__ == "__main__":

    inference = FraudDetectionInference(
        "/app/config.yaml"
    )

    inference.run_inference()

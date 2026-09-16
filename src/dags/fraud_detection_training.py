"""
Fraud Detection Model Training Pipeline V2

Goals:
- Same features in training and inference
- No fake behavioral features
- No SMOTE on categorical data
- XGBoost class imbalance handling
- Separate train / validation / test
- Threshold selected on validation only
- Strong fraud metrics
- MLflow tracking
- Model + threshold saved together
"""

import json
import logging
import os

import boto3
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml

from dotenv import load_dotenv
from kafka import KafkaConsumer
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./fraud_detection_model.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class FraudDetectionTraining:

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self, config_path="/app/config.yaml"):

        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "/usr/bin/git"

        load_dotenv(dotenv_path="/app/.env")

        self.config = self._load_config(config_path)

        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
            "AWS_ACCESS_KEY_ID", ""
        )
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "AWS_SECRET_ACCESS_KEY", ""
        )
        os.environ["AWS_S3_ENDPOINT_URL"] = self.config["mlflow"][
            "s3_endpoint_url"
        ]

        self._validate_environment()

        mlflow.set_tracking_uri(
            self.config["mlflow"]["tracking_uri"]
        )

        mlflow.set_experiment(
            self.config["mlflow"]["experiment_name"]
        )

    # ========================================================
    # Configuration
    # ========================================================

    def _load_config(self, config_path):

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Configuration file is empty.")

        logger.info("Configuration loaded successfully")

        return config

    # ========================================================
    # Environment
    # ========================================================

    def _validate_environment(self):

        self._check_minio_connection()

    # ========================================================
    # MinIO
    # ========================================================

    def _check_minio_connection(self):

        s3 = boto3.client(
            "s3",
            endpoint_url=self.config["mlflow"]["s3_endpoint_url"],
            aws_access_key_id=os.getenv(
                "AWS_ACCESS_KEY_ID",
                "minioadmin",
            ),
            aws_secret_access_key=os.getenv(
                "AWS_SECRET_ACCESS_KEY",
                "minioadmin",
            ),
        )

        buckets = s3.list_buckets()

        bucket_names = [
            b["Name"]
            for b in buckets.get("Buckets", [])
        ]

        logger.info(
            "Minio connection verified. Buckets: %s",
            bucket_names,
        )

        bucket = self.config["mlflow"].get(
            "bucket",
            "mlflow",
        )

        if bucket not in bucket_names:

            s3.create_bucket(Bucket=bucket)

            logger.info(
                "Created missing MLflow bucket: %s",
                bucket,
            )

    # ========================================================
    # Kafka
    # ========================================================

    def read_from_kafka(self):

        consumer = None

        try:

            kafka_config = self.config["kafka"]

            topic = kafka_config["topic"]

            bootstrap_servers = [
                x.strip()
                for x in kafka_config[
                    "bootstrap_servers"
                ].split(",")
                if x.strip()
            ]

            timeout_ms = int(
                kafka_config.get(
                    "timeout",
                    10000,
                )
            )

            max_messages = int(
                kafka_config.get(
                    "max_messages",
                    100000,
                )
            )

            poll_timeout_ms = int(
                kafka_config.get(
                    "poll_timeout_ms",
                    1000,
                )
            )

            consumer_kwargs = {
                "bootstrap_servers": bootstrap_servers,
                "value_deserializer": (
                    lambda x: json.loads(
                        x.decode("utf-8")
                    )
                ),
                "auto_offset_reset": "earliest",
                "enable_auto_commit": False,
                "consumer_timeout_ms": timeout_ms,
                "request_timeout_ms": 30000,
                "session_timeout_ms": 10000,
                "security_protocol": kafka_config.get(
                    "security_protocol",
                    "PLAINTEXT",
                ),
            }

            if consumer_kwargs["security_protocol"] == "SASL_SSL":

                consumer_kwargs.update(
                    {
                        "sasl_mechanism": "PLAIN",
                        "sasl_plain_username": (
                            kafka_config.get("username")
                            or os.getenv("KAFKA_USERNAME")
                        ),
                        "sasl_plain_password": (
                            kafka_config.get("password")
                            or os.getenv("KAFKA_PASSWORD")
                        ),
                    }
                )

            logger.info(
                "Creating Kafka consumer: topic=%s max_messages=%d",
                topic,
                max_messages,
            )

            consumer = KafkaConsumer(
                topic,
                **consumer_kwargs,
            )

            messages = []

            empty_polls = 0
            max_empty_polls = max(
                1,
                timeout_ms // max(poll_timeout_ms, 1),
            )

            while len(messages) < max_messages:

                records = consumer.poll(
                    timeout_ms=poll_timeout_ms
                )

                if not records:

                    empty_polls += 1

                    if empty_polls >= max_empty_polls:
                        break

                    continue

                empty_polls = 0

                for _, batch in records.items():

                    for msg in batch:

                        messages.append(msg.value)

                        if len(messages) >= max_messages:
                            break

                    if len(messages) >= max_messages:
                        break

                if len(messages) % 10000 < 1000:
                    logger.info(
                        "Kafka progress: %d/%d",
                        len(messages),
                        max_messages,
                    )

            if not messages:
                raise ValueError(
                    "No messages received from Kafka."
                )

            df = pd.DataFrame(messages)

            logger.info(
                "Kafka consumption completed: %d rows",
                len(df),
            )

            required = [
                "user_id",
                "amount",
                "currency",
                "merchant",
                "timestamp",
                "location",
                "is_fraud",
            ]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing required columns: {missing}"
                )

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce",
            )

            df["amount"] = pd.to_numeric(
                df["amount"],
                errors="coerce",
            )

            df["user_id"] = pd.to_numeric(
                df["user_id"],
                errors="coerce",
            )

            df["is_fraud"] = pd.to_numeric(
                df["is_fraud"],
                errors="coerce",
            )

            df = df.dropna(
                subset=[
                    "user_id",
                    "amount",
                    "merchant",
                    "timestamp",
                    "location",
                    "is_fraud",
                ]
            ).copy()

            df["user_id"] = df["user_id"].astype(int)
            df["is_fraud"] = df["is_fraud"].astype(int)

            fraud_rate = df["is_fraud"].mean()

            logger.info(
                "Valid dataset: rows=%d fraud_rate=%.4f%%",
                len(df),
                fraud_rate * 100,
            )

            return df

        finally:

            if consumer is not None:

                try:
                    consumer.close()
                except Exception:
                    pass

    # ========================================================
    # Feature Engineering
    # ========================================================

    def create_features(self, df):

        df = df.copy()

        df["transaction_hour"] = (
            df["timestamp"].dt.hour
        )

        df["is_night"] = (
            (df["transaction_hour"] >= 22)
            | (df["transaction_hour"] < 5)
        ).astype(int)

        df["is_weekend"] = (
            df["timestamp"].dt.dayofweek >= 5
        ).astype(int)

        df["transaction_day"] = (
            df["timestamp"].dt.day
        )

        # ----------------------------------------------------
        # Amount features
        # ----------------------------------------------------

        df["amount_log"] = np.log1p(
            df["amount"].clip(lower=0)
        )

        df["amount_over_500"] = (
            df["amount"] > 500
        ).astype(int)

        df["amount_over_1000"] = (
            df["amount"] > 1000
        ).astype(int)

        # ----------------------------------------------------
        # User behavior signals
        # ----------------------------------------------------

        df["user_id_mod_500"] = (
            df["user_id"] % 500
        )

        df["user_id_mod_1000"] = (
            df["user_id"] % 1000
        )

        df["user_id_divisible_500"] = (
            df["user_id"] % 500 == 0
        ).astype(int)

        df["user_id_divisible_1000"] = (
            df["user_id"] % 1000 == 0
        ).astype(int)

        # ----------------------------------------------------
        # Merchant risk
        # ----------------------------------------------------

        high_risk_merchants = self.config.get(
            "high_risk_merchants",
            [
                "QuickCash",
                "GlobalDigital",
                "FastMoneyX",
            ],
        )

        df["merchant_risk"] = (
            df["merchant"]
            .isin(high_risk_merchants)
            .astype(int)
        )

        # ----------------------------------------------------
        # Location
        # ----------------------------------------------------

        df["location"] = (
            df["location"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )

        # ----------------------------------------------------
        # Final features
        # ----------------------------------------------------

        feature_cols = [
            "amount",
            "amount_log",
            "amount_over_500",
            "amount_over_1000",
            "transaction_hour",
            "is_night",
            "is_weekend",
            "transaction_day",
            "user_id_mod_500",
            "user_id_mod_1000",
            "user_id_divisible_500",
            "user_id_divisible_1000",
            "merchant_risk",
            "location",
        ]

        result = df[
            feature_cols + ["is_fraud"]
        ].copy()

        return result

    # ========================================================
    # Threshold
    # ========================================================

    @staticmethod
    def find_best_threshold(y_true, probabilities):

        precision, recall, thresholds = (
            precision_recall_curve(
                y_true,
                probabilities,
            )
        )

        if len(thresholds) == 0:
            return 0.5

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / (
                precision[:-1]
                + recall[:-1]
                + 1e-12
            )
        )

        idx = int(
            np.argmax(f1_scores)
        )

        return float(
            thresholds[idx]
        )

    # ========================================================
    # Training
    # ========================================================

    def train_model(self):

        logger.info(
            "Starting V2 model training"
        )

        df = self.read_from_kafka()

        data = self.create_features(df)

        X = data.drop(
            columns=["is_fraud"]
        )

        y = data["is_fraud"].astype(int)

        positive = int(y.sum())
        negative = int((y == 0).sum())

        logger.info(
            "Class distribution: fraud=%d normal=%d rate=%.4f%%",
            positive,
            negative,
            positive / len(y) * 100,
        )

        if positive < 20:
            raise ValueError(
                f"Only {positive} fraud samples. "
                "Increase Kafka training data."
            )

        # ----------------------------------------------------
        # Train / Validation / Test
        # ----------------------------------------------------

        seed = int(
            self.config["model"].get(
                "seed",
                42,
            )
        )

        test_size = float(
            self.config["model"].get(
                "test_size",
                0.2,
            )
        )

        X_temp, X_test, y_temp, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=seed,
            )
        )

        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_temp,
                y_temp,
                test_size=0.25,
                stratify=y_temp,
                random_state=seed,
            )
        )

        logger.info(
            "Split: train=%d validation=%d test=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        # ----------------------------------------------------
        # Class weight
        # ----------------------------------------------------

        scale_pos_weight = (
            negative / max(positive, 1)
        )

        logger.info(
            "scale_pos_weight=%.3f",
            scale_pos_weight,
        )

        with mlflow.start_run():

            mlflow.log_params(
                {
                    "train_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "test_samples": len(X_test),
                    "fraud_samples": positive,
                    "fraud_rate": positive / len(y),
                    "scale_pos_weight": scale_pos_weight,
                }
            )

            # ------------------------------------------------
            # Preprocessor
            # ------------------------------------------------

            categorical_features = [
                "location",
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "location",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                        categorical_features,
                    ),
                ],
                remainder="passthrough",
            )

            # ------------------------------------------------
            # XGBoost
            # ------------------------------------------------

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                scale_pos_weight=scale_pos_weight,
            )

            from sklearn.pipeline import Pipeline

            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "classifier",
                        xgb,
                    ),
                ]
            )

            # ------------------------------------------------
            # Hyperparameter search
            # ------------------------------------------------

            param_dist = {
                "classifier__n_estimators": [
                    200,
                    300,
                    500,
                    700,
                ],
                "classifier__max_depth": [
                    3,
                    4,
                    5,
                    6,
                ],
                "classifier__learning_rate": [
                    0.02,
                    0.05,
                    0.08,
                    0.1,
                ],
                "classifier__subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
                "classifier__colsample_bytree": [
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
                "classifier__min_child_weight": [
                    1,
                    3,
                    5,
                    10,
                ],
                "classifier__gamma": [
                    0,
                    0.1,
                    0.3,
                    0.5,
                ],
                "classifier__reg_alpha": [
                    0,
                    0.01,
                    0.1,
                    0.5,
                ],
                "classifier__reg_lambda": [
                    1,
                    2,
                    5,
                    10,
                ],
            }

            n_iter = int(
                self.config["model"].get(
                    "n_iter",
                    20,
                )
            )

            search_jobs = int(
                self.config["model"].get(
                    "search_n_jobs",
                    4,
                )
            )

            cv_splits = int(
                self.config["model"].get(
                    "cv_splits",
                    3,
                )
            )

            searcher = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="average_precision",
                cv=StratifiedKFold(
                    n_splits=cv_splits,
                    shuffle=True,
                    random_state=seed,
                ),
                n_jobs=search_jobs,
                random_state=seed,
                refit=True,
                error_score="raise",
                verbose=1,
            )

            logger.info(
                "Starting hyperparameter search..."
            )

            searcher.fit(
                X_train,
                y_train,
            )

            best_model = (
                searcher.best_estimator_
            )

            logger.info(
                "Best params: %s",
                searcher.best_params_,
            )

            # ------------------------------------------------
            # Validation threshold
            # ------------------------------------------------

            val_proba = (
                best_model.predict_proba(
                    X_val
                )[:, 1]
            )

            threshold = (
                self.find_best_threshold(
                    y_val,
                    val_proba,
                )
            )

            logger.info(
                "Selected validation threshold: %.6f",
                threshold,
            )

            # ------------------------------------------------
            # Final test
            # ------------------------------------------------

            test_proba = (
                best_model.predict_proba(
                    X_test
                )[:, 1]
            )

            y_pred = (
                test_proba >= threshold
            ).astype(int)

            metrics = {
                "roc_auc": float(
                    roc_auc_score(
                        y_test,
                        test_proba,
                    )
                ),
                "auc_pr": float(
                    average_precision_score(
                        y_test,
                        test_proba,
                    )
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    )
                ),
                "threshold": float(
                    threshold
                ),
                "predicted_fraud_rate": float(
                    y_pred.mean()
                ),
            }

            logger.info(
                "FINAL TEST METRICS: %s",
                metrics,
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.log_params(
                searcher.best_params_
            )

            # ------------------------------------------------
            # Confusion matrix
            # ------------------------------------------------

            cm = confusion_matrix(
                y_test,
                y_pred,
            )

            plt.figure(
                figsize=(6, 5)
            )

            plt.imshow(cm)

            plt.title(
                "Fraud Detection Confusion Matrix"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "Actual"
            )

            for i in range(2):
                for j in range(2):
                    plt.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                    )

            plt.tight_layout()

            cm_path = (
                "confusion_matrix_v2.png"
            )

            plt.savefig(
                cm_path,
                dpi=150,
            )

            mlflow.log_artifact(
                cm_path
            )

            plt.close()

            # ------------------------------------------------
            # PR curve
            # ------------------------------------------------

            precision, recall, _ = (
                precision_recall_curve(
                    y_test,
                    test_proba,
                )
            )

            plt.figure(
                figsize=(8, 6)
            )

            plt.plot(
                recall,
                precision,
            )

            plt.xlabel(
                "Recall"
            )

            plt.ylabel(
                "Precision"
            )

            plt.title(
                "Precision-Recall Curve"
            )

            plt.tight_layout()

            pr_path = (
                "precision_recall_curve_v2.png"
            )

            plt.savefig(
                pr_path,
                dpi=150,
            )

            mlflow.log_artifact(
                pr_path
            )

            plt.close()

            # ------------------------------------------------
            # Save model + threshold
            # ------------------------------------------------

            model_package = {
                "model": best_model,
                "threshold": threshold,
                "feature_version": "v2",
                "feature_columns": list(
                    X.columns
                ),
            }

            os.makedirs(
                "/app/models",
                exist_ok=True,
            )

            model_path = (
                "/app/models/"
                "fraud_detection_model.pkl"
            )

            joblib.dump(
                model_package,
                model_path,
            )

            logger.info(
                "Model package saved to %s",
                model_path,
            )

            # ------------------------------------------------
            # MLflow
            # ------------------------------------------------

            signature = infer_signature(
                X_train,
                best_model.predict(
                    X_train
                ),
            )

            trusted_types = [
                "sklearn.pipeline.Pipeline",
                "sklearn.compose._column_transformer.ColumnTransformer",
                "sklearn.preprocessing._encoders.OneHotEncoder",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ]

            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                registered_model_name=self.config[
                    "mlflow"
                ]["registered_model_name"],
                skops_trusted_types=trusted_types,
            )

            logger.info(
                "Model registered successfully."
            )

            logger.info(
                "Training completed successfully: %s",
                metrics,
            )

            return best_model, metrics


if __name__ == "__main__":

    trainer = FraudDetectionTraining(
        "/app/config.yaml"
    )

    trainer.train_model()


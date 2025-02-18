import torch
from torch import nn

import boto3
import joblib
import pandas as pd
from io import BytesIO, StringIO
from datetime import datetime
from botocore.exceptions import ClientError


class Utilities:
    BUCKET_NAME = "mybucketingbucket"
    S3_CLIENT = boto3.client("s3", region_name="eu-west-3")
    MODEL_FOLDER = "models/"

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def init_xavier_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def clip_grad_norm_(module, max_grad_norm):
        params = [p for g in module.param_groups for p in g["params"]
                  if p.grad is not None]
        """print("Gradients before clipping:")
        for i, p in enumerate(params):
            print(f"Parameter {i}: Norm = {p.grad.norm():.6f}")"""
        total_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
        """print("\nGradients after clipping:")
        for i, p in enumerate(params):
            print(f"Parameter {i}: Norm = {p.grad.norm():.6f}")"""

    @staticmethod
    def clip_logits(logits, clip_value):
        return torch.clamp(logits, -clip_value, clip_value)

    @staticmethod
    def normalize_rewards(r):
        mean = r.mean()
        std = r.std() + 1e-8
        return (r - mean) / std

    def create_bucket(self):
        response = self.S3_CLIENT.list_buckets()

        bucket_names = [bucket["Name"] for bucket in response["Buckets"]]
        if self.BUCKET_NAME not in bucket_names:
            self.logger.info(
                f"Bucket {self.BUCKET_NAME} not found. Creating it now...")
            self.S3_CLIENT.create_bucket(Bucket=self.BUCKET_NAME)
            self.S3_CLIENT.put_object(Bucket=self.BUCKET_NAME, Key="model/")
            self.S3_CLIENT.put_object(Bucket=self.BUCKET_NAME, Key="data/")
            self.logger.info(
                f"Bucket {self.BUCKET_NAME} created successfully.")
        else:
            self.logger.info(f"Bucket {self.BUCKET_NAME} already exists.")

    def save_data(self, data: pd.DataFrame) -> None:
        self.create_bucket()

        try:
            self.logger.info("Saving data to S3...")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)

            self.S3_CLIENT.put_object(
                Bucket=self.BUCKET_NAME,
                Key=f"data/data_{timestamp}.csv",
                Body=csv_buffer.getvalue())
            self.logger.info("Data saved successfully to S3.")

        except ClientError as e:
            self.logger.error(f"Error saving data to S3: {e}")

    def load_data(self) -> pd.DataFrame:
        try:
            self.logger.info("Loading the latest data from S3...")

            response = self.S3_CLIENT.list_objects_v2(
                Bucket=self.BUCKET_NAME,
                Prefix="data/")
            list_data_files = [obj["Key"] for obj in response.get("Contents", [])
                               if obj["Key"].endswith(".csv")]

            if not list_data_files:
                self.logger.info("No data files found in the S3 bucket.")
                return None

            list_data_files.sort(reverse=True)

            latest_data_key = list_data_files[0]

            response = self.S3_CLIENT.get_object(
                Bucket=self.BUCKET_NAME, Key=latest_data_key)

            data = pd.read_csv(response["Body"], delimiter=",")

            self.logger.info(
                f"Data loaded successfully from {latest_data_key}.")
            return data

        except ClientError as e:
            self.logger.info(f"Error loading data from S3: {e}")
            return None

    def save_model(self, model_data) -> None:

        try:
            self.logger.info("Saving model to S3...")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            model_buffer = BytesIO()
            joblib.dump(model_data, model_buffer)
            model_buffer.seek(0)

            self.S3_CLIENT.put_object(
                Bucket=self.BUCKET_NAME,
                Key=f"models/model_{timestamp}.bin",
                Body=model_buffer.getvalue())
            self.logger.info("Model saved successfully.")

        except ClientError as e:
            self.logger.error(f"Error saving model to S3: {e}")

    def load_model(self) -> bytes:
        self.create_bucket()
        try:
            self.logger.info("Loading the latest model from S3...")

            response = self.S3_CLIENT.list_objects_v2(
                Bucket=self.BUCKET_NAME,
                Prefix=self.MODEL_FOLDER)

            list_models = [obj["Key"] for obj in response.get("Contents", [])
                           if obj["Key"].endswith(".bin")]

            if not list_models:
                self.logger.info("No models found in the S3 bucket.")
                return None

            list_models.sort(reverse=True)

            latest_model_key = list_models[0]

            response = self.S3_CLIENT.get_object(
                Bucket=self.BUCKET_NAME, Key=latest_model_key)

            model_bytes = response["Body"].read()

            model = joblib.load(BytesIO(model_bytes))

            self.logger.info(
                f"Model loaded successfully from {latest_model_key}.")
            return model

        except ClientError as e:
            self.logger.error(f"Error loading model from S3: {e}")
            return None

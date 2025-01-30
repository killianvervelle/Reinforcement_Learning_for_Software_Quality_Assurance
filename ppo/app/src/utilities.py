import io
import torch.optim as optim
import torch.nn as nn
import torch
import logging
import boto3
import pandas as pd
from io import StringIO
from datetime import datetime
from botocore.exceptions import ClientError


class Utilities:
    BUCKET_NAME = "mybucketingbucket"
    S3_CLIENT = boto3.client("s3", region_name="eu-west-3")
    MODEL_FOLDER = "models/"

    def __init__(self, logger):
        self.logger = logger

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
                               if obj["Key"].startswith("data/")]

            if not list_data_files:
                self.logger.error("No data files found in the S3 bucket.")
                return pd.DataFrame()

            list_data_files.sort(reverse=True)

            latest_data_key = list_data_files[0]

            response = self.S3_CLIENT.get_object(
                Bucket=self.BUCKET_NAME, Key=latest_data_key)

            data = pd.read_csv(response["Body"], delimiter=",")

            self.logger.info(
                f"Data loaded successfully from {latest_data_key}.")
            return data

        except ClientError as e:
            self.logger.error(f"Error loading data from S3: {e}")
            return pd.DataFrame()

    def save_model(self, model_data: bytes) -> None:
        self.create_bucket()

        try:
            self.logger.info("Saving model to S3...")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            self.S3_CLIENT.put_object(
                Bucket=self.BUCKET_NAME,
                Key=f"models/model_{timestamp}.bin",
                Body=model_data)
            self.logger.info("Model saved successfully.")

        except ClientError as e:
            self.logger.error(f"Error saving model to S3: {e}")

    def load_model(self) -> bytes:
        try:
            self.logger.info("Loading the latest model from S3...")

            response = self.S3_CLIENT.list_objects_v2(
                Bucket=self.BUCKET_NAME,
                Prefix=self.MODEL_FOLDER)

            list_models = [obj["Key"] for obj in response.get("Contents", [])
                           if obj["Key"].startswith("models/")]

            if not list_models:
                self.logger.error("No models found in the S3 bucket.")
                return b""

            list_models.sort(reverse=True)

            latest_model_key = list_models[0]

            response = self.S3_CLIENT.get_object(
                Bucket=self.BUCKET_NAME, Key=latest_model_key)

            model_data = response["Body"].read()

            buffer = io.BytesIO(model_data)
            model = torch.load(buffer, map_location=torch.device("cpu"))
            model.eval()

            self.logger.info(
                f"Model loaded successfully from {latest_model_key}.")
            return model

        except ClientError as e:
            self.logger.error(f"Error loading model from S3: {e}")
            return b""

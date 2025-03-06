from botocore.exceptions import ClientError
from datetime import datetime
from io import BytesIO
import pandas as pd
import joblib
import logging
import boto3
import json
import sys

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


sys.path.append("/opt")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

S3_BUCKET_NAME = "mybucketingbucket"
s3_client = boto3.client("s3")


def load_data():
    """Load the latest data file from S3."""
    try:
        logger.info("Loading the latest data from S3...")
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME, Prefix="data/")
        files = [obj["Key"] for obj in response.get(
            "Contents", []) if obj["Key"].startswith("data/")]

        if not files:
            logger.error("No data files found in the S3 bucket.")
            return pd.DataFrame()

        latest_file = sorted(files, reverse=True)[0]
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=latest_file)
        data = pd.read_csv(response["Body"], delimiter=",")
        logger.info(f"Data loaded successfully from {latest_file}.")

        return data

    except ClientError as e:
        logger.error(f"Error loading data from S3: {e}")

        return pd.DataFrame()


def train_model(df):
    """Train an SVR model and return the model binary and best parameters."""
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 1],
        'svr__gamma': ['scale', 0.01, 0.1, 1]
    }

    X = df[["cpu", "memory"]]
    y = df["responseTime"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    model_bytes = BytesIO()
    joblib.dump(best_model, model_bytes)

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.info(f"Trained SVM Regression\nMSE: {mse:.2f}\nR-squared: {r2:.2f}")

    return model_bytes, grid_search.best_params_


def save_model(model_data):
    """Save trained model to S3."""
    try:
        logger.info("Saving model to S3...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        key = f"models/model_{timestamp}.bin"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=key,
                             Body=model_data.getvalue())
        logger.info(f"Model saved successfully as {key}.")
    except ClientError as e:
        logger.error(f"Error saving model to S3: {e}")


def lambda_handler(event, context):
    logger.info("Starting ML job in AWS Lambda...")
    data = load_data()
    if data.empty:
        return {"statusCode": 500, "body": json.dumps("Failed to load data.")}

    model_bytes, _ = train_model(data)

    save_model(model_bytes)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Model trained and saved successfully."
        })
    }

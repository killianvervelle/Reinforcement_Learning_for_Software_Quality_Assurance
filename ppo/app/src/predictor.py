from datetime import datetime
import joblib
import logging
from io import BytesIO
from botocore.exceptions import ClientError

import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from ppo.app.src.utilities import Utilities


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(df):
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 1],
        'svr__gamma': ['scale', 0.01, 0.1, 1]
    }

    X = df[["cpu", "memory"]]
    y = df["responseTime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    svm_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

    grid_search = GridSearchCV(
        svm_pipeline, param_grid, cv=5, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    model_bytes = BytesIO()
    joblib.dump(best_model, model_bytes)

    svm_test_preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, svm_test_preds)
    r2 = r2_score(y_test, svm_test_preds)

    print(
        f"Tuned SVM Regression \nMSE: {mse:.2f}\nR-squared: {r2:.2f}")

    return model_bytes, grid_search.best_params_

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
            Body=model_data.getvalue())
        self.logger.info("Model saved successfully.")

    except ClientError as e:
        self.logger.error(f"Error saving model to S3: {e}")


if __name__ == "__main__":

    data = load_data()

    model_bytes, best_params_ = train_model(data)

    save_model(model_bytes)

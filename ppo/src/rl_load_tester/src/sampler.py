import os
import random
import time
import pandas as pd
import requests

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class Sampler:
    BASE_URL = os.getenv(
        "API_URL", "")

    def __init__(self):
        self.num_samples = 150

    def adjust_container_resources(self, cpu: int, memory: int) -> bool:
        try:
            response = requests.post(
                f"{self.BASE_URL}/adjust_container_resources/",
                params={"cpu": cpu, "memory": memory}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False

    def run_jmeter_test(self, threads: int, rampup: int, loops: int) -> int:
        try:
            response = requests.post(
                f"{self.BASE_URL}/run_jmeter_test_plan/",
                params={"threads": threads, "rampup": rampup, "loops": loops}
            )
            response.raise_for_status()
            data = response.json()
            return data["response_time"]
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return -1

    def generate_dataset(self):
        data = []
        test_cases = [(random.randint(35, 95), random.uniform(0.7, 2))
                      for _ in range(self.num_samples)]

        for cpu, memory in test_cases:
            self.adjust_container_resources(cpu=cpu, memory=memory)
            time.sleep(3)

            response_time = self.run_jmeter_test(threads=2, rampup=1, loops=1)

            data.append({"cpu": cpu, "memory": round(memory, 2),
                        "response_time": round(response_time)})

            print(f"\nSystem Metrics After JMeter Test:")
            print(f"CPU Usage: {cpu}%")
            print(f"Memory Usage: {round(memory, 2)} MB")
            print(f"Response Time: {round(response_time)} seconds")

        return pd.DataFrame(data)

    def train_model(self, df):
        X = df[["cpu", "memory"]]
        y = df["response_time"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        param_grid = {
            "XGB__learning_rate": [0.01, 0.1, 0.3],
            "XGB__n_estimators": [50, 100, 200],
            "XGB__max_depth": [3, 6, 9, 12],
            "XGB__reg_lambda": [0.01, 0.1, 1],
        }

        xgb_pipeline = Pipeline(steps=[("scaler", StandardScaler()),
                                       ("XGB", XGBRegressor())])

        model = GridSearchCV(xgb_pipeline, param_grid,
                             cv=5, scoring="neg_mean_absolute_error")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Mean Absolute Error: {mae}")

        return model.best_estimator_

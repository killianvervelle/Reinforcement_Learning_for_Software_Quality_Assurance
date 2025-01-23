import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import torch


class ResponseTimePredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.df = pd.DataFrame(self.data)

        self.process_data()

    def process_data(self):
        self.df["CPU"] = self.df["CPU"].apply(lambda x: int(x / 1000))
        self.df["Memory"] = self.df["Memory"].apply(
            lambda x: int("".join(x[:-1])) / 1000)
        self.df["ResponseTimeCpus"] = self.df["ResponseTimeCpus"].apply(
            lambda x: int(x * 1000))
        self.df["ResponseTimeMems"] = self.df["ResponseTimeMems"].apply(
            lambda x: int(x * 1000))

    def plot_response_time_vs_cpu(self):
        sns.lmplot(data=self.df, x="CPU", y="ResponseTimeCpus", aspect=1.5)
        plt.show()

        sns.lmplot(data=self.df, x="Memory", y="ResponseTimeCpus", aspect=1.5)
        plt.show()

    def plot_response_time_vs_memory(self):
        sns.lmplot(data=self.df, x="CPU", y="ResponseTimeMems", aspect=1.5)
        plt.show()

        sns.lmplot(data=self.df, x="Memory", y="ResponseTimeMems", aspect=1.5)
        plt.show()

    def train_svm_model(self, target_column):
        param_grid = {
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 1],
            'svr__gamma': ['scale', 0.01, 0.1, 1]
        }

        X = self.df[["CPU", "Memory"]]
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        svm_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

        grid_search = GridSearchCV(
            svm_pipeline, param_grid, cv=5, scoring='r2', verbose=1)
        grid_search.fit(X_train, y_train)

        print(f"Best Parameters for {target_column}:",
              grid_search.best_params_)
        best_model = grid_search.best_estimator_

        torch.save(best_model, f"best_model_{target_column}")

        svm_test_preds = best_model.predict(X_test)
        mse = mean_squared_error(y_test, svm_test_preds)
        r2 = r2_score(y_test, svm_test_preds)
        print(
            f"Tuned SVM Regression for {target_column}\nMSE: {mse:.2f}\nR-squared: {r2:.2f}")

        residuals = y_test - svm_test_preds

        plt.figure(figsize=(6, 3))
        plt.scatter(svm_test_preds, residuals, color="blue", alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f"Residuals vs Predicted Values for {target_column}")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.grid()
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.scatter(y_test, svm_test_preds, color="green",
                    alpha=0.6, label="Predictions")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
                 color='red', linestyle='--', label="Perfect Fit")
        plt.title(f"True vs Predicted Values for {target_column}")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid()
        plt.show()

        return best_model, grid_search.best_params_


predictor = ResponseTimePredictor("response_times.csv")

predictor.plot_response_time_vs_cpu()
predictor.plot_response_time_vs_memory()

best_model_mem, best_params_mem = predictor.train_svm_model("ResponseTimeMems")

best_model_cpu, best_params_cpu = predictor.train_svm_model("ResponseTimeCpus")

import os
import sys
import joblib
import logging
from io import BytesIO

import joblib
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from ppo.src.utilities import Utilities

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

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


if __name__ == "__main__":
    utilities = Utilities(logger)

    data = utilities.load_data()

    model_bytes, best_params_ = train_model(data)

    utilities.save_model(model_bytes)

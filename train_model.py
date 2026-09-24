"""
train_model.py
================
Trains and saves the car price prediction model.
Run from the project root: python model/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "car_price_prediction_.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_price_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

CURRENT_YEAR = 2026


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    df["Car Age"] = CURRENT_YEAR - df["Year"]
    return df


def build_pipeline():
    numeric = ["Year", "Engine Size", "Mileage", "Car Age"]
    categorical = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return preprocessor, numeric, categorical


def train():
    df = load_data()
    preprocessor, numeric, categorical = build_pipeline()

    features = numeric + categorical
    X = df[features]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42),
    }

    best_name, best_r2, best_pipe = None, -np.inf, None
    all_metrics = []

    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        all_metrics.append({"model": name, "r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)})

        if r2 > best_r2:
            best_r2, best_name, best_pipe = r2, name, pipe

    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": all_metrics}, f, indent=2)

    print(f"Best model: {best_name} (R2={best_r2:.4f})")
    print(f"Saved to: {MODEL_PATH}")
    return best_pipe, all_metrics


if __name__ == "__main__":
    train()

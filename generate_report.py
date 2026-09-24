"""
generate_report.py
====================
Regenerates all EDA/evaluation charts (into report_images/) and the
model comparison metrics (model/metrics.json) from scratch.

Run from the project root: python generate_report.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "car_price_prediction_.csv")
IMG_DIR = os.path.join(BASE_DIR, "report_images")
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_price_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
CURRENT_YEAR = 2026

os.makedirs(IMG_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()
    df["Car Age"] = CURRENT_YEAR - df["Year"]

    # --- EDA charts ---
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df["Price"], bins=40, kde=True, color="#4C72B0")
    plt.title("Distribution of Car Prices")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/price_distribution.png", dpi=130)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    order = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Brand", y="Price", order=order, hue="Brand", legend=False, palette="Blues_r")
    plt.title("Price Distribution by Brand")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/price_by_brand.png", dpi=130)
    plt.close()

    num_cols = ["Year", "Engine Size", "Mileage", "Car Age", "Price"]
    corr = df[num_cols].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/correlation_heatmap.png", dpi=130)
    plt.close()

    # --- Model training & comparison ---
    features = ["Year", "Engine Size", "Mileage", "Car Age", "Brand", "Fuel Type", "Transmission", "Condition", "Model"]
    X = df[features]
    y = df["Price"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["Year", "Engine Size", "Mileage", "Car Age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42),
    }

    results, best_name, best_r2, best_pipe = [], None, -np.inf, None
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({"model": name, "r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)})
        if r2 > best_r2:
            best_r2, best_name, best_pipe = r2, name, pipe

    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=results_df, x="model", y="r2", hue="model", legend=False, palette="Blues_d")
    plt.title("Model Comparison (R² Score)")
    plt.xticks(rotation=15)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/model_comparison.png", dpi=130)
    plt.close()

    best_preds = best_pipe.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_preds, alpha=0.4, s=15, color="#4C72B0")
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual Price ({best_name})")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/predicted_vs_actual.png", dpi=130)
    plt.close()

    print("Report regenerated.")
    print(json.dumps(results, indent=2))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()

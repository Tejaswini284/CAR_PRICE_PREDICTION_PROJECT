"""
app.py
=======
Flask API that serves car price predictions using the trained model.

Run:  python app.py
Then POST to http://localhost:5500/predict with JSON like:
{
  "Brand": "Toyota",
  "Year": 2018,
  "Engine Size": 2.0,
  "Fuel Type": "Petrol",
  "Transmission": "Automatic",
  "Mileage": 45000,
  "Condition": "Used",
  "Model": "Camry"
}
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_price_model.joblib")
CURRENT_YEAR = 2026

app = Flask(__name__)
CORS(app)  # allow the frontend (served from a different origin) to call this API

model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = ["Brand", "Year", "Engine Size", "Fuel Type", "Transmission", "Mileage", "Condition", "Model"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = {f: data[f] for f in REQUIRED_FIELDS}
    row["Car Age"] = CURRENT_YEAR - int(row["Year"])

    df = pd.DataFrame([row])

    try:
        prediction = float(model.predict(df)[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"predicted_price": round(prediction, 2)})


if __name__ == "__main__":
    app.run(debug=True, port=5500)

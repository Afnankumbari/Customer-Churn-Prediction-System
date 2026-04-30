"""
app.py  ──  ChurnShield Flask Backend
======================================
Python + Flask + Sklearn + HTML/CSS/JS

Routes:
  GET  /              → Dashboard home (KPIs + charts data)
  GET  /predict       → Prediction form page
  POST /api/predict   → JSON prediction endpoint (called by JS fetch)
  GET  /eda           → EDA visualizations page
  GET  /models        → Model performance page
  GET  /customers     → Customer data table page
  GET  /api/customers → JSON customer data (filterable)
  GET  /api/stats     → JSON dataset statistics

Run:
  python app.py
  → http://127.0.0.1:5000
"""

import os
import sys
import json
import pickle
import warnings
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from preprocess import engineer_features, load_data

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# ── Load model & data at startup ──────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
DATA_PATH  = os.path.join(ROOT, "data", "customers.csv")

_model_bundle = None
_df           = None


def get_model():
    global _model_bundle
    if _model_bundle is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
    return _model_bundle


def get_df():
    global _df
    if _df is None and os.path.exists(DATA_PATH):
        _df = load_data(DATA_PATH)
        _df["TotalCharges"] = pd.to_numeric(_df["TotalCharges"], errors="coerce")
    return _df


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all KPIs and chart data from the DataFrame."""
    churned     = int(df["Churn"].sum())
    total       = len(df)
    churn_rate  = round(df["Churn"].mean() * 100, 1)
    avg_monthly = round(df["MonthlyCharges"].mean(), 2)
    avg_tenure  = round(df["Tenure"].mean(), 1)

    # Churn by contract
    by_contract = (
        df.groupby("Contract")["Churn"].mean().mul(100).round(1).to_dict()
    )
    # Churn by internet
    by_internet = (
        df.groupby("InternetService")["Churn"].mean().mul(100).round(1).to_dict()
    )
    # Churn by payment
    by_payment = (
        df.groupby("PaymentMethod")["Churn"].mean().mul(100).round(1).to_dict()
    )
    # Avg charges/tenure by churn
    grp = df.groupby("Churn").agg(
        avg_monthly=("MonthlyCharges", "mean"),
        avg_tenure=("Tenure", "mean"),
        avg_total=("TotalCharges", "mean"),
    ).round(2)

    retained = grp.loc[0].to_dict() if 0 in grp.index else {}
    churned_grp = grp.loc[1].to_dict() if 1 in grp.index else {}

    # Monthly charges histogram bins
    bins = pd.cut(
        df["MonthlyCharges"].dropna(),
        bins=list(range(18, 125, 10)),
        right=False,
    )
    hist_all = bins.value_counts().sort_index()
    hist_churned = pd.cut(
        df[df["Churn"]==1]["MonthlyCharges"].dropna(),
        bins=list(range(18, 125, 10)),
        right=False,
    ).value_counts().sort_index()

    return {
        "total": total,
        "churned": churned,
        "retained": total - churned,
        "churn_rate": churn_rate,
        "avg_monthly": avg_monthly,
        "avg_tenure": avg_tenure,
        "by_contract": by_contract,
        "by_internet": by_internet,
        "by_payment": by_payment,
        "retained_stats": retained,
        "churned_stats": churned_grp,
        "hist_labels": [str(i) for i in hist_all.index],
        "hist_all": hist_all.values.tolist(),
        "hist_churned": hist_churned.reindex(hist_all.index, fill_value=0).values.tolist(),
    }


# ── HTML Page Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    df = get_df()
    model = get_model()
    if df is None or model is None:
        return render_template("setup.html")
    stats = compute_stats(df)
    return render_template("index.html", stats=stats, model=model)


@app.route("/predict")
def predict_page():
    model = get_model()
    return render_template("predict.html", model=model)


@app.route("/eda")
def eda_page():
    df = get_df()
    if df is None:
        return render_template("setup.html")
    stats = compute_stats(df)
    return render_template("eda.html", stats=stats)


@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/customers")
def customers_page():
    return render_template("customers.html")


# ── JSON API Routes ────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Receives customer data as JSON,
    runs sklearn pipeline, returns prediction.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        model_bundle = get_model()
        if model_bundle is None:
            return jsonify({"error": "Model not loaded. Run pipeline first."}), 500

        # Build input DataFrame exactly as training expects
        input_dict = {
            "Gender":           data.get("Gender", "Male"),
            "SeniorCitizen":    int(data.get("SeniorCitizen", 0)),
            "Partner":          data.get("Partner", "No"),
            "Dependents":       data.get("Dependents", "No"),
            "Tenure":           int(data.get("Tenure", 12)),
            "Age":              int(data.get("Age", 35)),
            "PhoneService":     data.get("PhoneService", "Yes"),
            "MultipleLines":    data.get("MultipleLines", "No"),
            "InternetService":  data.get("InternetService", "Fiber optic"),
            "OnlineSecurity":   data.get("OnlineSecurity", "No"),
            "OnlineBackup":     data.get("OnlineBackup", "No"),
            "DeviceProtection": data.get("DeviceProtection", "No"),
            "TechSupport":      data.get("TechSupport", "No"),
            "StreamingTV":      data.get("StreamingTV", "No"),
            "StreamingMovies":  data.get("StreamingMovies", "No"),
            "Contract":         data.get("Contract", "Month-to-month"),
            "PaperlessBilling": data.get("PaperlessBilling", "Yes"),
            "PaymentMethod":    data.get("PaymentMethod", "Electronic check"),
            "MonthlyCharges":   float(data.get("MonthlyCharges", 75)),
            "TotalCharges":     float(data.get("TotalCharges", 900)),
            "NumComplaints":    int(data.get("NumComplaints", 0)),
            "SupportCalls":     int(data.get("SupportCalls", 1)),
        }

        input_df = pd.DataFrame([input_dict])
        input_df = engineer_features(input_df)

        pipe  = model_bundle["model"]
        prob  = float(pipe.predict_proba(input_df)[0][1])
        pred  = int(prob >= 0.5)
        risk  = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        # Risk factors (human-readable explanations)
        risk_factors = []
        checks = [
            (data.get("Contract") == "Month-to-month",  "Month-to-month contract (+38% churn rate)"),
            (data.get("InternetService") == "Fiber optic", "Fiber optic internet (+28% churn rate)"),
            (data.get("PaymentMethod") == "Electronic check", "Electronic check payment (+28% churn rate)"),
            (int(data.get("Tenure", 12)) < 12,          "Short tenure < 12 months (high risk window)"),
            (data.get("PaperlessBilling") == "Yes",      "Paperless billing enabled"),
            (int(data.get("NumComplaints", 0)) > 0,      f"{data.get('NumComplaints')} open complaint(s)"),
            (int(data.get("SupportCalls", 0)) > 3,       f"{data.get('SupportCalls')} support calls (high)"),
            (int(data.get("SeniorCitizen", 0)) == 1,     "Senior citizen customer"),
            (data.get("OnlineSecurity") == "No",         "No online security subscribed"),
            (data.get("TechSupport") == "No",            "No tech support subscribed"),
        ]
        for is_risk, label in checks:
            risk_factors.append({"label": label, "risk": bool(is_risk)})

        return jsonify({
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "prediction": pred,
            "risk_level": risk,
            "model_name": model_bundle["name"],
            "model_auc": model_bundle["auc"],
            "risk_factors": risk_factors,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    df = get_df()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(compute_stats(df))


@app.route("/api/customers")
def api_customers():
    df = get_df()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    # Filters from query string
    churn_filter    = request.args.get("churn", "")
    contract_filter = request.args.get("contract", "")
    internet_filter = request.args.get("internet", "")
    search          = request.args.get("search", "").strip().upper()
    page            = int(request.args.get("page", 1))
    per_page        = int(request.args.get("per_page", 50))

    view = df.copy()
    if churn_filter in ("0", "1"):
        view = view[view["Churn"] == int(churn_filter)]
    if contract_filter:
        view = view[view["Contract"] == contract_filter]
    if internet_filter:
        view = view[view["InternetService"] == internet_filter]
    if search:
        view = view[view["CustomerID"].str.upper().str.contains(search, na=False)]

    total_filtered = len(view)
    start = (page - 1) * per_page
    page_data = view.iloc[start: start + per_page]

    cols = ["CustomerID", "Contract", "InternetService", "Tenure",
            "MonthlyCharges", "TotalCharges", "NumComplaints", "SupportCalls",
            "SeniorCitizen", "Churn"]
    records = page_data[cols].fillna("").to_dict(orient="records")

    return jsonify({
        "total": total_filtered,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, -(-total_filtered // per_page)),
        "records": records,
    })


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(os.path.join(ROOT, "assets"), filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🛡️  ChurnShield — Flask + Python + HTML/CSS")
    print("=" * 55)

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("\n⚠️  First run the pipeline:")
        print("    python run_pipeline.py\n")
    else:
        print(f"\n✅ Model loaded : {get_model()['name']}")
        print(f"✅ Dataset loaded: {len(get_df()):,} rows\n")

    print("🌐 Starting server → http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

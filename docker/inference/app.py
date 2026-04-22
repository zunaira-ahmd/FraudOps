"""
FraudOps Inference API
======================
Endpoints:
  POST /predict          — run fraud prediction on a single transaction
  POST /update-metrics   — receive evaluation metrics from the KFP pipeline
  POST /reload-model     — reload the best model from local artifacts dir
  GET  /health           — liveness check
  GET  /metrics          — Prometheus metrics (scraped by Prometheus)

Models are loaded from LOCAL_ARTIFACTS_DIR (env var, default /home/ubuntu/FraudOps/artifacts).
The app picks the most recently modified .joblib file whose name starts with "xgb_".
"""

import os
import glob
import time
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("fraudops")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/home/ubuntu/FraudOps/artifacts")
MODEL_PREFIX   = os.getenv("MODEL_PREFIX", "xgb_")   # which model to serve

# ─────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraudops_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraudops_request_latency_seconds",
    "Prediction endpoint latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FRAUD_PREDICTIONS = Counter(
    "fraudops_fraud_predictions_total",
    "Number of transactions predicted as fraud",
)
LEGIT_PREDICTIONS = Counter(
    "fraudops_legit_predictions_total",
    "Number of transactions predicted as legit",
)
PREDICTION_CONFIDENCE = Histogram(
    "fraudops_prediction_confidence",
    "Distribution of fraud probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Model-level metrics (updated by pipeline after each run)
MODEL_RECALL          = Gauge("fraudops_model_recall",          "Fraud recall of deployed model")
MODEL_AUC_ROC         = Gauge("fraudops_model_auc_roc",         "AUC-ROC of deployed model")
MODEL_F1              = Gauge("fraudops_model_f1",              "F1-score of deployed model")
MODEL_FPR             = Gauge("fraudops_model_false_positive_rate", "False positive rate")

# Data-level metrics
MISSING_VALUE_RATE    = Gauge("fraudops_missing_value_rate",    "Avg missing value rate in input data")
FEATURE_DRIFT_AMT     = Gauge("fraudops_feature_drift_TransactionAmt", "Drift score for TransactionAmt")
FEATURE_DRIFT_CARD1   = Gauge("fraudops_feature_drift_card1",  "Drift score for card1")
FEATURE_DRIFT_ADDR1   = Gauge("fraudops_feature_drift_addr1",  "Drift score for addr1")

# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
def load_latest_model():
    """
    Scans ARTIFACTS_DIR for files matching MODEL_PREFIX*.joblib
    and loads the most recently modified one.
    Returns (model, path) or (None, None) if none found.
    """
    pattern = os.path.join(ARTIFACTS_DIR, f"{MODEL_PREFIX}*.joblib")
    candidates = glob.glob(pattern)
    if not candidates:
        log.warning(f"No model files found matching: {pattern}")
        return None, None
    latest = max(candidates, key=os.path.getmtime)
    log.info(f"Loading model from: {latest}")
    model = joblib.load(latest)
    log.info("Model loaded successfully.")
    return model, latest


model, model_path = load_latest_model()

# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    """
    Accepts any transaction feature as an optional float.
    Only TransactionAmt is required as a demo field — in production
    you'd pass the full feature vector that matches your training schema.
    """
    TransactionAmt: float
    # All other fields are optional; missing ones are filled with 0.0
    # so the endpoint works with partial feature sets for demo purposes.
    model_config = {"extra": "allow"}


class MetricsPayload(BaseModel):
    recall:             float
    auc_roc:            float
    f1:                 float
    false_positive_rate: float
    feature_drift:      dict
    missing_value_rate: float


class PredictResponse(BaseModel):
    is_fraud:           bool
    fraud_probability:  float
    model_path:         Optional[str]


class HealthResponse(BaseModel):
    status:     str
    model_loaded: bool
    model_path: Optional[str]


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="FraudOps Inference API",
    description="Serves fraud predictions and exposes Prometheus metrics.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=(model is not None),
        model_path=model_path,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: TransactionFeatures):
    start = time.time()

    if model is None:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="No model loaded. Run the pipeline first.")

    try:
        # Build a single-row DataFrame from the request
        data = transaction.model_dump()
        df = pd.DataFrame([data])

        # Align to the feature set the model was trained on
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]

        prob = float(model.predict_proba(df)[0, 1])
        label = int(prob >= 0.5)

        PREDICTION_CONFIDENCE.observe(prob)
        if label == 1:
            FRAUD_PREDICTIONS.inc()
        else:
            LEGIT_PREDICTIONS.inc()

        REQUEST_COUNT.labels(endpoint="predict", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.time() - start)

        log.info(f"Prediction: fraud_prob={prob:.4f}  label={label}")
        return PredictResponse(
            is_fraud=bool(label),
            fraud_probability=round(prob, 6),
            model_path=model_path,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-metrics")
def update_metrics(payload: MetricsPayload):
    """
    Called by the KFP evaluate component after each pipeline run.
    Updates Prometheus gauges so Grafana dashboards reflect the latest model performance.
    """
    MODEL_RECALL.set(payload.recall)
    MODEL_AUC_ROC.set(payload.auc_roc)
    MODEL_F1.set(payload.f1)
    MODEL_FPR.set(payload.false_positive_rate)
    MISSING_VALUE_RATE.set(payload.missing_value_rate)

    if "TransactionAmt" in payload.feature_drift:
        FEATURE_DRIFT_AMT.set(payload.feature_drift["TransactionAmt"])
    if "card1" in payload.feature_drift:
        FEATURE_DRIFT_CARD1.set(payload.feature_drift["card1"])
    if "addr1" in payload.feature_drift:
        FEATURE_DRIFT_ADDR1.set(payload.feature_drift["addr1"])

    log.info(f"Metrics updated: recall={payload.recall:.4f}  auc={payload.auc_roc:.4f}  f1={payload.f1:.4f}")
    return {"status": "ok", "recall": payload.recall, "auc_roc": payload.auc_roc}


@app.post("/reload-model")
def reload_model():
    """
    Rescans ARTIFACTS_DIR and loads the most recently modified model.
    Call this after a pipeline run completes to hot-swap the model
    without restarting the service.
    """
    global model, model_path
    new_model, new_path = load_latest_model()
    if new_model is None:
        raise HTTPException(status_code=404, detail=f"No model found in {ARTIFACTS_DIR}")
    model, model_path = new_model, new_path
    log.info(f"Model reloaded from: {model_path}")
    return {"status": "ok", "model_path": model_path}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
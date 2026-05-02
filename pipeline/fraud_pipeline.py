from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics


# ─────────────────────────────────────────────
# COMPONENT 1: Data Ingestion
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "boto3", "requests"],
)
def ingest(
    transaction_path: str,
    identity_path: str,
    output_data: Output[Dataset],
):
    """
    Reads transaction and identity CSVs. Supports local hostPath files,
    HTTP/HTTPS URLs, and MinIO/S3 paths (e.g. minio://bucket/file.csv).
    Merges them on TransactionID, and writes a single merged CSV to KFP's artifact system.
    """
    import os
    import pandas as pd
    import requests
    from urllib.parse import urlparse

    def download_data(path, local_dest):
        if path.startswith("http://") or path.startswith("https://"):
            print(f"Downloading from URL: {path}")
            response = requests.get(path)
            response.raise_for_status()
            with open(local_dest, "wb") as f:
                f.write(response.content)
            return local_dest
        elif path.startswith("s3://") or path.startswith("minio://"):
            import boto3
            print(f"Downloading from S3/MinIO: {path}")
            parsed = urlparse(path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')

            # Use default Kubeflow MinIO settings if env vars are missing
            s3 = boto3.client(
                's3',
                endpoint_url=os.environ.get("S3_ENDPOINT", "http://minio-service.kubeflow.svc.cluster.local:9000"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
            )
            s3.download_file(bucket, key, local_dest)
            return local_dest
        else:
            print(f"Using local path: {path}")
            return path

    trans_file = download_data(transaction_path, "/tmp/transaction.csv")
    ident_file = download_data(identity_path, "/tmp/identity.csv")

    print("Reading transaction data...")
    trans = pd.read_csv(trans_file)

    print("Reading identity data...")
    identity = pd.read_csv(ident_file)

    print("Merging on TransactionID...")
    df = trans.merge(identity, on="TransactionID", how="left")

    print(f"Merged shape: {df.shape}")
    df.to_csv(output_data.path, index=False)
    print("Ingestion complete. Data is now in KFP artifacts.")


# ─────────────────────────────────────────────
# COMPONENT 2: Data Validation
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy"],
)
def validate(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    metrics: Output[Metrics],
):
    """
    Schema checks, basic statistics, and missing-value reporting.
    Logs key dataset health metrics to the KFP UI.
    """
    import pandas as pd

    df = pd.read_csv(input_data.path)

    # Schema checks
    assert "isFraud" in df.columns, "Target column 'isFraud' missing"
    assert "TransactionID" in df.columns, "TransactionID column missing"
    assert df.shape[0] > 0, "Dataset is empty"

    total = len(df)
    fraud = int(df["isFraud"].sum())
    fraud_rate = round(fraud / total * 100, 2)
    missing_pct = round(df.isnull().mean().mean() * 100, 2)
    n_features = df.shape[1] - 1  # exclude target

    print(f"Total records   : {total}")
    print(f"Fraud cases     : {fraud} ({fraud_rate}%)")
    print(f"Avg missing %   : {missing_pct}%")
    print(f"Feature count   : {n_features}")

    # Check for high-cardinality columns and report them
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    high_card = [c for c in cat_cols if df[c].nunique() > 100]
    print(f"High-cardinality categorical cols (>100 unique): {high_card}")

    metrics.log_metric("total_records", total)
    metrics.log_metric("fraud_cases", fraud)
    metrics.log_metric("fraud_rate_pct", fraud_rate)
    metrics.log_metric("avg_missing_pct", missing_pct)
    metrics.log_metric("n_features", n_features)
    metrics.log_metric("high_cardinality_cols", len(high_card))

    df.to_csv(output_data.path, index=False)
    print("Validation passed.")


# ─────────────────────────────────────────────
# COMPONENT 3: Preprocessing
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def preprocess(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    missing_threshold: float = 0.5,
):
    """
    Handles:
      - Dropping columns with >missing_threshold missing values
      - Median imputation for numerics
      - Mode imputation for categoricals
      - High-cardinality encoding: frequency encoding (>50 unique)
      - Low-cardinality encoding: label encoding
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(input_data.path)

    # Drop high-missing columns
    before = df.shape[1]
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    print(f"Dropped {before - df.shape[1]} columns exceeding {missing_threshold*100}% missing")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]

    # Advanced missing value strategy:
    # Numerics: median imputation (robust to outliers, better than mean for skewed fraud data)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categoricals: mode imputation
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Imputed {len(num_cols)} numeric cols with median")
    print(f"Imputed {len(cat_cols)} categorical cols with mode")

    # Encoding
    le = LabelEncoder()
    high_card_count = 0
    for col in cat_cols:
        if df[col].nunique() > 50:
            # Frequency encoding for high-cardinality features
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).fillna(0.0)
            high_card_count += 1
        else:
            df[col] = le.fit_transform(df[col].astype(str))

    print(f"Frequency-encoded {high_card_count} high-cardinality cols")
    print(f"Label-encoded {len(cat_cols) - high_card_count} low-cardinality cols")

    df.to_csv(output_data.path, index=False)
    print(f"Preprocessing complete. Final shape: {df.shape}")


# ─────────────────────────────────────────────
# COMPONENT 4: Feature Engineering
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def feature_engineering(
    input_data: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    test_size: float = 0.2,
):
    """
    Creates domain-relevant features from transaction data,
    then splits into stratified train/test sets.

    New features:
      - TransactionAmt_log   : log1p transform (reduces skew)
      - TransactionAmt_cent  : cents portion (fraud often has round amounts)
      - amt_to_card_ratio    : transaction amount relative to card1 (spending pattern)
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_cent"] = df["TransactionAmt"] % 1
        if "card1" in df.columns:
            df["amt_to_card_ratio"] = df["TransactionAmt"] / (df["card1"] + 1)

    df.drop(columns=["TransactionID"], inplace=True, errors="ignore")

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    print(f"Class distribution — Legit: {(y==0).sum()}, Fraud: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    train_df = X_train.copy()
    train_df["isFraud"] = y_train.values
    test_df = X_test.copy()
    test_df["isFraud"] = y_test.values

    train_df.to_csv(output_train.path, index=False)
    test_df.to_csv(output_test.path, index=False)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")
    print(f"Train fraud : {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test fraud  : {y_test.sum()} ({y_test.mean()*100:.2f}%)")


# ─────────────────────────────────────────────
# COMPONENT 5: Model Training
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "imbalanced-learn",
        "joblib",
    ],
)
def train(
    input_train: Input[Dataset],
    output_model_xgb: Output[Model],
    output_model_lgbm: Output[Model],
    output_model_hybrid: Output[Model],
    metrics: Output[Metrics],
    run_id: str,
    artifacts_dir: str,
    imbalance_strategy: str = "smote",
    cost_sensitive: bool = True,
):
    """
    Trains three models:
      1. XGBoost  (gradient boosting)
      2. LightGBM (gradient boosting, faster on large data)
      3. Hybrid   (RandomForest + SelectFromModel feature selection)

    Imbalance strategies compared:
      - smote         : synthetic oversampling of minority class
      - class_weight  : cost-sensitive weighting without resampling

    Cost-sensitive flag applies scale_pos_weight / class_weight
    equal to the negative-to-positive class ratio when True.

    Models are saved both to KFP artifact store (for evaluate step)
    and to artifacts_dir on the EC2 local filesystem.
    """
    import os
    import joblib
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    os.makedirs(artifacts_dir, exist_ok=True)

    df = pd.read_csv(input_train.path)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    fraud_weight = neg // pos
    print(f"Class distribution before resampling — Legit: {neg}, Fraud: {pos}")
    print(f"Computed class ratio (neg/pos): {fraud_weight}")
    print(f"Imbalance strategy : {imbalance_strategy}")
    print(f"Cost-sensitive     : {cost_sensitive}")

    # ── Imbalance handling ────────────────────────────────────────────
    if imbalance_strategy == "smote":
        print("\nApplying SMOTE oversampling...")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"After SMOTE — Legit: {(y_res==0).sum()}, Fraud: {(y_res==1).sum()}")
    elif imbalance_strategy == "undersample":
        print("\nApplying Random Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        print(f"After undersampling — Legit: {(y_res==0).sum()}, Fraud: {(y_res==1).sum()}")
    else:
        # class_weight strategy: no resampling, rely on model weights
        print("\nUsing class_weight strategy (no resampling).")
        X_res, y_res = X, y

    # cost-sensitive weight applied to models (scale_pos_weight / class_weight)
    fn_weight = fraud_weight if cost_sensitive else 1
    cw = {0: 1, 1: fn_weight} if cost_sensitive else "balanced"

    # Log imbalance strategy comparison note
    metrics.log_metric("imbalance_strategy", 0 if imbalance_strategy == "smote" else 1)
    metrics.log_metric("cost_sensitive", int(cost_sensitive))
    metrics.log_metric("class_ratio", fraud_weight)
    metrics.log_metric("train_samples_after_resampling", len(y_res))

    # ── XGBoost ──────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        device="cpu",
        scale_pos_weight=fn_weight,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_res, y_res)
    joblib.dump(xgb, output_model_xgb.path)
    joblib.dump(xgb, os.path.join(artifacts_dir, f"xgb_{run_id}.joblib"))
    print("XGBoost saved.")

    # ── LightGBM ─────────────────────────────────────────────────────
    print("\nTraining LightGBM...")
    lgbm = LGBMClassifier(
        device_type="cpu",
        scale_pos_weight=fn_weight,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_res, y_res)
    joblib.dump(lgbm, output_model_lgbm.path)
    joblib.dump(lgbm, os.path.join(artifacts_dir, f"lgbm_{run_id}.joblib"))
    print("LightGBM saved.")

    # ── Hybrid: RF + SelectFromModel feature selection ────────────────
    # This is the "hybrid" model: a RandomForest with built-in feature
    # selection via SelectFromModel, wrapped in a sklearn Pipeline.
    print("\nTraining Hybrid (RF + SelectFromModel)...")
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=100,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
        ),
        threshold="median",
    )
    rf_final = RandomForestClassifier(
        n_estimators=200,
        class_weight=cw,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    hybrid = Pipeline([("selector", selector), ("classifier", rf_final)])
    hybrid.fit(X_res, y_res)
    joblib.dump(hybrid, output_model_hybrid.path)
    joblib.dump(hybrid, os.path.join(artifacts_dir, f"hybrid_{run_id}.joblib"))
    print("Hybrid model saved.")

    print(f"\nAll models saved to {artifacts_dir}")


# ─────────────────────────────────────────────
# COMPONENT 6: Evaluation
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
        "shap",
        "matplotlib",
        "requests",
    ],
)
def evaluate(
    input_test: Input[Dataset],
    model_xgb: Input[Model],
    model_lgbm: Input[Model],
    model_hybrid: Input[Model],
    metrics: Output[Metrics],
    artifacts_dir: str,
    run_id: str,
    inference_api_url: str = "http://localhost:8000",
    recall_threshold: float = 0.75,
):
    """
    Evaluates all three models on the held-out test set.
    Reports: Precision, Recall, F1, AUC-ROC, Confusion Matrix, FPR.

    Also runs SHAP analysis on XGBoost and saves the summary plot
    to artifacts_dir on the local filesystem.

    Pushes metrics to the inference API for Prometheus/Grafana.
    Conditional deployment: logs deploy_decision based on recall_threshold.
    """
    import os
    import joblib
    import requests
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    os.makedirs(artifacts_dir, exist_ok=True)

    df = pd.read_csv(input_test.path)
    X_test = df.drop(columns=["isFraud"])
    y_test = df["isFraud"]

    models = {
        "XGBoost": joblib.load(model_xgb.path),
        "LightGBM": joblib.load(model_lgbm.path),
        "Hybrid (RF+Selection)": joblib.load(model_hybrid.path),
    }

    best_model_name = None
    best_recall = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    results = {}

    print("=" * 55)
    print("MODEL EVALUATION REPORT")
    print("=" * 55)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

        results[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "auc_roc": round(auc, 4),
            "fpr": fpr,
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
        }

        print(f"\nModel          : {name}")
        print(f"  Precision    : {p:.4f}")
        print(f"  Recall       : {r:.4f}")
        print(f"  F1-Score     : {f:.4f}")
        print(f"  AUC-ROC      : {auc:.4f}")
        print(f"  False Pos Rate: {fpr:.4f}")
        print("  Confusion Matrix:")
        print(f"    TP={tp}  FP={fp}")
        print(f"    FN={fn}  TN={tn}")

        # Business impact summary
        fraud_loss_prevented = tp * 1000  # assume $1000 avg fraud loss
        false_alarm_cost = fp * 10        # assume $10 per false alarm review cost
        net_benefit = fraud_loss_prevented - false_alarm_cost
        print("  Business Impact (estimated):")
        print(f"    Fraud losses prevented : ${fraud_loss_prevented:,}")
        print(f"    False alarm costs      : ${false_alarm_cost:,}")
        print(f"    Net benefit            : ${net_benefit:,}")

        if r > best_recall:
            best_recall = r
            best_model_name = name
            best_f1 = f
            best_auc = auc

    print("\n" + "=" * 55)
    print(f"Best model by recall: {best_model_name} (recall={best_recall:.4f})")

    best_res = results[best_model_name]

    # Log metrics to KFP
    metrics.log_metric("best_model", best_model_name)
    metrics.log_metric("best_recall", best_res["recall"])
    metrics.log_metric("best_precision", best_res["precision"])
    metrics.log_metric("best_f1", best_res["f1"])
    metrics.log_metric("best_auc_roc", best_res["auc_roc"])
    metrics.log_metric("false_positive_rate", best_res["fpr"])
    metrics.log_metric("recall_threshold", recall_threshold)

    # ── SHAP explainability ──────────────────────────────────────────
    print("\nRunning SHAP analysis on XGBoost...")
    xgb_model = models["XGBoost"]
    explainer = shap.TreeExplainer(xgb_model)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    shap_path = os.path.join(artifacts_dir, f"shap_summary_{run_id}.png")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — XGBoost ({run_id})")
    plt.tight_layout()
    plt.savefig(shap_path, dpi=100)
    plt.close()
    print(f"SHAP summary plot saved to: {shap_path}")

    # Top 10 features by mean absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(sample.columns, mean_shap), key=lambda x: x[1], reverse=True
    )
    print("\nTop 10 features by SHAP importance:")
    for feat, imp in feature_importance[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    # ── Feature drift scores ─────────────────────────────────────────
    drift_scores = {}
    baseline_mean = 100.0
    if "TransactionAmt" in X_test.columns:
        current_mean = float(X_test["TransactionAmt"].mean())
        drift_scores["TransactionAmt"] = round(
            abs(current_mean - baseline_mean) / baseline_mean, 4
        )
    if "card1" in X_test.columns:
        drift_scores["card1"] = round(float(X_test["card1"].std()) / 10000, 4)
    if "addr1" in X_test.columns:
        drift_scores["addr1"] = round(float(X_test["addr1"].std()) / 500, 4)

    missing_rate = round(float(df.isnull().mean().mean()), 4)

    # ── Push metrics to inference API ────────────────────────────────
    payload = {
        "recall": round(best_recall, 6),
        "auc_roc": round(best_auc, 6),
        "f1": round(best_f1, 6),
        "false_positive_rate": best_res["fpr"],
        "feature_drift": drift_scores,
        "missing_value_rate": missing_rate,
    }

    try:
        resp = requests.post(
            f"{inference_api_url}/update-metrics", json=payload, timeout=10
        )
        print(f"\nMetrics pushed to inference API: HTTP {resp.status_code}")
        print(f"Payload: {payload}")
    except Exception as e:
        print(f"\nWarning: Could not push metrics to inference API: {e}")
        print("Prometheus/Grafana will not reflect this run's metrics.")

    # ── Conditional deployment decision ─────────────────────────────
    deploy = best_recall >= recall_threshold
    metrics.log_metric("deploy_decision", int(deploy))

    print("\n" + "=" * 55)
    if deploy:
        print("DEPLOY DECISION: PASS")
        print(f"  Recall {best_recall:.4f} >= threshold {recall_threshold}")
        print(f"  Model '{best_model_name}' is approved for deployment.")
    else:
        print("DEPLOY DECISION: FAIL")
        print(f"  Recall {best_recall:.4f} < threshold {recall_threshold}")
        print("  Pipeline flagged for review. Retraining recommended.")
    print("=" * 55)


# ─────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────
@dsl.pipeline(
    name="FraudOps Pipeline",
    description=(
        "End-to-end fraud detection pipeline on the IEEE CIS dataset. "
        "Supports MinIO/S3 or local filesystem inputs. "
        "Covers: data ingestion, validation, preprocessing, feature engineering, "
        "training (XGBoost / LightGBM / Hybrid RF), evaluation, SHAP explainability, "
        "and conditional deployment."
    ),
)
def FraudOps_pipeline(
    # ── Data paths (MinIO / S3 / HTTP / Local) ────────────────────────
    transaction_path: str = "minio://fraudops-data/train_transaction.csv",
    identity_path: str = "minio://fraudops-data/train_identity.csv",
    # ── Artifact output directory (local EC2 filesystem) ─────────────
    artifacts_dir: str = "/home/ubuntu/FraudOps/artifacts",
    # ── Run configuration ─────────────────────────────────────────────
    run_id: str = "run-1",
    missing_threshold: float = 0.5,
    test_size: float = 0.2,
    # ── Imbalance strategy: "smote" | "undersample" | "class_weight" ─
    imbalance_strategy: str = "smote",
    cost_sensitive: bool = True,
    # ── Deployment gate ───────────────────────────────────────────────
    recall_threshold: float = 0.75,
    inference_api_url: str = "http://localhost:8000",
):
    # ── Step 1: Ingest ────────────────────────────────────────────────
    ingest_task = ingest(
        transaction_path=transaction_path,
        identity_path=identity_path,
    )
    ingest_task.set_retry(num_retries=2, backoff_duration="30s")

    # ── Step 2: Validate ──────────────────────────────────────────────
    validate_task = validate(
        input_data=ingest_task.outputs["output_data"],
    )
    validate_task.set_retry(num_retries=1, backoff_duration="15s")

    # ── Step 3: Preprocess ────────────────────────────────────────────
    preprocess_task = preprocess(
        input_data=validate_task.outputs["output_data"],
        missing_threshold=missing_threshold,
    )

    # ── Step 4: Feature Engineering ───────────────────────────────────
    fe_task = feature_engineering(
        input_data=preprocess_task.outputs["output_data"],
        test_size=test_size,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────
    train_task = train(
        input_train=fe_task.outputs["output_train"],
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        imbalance_strategy=imbalance_strategy,
        cost_sensitive=cost_sensitive,
    )
    train_task.set_memory_limit("14G")
    train_task.set_cpu_limit("4")
    train_task.set_retry(num_retries=1, backoff_duration="60s")

    # ── Step 6: Evaluate ──────────────────────────────────────────────
    evaluate_task = evaluate(
        input_test=fe_task.outputs["output_test"],
        model_xgb=train_task.outputs["output_model_xgb"],
        model_lgbm=train_task.outputs["output_model_lgbm"],
        model_hybrid=train_task.outputs["output_model_hybrid"],
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        inference_api_url=inference_api_url,
        recall_threshold=recall_threshold,
    )
    evaluate_task.set_memory_limit("8G")
    evaluate_task.set_cpu_limit("4")


# ─────────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=FraudOps_pipeline,
        package_path="pipeline/v1_pipeline.yaml",
    )
    print("Pipeline compiled to pipeline/v1_pipeline.yaml")
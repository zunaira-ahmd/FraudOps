import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics
 
# ── Component 1: Data Ingestion ──────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "kaggle"]
)
def data_ingestion(output_dataset: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv("/data/fraud-artifacts/train_transaction.csv")
    df.to_csv(output_dataset.path, index=False)
    print(f"Ingested {len(df)} rows")
 
 
# ── Component 2: Data Validation ─────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "great-expectations"]
)
def data_validation(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset]
):
    import pandas as pd
    df = pd.read_csv(input_dataset.path)
 
    required_cols = ["TransactionID", "isFraud", "TransactionAmt"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
 
    assert len(df) > 0, "Dataset is empty"
 
    print(f"Validation passed. Shape: {df.shape}")
    df.to_csv(output_dataset.path, index=False)
 
 
# ── Component 3: Data Preprocessing ──────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def data_preprocessing(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
 
    df = pd.read_csv(input_dataset.path)
 
    # Drop columns with >80% missing
    threshold = 0.8
    df = df[df.columns[df.isnull().mean() < threshold]]
 
    # Numeric imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "isFraud"]
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
 
    # Cat imputation
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col].fillna("MISSING", inplace=True)
 
    print(f"Preprocessing done. Shape: {df.shape}")
    df.to_csv(output_dataset.path, index=False)
 
 
# ── Component 4: Feature Engineering ─────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "category-encoders"]
)
def feature_engineering(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset]
):
    import pandas as pd
    import category_encoders as ce
 
    df = pd.read_csv(input_dataset.path)
 
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
 
    # Target encoding for high-cardinality categoricals
    encoder = ce.TargetEncoder(cols=cat_cols)
    df[cat_cols] = encoder.fit_transform(df[cat_cols], df["isFraud"])
 
    print(f"Feature engineering done. Columns: {df.shape[1]}")
    df.to_csv(output_dataset.path, index=False)
 
 
# ── Component 5: Model Training ───────────────────────────────────────────────
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn"]
)
def model_training(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics]
):
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
 
    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
 
    # SMOTE for class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
 
    model = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        n_estimators=200,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42
    )
    model.fit(X_train_res, y_train_res)
 
    with open(output_model.path, "wb") as f:
        pickle.dump(model, f)
 
    output_metrics.log_metric("train_samples", len(X_train_res))
    print("Model training complete")
 
 
# ── Component 6: Model Evaluation ────────────────────────────────────────────
# NOTE: No return value. Accuracy is written as a named Output[Metrics]
# so the pipeline can reference it by name via eval_task.outputs["accuracy_output"]
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "xgboost"]
)
def model_evaluation(
    input_dataset: Input[Dataset],
    input_model: Input[Model],
    output_metrics: Output[Metrics],
    accuracy_output: Output[Metrics]
):
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score,
        f1_score, roc_auc_score
    )
 
    df = pd.read_csv(input_dataset.path)
    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
 
    with open(input_model.path, "rb") as f:
        model = pickle.load(f)
 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    accuracy  = accuracy_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc_roc   = roc_auc_score(y_test, y_prob)
 
    # All metrics in the main output
    output_metrics.log_metric("accuracy",  accuracy)
    output_metrics.log_metric("recall",    recall)
    output_metrics.log_metric("precision", precision)
    output_metrics.log_metric("f1",        f1)
    output_metrics.log_metric("auc_roc",   auc_roc)
 
    # Accuracy also written separately so the pipeline can pass it downstream
    accuracy_output.log_metric("accuracy", accuracy)
 
    print(f"Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | AUC-ROC: {auc_roc:.4f}")
 
 
# ── Component 7: Conditional Deployment ──────────────────────────────────────
@component(base_image="python:3.9")
def conditional_deployment(accuracy_input: Input[Metrics], threshold: float = 0.85):
    acc_value = accuracy_input.metadata.get("accuracy", 0.0)
    if acc_value >= threshold:
        print(f"Accuracy {acc_value:.4f} >= {threshold}. Deploying model...")
        print("Model deployed successfully.")
    else:
        print(f"Accuracy {acc_value:.4f} < {threshold}. Deployment skipped.")
 
 
# ── Pipeline Definition ───────────────────────────────────────────────────────
@dsl.pipeline(
    name="Fraud Detection Pipeline",
    description="End-to-end fraud detection with conditional deployment"
)
def fraud_detection_pipeline(deployment_threshold: float = 0.85):
 
    ingest_task = data_ingestion()\
        .set_retry(num_retries=3, backoff_duration="30s")
 
    validate_task = data_validation(
        input_dataset=ingest_task.outputs["output_dataset"]
    ).set_retry(num_retries=2, backoff_duration="15s")
 
    preprocess_task = data_preprocessing(
        input_dataset=validate_task.outputs["output_dataset"]
    ).set_retry(num_retries=2, backoff_duration="15s")
 
    feature_task = feature_engineering(
        input_dataset=preprocess_task.outputs["output_dataset"]
    ).set_retry(num_retries=2, backoff_duration="15s")
 
    train_task = model_training(
        input_dataset=feature_task.outputs["output_dataset"]
    ).set_cpu_request("2")\
     .set_memory_request("6G")\
     .set_retry(num_retries=2, backoff_duration="60s")
 
    eval_task = model_evaluation(
        input_dataset=feature_task.outputs["output_dataset"],
        input_model=train_task.outputs["output_model"]
    ).set_retry(num_retries=1, backoff_duration="15s")
 
    # Reference accuracy by its named output, not .output
    conditional_deployment(
        accuracy_input=eval_task.outputs["accuracy_output"],
        threshold=deployment_threshold
    )
 
 
# ── Compile & Submit ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        fraud_detection_pipeline,
        "fraud_detection_pipeline.yaml"
    )
    print("Pipeline compiled to fraud_detection_pipeline.yaml")
 
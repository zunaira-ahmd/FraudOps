"""
Unit tests for FraudOps pipeline helper logic.
These tests run in CI (Stage 1) and do NOT require KFP or a Kubernetes cluster.
"""

import os
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers to build sample DataFrames
# ---------------------------------------------------------------------------

def _make_merged_df(n=200, fraud_frac=0.2):
    """Return a minimal merged transaction+identity DataFrame."""
    rng = np.random.default_rng(42)
    n_fraud = max(1, int(n * fraud_frac))

    records = []
    for i in range(n):
        is_fraud = 1 if i < n_fraud else 0
        records.append({
            "TransactionID": 2_000_000 + i,
            "isFraud": is_fraud,
            "TransactionAmt": float(rng.uniform(1, 5000)),
            "card1": int(rng.integers(1000, 18000)),
            "addr1": float(rng.integers(100, 500)),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"]),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stage 1 — Data Validation checks (mirrors the validate() component)
# ---------------------------------------------------------------------------

class TestDataValidation:
    def test_required_columns_present(self):
        df = _make_merged_df()
        assert "isFraud" in df.columns, "Target column missing"
        assert "TransactionID" in df.columns, "TransactionID missing"

    def test_non_empty(self):
        df = _make_merged_df()
        assert len(df) > 0, "Dataset is empty"

    def test_schema_missing_target_raises(self):
        df = _make_merged_df().drop(columns=["isFraud"])
        with pytest.raises(AssertionError, match="isFraud"):
            assert "isFraud" in df.columns, "Target column 'isFraud' missing"

    def test_fraud_rate_is_reasonable(self):
        df = _make_merged_df(n=500, fraud_frac=0.2)
        fraud_rate = df["isFraud"].mean()
        assert 0.01 < fraud_rate < 0.99, (
            f"Fraud rate suspiciously extreme: {fraud_rate}"
        )

    def test_missing_value_pct_computable(self):
        df = _make_merged_df()
        df.loc[df.index[:10], "addr1"] = np.nan
        missing_pct = df.isnull().mean().mean() * 100
        assert isinstance(missing_pct, float)
        assert 0.0 <= missing_pct <= 100.0


# ---------------------------------------------------------------------------
# Stage 1 — Schema: missing-value threshold check
# ---------------------------------------------------------------------------

class TestMissingValueSchema:
    def test_columns_above_threshold_dropped(self):
        df = _make_merged_df(n=100)
        df["sparse_feature"] = np.nan
        threshold = 0.5
        before = df.shape[1]
        df_clean = df.loc[:, df.isnull().mean() < threshold]
        after = df_clean.shape[1]
        assert after < before, "High-missing column was not dropped"
        assert "sparse_feature" not in df_clean.columns

    def test_clean_df_unchanged(self):
        df = _make_merged_df(n=100)
        threshold = 0.5
        df_clean = df.loc[:, df.isnull().mean() < threshold]
        assert df_clean.shape[1] == df.shape[1]


# ---------------------------------------------------------------------------
# Stage 1 — Preprocessing logic (mirrors the preprocess() component)
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def _preprocess(self, df, missing_threshold=0.5):
        df = df.loc[:, df.isnull().mean() < missing_threshold].copy()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def test_no_nulls_after_imputation(self):
        df = _make_merged_df(n=300)
        df.loc[df.index[:20], "TransactionAmt"] = np.nan
        df.loc[df.index[:5], "P_emaildomain"] = np.nan
        df_proc = self._preprocess(df)
        assert df_proc.isnull().sum().sum() == 0, (
            "NaN values remain after imputation"
        )

    def test_shape_preserved(self):
        df = _make_merged_df(n=300)
        df_proc = self._preprocess(df)
        assert df_proc.shape[0] == 300, "Row count changed after preprocessing"


# ---------------------------------------------------------------------------
# Stage 1 — Feature engineering logic
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_log_transform_created(self):
        df = _make_merged_df(n=200)
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        assert "TransactionAmt_log" in df.columns
        assert (df["TransactionAmt_log"] >= 0).all()

    def test_train_test_split_stratified(self):
        from sklearn.model_selection import train_test_split
        df = _make_merged_df(n=500, fraud_frac=0.2)
        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        assert abs(y_train.mean() - y_test.mean()) < 0.05, (
            "Stratification failed: fraud rates differ significantly"
        )
        assert len(X_train) + len(X_test) == 500


# ---------------------------------------------------------------------------
# Stage 1 — Inference app import smoke test
# ---------------------------------------------------------------------------

class TestInferenceAPIImport:
    def test_app_module_locatable(self):
        """Verify docker/inference/app.py exists and is readable."""
        app_path = os.path.join(
            os.path.dirname(__file__), "..", "docker", "inference", "app.py"
        )
        app_path = os.path.normpath(app_path)
        assert os.path.isfile(app_path), (
            f"Inference app not found at: {app_path}"
        )

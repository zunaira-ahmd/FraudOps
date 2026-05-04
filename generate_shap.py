import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

print("Loading local data...")
# Read just 30k rows to make it fast
df = pd.read_csv('data/train_transaction.csv', nrows=30000)

X = df.drop(columns=['isFraud', 'TransactionID'], errors='ignore')
y = df['isFraud']

print("Preprocessing...")
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X.fillna(-999, inplace=True)

print("Training model...")
model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X, y)

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
sample = X.sample(500, random_state=42)
shap_values = explainer.shap_values(sample)

print("Generating plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, sample, show=False)
plt.title("SHAP Feature Importance (FraudOps)")
plt.tight_layout()

output_path = 'shap_summary_final.png'
plt.savefig(output_path, dpi=120)
print(f"\nDone! Plot saved to {os.path.abspath(output_path)}")

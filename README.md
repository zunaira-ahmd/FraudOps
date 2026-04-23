# FraudOps: MLOps Pipeline for Fraud Detection

## Overview
FraudOps is a production-grade MLOps system for fraud detection, built using the IEEE CIS Fraud Detection dataset. This project implements a complete end-to-end machine learning pipeline from infrastructure provisioning to model monitoring and explainability. It leverages modern MLOps tools to ensure scalability, reproducibility, and automation.

## Features & Architecture

The system is deployed on an AWS EC2 instance running K3s (lightweight Kubernetes) and Kubeflow Pipelines. It features:
- **Kubeflow Pipelines (KFP)** for orchestrating a 7-step ML workflow.
- **Automated CI/CD** using GitHub Actions / Jenkins for continuous integration and continuous deployment.
- **Monitoring & Alerting** using Prometheus and Grafana to track system health, model metrics, and data drift.
- **Cost-Sensitive Learning** to handle imbalanced datasets and minimize the business cost of false negatives (missed fraud).
- **Model Explainability** using SHAP to interpret transaction predictions.

## Tech Stack
- **Infrastructure:** AWS EC2, K3s (Kubernetes)
- **Orchestration:** Kubeflow Pipelines
- **Machine Learning:** Python, scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Data Processing:** Pandas, category-encoders, Great Expectations
- **Monitoring:** Prometheus, Grafana
- **Package Management:** uv

## Pipeline Steps

The automated Kubeflow pipeline consists of 7 sequential stages:
1. **Data Ingestion:** Loads the IEEE CIS dataset from persistent volumes.
2. **Data Validation:** Uses Great Expectations to ensure schema integrity and data quality.
3. **Data Preprocessing:** Handles missing values and drops sparse columns.
4. **Feature Engineering:** Applies target encoding to high-cardinality categorical features.
5. **Model Training:** Trains XGBoost/LightGBM models, utilizing SMOTE for class imbalance.
6. **Model Evaluation:** Calculates metrics including Accuracy, Precision, Recall, F1, and AUC-ROC.
7. **Conditional Deployment:** Automatically deploys the model only if it exceeds predefined performance thresholds (e.g., Accuracy >= 0.85).

## Setup Instructions

### 1. Infrastructure Setup
Provision an AWS EC2 instance (e.g., `t3.xlarge` or `g4dn.xlarge` for GPU support) running Ubuntu 22.04 LTS.

### 2. Install Dependencies
Connect to your instance and install system requirements:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl wget git python3-pip python3-venv unzip
```

### 3. Initialize K3s and Kubeflow
Install K3s:
```bash
curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=~/.kube/config
```

Deploy Kubeflow Pipelines:
```bash
export PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

### 4. Project Setup
Initialize the project using `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv venv
source .venv/bin/activate
uv add kfp==2.2.0 pandas scikit-learn xgboost lightgbm imbalanced-learn category-encoders great-expectations shap matplotlib seaborn
```

### 5. Running the Pipeline
Compile and submit the pipeline:
```bash
python3 pipeline/fraud_pipeline.py
```
Access the Kubeflow UI via port-forwarding:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 --address=0.0.0.0 &
```

## Project Tasks
1. **Kubeflow Setup:** Provisioning PVs, resource quotas, and namespaces.
2. **Data Challenges:** Advanced imputation and imbalance handling.
3. **Model Selection:** Training and evaluating tree-based and hybrid models.
4. **Cost-Sensitive Learning:** Minimizing the cost of false negatives.
5. **CI/CD Pipeline:** Automating linting, testing, and deployment.
6. **Monitoring:** Creating Grafana dashboards for drift and health.
7. **Drift Simulation:** Evaluating the model against temporal shifts.
8. **Retraining Strategy:** Implementing threshold-based model retraining.
9. **Explainability:** Utilizing SHAP for model transparency.

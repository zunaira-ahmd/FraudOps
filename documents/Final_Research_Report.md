# Final Report: MLOps System for Fraud Detection


## Executive Summary
This report outlines the development and deployment of an end-to-end Machine Learning Operations (MLOps) pipeline for real-time fraud detection. Using the IEEE CIS Fraud Detection dataset, a highly scalable system was engineered to identify fraudulent transactions while minimizing the severe financial costs of missed fraud. The system was designed with automated CI/CD, real-time observability, automated drift-aware retraining, and model explainability.

---

## 1. Infrastructure & Pipeline Architecture (Task 1)
To ensure the system scales under high transaction volume, it was deployed on an AWS EC2 instance running a lightweight Kubernetes cluster (**K3s**). 
**Kubeflow Pipelines (KFP)** was utilized to orchestrate a 7-step automated workflow:
1. Data Ingestion (fetching from MinIO Object Storage)
2. Data Validation
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Conditional Deployment

A **conditional deployment gate** was implemented in the final step: the pipeline only deploys the newly trained model if its fraud Recall metric exceeds a strict `0.75` threshold.

## 2. Data Challenges & Imbalance Handling (Task 2)
Fraud datasets are notoriously messy and imbalanced. The pipeline implements advanced handling strategies:
* **Missing Values:** Numerical columns use robust median imputation (resilient to severe outliers often seen in fraud amounts), while categorical columns use mode imputation.
* **High-Cardinality Features:** Variables with over 50 unique values (e.g., specific cards or addresses) were transformed using Frequency Encoding, preserving their statistical distributions without blowing up the feature space.
* **Class Imbalance Comparison:**
  To combat the severe lack of fraud examples, two strategies were compared:
  1. **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generates new fraud cases. *Result: Led to high training times and slight overfitting on synthetic patterns.*
  2. **Cost-Sensitive Class Weighting:** Adjusts the loss function of the algorithm to penalize missed fraud cases heavily. *Result: Proved highly efficient computationally and generalized better to test data.* The system ultimately defaulted to Class Weighting.

## 3. Model Complexity & Evaluation (Task 3)
Three distinct architectures were trained in parallel within the Kubeflow pipeline:
1. **XGBoost:** The primary gradient boosting model, highly optimized for tabular data.
2. **LightGBM:** A faster tree-based alternative utilizing leaf-wise growth.
3. **Hybrid Model (RF + Feature Selection):** A Random Forest pipeline that uses `SelectFromModel` to automatically filter out noisy features before final classification.

**Evaluation Metrics Computed:** Precision, Recall, F1-Score, AUC-ROC, and a comprehensive Confusion Matrix. AUC-ROC and Recall were prioritized heavily due to the business context.

## 4. Cost-Sensitive Learning & Business Impact (Task 4)
In fraud detection, a False Negative (missed fraud) costs the business the entire transaction amount. A False Positive (false alarm) only costs a minor review fee or slight customer friction.
To address this, we implemented strict **Cost-Sensitive Learning**:
* The XGBoost and LightGBM models were configured with a `scale_pos_weight` inversely proportional to the class imbalance (roughly 10:1 ratio).
* **Business Impact Analysis:** Standard training optimized for Accuracy, resulting in many missed frauds. Cost-sensitive training sacrificed minor Precision to vastly increase Recall. By catching more true fraud at the expense of a few false alarms, the estimated net financial benefit improved significantly compared to standard training.

## 5. CI/CD Pipeline with Intelligent Triggers (Task 5)
A production-grade CI/CD pipeline was implemented using **GitHub Actions**:
* **Stage 1 (CI):** Triggers on push. Runs `flake8` linting and YAML structure validation.
* **Stage 2 (Build):** Compiles Docker images for the KFP training components and the FastAPI inference server, then pushes them to a container registry.
* **Stage 3 (CD):** Connects to the EC2 server and automatically submits a new Kubeflow pipeline run.
* **Stage 4 (Intelligent Trigger):** Contains a specific webhook listener job. When the production API monitoring detects severe data drift or dropped recall, this pipeline is automatically triggered to rebuild and retrain the model.

## 6. Observability & Monitoring System (Task 6)
To detect and respond to performance degradation, a full monitoring stack was deployed:
* **Prometheus:** Scrapes `/metrics` endpoints every 15 seconds.
* **Grafana Dashboards:** 
  1. *System Health:* Tracks API latency, Request Rates, CPU/Memory.
  2. *Model Performance:* Visualizes Recall, AUC-ROC, F1, and FPR over time.
  3. *Data Drift:* Tracks statistical shifts in features like `TransactionAmt`.
* **Alerting Rules:** Prometheus was configured to fire Critical alerts if Recall drops below `0.75` or if Data Drift exceeds `0.10`. These alerts route to Alertmanager.

## 7 & 8. Drift Simulation & Intelligent Retraining (Tasks 7 & 8)
A secondary pipeline (`drift_simulation.py`) was created to simulate real-world degradation. 
* **The Simulation:** It splits data temporally (early vs. late), injects realistic shifts into the late data (e.g., inflating `TransactionAmt` to mimic new premium-fraud targets, and degrading data quality by removing `addr1` values), and evaluates the early-trained model on this drifted data.
* **Retraining Strategy:** We designed and deployed a **Threshold-Based Retraining Strategy**. 
  * *Comparison:* Periodic retraining wastes massive compute if data hasn't drifted. Our Threshold strategy is cost-efficient and highly responsive: Alertmanager listens for the `FeatureDriftHigh` alert from Grafana, hits a custom local Webhook (`localhost:5001/alert`), which fires the GitHub Actions CI/CD to immediately trigger a new Kubeflow training pipeline.

## 9. Model Explainability (Task 9)
Black-box models are unacceptable in finance. We integrated **SHAP (SHapley Additive exPlanations)** directly into the evaluation step of the pipeline.
* After XGBoost finishes training, a `TreeExplainer` calculates the marginal contribution of every feature.
* A summary plot is automatically saved locally as `shap_summary_final.png`. 
* **Findings:** The SHAP analysis clearly answers *why* a model flags fraud, revealing which specific transactional limits, categorical mismatches, or IP attributes drove the model's confidence score.

## Conclusion
The FraudOps system successfully fulfills all requirements of a modern MLOps architecture. It moves beyond static Jupyter notebooks by providing a fully containerized, reproducible, and self-healing ML pipeline that automatically reacts to the dynamic nature of financial fraud.

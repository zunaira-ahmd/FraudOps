FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install inference dependencies directly (no uv; avoids Python-version mismatch
# between pyproject.toml requires-python>=3.12 and this 3.11 image).
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic \
        joblib \
        pandas \
        numpy \
        scikit-learn \
        xgboost \
        lightgbm \
        prometheus-client \
        boto3

COPY docker/inference/app.py .

ENV ARTIFACTS_DIR=/home/ubuntu/FraudOps/artifacts
ENV MODEL_PREFIX=xgb_

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
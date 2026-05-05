FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install training dependencies directly (no uv; avoids Python-version mismatch
# between pyproject.toml requires-python>=3.12 and this 3.11 image).
RUN pip install --no-cache-dir \
        pandas \
        numpy \
        scikit-learn \
        xgboost \
        lightgbm \
        imbalanced-learn \
        joblib \
        boto3 \
        kfp==2.15.0

COPY pipeline/ ./pipeline/

CMD ["python3"]
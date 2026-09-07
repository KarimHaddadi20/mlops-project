# Dockerfile for Churn Classifier
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ configs/
COPY src/ src/
COPY tests/ tests/

# Default command: run training
CMD ["python", "src/train.py", "--config", "configs/config.yaml"]

# Alternative commands:
# docker run <image> python src/evaluate.py --config configs/config.yaml
# docker run <image> python -m pytest tests/ -v

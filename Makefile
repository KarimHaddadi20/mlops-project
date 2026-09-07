# Makefile for Churn Classifier MLOps Project

PY = python
VENV = .venv
EXP ?= churn-classifier
CONFIG = configs/config.yaml

# Detect OS for activation script
ifeq ($(OS),Windows_NT)
	ACTIVATE = $(VENV)\Scripts\activate
	PIP = $(VENV)\Scripts\pip
	PYTHON = $(VENV)\Scripts\python
else
	ACTIVATE = $(VENV)/bin/activate
	PIP = $(VENV)/bin/pip
	PYTHON = $(VENV)/bin/python
endif

.PHONY: init install train evaluate test lint mlflow clean help

help:
	@echo "Available commands:"
	@echo "  make init       - Create virtual environment and install dependencies"
	@echo "  make install    - Install dependencies only"
	@echo "  make train      - Train the model with MLflow tracking"
	@echo "  make evaluate   - Evaluate the trained model and generate artifacts"
	@echo "  make test       - Run tests with pytest"
	@echo "  make lint       - Run ruff linter"
	@echo "  make mlflow     - Start MLflow UI"
	@echo "  make clean      - Remove generated files"
	@echo ""
	@echo "Environment variables:"
	@echo "  EXP             - MLflow experiment name (default: churn-classifier)"

init:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: $(VENV)\\Scripts\\activate"
	@echo "  Linux/Mac: source $(VENV)/bin/activate"

install:
	$(PIP) install -r requirements.txt

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) src/train.py --config $(CONFIG)

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PYTHON) src/evaluate.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

mlflow:
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf mlruns/
	rm -rf mlartifacts/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete

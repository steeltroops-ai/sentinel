# KIVE Project Makefile
# Usage: make <target>
# Example: make test, make train, make docker-up

.PHONY: help install test test-cov lint format clean data train train-fast docker-up docker-down docker-logs services health demo submit

# Default target
help:
	@echo "KIVE Project - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install all dependencies with uv"
	@echo "  make install-dev      Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-fast        Run tests in parallel (fast)"
	@echo "  make test-integration Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with black"
	@echo "  make check            Run lint + format check"
	@echo ""
	@echo "Data Generation:"
	@echo "  make data             Generate synthetic profiles (5000)"
	@echo "  make data-small       Generate small dataset (500)"
	@echo "  make data-large       Generate large dataset (10000)"
	@echo "  make validate-data    Validate signal distributions"
	@echo ""
	@echo "Training:"
	@echo "  make train            Train RL agent (3000 episodes, no tracking)"
	@echo "  make train-fast       Quick training (1000 episodes, no tracking)"
	@echo "  make train-mlflow     Train with MLflow (5000 episodes)"
	@echo "  make train-full       Full training with MLflow (10000 episodes)"
	@echo ""
	@echo "Docker & Services:"
	@echo "  make docker-up        Start all services with Docker Compose"
	@echo "  make docker-down      Stop all services"
	@echo "  make docker-build     Rebuild Docker images"
	@echo "  make docker-logs      View service logs"
	@echo "  make services         Start services locally (no Docker)"
	@echo "  make health           Check health of all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean artifacts and cache"
	@echo "  make clean-all        Deep clean (including data)"
	@echo "  make demo             Run interactive demo"
	@echo "  make notebook         Start Jupyter notebook server"
	@echo ""


# ============================================================================
# Setup & Installation
# ============================================================================

install:
	uv sync

install-dev:
	uv sync --all-extras

# ============================================================================
# Testing
# ============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov --cov-report=html --cov-report=term

test-fast:
	uv run pytest tests/ -v -n auto

test-integration:
	uv run pytest tests/test_integration.py -v

test-env:
	uv run pytest tests/test_env.py -v

# ============================================================================
# Code Quality
# ============================================================================

lint:
	uv run ruff check .

format:
	uv run black .

check: lint
	uv run black --check .

# ============================================================================
# Data Generation
# ============================================================================

data:
	uv run python data/synthetic_generator.py --n 5000 --fraud-ratio 0.4 --output data/synthetic_profiles.json

data-small:
	uv run python data/synthetic_generator.py --n 500 --fraud-ratio 0.4 --output data/synthetic_profiles.json

data-large:
	uv run python data/synthetic_generator.py --n 10000 --fraud-ratio 0.4 --output data/synthetic_profiles.json

data-verbose:
	uv run python data/synthetic_generator.py --n 5000 --fraud-ratio 0.4 --verbose --output data/synthetic_profiles.json

validate-data:
	uv run python data/validate_distribution.py

export-distributions:
	uv run python data/export_signal_distributions.py

# ============================================================================
# Training
# ============================================================================

train:
	uv run python services/orchestrator/train.py --n-episodes 3000 --run-name kive_ppo_v3 --output-dir artifacts/training --no-mlflow

train-fast:
	uv run python services/orchestrator/train.py --n-episodes 1000 --run-name kive_ppo_fast --output-dir artifacts/training --no-mlflow

train-full:
	uv run python services/orchestrator/train.py --n-episodes 10000 --run-name kive_ppo_full --output-dir artifacts/training

train-mlflow:
	uv run python services/orchestrator/train.py --n-episodes 5000 --run-name kive_ppo_mlflow --output-dir artifacts/training

# ============================================================================
# Docker & Services
# ============================================================================

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

services:
	@echo "Starting services locally (not implemented - use docker-up)"

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health || echo "TAV service down"
	@curl -s http://localhost:8002/health || echo "SVP service down"
	@curl -s http://localhost:8003/health || echo "FMD service down"
	@curl -s http://localhost:8004/health || echo "MDC service down"
	@curl -s http://localhost:8005/health || echo "TSI service down"
	@curl -s http://localhost:8006/health || echo "BES service down"
	@curl -s http://localhost:8007/health || echo "LQA service down"
	@curl -s http://localhost:8008/health || echo "CCS service down"
	@curl -s http://localhost:8009/health || echo "RSL service down"

# ============================================================================
# Utilities
# ============================================================================

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf artifacts/training/*
	rm -rf mlruns/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf data/synthetic_profiles.json
	rm -rf .venv

demo:
	@echo "Interactive demo not implemented yet"

notebook:
	uv run jupyter notebook

# ============================================================================
# Submission
# ============================================================================

submit-check:
	@echo "Checking submission readiness..."
	@echo ""
	@echo "1. Checking tests..."
	@uv run pytest tests/ -v --tb=no -q
	@echo ""
	@echo "2. Checking memo..."
	@test -f memo.md && echo "✓ memo.md exists" || echo "✗ memo.md missing"
	@echo ""
	@echo "3. Checking training artifacts..."
	@test -f artifacts/training/convergence_report.json && echo "✓ Training complete" || echo "✗ Need to run training"
	@echo ""
	@echo "4. Checking multi-modal doc..."
	@test -f docs/multimodal_live_evaluator.md && echo "✓ Multi-modal doc exists" || echo "✗ Multi-modal doc missing"
	@echo ""
	@echo "Submission checklist complete!"

submit-package:
	@echo "Packaging for submission..."
	@echo "1. Converting memo to PDF..."
	@echo "2. Creating submission archive..."
	@echo "Not implemented - manual submission required"

.PHONY: install dev-install test lint run-ui train help

install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pip install ruff pytest

test:
	pytest tests/ -v

lint:
	ruff check src/ app/ scripts/ tests/

format:
	ruff format src/ app/ scripts/ tests/

run-ui:
	python -m app.main

train:
	python scripts/train.py --data data/examples/midjourney_prompts.csv --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 1

docker-build:
	docker build -t lora-finetune-studio .

docker-run:
	docker-compose -f docker/docker-compose.yml up

help:
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  run-ui       Launch Gradio UI"
	@echo "  train        Run example training"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run with Docker Compose"

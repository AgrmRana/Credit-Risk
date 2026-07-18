.PHONY: install train api test lint format frontend-install frontend-build

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	. .venv/bin/activate && PYTHONPATH=backend python -m credit_risk_platform.training.train --dataset $${DATASET:-german}

api:
	. .venv/bin/activate && PYTHONPATH=backend uvicorn credit_risk_platform.api.main:app --reload

test:
	. .venv/bin/activate && PYTHONPATH=backend pytest

lint:
	. .venv/bin/activate && ruff check backend tests

format:
	. .venv/bin/activate && black backend tests

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build

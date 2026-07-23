.PHONY: install run train test lint format

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && streamlit run app.py

train:
	. .venv/bin/activate && python -m credit_risk_platform.training.train --dataset $${DATASET:-german}

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && ruff check credit_risk_platform tests scripts app.py

format:
	. .venv/bin/activate && black credit_risk_platform tests scripts app.py

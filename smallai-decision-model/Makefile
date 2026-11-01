.PHONY: setup data train eval benchmark test all clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

data:
	.venv/bin/python -m smallai generate --config configs/default.yaml

train:
	.venv/bin/python -m smallai train --config configs/default.yaml

eval:
	.venv/bin/python -m smallai evaluate --config configs/default.yaml

benchmark:
	.venv/bin/python -m smallai benchmark --config configs/default.yaml

test:
	.venv/bin/python -m pytest tests -q

all: train eval benchmark

clean:
	rm -rf data runs reports .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +

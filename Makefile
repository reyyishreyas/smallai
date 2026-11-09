.PHONY: setup test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

test:
	PYTHONPATH=src .venv/bin/python -m pytest tests -q

clean:
	rm -rf data runs .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +

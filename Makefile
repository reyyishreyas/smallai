.PHONY: help setup install test lint format check build clean

PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff

help:
	@echo "SmallAI — make targets"
	@echo ""
	@echo "  setup    create .venv and install the package with dev extras"
	@echo "  install  install the package (editable) into the existing venv"
	@echo "  test     run the test suite"
	@echo "  lint     run ruff"
	@echo "  format   format the code with ruff"
	@echo "  check    lint + tests (what CI runs)"
	@echo "  build    build sdist + wheel"
	@echo "  clean    remove caches and generated artifacts"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"
	@echo ""
	@echo "Ready. Try: $(PY) -m smallai quality --image <file>"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(RUFF) check src tests

format:
	$(RUFF) format src tests

check: lint test

build:
	$(PY) -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache .benchmarks htmlcov
	find . -name "__pycache__" -type d -not -path "./.venv/*" -exec rm -rf {} +
	find . -name "*.egg-info" -type d -not -path "./.venv/*" -exec rm -rf {} +

PYTHON ?= python3

.PHONY: install install-dev test lint format format-check profile validate train ci pre-commit-install

install:
	$(PYTHON) -m pip install -e . --no-deps

install-dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check .

format:
	black .

format-check:
	black --check .

profile:
	$(PYTHON) scripts/profile_solver.py --runs 5 --solver-type alternating

validate:
	$(PYTHON) scripts/run_validation.py --out validation.json

train:
	$(PYTHON) scripts/train_dummy.py --epochs 1 --batches 4 --max-iter 10

ci: test format-check lint

pre-commit-install:
	pre-commit install


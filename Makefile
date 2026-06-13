.PHONY: install install-dashboard test demo api preflight audit dataset-smoke api-smoke validate reproducibility monte-carlo monte-carlo-quick build docker-build clean clean-all

install:
	python -m pip install -e ".[dev]"

install-dashboard:
	python -m pip install -e ".[dev,dashboard]"

test:
	python -m pytest -q

demo:
	python scripts/run_pipeline.py --config configs/pipeline.yaml --mode demo

api:
	python scripts/serve_api.py --host 0.0.0.0 --port 8000 --reload

preflight:
	python scripts/preflight_check.py

audit:
	python scripts/local_audit.py --mode demo

dataset-smoke:
	python scripts/dataset_smoke_test.py --quick

api-smoke:
	python scripts/api_smoke_test.py

validate:
	python -m pytest -q
	python scripts/preflight_check.py
	python scripts/local_audit.py --mode demo
	python scripts/dataset_smoke_test.py --quick
	python scripts/api_smoke_test.py
	python scripts/reproducibility_check.py
	python scripts/monte_carlo_validate.py --trials-per-scenario 1

reproducibility:
	python scripts/reproducibility_check.py

monte-carlo:
	python scripts/monte_carlo_validate.py

monte-carlo-quick:
	python scripts/monte_carlo_validate.py --trials-per-scenario 1

build:
	python -m build

docker-build:
	docker build -t media-search-reliability:local .

clean:
	rm -rf data/processed data/synthetic .pytest_cache build dist *.egg-info src/*.egg-info .wheel-smoke artifacts/monte_carlo_work

clean-all: clean
	rm -rf artifacts reports
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

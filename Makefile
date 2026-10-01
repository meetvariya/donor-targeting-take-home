PY := uv run python

.PHONY: all setup data features train baseline policy prepare serve benchmark test clean

all: setup data features train baseline policy prepare  ## everything, from a fresh clone

setup:  ## install dependencies into .venv
	uv sync

data:  ## download the warehouse snapshot + upcoming requests (130 MB)
	$(PY) -m donor_targeting.download_data

features:  ## the DS team's feature sample -> data/warehouse/features.parquet
	$(PY) -m donor_targeting.features

train:  ## the existing model -> models/
	$(PY) -m donor_targeting.train

baseline:  ## BAU vs centroid precision/recall -> reports/baseline_report.md
	$(PY) -m donor_targeting.baseline

policy:  ## temporal targeting evaluation -> models/audience_policy.json
	$(PY) -m donor_targeting.policy --reports evidence

prepare:  ## compact active-audience serving artifacts
	$(PY) -m donor_targeting.serving_features

serve:  ## one local service with enforced 2 GB / 4 CPU limits
	docker compose up --build

benchmark:  ## actual HTTP replay, burst admission, and a separate 250k capacity case
	docker compose exec -T api /app/.venv/bin/python -m donor_targeting.benchmark --capacity

test:
	uv run pytest -q

clean:
	rm -rf data models reports

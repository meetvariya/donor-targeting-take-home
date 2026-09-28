PY := uv run python

.PHONY: all setup data features train baseline test clean

all: setup data features train baseline  ## everything, from a fresh clone

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

test:
	uv run pytest -q

clean:
	rm -rf data models reports

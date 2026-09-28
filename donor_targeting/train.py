"""Train the donation-propensity model (the data science team's existing work).

One feature, `centroid_cosine_similarity` (NULL when the person engaged with nothing in the 30
days before the send), one label: a donation attributed to that email within 7 days. Trained
on the sampled sends in features.parquet; validated out-of-time on the last month of sends.

    python -m donor_targeting.train     # -> models/donation_model.joblib, models/model_card.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from donor_targeting.config import MODELS_DIR, SNAPSHOT_AT
from donor_targeting.ml_harness import run
from donor_targeting.warehouse_io import modelling_frame

FEATURES = ["centroid_cosine_similarity"]
LABEL = "donated"
HOLDOUT_FROM = pd.Timestamp("2026-08-01", tz="UTC")


def main() -> None:
    df = modelling_frame()
    holdout = df["sent_at"] >= HOLDOUT_FROM
    result = run(df[FEATURES], df[[LABEL]], test_mask=holdout)
    print(f"Out-of-time holdout (sends from {HOLDOUT_FROM:%Y-%m-%d}):\n{result.summary()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, MODELS_DIR / "donation_model.joblib")
    card = {
        "model": "donation_model.joblib",
        "estimator": str(result.model),
        "features": FEATURES,
        "label": f"{LABEL}: attributed donation within 7 days of the send",
        "training_rows": f"sampled sends before {HOLDOUT_FROM:%Y-%m-%d}",
        "holdout_rows": f"sampled sends from {HOLDOUT_FROM:%Y-%m-%d} to the snapshot",
        "warehouse_snapshot": SNAPSHOT_AT.isoformat(),
        "holdout_metrics": {k: round(v, 4) for k, v in result.metrics.items()},
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"wrote {MODELS_DIR / 'donation_model.joblib'} and model_card.json")


if __name__ == "__main__":
    main()

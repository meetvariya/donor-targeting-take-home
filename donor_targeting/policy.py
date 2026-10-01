"""Evaluate a July-selected model policy and fixed safety references on August."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from donor_targeting.baseline import policy_metrics
from donor_targeting.config import BAU_SEND_FRACTION, MODELS_DIR, REPORTS_DIR, WAREHOUSE_DIR
from donor_targeting.ml_harness import default_model
from donor_targeting.serving_features import FEATURE_VERSION, file_sha256
from donor_targeting.warehouse_io import modelling_frame

FEATURES = ["centroid_cosine_similarity"]
POLICY_PATH = MODELS_DIR / "audience_policy.json"


def selected_positions(person_ids: np.ndarray, scores: np.ndarray, fraction: float) -> np.ndarray:
    """Select floor(n * fraction) people; equal scores resolve by ascending person id."""
    if not 0 < fraction <= 1:
        raise ValueError("audience fraction must be in (0, 1]")
    if len(person_ids) != len(scores):
        raise ValueError("scores and person ids must have equal length")
    key = np.where(np.isnan(scores), -np.inf, scores)
    order = np.lexsort((person_ids, -key))
    return order[: int(len(person_ids) * fraction)]


def replay_mask(frame: pd.DataFrame, scores: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros(len(frame))
    person_ids = frame["person_id"].to_numpy()
    for positions in frame.groupby("content_id", sort=False).indices.values():
        selected = selected_positions(person_ids[positions], scores[positions], fraction)
        mask[positions[selected]] = 1
    return mask


def summary(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    metrics = policy_metrics(frame, mask)
    return {
        "emailed_fraction": float(metrics["emailed"]),
        "donation_precision": float(metrics["precision"]),
        "donors_kept_vs_bau": float(metrics["recall"] / BAU_SEND_FRACTION),
        "dollars_kept_vs_bau": float(metrics["$ recall"] / BAU_SEND_FRACTION),
        "unsubscribes_avoided_vs_bau": float(1 - metrics["unsub share"] / BAU_SEND_FRACTION),
    }


def bootstrap_intervals(frame: pd.DataFrame, mask: np.ndarray, repetitions: int = 1000) -> dict:
    grouped = frame[["content_id", "donated", "amount", "unsubscribed"]].copy()
    grouped["selected_donors"] = mask * grouped["donated"]
    grouped["selected_dollars"] = mask * grouped["amount"]
    grouped["selected_unsubscribes"] = mask * grouped["unsubscribed"]
    totals = grouped.groupby("content_id").sum().to_numpy()
    rng = np.random.default_rng(42)
    samples = totals[rng.integers(len(totals), size=(repetitions, len(totals)))].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.column_stack(
            (
                samples[:, 3] / (samples[:, 0] * BAU_SEND_FRACTION),
                samples[:, 4] / (samples[:, 1] * BAU_SEND_FRACTION),
                1 - samples[:, 5] / (samples[:, 2] * BAU_SEND_FRACTION),
            )
        )
    return {
        name: np.nanquantile(ratios[:, position], [0.025, 0.975]).tolist()
        for position, name in enumerate(
            ("donors_kept_vs_bau", "dollars_kept_vs_bau", "unsubscribes_avoided_vs_bau")
        )
    }


def evaluate_policy(
    warehouse: Path = WAREHOUSE_DIR,
    models: Path = MODELS_DIR,
    reports: Path = REPORTS_DIR,
) -> dict:
    frame = modelling_frame(warehouse)
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    august_start = pd.Timestamp("2026-08-01", tz="UTC")
    training = frame[frame["sent_at"] < july_start]
    development = frame[(frame["sent_at"] >= july_start) & (frame["sent_at"] < august_start)]
    final = frame[frame["sent_at"] >= august_start]
    development_model = default_model().fit(training[FEATURES], training["donated"])
    development_scores = development_model.predict_proba(development[FEATURES])[:, 1]
    candidates = []
    for fraction in np.round(np.arange(0.5, 0.8001, 0.025), 3):
        mask = replay_mask(development, development_scores, float(fraction))
        candidates.append({"fraction": float(fraction), **summary(development, mask)})
    feasible = [
        candidate
        for candidate in candidates
        if candidate["donors_kept_vs_bau"] >= 1
        and candidate["dollars_kept_vs_bau"] >= 1
        and candidate["unsubscribes_avoided_vs_bau"] > 0
    ]
    if feasible:
        chosen = max(feasible, key=lambda candidate: candidate["unsubscribes_avoided_vs_bau"])
        ranking, fraction = "model", chosen["fraction"]
    else:
        ranking, fraction = "bau", BAU_SEND_FRACTION
        chosen = summary(development, np.full(len(development), BAU_SEND_FRACTION))
    model_path = Path(models) / "donation_model.joblib"
    model = joblib.load(model_path)
    policy = {
        "ranking": ranking,
        "fraction": fraction,
        "feature_version": FEATURE_VERSION,
        "model_sha256": file_sha256(model_path),
        "development_period": "2026-07-01 <= sent_at < 2026-08-01",
        "guardrail": "July point estimates retain >=100% BAU donors and dollars",
        "fallback": "approved BAU if no development candidate passes; never unknown eligibility",
    }
    candidate_policy = dict(policy)
    scores = model.predict_proba(final[FEATURES])[:, 1]
    mask = (
        replay_mask(final, scores, fraction)
        if ranking == "model"
        else np.full(len(final), fraction)
    )
    candidate_final = {
        "metrics": summary(final, mask),
        "email_bootstrap_95_percent": bootstrap_intervals(final, mask),
    }
    references = {
        "BAU": summary(final, np.full(len(final), BAU_SEND_FRACTION)),
        "engaged_last_30_days": summary(final, final[FEATURES[0]].notna().to_numpy(dtype=float)),
        "centroid_top_75": summary(final, replay_mask(final, final[FEATURES[0]].to_numpy(), 0.75)),
        "centroid_top_50": summary(final, replay_mask(final, final[FEATURES[0]].to_numpy(), 0.5)),
        "model_top_75": summary(final, replay_mask(final, scores, 0.75)),
    }
    metrics = candidate_final["metrics"]
    if (
        metrics["donors_kept_vs_bau"] < 1
        or metrics["dollars_kept_vs_bau"] < 1
        or metrics["unsubscribes_avoided_vs_bau"] <= 0
    ):
        reference = references["centroid_top_75"]
        if (
            reference["donors_kept_vs_bau"] >= 1
            and reference["dollars_kept_vs_bau"] >= 1
            and reference["unsubscribes_avoided_vs_bau"] > 0
        ):
            policy.update(
                ranking="centroid",
                fraction=0.75,
                release_decision="July winner failed August gate; use original fixed centroid 75% reference",
            )
            mask = replay_mask(final, final[FEATURES[0]].to_numpy(), 0.75)
        else:
            policy.update(
                ranking="bau",
                fraction=BAU_SEND_FRACTION,
                release_decision="No tested targeting policy passes; retain approved BAU",
            )
            mask = np.full(len(final), BAU_SEND_FRACTION)
    else:
        policy["release_decision"] = "July winner passes August point-estimate release gate"
    policy["version"] = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
    Path(models).mkdir(parents=True, exist_ok=True)
    (Path(models) / "audience_policy.json").write_text(json.dumps(policy, indent=2))
    content = pd.read_parquet(Path(warehouse) / "content.parquet", columns=["id", "program_area"])
    areas = final["content_id"].map(content.set_index("id")["program_area"])
    area_results = {}
    for area in areas.unique():
        positions = np.flatnonzero(areas.to_numpy() == area)
        subset = final.iloc[positions]
        area_results[str(area)] = {
            "n_sends": len(subset),
            "n_donors": int(subset["donated"].sum()),
            **summary(subset, mask[positions]),
        }
    result = {
        "policy": policy,
        "historical_all_data_diagnostic": summary(
            frame,
            replay_mask(frame, frame[FEATURES[0]].to_numpy(), 0.75)
            if policy["ranking"] == "centroid"
            else replay_mask(frame, model.predict_proba(frame[FEATURES])[:, 1], policy["fraction"])
            if policy["ranking"] == "model"
            else np.full(len(frame), BAU_SEND_FRACTION),
        ),
        "development": {"n_sends": len(development), "candidates": candidates, "chosen": chosen},
        "frozen_july_candidate": {"policy": candidate_policy, "august": candidate_final},
        "final_august": {
            "n_sends": len(final),
            "n_emails": int(final["content_id"].nunique()),
            "n_donors": int(final["donated"].sum()),
            "metrics": summary(final, mask),
            "email_bootstrap_95_percent": bootstrap_intervals(final, mask),
            "references": references,
            "program_areas": area_results,
        },
        "limitations": [
            "Point-estimate guardrails do not establish revenue non-inferiority.",
            "Only email-attributed gifts in the 7-day window are counted.",
            "Historical replay estimates single-email outcomes, not long-term causal effects.",
            "Program-area slices and email bootstrap have limited support in the small August sample.",
            "The safety reference is adopted after the August release gate; its results are diagnostic, not independent confirmation.",
        ],
    }
    Path(reports).mkdir(parents=True, exist_ok=True)
    (Path(reports) / "policy_evaluation.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "policy": policy,
                "august": result["final_august"]["metrics"],
                "intervals": result["final_august"]["email_bootstrap_95_percent"],
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE_DIR)
    parser.add_argument("--models", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    evaluate_policy(args.warehouse, args.models, args.reports)


if __name__ == "__main__":
    main()

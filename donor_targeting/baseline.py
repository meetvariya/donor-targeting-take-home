"""Precision and recall of the business-as-usual email rule vs targeting with the centroid.

BAU sends every email to a random 75% of the active list. Because history was sent to a random
75%, the sends we observed are an unbiased sample of the whole list, so any targeting rule can
be replayed on them. For each email we keep a share of its recipients and measure:

* precision   - share of emailed people who donated (attributed within 7 days)
* recall      - share of donors (who would have given had everyone been emailed) still reached
* $ recall    - the same, weighted by donation amount
* unsub share - share of unsubscribes still caused (lower is better: this is the churn)

The centroid rule ranks each email's recipients by `centroid_cosine_similarity` (people with
no engagement in the last 30 days last) and keeps the top k%.

    python -m donor_targeting.baseline    # prints the table, writes reports/baseline_report.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from donor_targeting.config import BAU_SEND_FRACTION, REPORTS_DIR
from donor_targeting.warehouse_io import modelling_frame


def top_share(df: pd.DataFrame, score: pd.Series, frac: float) -> np.ndarray:
    """1 for the top `frac` of each email's recipients by score (NULL scores rank last)."""
    tiebreak = np.random.default_rng(0).random(len(df)) * 1e-9
    key = score.fillna(-2.0) + tiebreak
    pct = key.groupby(df["content_id"]).rank(ascending=False, pct=True)
    return (pct <= frac).to_numpy(dtype=float)


def policy_metrics(df: pd.DataFrame, emailed: np.ndarray) -> dict[str, float]:
    """`emailed` is each row's probability of being emailed under the policy."""
    donated, amount, unsub = (df[c].to_numpy() for c in ("donated", "amount", "unsubscribed"))
    return {
        "emailed": emailed.mean(),
        "precision": (emailed * donated).sum() / emailed.sum(),
        "recall": (emailed * donated).sum() / donated.sum(),
        "$ recall": (emailed * amount).sum() / amount.sum(),
        "unsub share": (emailed * unsub).sum() / unsub.sum(),
    }


def report(df: pd.DataFrame) -> pd.DataFrame:
    cos = df["centroid_cosine_similarity"]
    ones = np.ones(len(df))
    policies = {
        "send to everyone": ones,
        f"BAU: random {BAU_SEND_FRACTION:.0%}": ones * BAU_SEND_FRACTION,
        "engaged in last 30 days": cos.notna().to_numpy(dtype=float),
        "centroid top 75%": top_share(df, cos, 0.75),
        "centroid top 50%": top_share(df, cos, 0.50),
        "centroid top 25%": top_share(df, cos, 0.25),
    }
    return pd.DataFrame({name: policy_metrics(df, e) for name, e in policies.items()}).T


def _fmt(table: pd.DataFrame) -> pd.DataFrame:
    out = table.map(lambda v: f"{v:.1%}")
    out["precision"] = table["precision"].map(lambda v: f"{v:.2%}")
    return out


def _markdown(table: pd.DataFrame) -> str:
    rows = [["policy", *table.columns], ["---"] + ["---:"] * len(table.columns)]
    rows += [[name, *vals] for name, vals in zip(table.index, table.to_numpy())]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main() -> None:
    df = modelling_frame()
    table = report(df)
    first, last = df["sent_at"].min(), df["sent_at"].max()
    context = (
        f"{len(df):,} sampled sends from {first:%Y-%m-%d} to {last:%Y-%m-%d} "
        f"({df['content_id'].nunique()} emails): {df['donated'].sum():,} donations, "
        f"{df['unsubscribed'].sum():,} unsubscribes."
    )
    print(context)
    print(_fmt(table).to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    body = _markdown(_fmt(table))
    (REPORTS_DIR / "baseline_report.md").write_text(
        f"# Baseline: BAU vs centroid targeting\n\n{context}\n\n{body}\n\n"
        "Precision = donation rate of the people emailed. Recall = share of all donors still "
        "reached. Unsub share = share of all unsubscribes still caused.\n"
    )
    print(f"wrote {REPORTS_DIR / 'baseline_report.md'}")


if __name__ == "__main__":
    main()

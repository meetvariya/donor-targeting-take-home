"""The data science team's feature pipeline (the "existing work").

For a historical send (a `timeline` row with type = 'sent') we compute:

* ``centroid_1m`` – the mean `dense_embedding` of the distinct content the person engaged
  with (opened or clicked) in the 30 days *before* the send. NULL if there was none.
* ``centroid_cosine_similarity`` – cosine similarity between that centroid and the embedding
  of the content being sent. NULL when the centroid is NULL.

This is written for offline training on a *sample* of sends.

    python -m donor_targeting.features             # writes data/warehouse/features.parquet
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from donor_targeting.config import (
    ATTRIBUTION_WINDOW_DAYS,
    CENTROID_LOOKBACK_DAYS,
    EMBEDDING_DIM,
    HISTORY_START,
    SNAPSHOT_AT,
    WAREHOUSE_DIR,
)
from donor_targeting.warehouse_io import load_content, load_timeline

FEATURE_SAMPLE_SIZE = 60_000
SAMPLE_SEED = 11


def engagements(timeline: pd.DataFrame) -> pd.DataFrame:
    """First engagement (open or click) per person x content."""
    e = timeline.loc[
        timeline["type"].isin(["opened", "clicked"]), ["person_id", "content_id", "occurred_at"]
    ]
    return (
        e.groupby(["person_id", "content_id"], as_index=False)["occurred_at"]
        .min()
        .rename(columns={"occurred_at": "engaged_at"})
    )


def centroid_weights(
    sends: pd.DataFrame, engaged: pd.DataFrame, content_ids: np.ndarray
) -> sparse.csr_matrix:
    """Sparse (n_sends x n_content) matrix whose rows average the content engaged with in the
    lookback window before each send."""
    lookback = pd.Timedelta(days=CENTROID_LOOKBACK_DAYS)
    s = sends[["person_id", "sent_at"]].reset_index(drop=True)
    s["row"] = np.arange(len(s))
    pairs = s.merge(engaged, on="person_id")
    pairs = pairs[
        (pairs["engaged_at"] < pairs["sent_at"])
        & (pairs["engaged_at"] >= pairs["sent_at"] - lookback)
    ]
    col = pd.Index(content_ids).get_indexer(pairs["content_id"])
    counts = np.bincount(pairs["row"], minlength=len(s))
    weights = 1.0 / counts[pairs["row"].to_numpy()]
    return sparse.csr_matrix(
        (weights, (pairs["row"].to_numpy(), col)), shape=(len(s), len(content_ids))
    )


def compute_features(
    sends: pd.DataFrame, engaged: pd.DataFrame, content: pd.DataFrame
) -> pd.DataFrame:
    """`sends` needs timeline_id, person_id, content_id, sent_at."""
    content_ids = content["id"].to_numpy()
    emb = np.stack(content["dense_embedding"].to_numpy()).astype(np.float32)
    w = centroid_weights(sends, engaged, content_ids)
    centroids = np.asarray(w @ emb, dtype=np.float32)
    has = np.asarray(w.sum(axis=1)).ravel() > 0
    sent_emb = emb[pd.Index(content_ids).get_indexer(sends["content_id"])]
    norms = np.linalg.norm(centroids, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.einsum("ij,ij->i", centroids, sent_emb) / norms
    sim[~has] = np.nan
    return pd.DataFrame(
        {
            "timeline_id": sends["timeline_id"].to_numpy(),
            "centroid_1m": list(np.where(has[:, None], centroids, np.nan)),
            "centroid_cosine_similarity": sim.astype(np.float32),
        }
    )


def eligible_sends(timeline: pd.DataFrame) -> pd.DataFrame:
    """Sends with a full 30-day lookback and a fully observed 7-day attribution window."""
    first = pd.Timestamp(HISTORY_START, tz="UTC") + pd.Timedelta(days=CENTROID_LOOKBACK_DAYS)
    last = pd.Timestamp(SNAPSHOT_AT) - pd.Timedelta(days=ATTRIBUTION_WINDOW_DAYS)
    s = timeline.loc[timeline["type"] == "sent", ["id", "person_id", "content_id", "occurred_at"]]
    s = s[(s["occurred_at"] >= first) & (s["occurred_at"] < last)]
    return s.rename(columns={"id": "timeline_id", "occurred_at": "sent_at"}).reset_index(drop=True)


def write_features(df: pd.DataFrame, path) -> None:
    has = df["centroid_cosine_similarity"].notna().to_numpy()
    # A variable-length list: parquet readers cannot read back NULLs in a fixed-size list column.
    flat = np.stack(df["centroid_1m"].to_numpy()[has]).astype(np.float32).ravel()
    offsets = np.r_[0, np.cumsum(np.where(has, EMBEDDING_DIM, 0))].astype(np.int32)
    centroid = pa.ListArray.from_arrays(pa.array(offsets), pa.array(flat), mask=pa.array(~has))
    table = pa.table(
        {
            "timeline_id": pa.array(df["timeline_id"].to_numpy(dtype=np.int64)),
            "centroid_1m": centroid,
            "centroid_cosine_similarity": pa.array(
                df["centroid_cosine_similarity"].to_numpy(), mask=~has
            ),
        }
    )
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--warehouse", default=str(WAREHOUSE_DIR))
    parser.add_argument("--sample", type=int, default=FEATURE_SAMPLE_SIZE)
    args = parser.parse_args()
    t0 = time.time()
    timeline = load_timeline(args.warehouse)
    content = load_content(args.warehouse)
    sends = eligible_sends(timeline)
    sends = sends.sample(n=min(args.sample, len(sends)), random_state=SAMPLE_SEED).sort_values(
        "timeline_id"
    )
    feats = compute_features(sends, engagements(timeline), content)
    write_features(feats, f"{args.warehouse}/features.parquet")
    print(
        f"features.parquet: {len(feats):,} sends "
        f"({feats['centroid_cosine_similarity'].notna().mean():.0%} with a centroid) in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

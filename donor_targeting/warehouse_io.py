"""Small helpers for reading the warehouse parquet files into pandas (offline use)."""

from __future__ import annotations

import pandas as pd

from donor_targeting.config import ATTRIBUTION_WINDOW_DAYS, WAREHOUSE_DIR


def _path(warehouse, name: str) -> str:
    return f"{warehouse}/{name}.parquet"


def load_timeline(warehouse=WAREHOUSE_DIR, columns=None) -> pd.DataFrame:
    df = pd.read_parquet(_path(warehouse, "timeline"), columns=columns)
    if "type" in df:
        df["type"] = df["type"].astype(str)
    return df


def load_content(warehouse=WAREHOUSE_DIR) -> pd.DataFrame:
    return pd.read_parquet(_path(warehouse, "content"))


def load_transactions(warehouse=WAREHOUSE_DIR) -> pd.DataFrame:
    return pd.read_parquet(_path(warehouse, "transactions"))


def load_people(warehouse=WAREHOUSE_DIR) -> pd.DataFrame:
    return pd.read_parquet(_path(warehouse, "people"))


def load_features(warehouse=WAREHOUSE_DIR, with_centroid: bool = False) -> pd.DataFrame:
    cols = None if with_centroid else ["timeline_id", "centroid_cosine_similarity"]
    return pd.read_parquet(_path(warehouse, "features"), columns=cols)


def attributed_amount(sends: pd.DataFrame, transactions: pd.DataFrame) -> pd.Series:
    """Donation amount attributed to each send: a transaction by the same person for the same
    content_id within the attribution window. `sends` needs person_id, content_id, sent_at."""
    tx = transactions.dropna(subset=["content_id"]).astype({"content_id": "int64"})
    m = (
        sends[["person_id", "content_id", "sent_at"]]
        .reset_index()
        .merge(
            tx[["person_id", "content_id", "transaction_date", "amount"]],
            on=["person_id", "content_id"],
        )
    )
    window = pd.Timedelta(days=ATTRIBUTION_WINDOW_DAYS)
    m = m[(m["transaction_date"] >= m["sent_at"]) & (m["transaction_date"] < m["sent_at"] + window)]
    amount = m.groupby("index")["amount"].sum()
    return amount.reindex(sends.index, fill_value=0.0).rename("amount")


def donation_labels(sends: pd.DataFrame, transactions: pd.DataFrame) -> pd.Series:
    """1 if the send has an attributed donation (see `attributed_amount`)."""
    return (attributed_amount(sends, transactions) > 0).astype(int).rename("donated")


def unsubscribe_labels(sends: pd.DataFrame, unsubscribes: pd.DataFrame) -> pd.Series:
    """1 if the person unsubscribed from the sent content (`unsubscribes`: person_id,
    content_id of the 'unsubscribed' timeline events)."""
    keys = pd.MultiIndex.from_frame(unsubscribes[["person_id", "content_id"]])
    hit = pd.MultiIndex.from_frame(sends[["person_id", "content_id"]]).isin(keys)
    return pd.Series(hit.astype(int), index=sends.index, name="unsubscribed")


def modelling_frame(warehouse=WAREHOUSE_DIR) -> pd.DataFrame:
    """The modelling table: one row per sampled historical send in features.parquet with its
    centroid feature and outcomes (amount, donated, unsubscribed)."""
    feats = load_features(warehouse)
    sends = pd.read_parquet(
        _path(warehouse, "timeline"),
        columns=["id", "person_id", "content_id", "occurred_at"],
        filters=[("id", "in", feats["timeline_id"].tolist())],
    ).rename(columns={"id": "timeline_id", "occurred_at": "sent_at"})
    df = feats.merge(sends, on="timeline_id").sort_values("timeline_id", ignore_index=True)
    unsubscribes = pd.read_parquet(
        _path(warehouse, "timeline"),
        columns=["person_id", "content_id"],
        filters=[("type", "=", "unsubscribed")],
    )
    df["amount"] = attributed_amount(df, load_transactions(warehouse))
    df["donated"] = (df["amount"] > 0).astype(int)
    df["unsubscribed"] = unsubscribe_labels(df, unsubscribes)
    return df

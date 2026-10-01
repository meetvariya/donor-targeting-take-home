import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from donor_targeting.config import SNAPSHOT_AT, WAREHOUSE_DIR


@pytest.fixture(scope="session")
def small_warehouse(tmp_path_factory):
    """The timeline of 3,000 random people from the downloaded warehouse, plus all content."""
    if not (WAREHOUSE_DIR / "timeline.parquet").exists():
        pytest.fail("no data/warehouse/: run `make data` first", pytrace=False)
    out = tmp_path_factory.mktemp("warehouse")
    ids = pq.read_table(WAREHOUSE_DIR / "people.parquet", columns=["id"])["id"].to_numpy()
    ids = np.random.default_rng(0).choice(ids, 3_000, replace=False).tolist()
    timeline = pq.read_table(WAREHOUSE_DIR / "timeline.parquet", filters=[("person_id", "in", ids)])
    pq.write_table(timeline, out / "timeline.parquet")
    shutil.copy(WAREHOUSE_DIR / "content.parquet", out)
    return out


@pytest.fixture
def serving_warehouse(tmp_path):
    out = tmp_path / "warehouse"
    out.mkdir()
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(4, 1024)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    pd.DataFrame({"id": [1001, 1002, 1003, 1004], "dense_embedding": list(embeddings)}).to_parquet(
        out / "content.parquet", index=False
    )
    pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6],
         "status": ["active", "active", "unsubscribed", "bounced", "active", "active"]}
    ).to_parquet(out / "people.parquet", index=False)
    offsets = [1, 0.5, 30, 1, 1, 31, 1]
    pd.DataFrame(
        {"id": list(range(1, 8)), "person_id": [1, 1, 2, 3, 4, 6, 6],
         "content_id": [1001, 1001, 1002, 1003, 1003, 1002, 1002],
         "type": ["opened", "clicked", "opened", "opened", "opened", "opened", "clicked"],
         "occurred_at": [pd.Timestamp(SNAPSHOT_AT) - pd.Timedelta(days=offset) for offset in offsets]}
    ).to_parquet(out / "timeline.parquet", index=False)
    return out

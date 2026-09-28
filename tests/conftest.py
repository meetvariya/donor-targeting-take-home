import shutil

import numpy as np
import pyarrow.parquet as pq
import pytest

from donor_targeting.config import WAREHOUSE_DIR


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

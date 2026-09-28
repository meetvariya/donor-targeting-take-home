"""The data warehouse, mocked as DuckDB views over data/warehouse/*.parquet.

    from donor_targeting.mocks.warehouse import connect

    con = connect()
    con.sql("select type, count(*) from timeline group by all").show()

The real warehouse is refreshed nightly at 01:00 America/New_York. These files are the refresh
of SNAPSHOT_AT (2026-09-01 01:00 ET): nothing that happened after it is in them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from donor_targeting.config import SNAPSHOT_AT, WAREHOUSE_DIR

TABLES = ("timeline", "transactions", "content", "people", "features")


def connect(warehouse: Path = WAREHOUSE_DIR, **config) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with one view per warehouse table. `config` is passed to DuckDB as
    connection settings."""
    con = duckdb.connect(config=config)
    for table in TABLES:
        path = Path(warehouse) / f"{table}.parquet"
        if path.exists():
            con.execute(f"create view {table} as select * from read_parquet('{path}')")
    con.execute(f"create macro snapshot_at() as timestamptz '{SNAPSHOT_AT.isoformat()}'")
    return con

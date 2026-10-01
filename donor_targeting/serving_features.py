"""Memory-bounded feature preparation and exact centroid scoring for serving."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import sparse

from donor_targeting.config import (
    CENTROID_LOOKBACK_DAYS,
    DATA_DIR,
    EMBEDDING_DIM,
    SNAPSHOT_AT,
    WAREHOUSE_DIR,
)
from donor_targeting.mocks.warehouse import connect

SERVING_DIR = DATA_DIR / "serving"
FEATURE_VERSION = 1
ARRAY_NAMES = ("person_ids", "content_ids", "embeddings", "indptr", "indices", "engaged_ns")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(warehouse: Path) -> dict:
    return {
        name: {
            "size": (warehouse / f"{name}.parquet").stat().st_size,
            "mtime_ns": (warehouse / f"{name}.parquet").stat().st_mtime_ns,
        }
        for name in ("people", "content", "timeline")
    }


@dataclass(frozen=True)
class ServingSnapshot:
    generation: str
    snapshot_at: datetime
    person_ids: np.ndarray
    content_ids: np.ndarray
    embeddings: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    engaged_ns: np.ndarray
    metadata: dict

    @property
    def array_bytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in ARRAY_NAMES)

    def score(self, query: np.ndarray, as_of: datetime) -> np.ndarray:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("feature time must be timezone-aware")
        if as_of < self.snapshot_at:
            raise ValueError("request predates the eligibility snapshot")
        if as_of - self.snapshot_at >= timedelta(hours=24):
            raise ValueError("warehouse snapshot is stale for this request")
        upper_ns = int(as_of.timestamp() * 1_000_000_000)
        lower_ns = upper_ns - CENTROID_LOOKBACK_DAYS * 86_400 * 1_000_000_000
        valid = (self.engaged_ns >= lower_ns) & (self.engaged_ns < upper_ns)
        weights = sparse.csr_matrix(
            (valid.astype(np.float32), self.indices, self.indptr),
            shape=(len(self.person_ids), len(self.content_ids)),
        )
        return compact_cosine(weights, self.embeddings, query)


def load_snapshot(directory: Path = SERVING_DIR) -> ServingSnapshot:
    directory = Path(directory)
    generation = json.loads((directory / "current.json").read_text())["generation"]
    if len(generation) != 32 or any(
        character not in "0123456789abcdef" for character in generation
    ):
        raise ValueError("invalid serving generation")
    root = directory / generation
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata["feature_version"] != FEATURE_VERSION or metadata["embedding_dim"] != EMBEDDING_DIM:
        raise ValueError("incompatible serving feature version or embedding dimension")
    arrays = {}
    for name in ARRAY_NAMES:
        path = root / f"{name}.npy"
        if file_sha256(path) != metadata["sha256"][name]:
            raise ValueError(f"corrupt serving array: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    person_ids, content_ids = arrays["person_ids"], arrays["content_ids"]
    indptr, indices, engaged_ns = arrays["indptr"], arrays["indices"], arrays["engaged_ns"]
    if (
        len(person_ids) == 0
        or np.any(np.diff(person_ids) <= 0)
        or len(np.unique(content_ids)) != len(content_ids)
        or arrays["embeddings"].shape != (len(content_ids), EMBEDDING_DIM)
        or not np.isfinite(arrays["embeddings"]).all()
        or indptr.shape != (len(person_ids) + 1,)
        or indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(np.diff(indptr) < 0)
        or len(indices) != len(engaged_ns)
        or np.any((indices < 0) | (indices >= len(content_ids)))
    ):
        raise ValueError("invalid serving array structure")
    return ServingSnapshot(
        generation=generation,
        snapshot_at=datetime.fromisoformat(metadata["snapshot_at"]),
        metadata=metadata,
        **arrays,
    )


def prepare_snapshot(
    warehouse: Path = WAREHOUSE_DIR,
    directory: Path = SERVING_DIR,
    *,
    snapshot_at: datetime = SNAPSHOT_AT,
    force: bool = False,
) -> ServingSnapshot:
    """Publish bounded serving artifacts after a completed warehouse refresh."""
    warehouse, directory = Path(warehouse), Path(directory)
    fingerprint = source_fingerprint(warehouse)
    if not force and (directory / "current.json").exists():
        current = load_snapshot(directory)
        if current.metadata["source"] == fingerprint and current.snapshot_at == snapshot_at:
            return current
    directory.mkdir(parents=True, exist_ok=True)
    with connect(
        warehouse, memory_limit="512MB", threads=2, temp_directory=str(directory / "spill")
    ) as connection:
        connection.execute(
            "CREATE TEMP VIEW active_people AS SELECT id FROM people WHERE status = 'active'"
        )
        person_ids = connection.sql("SELECT id FROM active_people ORDER BY id").fetchnumpy()["id"]
        content = connection.sql("SELECT id, dense_embedding FROM content ORDER BY id").fetchdf()
        engaged = connection.execute(
            """
            SELECT timeline.person_id, timeline.content_id, MIN(occurred_at) AS engaged_at
            FROM timeline INNER JOIN active_people ON timeline.person_id = active_people.id
            WHERE type IN ('opened', 'clicked') AND occurred_at < ?
            GROUP BY timeline.person_id, timeline.content_id
            HAVING MIN(occurred_at) >= ? - INTERVAL '30 days'
            ORDER BY timeline.person_id, timeline.content_id
            """,
            [snapshot_at, snapshot_at],
        ).fetchdf()
    import pandas as pd

    content_ids = content["id"].to_numpy(dtype=np.int64)
    rows = np.searchsorted(person_ids, engaged["person_id"].to_numpy())
    indices = pd.Index(content_ids).get_indexer(engaged["content_id"]).astype(np.int32)
    if np.any(indices < 0):
        raise ValueError("engagement references unknown content")
    indptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(person_ids)))].astype(np.int32)
    arrays = {
        "person_ids": np.asarray(person_ids, dtype=np.int64),
        "content_ids": content_ids,
        "embeddings": np.stack(content["dense_embedding"]).astype(np.float32),
        "indptr": indptr,
        "indices": indices,
        "engaged_ns": pd.DatetimeIndex(engaged["engaged_at"]).as_unit("ns").asi8,
    }
    if (
        len(person_ids) == 0
        or np.any(np.diff(person_ids) <= 0)
        or len(np.unique(content_ids)) != len(content_ids)
        or arrays["embeddings"].shape != (len(content_ids), EMBEDDING_DIM)
        or not np.isfinite(arrays["embeddings"]).all()
    ):
        raise ValueError("invalid active IDs or historical embedding dimensions/values")
    generation = uuid.uuid4().hex
    root = directory / generation
    root.mkdir()
    try:
        for name, array in arrays.items():
            np.save(root / f"{name}.npy", array, allow_pickle=False)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "embedding_dim": EMBEDDING_DIM,
            "snapshot_at": snapshot_at.isoformat(),
            "source": fingerprint,
            "active_count": len(person_ids),
            "engagement_count": len(indices),
            "array_bytes": sum(array.nbytes for array in arrays.values()),
            "sha256": {name: file_sha256(root / f"{name}.npy") for name in ARRAY_NAMES},
        }
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2))
        if source_fingerprint(warehouse) != fingerprint:
            raise ValueError("warehouse changed during preparation; retry after refresh completes")
        pointer = directory / f"{generation}.json.tmp"
        pointer.write_text(json.dumps({"generation": generation}))
        os.replace(pointer, directory / "current.json")
    except Exception:
        shutil.rmtree(root)
        raise
    return load_snapshot(directory)


def compact_cosine(
    weights: sparse.csr_matrix,
    embeddings: np.ndarray,
    query: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute centroid cosine using the small content Gram matrix, not dense centroids."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if embeddings.ndim != 2 or weights.shape[1] != len(embeddings):
        raise ValueError("weights and embeddings must have matching content columns")
    if query.shape != (embeddings.shape[1],) or not np.isfinite(query).all():
        raise ValueError("query must be a finite vector with the embedding dimension")
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or not np.isfinite(query_norm):
        raise ValueError("query must have a finite nonzero norm")
    if not np.isfinite(embeddings).all():
        raise ValueError("historical embeddings must be finite")
    gram = embeddings @ embeddings.T
    numerator = np.asarray(weights @ (embeddings @ (query / query_norm))).ravel()
    similarities = np.full(weights.shape[0], np.nan, dtype=np.float32)
    for start in range(0, weights.shape[0], batch_size):
        stop = min(start + batch_size, weights.shape[0])
        batch = weights[start:stop]
        squared_norm = np.asarray(batch.multiply(batch @ gram).sum(axis=1)).ravel()
        norms = np.sqrt(np.maximum(squared_norm, 0))
        valid = norms > 1e-12
        values = np.full(stop - start, np.nan)
        values[valid] = numerator[start:stop][valid] / norms[valid]
        similarities[start:stop] = np.clip(values, -1, 1)
    return similarities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE_DIR)
    parser.add_argument("--output", type=Path, default=SERVING_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    snapshot = prepare_snapshot(args.warehouse, args.output, force=args.force)
    print(json.dumps({"generation": snapshot.generation, **snapshot.metadata}, indent=2))


if __name__ == "__main__":
    main()

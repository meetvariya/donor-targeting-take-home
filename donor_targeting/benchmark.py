"""Measure actual HTTP-to-CRM delivery and whole-container resource usage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from donor_targeting.config import (
    CRM_OUTBOX_DIR,
    DATA_DIR,
    MODELS_DIR,
    REPO_ROOT,
    REQUESTS_DIR,
    WAREHOUSE_DIR,
)
from donor_targeting.serving_features import (
    SERVING_DIR,
    file_sha256,
    load_snapshot,
    prepare_snapshot,
)


def http_json(url: str, token: str, payload: dict | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def complete_request(base_url: str, token: str, payload: dict) -> dict:
    started = time.perf_counter()
    status, job = http_json(f"{base_url}/audiences", token, payload)
    if status != 202:
        return {"request_id": payload["request_id"], "http_status": status, "response": job}
    deadline = time.monotonic() + 180
    while job["status"] not in ("succeeded", "failed", "delivery_unknown"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"benchmark deadline exceeded: {payload['request_id']}")
        threading.Event().wait(0.025)
        _, job = http_json(f"{base_url}/requests/{payload['request_id']}", token)
    return {
        "request_id": payload["request_id"],
        "http_status": status,
        "client_elapsed_seconds": time.perf_counter() - started,
        "job": job,
    }


def validate_output(record: dict, outbox: Path, eligible: np.ndarray, content_id: int) -> dict:
    if record.get("http_status") != 202 or record["job"]["status"] != "succeeded":
        raise AssertionError(f"delivery did not complete: {record}")
    result = record["job"]["result"]
    if not result["sla_met"] or record["client_elapsed_seconds"] >= 180:
        raise AssertionError("receipt-to-CRM deadline was exceeded")
    root = Path(outbox) / result["list_id"]
    members = np.loadtxt(
        root.with_suffix(".csv"), delimiter=",", skiprows=1, dtype=np.int64, ndmin=1
    )
    metadata = json.loads(root.with_suffix(".json").read_text())
    unique = len(np.unique(members)) == len(members)
    active_only = bool(np.isin(members, eligible).all())
    if not unique or not active_only:
        raise AssertionError("CRM output has duplicate or ineligible IDs")
    if (
        len(members) != result["member_count"]
        or len(members) != metadata["member_count"]
        or metadata["content_id"] != content_id
    ):
        raise AssertionError("CRM output count or content metadata is inconsistent")
    return {"member_count": len(members), "unique": unique, "active_only": active_only}


def resource_evidence() -> dict:
    root = Path("/sys/fs/cgroup")

    def text(name):
        try:
            return (root / name).read_text().strip()
        except OSError:
            return None

    peak, maximum = text("memory.peak"), text("memory.max")
    process_peaks = {}
    proc = Path("/proc")
    if proc.exists():
        for status in proc.glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1) for line in status.read_text().splitlines() if ":" in line
                )
                if "VmHWM" in fields:
                    process_peaks[status.parent.name] = int(fields["VmHWM"].split()[0]) * 1024
            except (OSError, ValueError):
                continue
    return {
        "memory_peak_bytes": int(peak) if peak and peak.isdigit() else None,
        "memory_limit_bytes": int(maximum) if maximum and maximum.isdigit() else None,
        "cpu_max": text("cpu.max"),
        "memory_events": text("memory.events"),
        "process_peak_rss_bytes": process_peaks,
        "note": "cgroup peak includes API, benchmark, fixture preparation, file cache, and capacity child",
    }


def capacity_fixture(root: Path, count: int = 250_000) -> tuple[Path, Path, Path]:
    """Copy realistic engagement patterns onto unique synthetic active IDs, isolated from evaluation."""
    source = load_snapshot(SERVING_DIR)
    warehouse, models, serving = root / "warehouse", root / "models", root / "serving"
    warehouse.mkdir(parents=True)
    models.mkdir()
    ids = np.arange(1, count + 1, dtype=np.int64)
    source_rows = np.arange(count) % len(source.person_ids)
    counts = np.diff(source.indptr)[source_rows]
    rows = np.repeat(ids, counts)
    edge_count = int(counts.sum())
    edge_positions = np.empty(edge_count, dtype=np.int32)
    original_count = min(count, len(source.person_ids))
    original_edges = int(source.indptr[original_count])
    edge_positions[:original_edges] = np.arange(original_edges)
    cursor = original_edges
    for row in source_rows[original_count:]:
        positions = np.arange(source.indptr[row], source.indptr[row + 1])
        edge_positions[cursor : cursor + len(positions)] = positions
        cursor += len(positions)
    pq.write_table(
        pa.table({"id": ids, "status": pa.repeat(pa.scalar("active"), count)}),
        warehouse / "people.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": np.arange(1, edge_count + 1, dtype=np.int64),
                "person_id": rows,
                "content_id": source.content_ids[source.indices[edge_positions]],
                "occurred_at": pa.array(
                    source.engaged_ns[edge_positions], type=pa.timestamp("ns", tz="UTC")
                ),
                "type": pa.repeat(pa.scalar("opened"), edge_count),
            }
        ),
        warehouse / "timeline.parquet",
    )
    shutil.copy2(WAREHOUSE_DIR / "content.parquet", warehouse / "content.parquet")
    shutil.copy2(MODELS_DIR / "donation_model.joblib", models / "donation_model.joblib")
    policy = json.loads((MODELS_DIR / "audience_policy.json").read_text())
    policy.update(
        fraction=1.0, ranking="centroid", version="capacity-import-only-not-business-policy"
    )
    (models / "audience_policy.json").write_text(json.dumps(policy))
    prepare_snapshot(warehouse, serving, snapshot_at=source.snapshot_at)
    return warehouse, models, serving


def run_capacity(root: Path, token: str, payload: dict) -> dict:
    preparation_started = time.perf_counter()
    warehouse, models, serving = capacity_fixture(root)
    preparation_seconds = time.perf_counter() - preparation_started
    env = {
        **os.environ,
        "DONOR_API_TOKEN": token,
        "PORT": "8001",
        "DONOR_CLOCK_MODE": "demo",
        "DONOR_WAREHOUSE": str(warehouse),
        "DONOR_MODELS": str(models),
        "DONOR_SERVING": str(serving),
        "DONOR_STATE": str(root / "state"),
        "DONOR_OUTBOX": str(root / "outbox"),
    }
    startup_started = time.perf_counter()
    with (root / "server.log").open("wb") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "donor_targeting.service"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while time.perf_counter() - startup_started < 60:
                if child.poll() is not None:
                    raise RuntimeError(f"capacity service failed; inspect {root / 'server.log'}")
                try:
                    status, _ = http_json("http://127.0.0.1:8001/readyz", token)
                    if status == 200:
                        break
                except urllib.error.URLError:
                    pass
                threading.Event().wait(0.05)
            else:
                raise TimeoutError("capacity service did not become ready")
            startup_seconds = time.perf_counter() - startup_started
            record = complete_request("http://127.0.0.1:8001", token, payload)
            validation = validate_output(
                record, root / "outbox", np.arange(1, 250001), payload["content"]["content_id"]
            )
            if validation["member_count"] != 250_000:
                raise AssertionError(
                    "capacity test did not import exactly 250,000 unique active people"
                )
            return {
                "fixture_preparation_seconds": preparation_seconds,
                "startup_seconds": startup_seconds,
                "record": record,
                "validation": validation,
                "resources": resource_evidence(),
                "note": "Isolated fixture and fraction=1 test maximum CRM payload, not the live targeting policy",
            }
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=Path, default=REQUESTS_DIR)
    parser.add_argument("--outbox", type=Path, default=CRM_OUTBOX_DIR)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "evidence" / "benchmark.json")
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("DONOR_API_TOKEN", "local-demo-token-change-before-deploy")
    run_id = uuid.uuid4().hex[:10]
    root = DATA_DIR / "benchmark" / run_id
    root.mkdir(parents=True)
    snapshot = load_snapshot(SERVING_DIR)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.requests.glob("*.json"))
    ]
    if len(payloads) != 12:
        raise AssertionError("expected the 12 provided requests")
    status, readiness = http_json(f"{args.base_url}/readyz", token)
    if status != 200:
        raise RuntimeError("service is not ready")
    report = {
        "run_id": run_id,
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "duckdb", "fastapi")
        },
        "code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("service.py", "serving_features.py", "policy.py", "benchmark.py")
        },
        "readiness": readiness,
        "resources_before": resource_evidence(),
        "requests": [],
    }
    for payload in payloads:
        request = {**payload, "request_id": f"{payload['request_id']}_bench_{run_id}"}
        record = complete_request(args.base_url, token, request)
        record["validation"] = validate_output(
            record, args.outbox, snapshot.person_ids, payload["content"]["content_id"]
        )
        report["requests"].append(record)
    burst_payloads = [
        {**payloads[position % 12], "request_id": f"burst_{run_id}_{position}"}
        for position in range(12)
    ]
    with ThreadPoolExecutor(max_workers=12) as pool:
        burst = list(
            pool.map(
                lambda payload: complete_request(args.base_url, token, payload), burst_payloads
            )
        )
    for payload, record in zip(burst_payloads, burst):
        if record["http_status"] == 202:
            record["validation"] = validate_output(
                record, args.outbox, snapshot.person_ids, payload["content"]["content_id"]
            )
        elif record["http_status"] != 429:
            raise AssertionError("burst returned an unexpected admission result")
    report["burst"] = {
        "submitted": 12,
        "accepted": sum(record["http_status"] == 202 for record in burst),
        "capacity_rejected": sum(record["http_status"] == 429 for record in burst),
        "records": burst,
    }
    cold_started = time.perf_counter()
    cold = prepare_snapshot(WAREHOUSE_DIR, root / "cold_serving", force=True)
    report["cold_real_preparation"] = {
        "seconds": time.perf_counter() - cold_started,
        "active_count": len(cold.person_ids),
        "array_bytes": cold.array_bytes,
    }
    if args.capacity:
        request = {**payloads[2], "request_id": f"capacity_{run_id}"}
        report["capacity_250k"] = run_capacity(root / "capacity", token, request)
    report["resources_after"] = resource_evidence()
    peak = report["resources_after"]["memory_peak_bytes"]
    limit = report["resources_after"]["memory_limit_bytes"]
    if peak is not None and limit is not None and peak >= limit:
        raise AssertionError("container peak reached the enforced memory limit")
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    times = [record["job"]["result"]["elapsed_seconds"] for record in report["requests"]]
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(args.output),
                "real_requests": len(times),
                "max_real_elapsed_seconds": max(times),
                "burst_accepted": report["burst"]["accepted"],
                "burst_rejected": report["burst"]["capacity_rejected"],
                "memory_peak_bytes": peak,
                "memory_limit_bytes": limit,
                "capacity_elapsed_seconds": report.get("capacity_250k", {})
                .get("record", {})
                .get("job", {})
                .get("result", {})
                .get("elapsed_seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

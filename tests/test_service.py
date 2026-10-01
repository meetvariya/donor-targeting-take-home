import copy
import json
import os
import threading
import time
from dataclasses import replace
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from donor_targeting.benchmark import validate_output
from donor_targeting.config import SNAPSHOT_AT
from donor_targeting.ml_harness import default_model
from donor_targeting.mocks.crm import MockCRM
from donor_targeting.service import AudienceRequest, JobStore, Runtime, Settings, create_app
from donor_targeting.serving_features import file_sha256, load_snapshot, prepare_snapshot

TOKEN = "test-token-not-a-production-secret"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def settings(serving_warehouse, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    features = pd.DataFrame({"centroid_cosine_similarity": [np.nan, -0.5, 0, 0.5, 1] * 20})
    labels = np.array([0, 0, 0, 1, 1] * 20)
    model = default_model().fit(features, labels)
    joblib.dump(model, models / "donation_model.joblib")
    (models / "audience_policy.json").write_text(
        json.dumps(
            {
                "ranking": "centroid",
                "fraction": 0.75,
                "feature_version": 1,
                "model_sha256": file_sha256(models / "donation_model.joblib"),
                "version": "test-policy",
            }
        )
    )
    return Settings(
        api_token=TOKEN,
        warehouse=serving_warehouse,
        models=models,
        serving=tmp_path / "serving",
        state=tmp_path / "state",
        outbox=tmp_path / "outbox",
    )


@pytest.fixture
def payload():
    vector = np.zeros(1024)
    vector[0] = 1
    return {
        "request_id": "test_2003",
        "org_id": "demo-charity",
        "requested_at": (SNAPSHOT_AT + timedelta(hours=11)).isoformat(),
        "content": {
            "content_id": 2003,
            "subject": "Clean water appeal",
            "program_area": "clean_water",
            "dense_embedding": vector.tolist(),
        },
        "audience": {"type": "all_active_subscribers"},
        "destination": {"crm": "mock", "list_name": "clean_water appeal (2003)"},
    }


def wait_for_result(client, request_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/requests/{request_id}", headers=HEADERS)
        job = response.json()
        if job["status"] in ("succeeded", "failed", "delivery_unknown"):
            return job
        threading.Event().wait(0.01)
    pytest.fail("job did not finish")


def test_api_delivers_once_and_persists_result(settings, payload):
    crm = MockCRM(settings.outbox, latency_s=0)
    with TestClient(create_app(settings, crm=crm)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz", headers=HEADERS).json()["active_count"] == 4
        accepted = client.post("/audiences", json=payload, headers=HEADERS)
        assert accepted.status_code == 202
        result = wait_for_result(client, payload["request_id"])
        assert result["status"] == "succeeded"
        assert result["result"]["member_count"] == 3
        assert result["result"]["sla_met"]
        members = pd.read_csv(settings.outbox / f"{result['result']['list_id']}.csv")["person_id"]
        assert members.is_unique
        assert set(members) <= {1, 2, 5, 6}
        duplicate = client.post("/audiences", json=payload, headers=HEADERS)
        assert duplicate.json()["result"]["list_id"] == result["result"]["list_id"]
        conflict = copy.deepcopy(payload)
        conflict["content"]["subject"] = "Different appeal"
        assert client.post("/audiences", json=conflict, headers=HEADERS).status_code == 409
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        assert (
            client.get(f"/requests/{payload['request_id']}", headers=HEADERS).json()["status"]
            == "succeeded"
        )
    assert len(list(settings.outbox.glob("*.csv"))) == 1


@pytest.mark.parametrize("change", ["dimension", "zero", "org", "program", "naive_time", "stale"])
def test_invalid_requests_are_rejected(settings, payload, change):
    if change == "dimension":
        payload["content"]["dense_embedding"] = [1.0]
    elif change == "zero":
        payload["content"]["dense_embedding"] = [0.0] * 1024
    elif change == "org":
        payload["org_id"] = "other-charity"
    elif change == "program":
        payload["content"]["program_area"] = "unknown"
    elif change == "naive_time":
        payload["requested_at"] = "2026-09-01T16:00:00"
    elif change == "stale":
        payload["requested_at"] = (SNAPSHOT_AT + timedelta(days=1)).isoformat()
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        response = client.post("/audiences", json=payload, headers=HEADERS)
        assert response.status_code == (503 if change == "stale" else 422)
    assert not list(settings.outbox.glob("*.csv"))


def test_api_requires_auth_and_bounds_body(settings, payload):
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        assert client.post("/audiences", json=payload).status_code == 401
        assert client.get("/requests/test_2003").status_code == 401
        assert client.post("/audiences", content=b"x" * 65537, headers=HEADERS).status_code == 413


def test_restart_does_not_blindly_retry_uncertain_delivery(settings, payload):
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        client.post("/audiences", json=payload, headers=HEADERS)
        wait_for_result(client, payload["request_id"])
    store = JobStore(settings.state)
    store.update(payload["request_id"], "delivering")
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        job = client.get(f"/requests/{payload['request_id']}", headers=HEADERS).json()
        assert job["status"] == "delivery_unknown"
    assert len(list(settings.outbox.glob("*.csv"))) == 1


def test_crm_exception_is_unknown_not_retried(settings, payload):
    class FailingCRM(MockCRM):
        def create_list(self, name, person_ids, content_id=None):
            raise TimeoutError("acknowledgement unavailable")

    with TestClient(create_app(settings, crm=FailingCRM(settings.outbox, latency_s=0))) as client:
        client.post("/audiences", json=payload, headers=HEADERS)
        job = wait_for_result(client, payload["request_id"])
        assert job["status"] == "delivery_unknown"


def test_capacity_is_rejected_before_acceptance(settings, payload):
    entered, release = threading.Event(), threading.Event()

    class BlockingCRM(MockCRM):
        def create_list(self, name, person_ids, content_id=None):
            entered.set()
            release.wait(3)
            return super().create_list(name, person_ids, content_id)

    with TestClient(
        create_app(replace(settings, max_pending=1), crm=BlockingCRM(settings.outbox, latency_s=0))
    ) as client:
        try:
            assert client.post("/audiences", json=payload, headers=HEADERS).status_code == 202
            assert entered.wait(2)
            payload["request_id"] = "second_request"
            assert client.post("/audiences", json=payload, headers=HEADERS).status_code == 429
            assert client.get("/requests/second_request", headers=HEADERS).status_code == 404
        finally:
            release.set()


def test_changed_warehouse_fails_closed(settings, payload):
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        source = settings.warehouse / "people.parquet"
        os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1_000_000))
        assert client.post("/audiences", json=payload, headers=HEADERS).status_code == 503
        assert client.get("/readyz", headers=HEADERS).status_code == 503


def test_corrupt_serving_generation_is_rejected(settings):
    snapshot = prepare_snapshot(settings.warehouse, settings.serving)
    source = settings.serving / snapshot.generation / "indices.npy"
    with source.open("r+b") as artifact:
        artifact.seek(-1, 2)
        artifact.write(b"\xff")
    with pytest.raises(ValueError, match="corrupt"):
        load_snapshot(settings.serving)


def test_expired_request_is_failed_without_crm_call(settings, payload):
    with TestClient(
        create_app(
            replace(settings, sla_seconds=2, admission_budget_seconds=1),
            crm=MockCRM(settings.outbox, latency_s=0),
        )
    ) as client:
        assert client.post("/audiences", json=payload, headers=HEADERS).status_code == 202
        assert wait_for_result(client, payload["request_id"])["status"] == "failed"
    assert not list(settings.outbox.glob("*.csv"))


def test_benchmark_detects_invalid_crm_output(tmp_path):
    (tmp_path / "list_test.csv").write_text("person_id\n1\n1\n99\n")
    (tmp_path / "list_test.json").write_text(json.dumps({"member_count": 3, "content_id": 2003}))
    record = {
        "http_status": 202,
        "client_elapsed_seconds": 2.1,
        "job": {
            "status": "succeeded",
            "result": {"sla_met": True, "list_id": "list_test", "member_count": 3},
        },
    }
    with pytest.raises(AssertionError, match="duplicate or ineligible"):
        validate_output(record, tmp_path, np.array([1, 2, 3]), 2003)


def test_queued_job_recovers_after_restart(settings, payload):
    runtime = Runtime(settings, MockCRM(settings.outbox, latency_s=0))
    request = AudienceRequest.model_validate(payload)
    runtime.store.admit(request, request.requested_at, time.time(), runtime, time.perf_counter())
    runtime.store.update(request.request_id, "running")
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        assert wait_for_result(client, request.request_id)["status"] == "succeeded"
    assert len(list(settings.outbox.glob("*.csv"))) == 1


def test_failed_preparation_keeps_previous_generation(settings):
    original = prepare_snapshot(settings.warehouse, settings.serving)
    path = settings.warehouse / "content.parquet"
    content = pd.read_parquet(path)
    content["dense_embedding"] = [np.ones(1023, dtype=np.float32)] * len(content)
    content.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="embedding dimensions"):
        prepare_snapshot(settings.warehouse, settings.serving, force=True)
    assert load_snapshot(settings.serving).generation == original.generation


def test_nonfinite_embedding_returns_clean_validation_error(settings, payload):
    payload["content"]["dense_embedding"][0] = float("nan")
    with TestClient(create_app(settings, crm=MockCRM(settings.outbox, latency_s=0))) as client:
        response = client.post(
            "/audiences",
            content=json.dumps(payload),
            headers={**HEADERS, "Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "input" not in response.json()["detail"][0]


def test_retention_keeps_unreconciled_imports(settings, payload):
    runtime = Runtime(settings, MockCRM(settings.outbox, latency_s=0))
    for status in ("succeeded", "failed", "delivery_unknown"):
        request = AudienceRequest.model_validate({**payload, "request_id": status})
        runtime.store.admit(
            request, request.requested_at, time.time(), runtime, time.perf_counter()
        )
        runtime.store.update(request.request_id, status)
    with runtime.store.connection() as connection:
        connection.execute("UPDATE jobs SET finished_at = ?", (time.time() - 8 * 86400,))
    runtime.store.purge(7)
    assert runtime.store.get("succeeded") is None
    assert runtime.store.get("failed") is None
    assert runtime.store.get("delivery_unknown")["status"] == "delivery_unknown"

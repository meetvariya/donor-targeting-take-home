"""Authenticated audience API with a durable, bounded, single-worker delivery loop."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from donor_targeting.config import (
    CRM_OUTBOX_DIR,
    DATA_DIR,
    MODELS_DIR,
    PROGRAM_AREAS,
    WAREHOUSE_DIR,
)
from donor_targeting.mocks.crm import MockCRM
from donor_targeting.policy import selected_positions
from donor_targeting.serving_features import (
    FEATURE_VERSION,
    SERVING_DIR,
    file_sha256,
    prepare_snapshot,
    source_fingerprint,
)

LOGGER = logging.getLogger("donor_targeting")
IDENTIFIER = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Content(StrictModel):
    content_id: int = Field(gt=0, le=2**63 - 1)
    subject: str = Field(min_length=1, max_length=2000)
    program_area: str
    dense_embedding: list[float] = Field(min_length=1024, max_length=1024)

    @field_validator("program_area")
    @classmethod
    def valid_program(cls, value: str) -> str:
        if value not in PROGRAM_AREAS:
            raise ValueError("unknown program area")
        return value

    @field_validator("dense_embedding")
    @classmethod
    def valid_embedding(cls, value: list[float]) -> list[float]:
        vector = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("embedding must be finite and have nonzero norm")
        return value


class Audience(StrictModel):
    type: Literal["all_active_subscribers"]


class Destination(StrictModel):
    crm: Literal["mock"]
    list_name: str = Field(min_length=1, max_length=200)


class AudienceRequest(StrictModel):
    request_id: IDENTIFIER
    org_id: Literal["demo-charity"]
    requested_at: datetime
    content: Content
    audience: Audience
    destination: Destination

    @field_validator("requested_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must include a timezone")
        return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Settings:
    api_token: str
    warehouse: Path = WAREHOUSE_DIR
    models: Path = MODELS_DIR
    serving: Path = SERVING_DIR
    state: Path = DATA_DIR / "service_state"
    outbox: Path = CRM_OUTBOX_DIR
    clock_mode: Literal["demo", "live"] = "demo"
    max_pending: int = 3
    sla_seconds: float = 180
    admission_budget_seconds: float = 45
    retention_days: int = 7

    def __post_init__(self):
        if len(self.api_token) < 16:
            raise ValueError("DONOR_API_TOKEN must contain at least 16 characters")
        if self.max_pending < 1 or self.admission_budget_seconds <= 0:
            raise ValueError("capacity and per-job admission budget must be positive")
        if self.clock_mode not in ("demo", "live"):
            raise ValueError("clock mode must be demo or live")
        if self.clock_mode == "live" and self.api_token.startswith("local-demo-"):
            raise ValueError(
                "live mode requires a private token, not the local demonstration token"
            )

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_token=os.environ.get("DONOR_API_TOKEN", ""),
            warehouse=Path(os.environ.get("DONOR_WAREHOUSE", str(WAREHOUSE_DIR))),
            models=Path(os.environ.get("DONOR_MODELS", str(MODELS_DIR))),
            serving=Path(os.environ.get("DONOR_SERVING", str(SERVING_DIR))),
            state=Path(os.environ.get("DONOR_STATE", str(DATA_DIR / "service_state"))),
            outbox=Path(os.environ.get("DONOR_OUTBOX", str(CRM_OUTBOX_DIR))),
            clock_mode=os.environ.get("DONOR_CLOCK_MODE", "demo"),
        )


class JobStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.database = directory / "jobs.sqlite3"
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL,
                    payload TEXT NOT NULL, feature_at TEXT NOT NULL,
                    accepted_at REAL NOT NULL, deadline_at REAL NOT NULL,
                    generation TEXT NOT NULL, model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL, status TEXT NOT NULL,
                    started_at REAL, finished_at REAL, result TEXT, error TEXT
                )"""
            )

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, request_id: str):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row else None

    def admit(
        self,
        payload: AudienceRequest,
        feature_at: datetime,
        received_at: float,
        runtime: Runtime,
        received_monotonic: float,
    ):
        serialized = json.dumps(
            payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        payload_hash = hashlib.sha256(serialized.encode()).hexdigest()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM jobs WHERE request_id = ?", (payload.request_id,)
            ).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise HTTPException(409, "request_id already has a different payload")
                return dict(existing)
            pending = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running', 'delivering')"
            ).fetchone()[0]
            settings = runtime.settings
            if (
                pending >= settings.max_pending
                or (pending + 1) * settings.admission_budget_seconds > settings.sla_seconds
            ):
                raise HTTPException(
                    429, "capacity reserved; retry later", headers={"Retry-After": "45"}
                )
            connection.execute(
                """INSERT INTO jobs (request_id, payload_hash, payload, feature_at,
                    accepted_at, deadline_at, generation, model_version, policy_version, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')""",
                (
                    payload.request_id,
                    payload_hash,
                    serialized,
                    feature_at.isoformat(),
                    received_at,
                    received_at + settings.sla_seconds,
                    runtime.snapshot.generation,
                    runtime.model_version,
                    runtime.policy["version"],
                ),
            )
            runtime.receipt_monotonic[payload.request_id] = received_monotonic
        return self.get(payload.request_id)

    def recover(self):
        with self.connection() as connection:
            connection.execute("UPDATE jobs SET status = 'queued' WHERE status = 'running'")
            connection.execute(
                """UPDATE jobs SET status = 'delivery_unknown', error = ?, finished_at = ?
                    WHERE status = 'delivering'""",
                (
                    "CRM may have accepted the list before restart; reconcile before retry",
                    time.time(),
                ),
            )

    def claim(self):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY accepted_at LIMIT 1"
            ).fetchone()
            if row:
                connection.execute(
                    "UPDATE jobs SET status = 'running', started_at = ? WHERE request_id = ?",
                    (time.time(), row["request_id"]),
                )
        return dict(row) if row else None

    def update(self, request_id: str, status: str, *, result=None, error=None):
        finished_at = time.time() if status in ("succeeded", "failed", "delivery_unknown") else None
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? WHERE request_id = ?",
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    error,
                    finished_at,
                    request_id,
                ),
            )

    def purge(self, retention_days: int):
        cutoff = time.time() - retention_days * 86400
        with self.connection() as connection:
            expired = connection.execute(
                "SELECT request_id FROM jobs WHERE finished_at < ? AND status IN ('succeeded', 'failed')",
                (cutoff,),
            ).fetchall()
            connection.execute(
                "DELETE FROM jobs WHERE finished_at < ? AND status IN ('succeeded', 'failed')",
                (cutoff,),
            )
        for row in expired:
            artifact = (
                self.directory / f"audience_{hashlib.sha256(row[0].encode()).hexdigest()}.npy"
            )
            artifact.unlink(missing_ok=True)


class Runtime:
    def __init__(self, settings: Settings, crm: MockCRM | None = None):
        self.settings = settings
        self.snapshot = prepare_snapshot(settings.warehouse, settings.serving)
        model_path = settings.models / "donation_model.joblib"
        self.model_version = file_sha256(model_path)
        self.model = joblib.load(model_path)
        self.policy = json.loads((settings.models / "audience_policy.json").read_text())
        if (
            self.policy["model_sha256"] != self.model_version
            or self.policy["feature_version"] != FEATURE_VERSION
            or self.policy["ranking"] not in ("model", "centroid", "bau")
            or not 0 < self.policy["fraction"] <= 1
        ):
            raise ValueError("model/policy artifacts are incompatible; rebuild the policy")
        self.store = JobStore(settings.state)
        self.crm = crm if crm is not None else MockCRM(settings.outbox)
        self.receipt_monotonic: dict[str, float] = {}
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread = threading.Thread(target=self.work, name="audience-worker", daemon=True)
        self.store.recover()
        self.store.purge(settings.retention_days)

    def fresh_source(self) -> bool:
        try:
            return source_fingerprint(self.settings.warehouse) == self.snapshot.metadata["source"]
        except OSError:
            return False

    def feature_time(self, payload: AudienceRequest) -> datetime:
        if self.settings.clock_mode == "demo":
            feature_at = payload.requested_at
        else:
            feature_at = datetime.now(timezone.utc)
            if abs((feature_at - payload.requested_at).total_seconds()) > 300:
                raise HTTPException(422, "requested_at is outside the live clock-skew allowance")
        if feature_at < self.snapshot.snapshot_at:
            raise HTTPException(422, "request predates available eligibility data")
        if feature_at - self.snapshot.snapshot_at >= timedelta(hours=24):
            raise HTTPException(
                503, "eligibility snapshot is stale; await the next completed refresh"
            )
        if not self.fresh_source():
            raise HTTPException(503, "warehouse changed; prepare a new generation and restart")
        return feature_at

    def work(self):
        while not self.stop_event.is_set():
            job = self.store.claim()
            if job is None:
                self.wake_event.wait(0.25)
                self.wake_event.clear()
                continue
            self.deliver(job)

    def deliver(self, job: dict):
        request_id = job["request_id"]
        started = time.perf_counter()
        receipt = self.receipt_monotonic.get(request_id)
        queue_seconds = max(
            0, started - receipt if receipt is not None else time.time() - job["accepted_at"]
        )

        def remaining_seconds():
            return (
                self.settings.sla_seconds - (time.perf_counter() - receipt)
                if receipt is not None
                else job["deadline_at"] - time.time()
            )

        stage = "running"
        try:
            if (
                job["generation"] != self.snapshot.generation
                or job["model_version"] != self.model_version
                or job["policy_version"] != self.policy["version"]
            ):
                raise ValueError("accepted artifacts changed; request cannot be silently rescored")
            if not self.fresh_source():
                raise ValueError("warehouse changed before delivery")
            if remaining_seconds() <= 3:
                raise TimeoutError("insufficient deadline remaining for CRM completion")
            payload = AudienceRequest.model_validate_json(job["payload"])
            feature_at = datetime.fromisoformat(job["feature_at"])
            similarities = self.snapshot.score(
                np.asarray(payload.content.dense_embedding), feature_at
            )
            probabilities = self.model.predict_proba(
                pd.DataFrame({"centroid_cosine_similarity": similarities})
            )[:, 1]
            ranking = self.policy["ranking"]
            if ranking == "model":
                scores = probabilities
            elif ranking == "centroid":
                scores = similarities
            else:
                seed = int(hashlib.sha256(request_id.encode()).hexdigest()[:16], 16)
                scores = np.random.default_rng(seed).random(len(similarities))
            positions = selected_positions(
                self.snapshot.person_ids, scores, self.policy["fraction"]
            )
            selected = self.snapshot.person_ids[positions]
            if len(np.unique(selected)) != len(selected):
                raise ValueError("audience contains duplicate person ids")
            artifact = (
                self.store.directory
                / f"audience_{hashlib.sha256(request_id.encode()).hexdigest()}.npy"
            )
            temporary = artifact.with_suffix(".tmp")
            with temporary.open("wb") as destination:
                np.save(destination, selected, allow_pickle=False)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, artifact)
            audience_sha256 = file_sha256(artifact)
            scoring_seconds = time.perf_counter() - started
            if remaining_seconds() <= 3:
                raise TimeoutError("deadline exhausted during scoring")
            if not self.fresh_source():
                raise ValueError("warehouse changed during scoring")
            self.store.update(request_id, "delivering")
            stage = "delivering"
            crm_started = time.perf_counter()
            list_id = self.crm.create_list(
                payload.destination.list_name, selected, content_id=payload.content.content_id
            )
            crm_seconds = time.perf_counter() - crm_started
            elapsed = (
                time.perf_counter() - receipt
                if receipt is not None
                else time.time() - job["accepted_at"]
            )
            result = {
                "list_id": list_id,
                "member_count": len(selected),
                "active_count": len(self.snapshot.person_ids),
                "snapshot_at": self.snapshot.snapshot_at.isoformat(),
                "feature_at": feature_at.isoformat(),
                "generation": self.snapshot.generation,
                "model_version": self.model_version,
                "policy_version": self.policy["version"],
                "ranking": ranking,
                "audience_fraction": self.policy["fraction"],
                "missing_feature_fraction": float(np.isnan(similarities).mean()),
                "queue_seconds": queue_seconds,
                "scoring_seconds": scoring_seconds,
                "crm_seconds": crm_seconds,
                "elapsed_seconds": elapsed,
                "sla_met": elapsed < self.settings.sla_seconds,
                "audience_sha256": audience_sha256,
            }
            self.store.update(request_id, "succeeded", result=result)
            LOGGER.info(
                json.dumps(
                    {
                        "event": "audience_delivered",
                        "request_id": request_id,
                        "member_count": len(selected),
                        "elapsed_seconds": elapsed,
                        "sla_met": result["sla_met"],
                        "policy_version": self.policy["version"],
                    }
                )
            )
        except Exception as error:
            status = "delivery_unknown" if stage == "delivering" else "failed"
            self.store.update(request_id, status, error=f"{type(error).__name__}: {error}")
            LOGGER.error(
                json.dumps(
                    {"event": status, "request_id": request_id, "error_type": type(error).__name__}
                )
            )
        finally:
            self.receipt_monotonic.pop(request_id, None)


def public_job(job: dict) -> dict:
    return {
        "request_id": job["request_id"],
        "status": job["status"],
        "status_url": f"/requests/{job['request_id']}",
        "accepted_at": datetime.fromtimestamp(job["accepted_at"], timezone.utc).isoformat(),
        "deadline_at": datetime.fromtimestamp(job["deadline_at"], timezone.utc).isoformat(),
        "result": json.loads(job["result"]) if job["result"] else None,
        "error": job["error"],
    }


class BodyLimit:
    def __init__(self, app, limit: int = 65536):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["received_at"] = time.time()
        state["received_monotonic"] = time.perf_counter()
        headers = dict(scope["headers"])
        try:
            if int(headers.get(b"content-length", b"0")) > self.limit:
                await JSONResponse({"detail": "request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return
        except ValueError:
            await JSONResponse({"detail": "invalid content length"}, status_code=400)(
                scope, receive, send
            )
            return
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.limit:
                raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None, *, crm: MockCRM | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = Runtime(settings, crm)
        app.state.runtime = runtime
        runtime.thread.start()
        try:
            yield
        finally:
            runtime.stop_event.set()
            runtime.wake_event.set()
            runtime.thread.join(timeout=5)

    app = FastAPI(title="Donor audience service", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(BodyLimit)
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        details = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse({"detail": details}, status_code=422)

    def authorized(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not hmac.compare_digest(
            credentials.credentials.encode(), settings.api_token.encode()
        ):
            raise HTTPException(
                401, "valid bearer token required", headers={"WWW-Authenticate": "Bearer"}
            )

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz", dependencies=[Depends(authorized)])
    def ready(request: Request):
        runtime = request.app.state.runtime
        if not runtime.thread.is_alive() or not runtime.fresh_source():
            raise HTTPException(503, "worker or serving generation is not ready")
        if settings.clock_mode == "live" and datetime.now(
            timezone.utc
        ) - runtime.snapshot.snapshot_at >= timedelta(hours=24):
            raise HTTPException(503, "warehouse snapshot is stale")
        return {
            "status": "ready",
            "active_count": len(runtime.snapshot.person_ids),
            "array_bytes": runtime.snapshot.array_bytes,
            "generation": runtime.snapshot.generation,
            "policy_version": runtime.policy["version"],
            "clock_mode": settings.clock_mode,
        }

    @app.post("/audiences", status_code=202, dependencies=[Depends(authorized)])
    def submit(payload: AudienceRequest, request: Request):
        runtime = request.app.state.runtime
        if not runtime.thread.is_alive():
            raise HTTPException(503, "delivery worker is unavailable")
        received_at = request.state.received_at
        existing = runtime.store.get(payload.request_id)
        feature_at = (
            datetime.fromisoformat(existing["feature_at"])
            if existing
            else runtime.feature_time(payload)
        )
        job = runtime.store.admit(
            payload, feature_at, received_at, runtime, request.state.received_monotonic
        )
        runtime.wake_event.set()
        return public_job(job)

    @app.get("/requests/{request_id}", dependencies=[Depends(authorized)])
    def status(request_id: str, request: Request):
        job = request.app.state.runtime.store.get(request_id)
        if job is None:
            raise HTTPException(404, "unknown request")
        return public_job(job)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)


if __name__ == "__main__":
    main()

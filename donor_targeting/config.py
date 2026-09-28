"""Shared constants: paths, the program-area enum, and the simulated calendar."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
REQUESTS_DIR = DATA_DIR / "requests"
CRM_OUTBOX_DIR = DATA_DIR / "crm_outbox"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

EMBEDDING_DIM = 1024

PROGRAM_AREAS: tuple[str, ...] = (
    "disaster_relief",
    "hunger_relief",
    "clean_water",
    "education",
    "child_health",
    "maternal_health",
    "refugee_support",
    "climate_resilience",
    "mental_health",
    "housing",
    "economic_empowerment",
    "medical_research",
)

# Simulated calendar. The warehouse snapshot contains everything that happened before
# SNAPSHOT_AT (2026-09-01 01:00 America/New_York == 05:00 UTC).
HISTORY_START = date(2026, 3, 1)
HISTORY_END = date(2026, 8, 31)
SNAPSHOT_AT = datetime(2026, 9, 1, 5, 0, tzinfo=timezone.utc)

CENTROID_LOOKBACK_DAYS = 30
ATTRIBUTION_WINDOW_DAYS = 7
BAU_SEND_FRACTION = 0.75

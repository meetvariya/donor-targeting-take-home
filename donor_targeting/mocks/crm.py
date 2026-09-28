"""A stand-in for the customer's CRM list-import API.

    crm = MockCRM()
    list_id = crm.create_list("Clean water appeal (2003)", person_ids, content_id=2003)
    crm.get_list(list_id)  # {"member_count": ..., ...}

`create_list` sends the whole list in a single request, which takes about 2 seconds whatever
the list's size. Duplicate ids are ignored.

Members are written to data/crm_outbox/<list_id>.csv, list metadata to <list_id>.json. We score
your lists from there.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from donor_targeting.config import CRM_OUTBOX_DIR


class MockCRM:
    def __init__(self, outbox: Path = CRM_OUTBOX_DIR, *, latency_s: float = 2.0):
        self.outbox = Path(outbox)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.latency_s = latency_s
        self._lock = threading.Lock()
        self._meta: dict[str, dict] = {}

    def create_list(
        self, name: str, person_ids: Iterable[int], content_id: int | None = None
    ) -> str:
        """Create a list with these members in one request; returns its id."""
        members = list(dict.fromkeys(int(p) for p in person_ids))
        time.sleep(self.latency_s)
        list_id = f"list_{uuid.uuid4().hex[:12]}"
        (self.outbox / f"{list_id}.csv").write_text(
            "person_id\n" + "".join(f"{p}\n" for p in members)
        )
        meta = {
            "list_id": list_id,
            "name": name,
            "content_id": content_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "member_count": len(members),
        }
        (self.outbox / f"{list_id}.json").write_text(json.dumps(meta, indent=2))
        with self._lock:
            self._meta[list_id] = meta
        return list_id

    def get_list(self, list_id: str) -> dict:
        with self._lock:
            return dict(self._meta[list_id])

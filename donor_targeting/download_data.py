"""Download the warehouse snapshot and the 12 upcoming requests into data/.

    python -m donor_targeting.download_data     # -> data/warehouse/, data/requests/

Everyone gets the same snapshot, so your numbers will match ours. The archive is checked against
its SHA-256 before it is unpacked, and not downloaded again once data/ holds it.
"""

from __future__ import annotations

import hashlib
import tarfile
import time
import urllib.request
from pathlib import Path

from donor_targeting.config import DATA_DIR

DATA_URL = "https://chorus-ai-ds-takehome.s3.us-east-2.amazonaws.com/donor-targeting/data-2026-09-01.tar.gz"
DATA_SHA256 = "d36104bf36acd686d09f9d6035f0335d12876b6f4fc5a840954e2eb7e0a4fb0e"


def download(out: Path = DATA_DIR, url: str = DATA_URL, sha256: str = DATA_SHA256) -> None:
    marker = out / ".snapshot_sha256"
    if marker.exists() and marker.read_text() == sha256:
        print(f"{out} already holds the snapshot")
        return
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "snapshot.tar.gz.part"
    t0 = time.time()
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, open(archive, "wb") as f:
        print(f"downloading {url} ({int(resp.headers['Content-Length']) / 1e6:.0f} MB)")
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        archive.unlink()
        raise SystemExit(f"checksum mismatch for {url}: got {digest.hexdigest()}, want {sha256}")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
        n = len(tar.getnames())
    archive.unlink()
    marker.write_text(sha256)
    print(f"unpacked {n} files into {out} in {time.time() - t0:.0f} s")


def main() -> None:
    download()


if __name__ == "__main__":
    main()

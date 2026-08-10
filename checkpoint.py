"""Atomic JSON checkpoint persistence."""

import json
import os
from datetime import datetime, timezone
from typing import Iterable, Set


def load_checkpoint(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_lattes_ids", []))


def save_checkpoint(path: str, processed_ids: Iterable[str]) -> None:
    tmp_path = path + ".tmp"
    payload = {
        "processed_lattes_ids": sorted(processed_ids),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

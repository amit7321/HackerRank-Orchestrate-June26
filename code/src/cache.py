"""Disk cache for per-image analysis (and other structured calls), keyed by a
caller-supplied string (normally sha256(image bytes) + prompt version + model
+ claim_object). Avoids re-running expensive vision calls across pipeline
reruns and across the two evaluation configs when they share a cache key.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from . import config


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return config.CACHE_DIR / f"{digest}.json"


def make_key(*parts: str) -> str:
    return "|".join(parts)


def get(key: str) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def set(key: str, value: Any) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    path.write_text(json.dumps(value), encoding="utf-8")


def image_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

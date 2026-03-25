"""Simple in-memory per-IP rate limiter for API abuse prevention."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

# Defaults: 30 requests per 60 seconds per IP.
WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "30"))

_REQUESTS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()


def is_rate_limited(client_id: str) -> bool:
    """Return True when the client exceeds configured request limits."""
    now = time.monotonic()

    with _LOCK:
        bucket = _REQUESTS[client_id]

        while bucket and now - bucket[0] > WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= MAX_REQUESTS:
            return True

        bucket.append(now)
        return False

"""
===========================================================================
VIDYUT Rate Limit Middleware
===========================================================================
Token-bucket per client IP (or API key prefix). Uses in-process counter — fine
for single-replica hackathon deployment. For multi-replica, swap to Redis.

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter. Allows N requests per 60s per client.

    Identifier priority:
      1. API key prefix (request.state.actor if set by auth)
      2. X-Forwarded-For header
      3. request.client.host
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 120,
        window_seconds: int = 60,
        exempt_paths=("/healthz", "/readyz", "/metrics"),
    ):
        super().__init__(app)
        self.limit = max(1, int(requests_per_minute))
        self.window = max(1, int(window_seconds))
        self.exempt_paths = tuple(exempt_paths)
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        client_id = self._client_id(request)
        now = time.time()

        with self._lock:
            bucket = self._buckets[client_id]
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = int(self.window - (now - bucket[0])) + 1
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": (
                            f"Rate limit {self.limit}/min exceeded for {client_id}. "
                            f"Retry in {retry_after}s."
                        ),
                        "limit": self.limit,
                        "window_seconds": self.window,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            bucket.append(now)
            remaining = self.limit - len(bucket)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _client_id(request: Request) -> str:
        actor = getattr(request.state, "actor", None)
        if actor and actor not in ("anonymous", "anonymous-no-auth"):
            return actor
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
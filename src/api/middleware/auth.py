"""
===========================================================================
VIDYUT Auth Middleware
===========================================================================
Simple API-key authentication for hackathon scope. In production, swap for
OAuth2 / JWT (entry point already isolated for that).

Header: X-API-Key: <key>
or:    Authorization: Bearer <key>

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication."""

    def __init__(
        self,
        app,
        api_key: str = "",
        exempt_paths: Iterable[str] = (),
    ):
        super().__init__(app)
        self.api_key = api_key.strip()
        self.exempt_paths: Tuple[str, ...] = tuple(exempt_paths)
        self._enabled = bool(self.api_key)
        if not self._enabled:
            logger.warning(
                "AuthMiddleware: VIDYUT_API_KEY is empty → auth DISABLED. "
                "Set VIDYUT_API_KEY in production."
            )

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Exempt paths skip auth
        if any(path == p or path.startswith(p) for p in self.exempt_paths):
            request.state.actor = "anonymous"
            return await call_next(request)

        if not self._enabled:
            request.state.actor = "anonymous-no-auth"
            return await call_next(request)

        provided = self._extract_key(request)
        if not provided:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Missing API key. Send X-API-Key header.",
                },
            )
        if not self._secure_compare(provided, self.api_key):
            return JSONResponse(
                status_code=403,
                content={"error": "forbidden", "message": "Invalid API key."},
            )

        request.state.actor = f"api-key:{provided[:6]}"
        return await call_next(request)

    @staticmethod
    def _extract_key(request: Request) -> str:
        key = request.headers.get("X-API-Key", "").strip()
        if key:
            return key
        auth = request.headers.get("Authorization", "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    @staticmethod
    def _secure_compare(a: str, b: str) -> bool:
        if len(a) != len(b):
            return False
        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)
        return result == 0
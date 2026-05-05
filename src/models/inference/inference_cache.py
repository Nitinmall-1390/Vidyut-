"""
VIDYUT Inference Cache
===========================================================================
Redis-backed cache for prediction results. Avoids redundant model inference
for recently computed consumer/feeder predictions.
===========================================================================
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from src.config.settings import get_settings
from src.utils.logger import get_logger

log = get_logger("vidyut.inference_cache")
settings = get_settings()

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    log.warning("redis package not installed. InferenceCache will be no-op.")


class InferenceCache:
    """
    Redis-backed inference result cache.

    If Redis is unavailable (dev mode, tests), falls back to an in-process
    dictionary cache (non-persistent, single-process only).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        key_prefix: str = "vidyut:inference:",
    ) -> None:
        self.host = host or settings.redis_host
        self.port = port or settings.redis_port
        self.db = db if db is not None else settings.redis_db
        self.password = password or settings.redis_password or None
        self.ttl = ttl_seconds or settings.redis_ttl_seconds
        self.key_prefix = key_prefix
        self._client: Optional[Any] = None
        self._fallback_cache: Dict[str, str] = {}
        self._use_redis = False

        self._connect()

    def _connect(self) -> None:
        if not REDIS_AVAILABLE:
            log.info("Redis unavailable. Using in-process fallback cache.")
            return
        try:
            client = redis_lib.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password or None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            self._client = client
            self._use_redis = True
            log.info("Redis connected: %s:%d db=%d", self.host, self.port, self.db)
        except Exception as exc:
            log.warning("Redis connection failed (%s). Using in-process cache.", exc)
            self._use_redis = False

    @staticmethod
    def _make_cache_key(payload: Dict) -> str:
        """Deterministic hash of a prediction request payload."""
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def get(self, payload: Dict) -> Optional[Dict]:
        """
        Retrieve cached prediction result.

        Returns None on cache miss.
        """
        key = self.key_prefix + self._make_cache_key(payload)
        try:
            if self._use_redis:
                raw = self._client.get(key)
            else:
                raw = self._fallback_cache.get(key)

            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log.debug("Cache get error: %s", exc)
            return None

    def set(self, payload: Dict, result: Dict) -> bool:
        """
        Store a prediction result in the cache.

        Returns True on success, False on error.
        """
        key = self.key_prefix + self._make_cache_key(payload)
        value = json.dumps(result, default=str)
        try:
            if self._use_redis:
                self._client.setex(key, self.ttl, value)
            else:
                self._fallback_cache[key] = value
            return True
        except Exception as exc:
            log.debug("Cache set error: %s", exc)
            return False

    def invalidate(self, payload: Dict) -> bool:
        """Delete a specific cached result."""
        key = self.key_prefix + self._make_cache_key(payload)
        try:
            if self._use_redis:
                self._client.delete(key)
            else:
                self._fallback_cache.pop(key, None)
            return True
        except Exception:
            return False

    def flush_all(self, pattern: Optional[str] = None) -> int:
        """Flush all Vidyut cache keys. Returns count deleted."""
        try:
            if self._use_redis:
                scan_pattern = self.key_prefix + (pattern or "*")
                keys = self._client.keys(scan_pattern)
                if keys:
                    return self._client.delete(*keys)
                return 0
            else:
                count = len(self._fallback_cache)
                self._fallback_cache.clear()
                return count
        except Exception:
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Return cache health status."""
        if self._use_redis:
            try:
                info = self._client.info()
                return {
                    "backend": "redis",
                    "connected": True,
                    "used_memory": info.get("used_memory_human"),
                    "keyspace": info.get("keyspace"),
                }
            except Exception as exc:
                return {"backend": "redis", "connected": False, "error": str(exc)}
        else:
            return {
                "backend": "in-process",
                "connected": True,
                "keys_cached": len(self._fallback_cache),
            }

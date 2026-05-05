"""
===========================================================================
VIDYUT Redis Cache
===========================================================================
High-performance caching layer with automatic fallback to in-memory LRU
when Redis is unavailable. The system MUST keep working if Redis dies —
caching is an optimization, not a critical path.

Features:
  - Transparent JSON + pickle serialization (handles numpy, pandas, sklearn objects)
  - TTL per namespace (forecasts: 15min, SHAP: 24h, weather: 7d)
  - Bulk operations (mget, mset) for batch inference
  - Pattern-based invalidation (delete by namespace/entity/version)
  - Hit/miss metrics for observability
  - Graceful degradation — never raises if cache fails
  - Thread-safe singleton via get_cache()

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Optional Redis import — system works without it
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception
    REDIS_AVAILABLE = False

from src.cache.cache_key_builder import CacheKeyBuilder, CacheNamespace

logger = logging.getLogger(__name__)


# ===========================================================================
# Backend types
# ===========================================================================
class CacheBackend(str, Enum):
    """Cache backend types."""

    REDIS = "redis"
    MEMORY = "memory"
    DISABLED = "disabled"


# ===========================================================================
# Default TTLs per namespace (in seconds)
# ===========================================================================
DEFAULT_TTL_BY_NAMESPACE: Dict[str, int] = {
    CacheNamespace.DEMAND_FORECAST.value: 900,        # 15 min — forecasts go stale fast
    CacheNamespace.THEFT_SCORE.value: 3600,           # 1 hour
    CacheNamespace.ANOMALY_SCORE.value: 3600,         # 1 hour
    CacheNamespace.SHAP_EXPLANATION.value: 86400,     # 24 hours — SHAP is expensive, cache aggressively
    CacheNamespace.FEATURE_VECTOR.value: 1800,        # 30 min
    CacheNamespace.RULE_FLAGS.value: 3600,            # 1 hour
    CacheNamespace.CONFIDENCE_SCORE.value: 3600,      # 1 hour
    CacheNamespace.NETWORK_RING.value: 21600,         # 6 hours — graphs are expensive
    CacheNamespace.WEATHER_DATA.value: 604800,        # 7 days — historical weather never changes
    CacheNamespace.API_RESPONSE.value: 300,           # 5 min
    CacheNamespace.MODEL_METADATA.value: 86400,       # 24 hours
}
DEFAULT_TTL_FALLBACK = 600  # 10 min for un-namespaced keys


# ===========================================================================
# Metrics
# ===========================================================================
@dataclass
class CacheMetrics:
    """Tracks cache performance counters."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    fallback_count: int = 0  # How many times we fell back to memory
    started_at: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "fallback_count": self.fallback_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }

    def reset(self) -> None:
        self.__init__()


# ===========================================================================
# In-memory LRU fallback
# ===========================================================================
class _LRUMemoryCache:
    """
    Thread-safe in-memory LRU cache used when Redis is unavailable.
    Stores (value, expires_at) tuples. Evicts oldest when capacity is hit.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._store: "OrderedDict[str, Tuple[bytes, float]]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)  # mark as recently used
            return value

    def set(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> None:
        with self._lock:
            expires_at = time.time() + ttl_seconds if ttl_seconds else 0
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # evict LRU

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def delete_pattern(self, pattern: str) -> int:
        """Glob-style pattern deletion. Supports trailing '*' wildcards."""
        with self._lock:
            prefix = pattern.rstrip("*")
            keys_to_delete = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                self._store.pop(k, None)
            return len(keys_to_delete)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def keys(self, pattern: str = "*") -> List[str]:
        with self._lock:
            prefix = pattern.rstrip("*")
            return [k for k in self._store.keys() if k.startswith(prefix)]


# ===========================================================================
# Main Redis cache class
# ===========================================================================
class RedisCache:
    """
    Production-grade cache for Vidyut. Wraps Redis with intelligent fallback
    to in-memory LRU. Use via get_cache() singleton in normal operation.

    Example:
        cache = get_cache()
        result = cache.get_or_compute(
            key=builder.theft_score_key("C12345", "2024-11-01"),
            compute_fn=lambda: model.predict(features),
            ttl=3600,
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        max_connections: int = 50,
        memory_fallback_size: int = 10_000,
        enabled: bool = True,
        key_builder: Optional[CacheKeyBuilder] = None,
    ):
        self.enabled = enabled
        self.host = host
        self.port = port
        self.db = db
        self.metrics = CacheMetrics()
        self.key_builder = key_builder or CacheKeyBuilder()

        # In-memory fallback always available
        self._memory = _LRUMemoryCache(max_size=memory_fallback_size)
        self._memory_only_warning_logged = False

        # Redis client (may be None if unavailable)
        self._redis: Optional["redis.Redis"] = None
        self._backend = CacheBackend.DISABLED

        if not enabled:
            logger.info("Vidyut cache: DISABLED by configuration")
            return

        if not REDIS_AVAILABLE:
            logger.warning(
                "Vidyut cache: redis-py not installed → using in-memory LRU only"
            )
            self._backend = CacheBackend.MEMORY
            return

        # Try to connect to Redis
        try:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                max_connections=max_connections,
                decode_responses=False,  # we handle bytes ourselves
            )
            self._redis = redis.Redis(connection_pool=pool)
            self._redis.ping()
            self._backend = CacheBackend.REDIS
            logger.info(
                "Vidyut cache: connected to Redis at %s:%d/db%d", host, port, db
            )
        except (RedisError, RedisConnectionError, OSError) as e:
            logger.warning(
                "Vidyut cache: Redis connection failed (%s) → falling back to in-memory LRU",
                e,
            )
            self._redis = None
            self._backend = CacheBackend.MEMORY

    # -----------------------------------------------------------------------
    # Public API: get / set / delete / exists
    # -----------------------------------------------------------------------
    @property
    def backend(self) -> CacheBackend:
        return self._backend

    @property
    def is_redis(self) -> bool:
        return self._backend == CacheBackend.REDIS

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key. Returns `default` on miss or error."""
        if not self.enabled:
            return default
        try:
            raw = self._raw_get(key)
            if raw is None:
                self.metrics.misses += 1
                return default
            self.metrics.hits += 1
            self.metrics.bytes_read += len(raw)
            return self._deserialize(raw)
        except Exception as e:
            logger.debug("Cache get error for key=%s: %s", key, e)
            self.metrics.errors += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a value with optional TTL. Returns True on success."""
        if not self.enabled:
            return False
        try:
            ttl = ttl if ttl is not None else self._infer_ttl(key)
            payload = self._serialize(value)
            self._raw_set(key, payload, ttl)
            self.metrics.sets += 1
            self.metrics.bytes_written += len(payload)
            return True
        except Exception as e:
            logger.debug("Cache set error for key=%s: %s", key, e)
            self.metrics.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete a single key. Returns True if deleted."""
        if not self.enabled:
            return False
        try:
            deleted = self._raw_delete(key)
            if deleted:
                self.metrics.deletes += 1
            return deleted
        except Exception as e:
            logger.debug("Cache delete error for key=%s: %s", key, e)
            self.metrics.errors += 1
            return False

    def exists(self, key: str) -> bool:
        """Check whether a key exists (without fetching value)."""
        if not self.enabled:
            return False
        try:
            if self.is_redis:
                return bool(self._redis.exists(key))
            return self._memory.get(key) is not None
        except Exception as e:
            logger.debug("Cache exists error for key=%s: %s", key, e)
            self.metrics.errors += 1
            return False

    # -----------------------------------------------------------------------
    # Bulk operations — critical for batch inference
    # -----------------------------------------------------------------------
    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Bulk-fetch multiple keys. Returns {key: value} for hits only."""
        if not self.enabled or not keys:
            return {}
        results: Dict[str, Any] = {}
        try:
            if self.is_redis:
                raw_values = self._redis.mget(keys)
                for k, raw in zip(keys, raw_values):
                    if raw is not None:
                        try:
                            results[k] = self._deserialize(raw)
                            self.metrics.hits += 1
                            self.metrics.bytes_read += len(raw)
                        except Exception:
                            self.metrics.errors += 1
                    else:
                        self.metrics.misses += 1
            else:
                for k in keys:
                    raw = self._memory.get(k)
                    if raw is not None:
                        try:
                            results[k] = self._deserialize(raw)
                            self.metrics.hits += 1
                        except Exception:
                            self.metrics.errors += 1
                    else:
                        self.metrics.misses += 1
        except Exception as e:
            logger.debug("Cache mget error: %s", e)
            self.metrics.errors += 1
        return results

    def mset(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Bulk-store multiple {key: value} pairs. Returns count successfully set."""
        if not self.enabled or not items:
            return 0
        success = 0
        try:
            if self.is_redis:
                # Use pipeline for atomicity + speed
                pipe = self._redis.pipeline(transaction=False)
                for k, v in items.items():
                    payload = self._serialize(v)
                    item_ttl = ttl if ttl is not None else self._infer_ttl(k)
                    if item_ttl and item_ttl > 0:
                        pipe.setex(k, item_ttl, payload)
                    else:
                        pipe.set(k, payload)
                    self.metrics.bytes_written += len(payload)
                pipe.execute()
                success = len(items)
                self.metrics.sets += success
            else:
                for k, v in items.items():
                    item_ttl = ttl if ttl is not None else self._infer_ttl(k)
                    payload = self._serialize(v)
                    self._memory.set(k, payload, item_ttl)
                    self.metrics.bytes_written += len(payload)
                    success += 1
                self.metrics.sets += success
        except Exception as e:
            logger.debug("Cache mset error: %s", e)
            self.metrics.errors += 1
        return success

    # -----------------------------------------------------------------------
    # Compute-or-fetch — the most important method
    # -----------------------------------------------------------------------
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Any:
        """
        Cache-aside pattern: return cached value, or compute+cache if missing.
        This is the workhorse for prediction caching.

        Args:
            key: Cache key (use CacheKeyBuilder)
            compute_fn: Zero-arg callable that produces the value on miss
            ttl: Optional TTL override (else inferred from namespace)
            force_refresh: If True, skip lookup and recompute

        Returns:
            The cached or freshly computed value
        """
        if not force_refresh:
            cached = self.get(key, default=_SENTINEL)
            if cached is not _SENTINEL:
                return cached

        # Cache miss → compute
        value = compute_fn()
        self.set(key, value, ttl=ttl)
        return value

    # -----------------------------------------------------------------------
    # Pattern-based invalidation
    # -----------------------------------------------------------------------
    def delete_pattern(self, pattern: str, batch_size: int = 500) -> int:
        """
        Delete all keys matching a glob pattern. Used for invalidating
        an entire namespace or model version.

        Returns: number of keys deleted.
        """
        if not self.enabled:
            return 0
        try:
            if self.is_redis:
                deleted_total = 0
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(
                        cursor=cursor, match=pattern, count=batch_size
                    )
                    if keys:
                        deleted_total += self._redis.delete(*keys)
                    if cursor == 0:
                        break
                self.metrics.deletes += deleted_total
                return deleted_total
            else:
                deleted = self._memory.delete_pattern(pattern)
                self.metrics.deletes += deleted
                return deleted
        except Exception as e:
            logger.debug("Cache delete_pattern error for pattern=%s: %s", pattern, e)
            self.metrics.errors += 1
            return 0

    def invalidate_namespace(
        self,
        namespace: Union[CacheNamespace, str],
        version_only: bool = True,
    ) -> int:
        """Invalidate all keys in a namespace (optionally only current version)."""
        if version_only:
            pattern = self.key_builder.version_pattern(namespace)
        else:
            pattern = self.key_builder.namespace_pattern(namespace)
        deleted = self.delete_pattern(pattern)
        logger.info(
            "Cache invalidated namespace=%s pattern=%s deleted=%d",
            namespace, pattern, deleted,
        )
        return deleted

    def invalidate_entity(
        self,
        namespace: Union[CacheNamespace, str],
        entity_id: str,
    ) -> int:
        """Invalidate all keys for a specific entity in a namespace."""
        pattern = self.key_builder.entity_pattern(namespace, entity_id)
        return self.delete_pattern(pattern)

    def clear_all(self) -> int:
        """Delete every Vidyut key. DESTRUCTIVE — use with care."""
        return self.delete_pattern(self.key_builder.all_vidyut_pattern())

    # -----------------------------------------------------------------------
    # Health & introspection
    # -----------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Returns cache health status — used by /healthz endpoint."""
        status: Dict[str, Any] = {
            "enabled": self.enabled,
            "backend": self._backend.value,
            "redis_available": REDIS_AVAILABLE,
        }
        if self.is_redis and self._redis is not None:
            try:
                start = time.time()
                self._redis.ping()
                status["ping_ms"] = round((time.time() - start) * 1000, 2)
                info = self._redis.info(section="memory")
                status["redis_used_memory_mb"] = round(
                    info.get("used_memory", 0) / (1024 * 1024), 2
                )
                status["healthy"] = True
            except Exception as e:
                status["healthy"] = False
                status["error"] = str(e)
        else:
            status["healthy"] = self.enabled
            status["memory_size"] = self._memory.size()
        status["metrics"] = self.metrics.to_dict()
        return status

    def get_metrics(self) -> Dict[str, Any]:
        """Return current cache metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset metrics counters (e.g., after deployment)."""
        self.metrics.reset()

    # -----------------------------------------------------------------------
    # Internal: backend dispatch
    # -----------------------------------------------------------------------
    def _raw_get(self, key: str) -> Optional[bytes]:
        if self.is_redis:
            try:
                return self._redis.get(key)
            except (RedisError, RedisConnectionError, OSError) as e:
                self._fallback_to_memory(reason=str(e))
        return self._memory.get(key)

    def _raw_set(self, key: str, payload: bytes, ttl: Optional[int]) -> None:
        if self.is_redis:
            try:
                if ttl and ttl > 0:
                    self._redis.setex(key, ttl, payload)
                else:
                    self._redis.set(key, payload)
                return
            except (RedisError, RedisConnectionError, OSError) as e:
                self._fallback_to_memory(reason=str(e))
        self._memory.set(key, payload, ttl)

    def _raw_delete(self, key: str) -> bool:
        if self.is_redis:
            try:
                return bool(self._redis.delete(key))
            except (RedisError, RedisConnectionError, OSError) as e:
                self._fallback_to_memory(reason=str(e))
        return self._memory.delete(key)

    def _fallback_to_memory(self, reason: str) -> None:
        """Switch to memory backend after Redis failure."""
        self.metrics.fallback_count += 1
        if not self._memory_only_warning_logged:
            logger.warning(
                "Vidyut cache: Redis operation failed (%s) — degrading to in-memory LRU",
                reason,
            )
            self._memory_only_warning_logged = True
        self._backend = CacheBackend.MEMORY
        self._redis = None

    # -----------------------------------------------------------------------
    # Internal: serialization
    # -----------------------------------------------------------------------
    @staticmethod
    def _serialize(value: Any) -> bytes:
        """
        Serialize using pickle (handles numpy arrays, pandas, sklearn outputs).
        We trust our own cache (private network) so pickle is safe here.
        """
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        return pickle.loads(raw)

    # -----------------------------------------------------------------------
    # Internal: TTL inference from key namespace
    # -----------------------------------------------------------------------
    @staticmethod
    def _infer_ttl(key: str) -> int:
        """Infer TTL from the namespace component of a Vidyut key."""
        # Key format: vidyut:<namespace>:v<ver>:<entity>:<hash>
        try:
            parts = key.split(":", 3)
            if len(parts) >= 2 and parts[0] == "vidyut":
                return DEFAULT_TTL_BY_NAMESPACE.get(parts[1], DEFAULT_TTL_FALLBACK)
        except Exception:
            pass
        return DEFAULT_TTL_FALLBACK

    # -----------------------------------------------------------------------
    # Context manager support
    # -----------------------------------------------------------------------
    def close(self) -> None:
        """Close Redis connection pool gracefully."""
        if self._redis is not None:
            try:
                self._redis.close()
                if hasattr(self._redis, "connection_pool"):
                    self._redis.connection_pool.disconnect()
            except Exception as e:
                logger.debug("Cache close error: %s", e)
        self._memory.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Sentinel for missing-value detection in get_or_compute
_SENTINEL = object()


# ===========================================================================
# Singleton accessor — reads config from env vars
# ===========================================================================
_cache_instance: Optional[RedisCache] = None
_cache_lock = threading.Lock()


def get_cache(force_new: bool = False) -> RedisCache:
    """
    Returns the global RedisCache singleton, initialized from environment vars.
    Thread-safe. Use this in API routes, inference, training scripts.

    Environment variables read:
        REDIS_HOST           (default: localhost)
        REDIS_PORT           (default: 6379)
        REDIS_DB             (default: 0)
        REDIS_PASSWORD       (default: None)
        CACHE_ENABLED        (default: true)
        CACHE_MEMORY_SIZE    (default: 10000)
        MODEL_VERSION        (default: v2)
    """
    global _cache_instance
    if _cache_instance is not None and not force_new:
        return _cache_instance

    with _cache_lock:
        if _cache_instance is not None and not force_new:
            return _cache_instance

        enabled = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD") or None
        memory_size = int(os.getenv("CACHE_MEMORY_SIZE", "10000"))
        model_version = os.getenv("MODEL_VERSION", "v2")

        _cache_instance = RedisCache(
            host=host,
            port=port,
            db=db,
            password=password,
            enabled=enabled,
            memory_fallback_size=memory_size,
            key_builder=CacheKeyBuilder(model_version=model_version),
        )
    return _cache_instance


def reset_cache() -> None:
    """Drop the singleton (mainly for tests)."""
    global _cache_instance
    with _cache_lock:
        if _cache_instance is not None:
            _cache_instance.close()
        _cache_instance = None


# ===========================================================================
# Decorator: cache the output of any function
# ===========================================================================
def cached(
    key_fn: Callable[..., str],
    ttl: Optional[int] = None,
):
    """
    Decorator to cache a function's output using a key derived from its args.

    Example:
        @cached(
            key_fn=lambda consumer_id, date: get_cache().key_builder.theft_score_key(consumer_id, date),
            ttl=3600,
        )
        def predict_theft(consumer_id: str, date: str) -> float:
            return model.predict(...)
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            try:
                key = key_fn(*args, **kwargs)
            except Exception as e:
                logger.debug("cached() key_fn failed (%s) — bypassing cache", e)
                return fn(*args, **kwargs)
            return cache.get_or_compute(
                key=key,
                compute_fn=lambda: fn(*args, **kwargs),
                ttl=ttl,
            )
        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapper
    return decorator
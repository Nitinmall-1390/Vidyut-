"""
===========================================================================
VIDYUT Cache Module
===========================================================================
Redis-based caching layer for Vidyut. Provides high-performance caching for:
  - Model predictions (demand forecast, theft scores)
  - SHAP explanations (expensive to compute)
  - Feature vectors (avoid recomputation)
  - API responses (rate-limit-friendly)

Falls back gracefully to in-memory LRU cache if Redis is unavailable,
ensuring the system never breaks due to cache layer failure.

Author: Vidyut Team
License: MIT
===========================================================================
"""

from src.cache.redis_cache import RedisCache, get_cache, CacheBackend
from src.cache.cache_key_builder import CacheKeyBuilder, CacheNamespace

__all__ = [
    "RedisCache",
    "get_cache",
    "CacheBackend",
    "CacheKeyBuilder",
    "CacheNamespace",
]
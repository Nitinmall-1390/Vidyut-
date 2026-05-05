"""
===========================================================================
VIDYUT Cache Key Builder
===========================================================================
Constructs deterministic, collision-free cache keys following a strict
namespacing convention:

    vidyut:<namespace>:<version>:<entity>:<hash>

Why deterministic hashing matters:
  - Same input → same key → cache hit (no duplicate computation)
  - Version bump → automatic invalidation of stale cache
  - Namespacing prevents collisions across modules

Author: Vidyut Team
License: MIT
===========================================================================
"""

import hashlib
import json
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ===========================================================================
# Cache Namespaces — strict enum prevents typos in cache keys
# ===========================================================================
class CacheNamespace(str, Enum):
    """Cache namespaces for different data types in Vidyut."""

    DEMAND_FORECAST = "demand_forecast"
    THEFT_SCORE = "theft_score"
    ANOMALY_SCORE = "anomaly_score"
    SHAP_EXPLANATION = "shap_explanation"
    FEATURE_VECTOR = "feature_vector"
    RULE_FLAGS = "rule_flags"
    CONFIDENCE_SCORE = "confidence_score"
    NETWORK_RING = "network_ring"
    WEATHER_DATA = "weather_data"
    API_RESPONSE = "api_response"
    MODEL_METADATA = "model_metadata"


# ===========================================================================
# Cache Key Builder
# ===========================================================================
class CacheKeyBuilder:
    """
    Builds standardized cache keys for Vidyut.

    Key format:
        vidyut:<namespace>:v<version>:<entity_id>:<param_hash>

    Examples:
        vidyut:demand_forecast:v2:feeder_BNG_F001:a3f8c9
        vidyut:theft_score:v2:consumer_C12345:b7e1d4
        vidyut:shap_explanation:v2:consumer_C12345:c9a2f1
    """

    PREFIX = "vidyut"
    HASH_LENGTH = 12  # 12 hex chars = 48 bits, collision prob ≈ 0 for our scale

    def __init__(self, model_version: str = "v2"):
        """
        Args:
            model_version: Model version tag (e.g., "v1", "v2"). Bumping this
                           invalidates all cached predictions automatically.
        """
        if not model_version.startswith("v"):
            model_version = f"v{model_version}"
        self.model_version = model_version

    # -----------------------------------------------------------------------
    # Public builders — one per namespace for clarity
    # -----------------------------------------------------------------------
    def demand_forecast_key(
        self,
        feeder_id: str,
        forecast_start: Union[str, datetime, date],
        horizon_hours: int,
        weather_features: Optional[Dict[str, float]] = None,
    ) -> str:
        """Build cache key for demand forecast result."""
        params = {
            "start": self._normalize_datetime(forecast_start),
            "horizon": int(horizon_hours),
            "weather": self._normalize_dict(weather_features or {}),
        }
        return self._build(
            namespace=CacheNamespace.DEMAND_FORECAST,
            entity_id=self._sanitize(feeder_id),
            params=params,
        )

    def theft_score_key(
        self,
        consumer_id: str,
        evaluation_date: Union[str, datetime, date],
        feature_hash: Optional[str] = None,
    ) -> str:
        """Build cache key for theft probability score."""
        params = {
            "date": self._normalize_datetime(evaluation_date),
            "feat": feature_hash or "default",
        }
        return self._build(
            namespace=CacheNamespace.THEFT_SCORE,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def anomaly_score_key(
        self,
        consumer_id: str,
        sequence_end_date: Union[str, datetime, date],
        sequence_length_days: int = 14,
    ) -> str:
        """Build cache key for anomaly score (LSTM-AE + IsoForest intersection)."""
        params = {
            "end": self._normalize_datetime(sequence_end_date),
            "seq_len": int(sequence_length_days),
        }
        return self._build(
            namespace=CacheNamespace.ANOMALY_SCORE,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def shap_explanation_key(
        self,
        consumer_id: str,
        model_name: str,
        feature_hash: str,
    ) -> str:
        """Build cache key for SHAP explanation (expensive — high TTL)."""
        params = {
            "model": self._sanitize(model_name),
            "feat": feature_hash,
        }
        return self._build(
            namespace=CacheNamespace.SHAP_EXPLANATION,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def feature_vector_key(
        self,
        consumer_id: str,
        as_of_date: Union[str, datetime, date],
        feature_set: str = "default",
    ) -> str:
        """Build cache key for engineered feature vector."""
        params = {
            "date": self._normalize_datetime(as_of_date),
            "set": self._sanitize(feature_set),
        }
        return self._build(
            namespace=CacheNamespace.FEATURE_VECTOR,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def rule_flags_key(
        self,
        consumer_id: str,
        evaluation_date: Union[str, datetime, date],
    ) -> str:
        """Build cache key for rule-based flag results."""
        params = {"date": self._normalize_datetime(evaluation_date)}
        return self._build(
            namespace=CacheNamespace.RULE_FLAGS,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def confidence_score_key(
        self,
        consumer_id: str,
        prediction_id: str,
    ) -> str:
        """Build cache key for confidence score (0-100)."""
        params = {"pred": self._sanitize(prediction_id)}
        return self._build(
            namespace=CacheNamespace.CONFIDENCE_SCORE,
            entity_id=self._sanitize(consumer_id),
            params=params,
        )

    def network_ring_key(
        self,
        community_id: str,
        snapshot_date: Union[str, datetime, date],
    ) -> str:
        """Build cache key for theft ring detection result."""
        params = {"date": self._normalize_datetime(snapshot_date)}
        return self._build(
            namespace=CacheNamespace.NETWORK_RING,
            entity_id=self._sanitize(community_id),
            params=params,
        )

    def weather_data_key(
        self,
        latitude: float,
        longitude: float,
        date_value: Union[str, datetime, date],
    ) -> str:
        """Build cache key for NASA POWER weather data (rounded to 0.5 deg grid)."""
        # Round lat/lon to NASA POWER's 0.5° resolution to maximize cache hits
        lat_rounded = round(latitude * 2) / 2
        lon_rounded = round(longitude * 2) / 2
        params = {
            "lat": lat_rounded,
            "lon": lon_rounded,
            "date": self._normalize_datetime(date_value),
        }
        entity = f"grid_{lat_rounded:.1f}_{lon_rounded:.1f}"
        return self._build(
            namespace=CacheNamespace.WEATHER_DATA,
            entity_id=entity,
            params=params,
        )

    def api_response_key(
        self,
        endpoint: str,
        query_params: Dict[str, Any],
    ) -> str:
        """Build cache key for full API response."""
        return self._build(
            namespace=CacheNamespace.API_RESPONSE,
            entity_id=self._sanitize(endpoint.strip("/").replace("/", "_")),
            params=self._normalize_dict(query_params),
        )

    def model_metadata_key(self, model_name: str) -> str:
        """Build cache key for model metadata (loaded once per model version)."""
        return self._build(
            namespace=CacheNamespace.MODEL_METADATA,
            entity_id=self._sanitize(model_name),
            params={},
        )

    # -----------------------------------------------------------------------
    # Generic builder (escape hatch for custom keys)
    # -----------------------------------------------------------------------
    def custom_key(
        self,
        namespace: Union[CacheNamespace, str],
        entity_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a custom cache key — use sparingly, prefer typed builders."""
        ns_value = namespace.value if isinstance(namespace, CacheNamespace) else namespace
        return self._build(
            namespace=ns_value,
            entity_id=self._sanitize(entity_id),
            params=params or {},
        )

    # -----------------------------------------------------------------------
    # Internal: deterministic hashing & normalization
    # -----------------------------------------------------------------------
    def _build(
        self,
        namespace: Union[CacheNamespace, str],
        entity_id: str,
        params: Dict[str, Any],
    ) -> str:
        """Assemble final cache key with deterministic hash of params."""
        ns_value = namespace.value if isinstance(namespace, CacheNamespace) else namespace
        param_hash = self._hash_params(params)
        return f"{self.PREFIX}:{ns_value}:{self.model_version}:{entity_id}:{param_hash}"

    def _hash_params(self, params: Dict[str, Any]) -> str:
        """
        Deterministically hash params dict.
        Uses sorted JSON to guarantee same dict → same hash, regardless of key order.
        """
        if not params:
            return "0" * self.HASH_LENGTH

        normalized = self._normalize_dict(params)
        canonical = json.dumps(normalized, sort_keys=True, default=self._json_default)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[: self.HASH_LENGTH]

    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize dict values for stable hashing."""
        result = {}
        for k, v in sorted(d.items()):
            result[str(k)] = self._normalize_value(v)
        return result

    def _normalize_value(self, v: Any) -> Any:
        """Convert value to a JSON-serializable, hash-stable form."""
        if v is None:
            return None
        if isinstance(v, (str, bool)):
            return v
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            # Round to 6 decimals to avoid float-precision cache misses
            if np.isnan(v):
                return "NaN"
            return round(float(v), 6)
        if isinstance(v, (datetime, date, pd.Timestamp)):
            return self._normalize_datetime(v)
        if isinstance(v, dict):
            return self._normalize_dict(v)
        if isinstance(v, (list, tuple, set)):
            return [self._normalize_value(x) for x in v]
        if isinstance(v, np.ndarray):
            return [self._normalize_value(x) for x in v.tolist()]
        # Fallback: stringify
        return str(v)

    @staticmethod
    def _normalize_datetime(dt: Union[str, datetime, date, pd.Timestamp]) -> str:
        """Normalize datetime-like to ISO-8601 string (date-only granularity by default)."""
        if isinstance(dt, str):
            return dt
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if isinstance(dt, datetime):
            return dt.replace(microsecond=0).isoformat()
        if isinstance(dt, date):
            return dt.isoformat()
        return str(dt)

    @staticmethod
    def _sanitize(value: str) -> str:
        """Sanitize entity ID — remove characters that break Redis key conventions."""
        if value is None:
            return "none"
        s = str(value).strip()
        # Replace problematic chars; keep alphanumeric, underscore, hyphen, dot
        return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)

    @staticmethod
    def _json_default(o: Any) -> Any:
        """Fallback JSON serializer for non-standard types."""
        if isinstance(o, (datetime, date, pd.Timestamp)):
            return o.isoformat() if hasattr(o, "isoformat") else str(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # -----------------------------------------------------------------------
    # Pattern builders for bulk invalidation
    # -----------------------------------------------------------------------
    def namespace_pattern(self, namespace: Union[CacheNamespace, str]) -> str:
        """Build a glob pattern for invalidating all keys in a namespace."""
        ns_value = namespace.value if isinstance(namespace, CacheNamespace) else namespace
        return f"{self.PREFIX}:{ns_value}:*"

    def version_pattern(self, namespace: Union[CacheNamespace, str]) -> str:
        """Build a glob pattern for invalidating all keys in a namespace+version."""
        ns_value = namespace.value if isinstance(namespace, CacheNamespace) else namespace
        return f"{self.PREFIX}:{ns_value}:{self.model_version}:*"

    def entity_pattern(
        self,
        namespace: Union[CacheNamespace, str],
        entity_id: str,
    ) -> str:
        """Build a glob pattern for invalidating all keys for a specific entity."""
        ns_value = namespace.value if isinstance(namespace, CacheNamespace) else namespace
        return f"{self.PREFIX}:{ns_value}:{self.model_version}:{self._sanitize(entity_id)}:*"

    def all_vidyut_pattern(self) -> str:
        """Build a glob pattern matching every Vidyut cache key (use with care!)."""
        return f"{self.PREFIX}:*"

    # -----------------------------------------------------------------------
    # Convenience: fingerprint a feature vector → reusable hash
    # -----------------------------------------------------------------------
    def fingerprint_features(
        self,
        features: Union[Dict[str, Any], pd.Series, np.ndarray, List[float]],
    ) -> str:
        """
        Generate a stable short hash of a feature vector. Use this when feature
        values directly determine the cached prediction (e.g., for SHAP).
        """
        if isinstance(features, pd.Series):
            features = features.to_dict()
        elif isinstance(features, np.ndarray):
            features = features.tolist()
        elif isinstance(features, list):
            features = {f"f{i}": v for i, v in enumerate(features)}

        if not isinstance(features, dict):
            features = {"value": features}

        return self._hash_params(features)
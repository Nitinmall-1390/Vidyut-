"""
VIDYUT Model Registry / Versioning
===========================================================================
Lightweight file-based model registry. Manages versioned model artefacts
in data/models/v{N}/ with a LATEST symlink pointing to the current best.
===========================================================================
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import get_settings
from src.utils.logger import get_logger

log = get_logger("vidyut.versioning")
settings = get_settings()


class ModelVersion:
    """Metadata container for a versioned model artefact."""

    def __init__(
        self,
        model_name: str,
        version: str,
        task: str,
        metrics: Dict[str, float],
        artefact_path: Path,
        notes: str = "",
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.task = task
        self.metrics = metrics
        self.artefact_path = Path(artefact_path)
        self.notes = notes
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "task": self.task,
            "metrics": self.metrics,
            "artefact_path": str(self.artefact_path),
            "notes": self.notes,
            "created_at": self.created_at,
        }


class ModelRegistry:
    """
    File-system model registry.

    Structure:
      data/models/
        v1/
          model_metadata.json
          demand_FEEDER_000_ensemble_meta.joblib
          …
        v2/
          model_metadata.json
          …
        LATEST -> v2/   (symlink or JSON pointer on Windows)

    Usage:
        registry = ModelRegistry()
        registry.register(version, metadata)
        registry.promote_to_latest(version)
        latest_dir = registry.get_latest_dir()
    """

    METADATA_FILE = "model_metadata.json"

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir or settings.data_models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        return self.models_dir / version

    def list_versions(self) -> List[str]:
        """Return all version directory names sorted alphabetically."""
        return sorted(
            d.name for d in self.models_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def register(
        self,
        version: str,
        model_versions: List[ModelVersion],
    ) -> Path:
        """
        Register a new model version with its metadata.

        Parameters
        ----------
        version : str
            e.g. "v1", "v2"
        model_versions : list of ModelVersion
            All artefacts belonging to this version.

        Returns
        -------
        Path to the version directory.
        """
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "models": [mv.to_dict() for mv in model_versions],
        }
        metadata_path = version_dir / self.METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        log.info("Registered model version '%s' at %s", version, version_dir)
        return version_dir

    def get_metadata(self, version: str) -> Dict:
        """Load metadata JSON for a given version."""
        metadata_path = self._version_dir(version) / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata for version '{version}'.")
        with open(metadata_path) as f:
            return json.load(f)

    def promote_to_latest(self, version: str) -> None:
        """
        Update the LATEST pointer to the given version.
        Uses a JSON file as the pointer (cross-platform; avoids symlink issues).
        """
        version_dir = self._version_dir(version)
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")

        latest_file = self.models_dir / "LATEST.json"
        with open(latest_file, "w") as f:
            json.dump({"version": version}, f)

        # Also attempt a symlink for Unix environments
        symlink_path = self.models_dir / "LATEST"
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(version_dir.resolve())
        except (OSError, NotImplementedError):
            pass  # Windows without admin privileges — use JSON pointer instead

        log.info("LATEST → %s", version)

    def get_latest_dir(self) -> Path:
        """Return the Path to the LATEST version directory."""
        latest_file = self.models_dir / "LATEST.json"
        if latest_file.exists():
            with open(latest_file) as f:
                data = json.load(f)
            return self._version_dir(data["version"])

        # Fallback: last version alphabetically
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError("No model versions registered.")
        return self._version_dir(versions[-1])

    def get_latest_version_name(self) -> str:
        latest_dir = self.get_latest_dir()
        return latest_dir.name

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
        metric: str = "mape",
    ) -> Dict:
        """
        Compare two versions on a specific metric.
        Lower is better for error metrics, higher for quality metrics.
        """
        meta_a = self.get_metadata(version_a)
        meta_b = self.get_metadata(version_b)

        def _extract_metric(meta: Dict) -> Optional[float]:
            for m in meta.get("models", []):
                val = m.get("metrics", {}).get(metric)
                if val is not None:
                    return float(val)
            return None

        val_a = _extract_metric(meta_a)
        val_b = _extract_metric(meta_b)

        return {
            "metric": metric,
            version_a: val_a,
            version_b: val_b,
            "winner": (
                version_a if (val_a is not None and val_b is not None and val_a < val_b)
                else version_b
            ),
        }

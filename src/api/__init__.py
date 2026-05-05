"""
===========================================================================
VIDYUT API Module
===========================================================================
FastAPI service exposing Vidyut models to BESCOM systems and the dashboard.
Endpoints: /demand, /anomaly, /theft, /explain, /healthz, /metrics.

Author: Vidyut Team
License: MIT
===========================================================================
"""

from src.api.app import create_app, app

__all__ = ["create_app", "app"]
"""
===========================================================================
VIDYUT FastAPI Application Factory
===========================================================================
Production-ready FastAPI app with:
  - CORS for dashboard
  - GZip compression
  - Structured request logging
  - Auth + rate-limit middleware
  - /healthz, /readyz, /metrics
  - OpenAPI docs at /docs and /redoc

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import anomaly, demand, explain, theft
from src.audit.logger import get_audit_logger
from src.cache.redis_cache import get_cache
from src.models.inference.model_loader import get_model_loader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("Vidyut API starting up...")
    # Warm up singletons
    cache = get_cache()
    loader = get_model_loader()
    audit = get_audit_logger()
    app.state.cache = cache
    app.state.model_loader = loader
    app.state.audit = audit
    app.state.started_at = time.time()
    logger.info("Vidyut API ready. Cache backend=%s", cache.backend.value)
    yield
    logger.info("Vidyut API shutting down...")
    try:
        cache.close()
    except Exception:
        pass


def create_app(
    enable_auth: bool = True,
    enable_rate_limit: bool = True,
) -> FastAPI:
    """Application factory — used by Uvicorn entrypoint and tests."""
    app = FastAPI(
        title="Vidyut API",
        description=(
            "AI-Powered Smart Meter Intelligence & Theft Detection for BESCOM. "
            "All endpoints return explainable, auditable predictions."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ---- Middleware (order matters: outer → inner) ----
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "120")),
        )
    if enable_auth:
        app.add_middleware(
            AuthMiddleware,
            api_key=os.getenv("VIDYUT_API_KEY", ""),
            exempt_paths=("/healthz", "/readyz", "/docs", "/redoc",
                          "/openapi.json", "/metrics"),
        )

    # ---- Request ID + access log ----
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            logger.exception(
                "Unhandled error req_id=%s path=%s elapsed_ms=%.1f",
                request_id, request.url.path, elapsed,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": str(exc),
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        elapsed_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"

        try:
            actor = getattr(request.state, "actor", "anonymous")
            get_audit_logger().log_api_call(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                latency_ms=elapsed_ms,
                actor=actor,
                request_id=request_id,
            )
        except Exception as e:
            logger.debug("audit on api call failed: %s", e)

        logger.info(
            "%s %s %d %.1fms id=%s",
            request.method, request.url.path,
            response.status_code, elapsed_ms, request_id,
        )
        return response

    # ---- Routers ----
    app.include_router(demand.router, prefix="/api/v1/demand", tags=["demand"])
    app.include_router(anomaly.router, prefix="/api/v1/anomaly", tags=["anomaly"])
    app.include_router(theft.router, prefix="/api/v1/theft", tags=["theft"])
    app.include_router(explain.router, prefix="/api/v1/explain", tags=["explain"])

    # ---- Health endpoints ----
    @app.get("/healthz", tags=["meta"])
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "service": "vidyut-api", "version": app.version}

    @app.get("/readyz", tags=["meta"])
    async def readyz(request: Request) -> Dict[str, Any]:
        cache_health = request.app.state.cache.health_check()
        return {
            "ready": True,
            "uptime_seconds": round(time.time() - request.app.state.started_at, 1),
            "cache": cache_health,
            "models_loaded": request.app.state.model_loader.list_loaded(),
        }

    @app.get("/metrics", response_class=PlainTextResponse, tags=["meta"])
    async def metrics(request: Request) -> str:
        cache_metrics = request.app.state.cache.get_metrics()
        lines = [
            "# HELP vidyut_cache_hits_total Cache hits",
            "# TYPE vidyut_cache_hits_total counter",
            f"vidyut_cache_hits_total {cache_metrics['hits']}",
            "# HELP vidyut_cache_misses_total Cache misses",
            "# TYPE vidyut_cache_misses_total counter",
            f"vidyut_cache_misses_total {cache_metrics['misses']}",
            "# HELP vidyut_cache_hit_rate Cache hit rate",
            "# TYPE vidyut_cache_hit_rate gauge",
            f"vidyut_cache_hit_rate {cache_metrics['hit_rate']}",
            "# HELP vidyut_uptime_seconds API uptime",
            "# TYPE vidyut_uptime_seconds gauge",
            f"vidyut_uptime_seconds {time.time() - request.app.state.started_at:.1f}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/", tags=["meta"])
    async def root() -> Dict[str, Any]:
        return {
            "service": "Vidyut API",
            "version": app.version,
            "docs": "/docs",
            "endpoints": [
                "/api/v1/demand", "/api/v1/anomaly",
                "/api/v1/theft", "/api/v1/explain",
            ],
        }

    return app


# Module-level instance for `uvicorn src.api.app:app`
app = create_app()
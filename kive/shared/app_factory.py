"""
Shared FastAPI app factory for all KIVE signal services.
Each service's main.py calls build_app() with its detector class and service config.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Type

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import (
    FlagDetail,
    HealthResponse,
    ProbeSuggestion,
    SignalRequest,
    SignalResponse,
)


def build_app(
    service_name: str,       # "TAV" | "SVP" | "FMD" | "MDC" | "TSI" | "BES" | "LQA" | "CCS" | "RSL"
    service_version: str,
    weight: float,
    detector_class: type,
    detector_kwargs: Optional[dict] = None,
) -> FastAPI:
    """
    Factory that returns a configured FastAPI application for a signal service.
    Handles: lifespan, health endpoint, signal endpoint, error handling.
    """
    _start_time = time.time()
    _detector = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _detector
        _detector = detector_class(**(detector_kwargs or {}))
        await _detector.initialize()
        yield
        await _detector.close()

    app = FastAPI(
        title=f"KIVE {service_name} Signal Service",
        description=f"KIVE {service_name} fraud detection signal — weight {weight}",
        version=service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "signal": service_name,
                "code": "INTERNAL_ERROR",
                "detail": str(exc),
            },
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=service_name,
            version=service_version,
            uptime_seconds=round(time.time() - _start_time, 1),
        )

    @app.post(
        f"/api/v1/signals/{service_name.lower()}",
        response_model=SignalResponse,
        tags=["signals"],
    )
    async def detect(request: SignalRequest) -> SignalResponse:
        t0 = time.time()
        try:
            result = await _detector.analyze(request)
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={"signal": service_name, "code": "INVALID_INPUT", "detail": str(e)},
            )

        # Unpack result dataclass into SignalResponse
        flags = getattr(result, "flags", [])
        probe = getattr(result, "probe_suggestion", None)

        return SignalResponse(
            signal=service_name,
            score=result.score,
            confidence=result.confidence,
            weight=weight,
            flags=flags if flags else [],
            probe_suggestion=probe,
            latency_ms=int((time.time() - t0) * 1000),
            model_version=f"{service_name.lower()}-v{service_version}",
        )

    return app

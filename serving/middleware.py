"""HTTP middleware: request-id, security headers, metrics, OTel."""
from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import record_http
from ml_sentinel.core.settings import get_settings
from ml_sentinel.core.utils import new_request_id

logger = get_logger("serving.middleware")
settings = get_settings()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = req_id
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            record_http(request.method, request.url.path, 500, elapsed)
            logger.exception("request_failed", request_id=req_id,
                             method=request.method, path=request.url.path,
                             latency_ms=int(elapsed * 1000))
            raise
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = req_id
        record_http(request.method, request.url.path, response.status_code, elapsed)
        logger.info("http_request", request_id=req_id, method=request.method,
                    path=request.url.path, status=response.status_code,
                    latency_ms=int(elapsed * 1000),
                    client=request.client.host if request.client else "unknown")
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains; preload",
            )
        return response

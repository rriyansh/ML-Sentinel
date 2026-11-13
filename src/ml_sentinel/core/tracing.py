"""
OpenTelemetry tracing initialization.

Sets up the OTLP exporter + FastAPI/SQLAlchemy/requests auto-instrumentation.
Uses W3C Trace Context so spans correlate across Airflow → MLflow → serving.
"""
from __future__ import annotations

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)
_DONE = False


def setup_tracing(app=None) -> None:
    global _DONE
    if _DONE:
        return

    settings = get_settings()
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    except ImportError:
        logger.warning("otel_not_installed_skipping")
        return

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.app_version,
        "deployment.environment": settings.environment,
    })
    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(settings.otel_traces_sampler_arg),
    )
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # FastAPI instrumentation
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
        except ImportError:
            pass

    # SQLAlchemy
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        from ml_sentinel.infrastructure.database import get_engine

        SQLAlchemyInstrumentor().instrument(engine=get_engine())
    except ImportError:
        pass

    # HTTPX (outbound)
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument()
    except ImportError:
        pass

    _DONE = True
    logger.info("tracing_initialized",
                endpoint=settings.otel_exporter_otlp_endpoint,
                sample_rate=settings.otel_traces_sampler_arg)


def get_tracer(name: str = "ml-sentinel"):
    from opentelemetry import trace

    return trace.get_tracer(name)

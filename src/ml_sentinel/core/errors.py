"""Domain-specific exceptions with HTTP status mapping."""
from __future__ import annotations


class MLSentinelError(Exception):
    """Base exception."""
    http_status = 500
    error_code = "internal_error"


class ConfigurationError(MLSentinelError):
    http_status = 500
    error_code = "configuration_error"


class DataValidationError(MLSentinelError):
    http_status = 422
    error_code = "data_validation_error"


class ModelNotFoundError(MLSentinelError):
    http_status = 404
    error_code = "model_not_found"


class ModelLoadError(MLSentinelError):
    http_status = 503
    error_code = "model_load_error"


class DriftCheckError(MLSentinelError):
    http_status = 500
    error_code = "drift_check_error"


class QualityGateError(MLSentinelError):
    http_status = 409
    error_code = "quality_gate_error"


class AuthenticationError(MLSentinelError):
    http_status = 401
    error_code = "authentication_error"


class AuthorizationError(MLSentinelError):
    http_status = 403
    error_code = "authorization_error"


class RateLimitError(MLSentinelError):
    http_status = 429
    error_code = "rate_limit_error"


class CircuitOpenError(MLSentinelError):
    http_status = 503
    error_code = "circuit_open"

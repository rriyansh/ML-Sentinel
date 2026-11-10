"""Domain-specific exceptions."""

from __future__ import annotations


class MLSentinelError(Exception):
    """Base exception for all ML-Sentinel errors."""


class ConfigurationError(MLSentinelError):
    """Raised when runtime configuration is invalid."""


class DataValidationError(MLSentinelError):
    """Raised when input data fails schema or quality validation."""


class ModelNotFoundError(MLSentinelError):
    """Raised when the requested model version cannot be located."""


class ModelLoadError(MLSentinelError):
    """Raised when a model fails to load from the registry."""


class DriftCheckError(MLSentinelError):
    """Raised when drift detection cannot be executed."""


class QualityGateError(MLSentinelError):
    """Raised when a challenger fails the promotion gate."""


class AuthenticationError(MLSentinelError):
    """Raised when a request cannot be authenticated."""


class RateLimitError(MLSentinelError):
    """Raised when a client exceeds its rate limit."""

"""
Pluggable secret backend:
- ``env``   : read from environment (default)
- ``vault`` : HashiCorp Vault (KV v2)
- ``aws``   : AWS Secrets Manager
- ``gcp``   : GCP Secret Manager

The facade caches resolved values per process to avoid repeated calls.
"""
from __future__ import annotations

from functools import lru_cache

from ml_sentinel.core.errors import ConfigurationError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)


class SecretsBackend:
    def get(self, key: str) -> str:
        raise NotImplementedError


class EnvBackend(SecretsBackend):
    def get(self, key: str) -> str:
        import os

        val = os.getenv(key)
        if val is None:
            raise ConfigurationError(f"Missing env var: {key}")
        return val


class VaultBackend(SecretsBackend):
    def __init__(self) -> None:
        import hvac

        s = get_settings()
        if not s.vault_addr or not s.vault_token.get_secret_value():
            raise ConfigurationError("Vault backend requires VAULT_ADDR + VAULT_TOKEN")
        self._client = hvac.Client(url=s.vault_addr, token=s.vault_token.get_secret_value())
        self._path = s.vault_secret_path

    def get(self, key: str) -> str:
        try:
            resp = self._client.secrets.kv.v2.read_secret_version(path=self._path)
            return resp["data"]["data"][key]
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Vault read failed for {key}: {exc}") from exc


class AWSBackend(SecretsBackend):
    def __init__(self) -> None:
        import boto3

        self._client = boto3.client("secretsmanager")

    def get(self, key: str) -> str:
        import json

        try:
            resp = self._client.get_secret_value(SecretId=key)
            if "SecretString" in resp:
                try:
                    return json.loads(resp["SecretString"])["value"]
                except (json.JSONDecodeError, KeyError):
                    return resp["SecretString"]
            raise ConfigurationError(f"No SecretString for {key}")
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"AWS Secrets Manager read failed for {key}: {exc}") from exc


class GCPBackend(SecretsBackend):
    def __init__(self) -> None:
        from google.cloud import secretmanager

        self._client = secretmanager.SecretManagerServiceClient()

    def get(self, key: str) -> str:
        try:
            name = f"projects/{key.split('/')[0]}/secrets/{key.split('/')[1]}/versions/latest"
            resp = self._client.access_secret_version(request={"name": name})
            return resp.payload.data.decode()
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"GCP Secret Manager read failed for {key}: {exc}") from exc


def _factory() -> SecretsBackend:
    backend = get_settings().secrets_backend
    if backend == "env":
        return EnvBackend()
    if backend == "vault":
        return VaultBackend()
    if backend == "aws":
        return AWSBackend()
    if backend == "gcp":
        return GCPBackend()
    raise ConfigurationError(f"Unknown secrets backend: {backend}")


@lru_cache(maxsize=256)
def get_secret(key: str) -> str:
    logger.info("secret_resolved", key=key, backend=get_settings().secrets_backend)
    return _factory().get(key)

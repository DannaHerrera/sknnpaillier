"""Synthetic fixtures configured independently of examples and shell cwd."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
    resolve_key_bits,
)


def load_test_settings() -> tuple[int, int, int | None, str]:
    """Load tests/.env with shell precedence and validate fixture settings."""
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    bits = resolve_key_bits(os.getenv("SECURITY_LEVEL", "test"))
    try:
        capacity = int(os.getenv("SKNN_TEST_LEAF_CAPACITY", "2"))
        raw_depth = os.getenv("SKNN_TEST_MAX_DEPTH", "").strip()
        depth = int(raw_depth) if raw_depth else None
    except ValueError as error:
        raise ValueError("Parámetros del árbol de prueba: enteros requeridos.") from error
    if capacity < 1 or (depth is not None and depth < 0):
        raise ValueError("Capacidad >= 1 y profundidad >= 0 requeridas.")
    disabled = os.getenv("SKNN_LOG_DISABLED", "1").lower()
    if disabled not in ("0", "1", "true", "false", "yes", "no"):
        raise ValueError("SKNN_LOG_DISABLED debe ser una bandera booleana.")
    return bits, capacity, depth, disabled


@pytest.fixture(scope="session")
def suite_settings():
    return load_test_settings()


@pytest.fixture
def settings_loader():
    return load_test_settings


@pytest.fixture(autouse=True)
def configure_logging(monkeypatch, suite_settings):
    monkeypatch.setenv("SKNN_LOG_DISABLED", suite_settings[3])


@pytest.fixture(scope="session")
def key_manager(suite_settings):
    with patch.dict("os.environ", {"SKNN_LOG_DISABLED": suite_settings[3]}):
        return KeyManagementModule(suite_settings[0])


@pytest.fixture
def transformer(key_manager):
    return SecureDataTransformationModule(key_manager.public_key)


@pytest.fixture
def cloud(transformer, key_manager):
    instance = CloudBQueryProcessingModule(transformer, key_manager)
    transformer.set_cloud(instance)
    return instance

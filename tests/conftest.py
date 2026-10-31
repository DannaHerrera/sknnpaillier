"""Small synthetic fixtures; keys are deliberately for tests only."""

from unittest.mock import patch

import pytest

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
)


@pytest.fixture(autouse=True)
def disable_logging(monkeypatch):
    monkeypatch.setenv("SKNN_LOG_DISABLED", "1")


@pytest.fixture(scope="session")
def key_manager():
    with patch.dict("os.environ", {"SKNN_LOG_DISABLED": "1"}):
        return KeyManagementModule(128)


@pytest.fixture
def transformer(key_manager):
    return SecureDataTransformationModule(key_manager.public_key)


@pytest.fixture
def cloud(transformer, key_manager):
    instance = CloudBQueryProcessingModule(transformer, key_manager)
    transformer.set_cloud(instance)
    return instance

import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from sknnpaillier import resolve_key_bits


@pytest.mark.parametrize("level, bits", [
    ("test", 128), (80, 1024), (112, 2048), (128, 3072),
    ("80", 1024), ("112", 2048), ("128", 3072),
])
def test_security_level(level, bits):
    assert resolve_key_bits(level) == bits


@pytest.mark.parametrize("level", [
    0, 256, 1024, 2048, 3072, -1, "standard", "legacy", "TEST", " test ",
    "", "0", "1024", "2048", "3072", "080", "80.0", 80.0, True, False, None, [], {},
])
def test_invalid_security_level(level):
    with pytest.raises(ValueError, match="SECURITY_LEVEL"):
        resolve_key_bits(level)


@pytest.fixture
def example_module(request, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / request.param
    loader = Mock(side_effect=AssertionError("dotenv loaded during import"))
    monkeypatch.setattr("dotenv.load_dotenv", loader)
    spec = importlib.util.spec_from_file_location("example_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader.assert_not_called()
    from dotenv.main import load_dotenv
    monkeypatch.setattr(module, "load_dotenv", load_dotenv)
    return module


def clear_configuration(monkeypatch):
    for key in list(os.environ):
        if key == "SECURITY_LEVEL" or key.startswith(
            ("SKNN_EXAMPLE_", "SKNN_TEST_", "SKNN_LOG_", "PYTHON_DOTENV_")
        ):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("example_module", [
    "basic_usage.py", "float_scaling_usage.py",
], indirect=True)
def test_example_local_env(example_module, tmp_path, monkeypatch):
    clear_configuration(monkeypatch)
    monkeypatch.setattr(example_module, "__file__", str(tmp_path / "example.py"))
    monkeypatch.chdir(tmp_path.parent)
    assert example_module.load_settings() == (1024, 3, 2, None)
    (tmp_path / ".env").write_text(
        "SECURITY_LEVEL=112\nSKNN_EXAMPLE_K=5\nSKNN_EXAMPLE_MAX_DEPTH=0\n"
    )
    assert example_module.load_settings() == (2048, 5, 2, 0)
    monkeypatch.setenv("SECURITY_LEVEL", "128")
    assert example_module.load_settings()[0] == 3072
    monkeypatch.setenv("SECURITY_LEVEL", "1024")
    with pytest.raises(ValueError, match="SECURITY_LEVEL"):
        example_module.load_settings()
    monkeypatch.setenv("SECURITY_LEVEL", "test")
    monkeypatch.setenv("SKNN_EXAMPLE_K", "2")
    with pytest.raises(ValueError, match="impar"):
        example_module.load_settings()


def test_suite_settings(settings_loader, tmp_path, monkeypatch):
    clear_configuration(monkeypatch)
    monkeypatch.setitem(settings_loader.__globals__, "__file__",
                        str(tmp_path / "conftest.py"))
    assert settings_loader() == (128, 2, None, "1")
    (tmp_path / ".env").write_text(
        "SECURITY_LEVEL=112\nSKNN_TEST_LEAF_CAPACITY=1\n"
    )
    assert settings_loader() == (2048, 1, None, "1")
    monkeypatch.setenv("SECURITY_LEVEL", "test")
    assert settings_loader()[0] == 128
    monkeypatch.setenv("SKNN_TEST_LEAF_CAPACITY", "0")
    with pytest.raises(ValueError, match="Capacidad"):
        settings_loader()


def test_configured_tree(cloud, transformer, key_manager, suite_settings):
    _, capacity, depth, _ = suite_settings
    transformer.send_database_to_cloud(
        np.array([[0, 0], [2, 2], [5, 5]]),
        leaf_capacity=capacity, max_depth=depth,
    )
    ids, _, distances = transformer.send_query_to_cloud(np.array([1, 1]), 1)
    assert ids == [0]
    assert [key_manager.decrypt(value) for value in distances] == [2]


@pytest.mark.parametrize("example_module", [
    "basic_usage.py", "float_scaling_usage.py",
], indirect=True)
def test_level_resolved_before_key_creation(
    example_module, tmp_path, monkeypatch,
):
    clear_configuration(monkeypatch)
    monkeypatch.setenv("SECURITY_LEVEL", "128")
    # Stop before cryptography: exercise the actual main without a large key.
    monkeypatch.setattr(example_module, "__file__", str(tmp_path / "example.py"))
    constructor = Mock(side_effect=RuntimeError("stop before key generation"))
    monkeypatch.setattr(example_module, "KeyManagementModule", constructor)
    with pytest.raises(RuntimeError, match="stop before key generation"):
        example_module.main()
    constructor.assert_called_once_with(3072)

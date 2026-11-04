"""Check fixed-point contracts and the relocated mathematical helpers."""

import importlib
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import sknnpaillier
from sknnpaillier import key_m, utils


@pytest.mark.parametrize("values", [
    [0.0, 0.125, 1.0],
    [[-1.25, 0.0], [0.375, 2.5]],
    np.empty((0, 2)),
])
def test_quantize_shape_type_and_values(values):
    original = np.asarray(values, dtype=float)
    result = utils.quantize(original, scale_factor=1000)
    assert result.shape == original.shape
    assert result.dtype == np.dtype(int)
    np.testing.assert_array_equal(result, np.rint(original * 1000).astype(int))


def test_quantize_half_to_even():
    values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(
        utils.quantize(values, scale_factor=1), [-2, -2, 0, 0, 2, 2],
    )


@pytest.mark.parametrize("scale", [1, 10, 1000, np.int64(100)])
def test_quantization_error(scale):
    values = np.random.default_rng(42).uniform(-2.0, 2.0, size=(20, 3))
    restored = utils.quantize(values, scale_factor=scale) / scale
    assert np.all(np.abs(restored - values) <= 0.5 / scale + 1e-12)


@pytest.mark.parametrize("scale", [0, -1, 1.5, True, np.bool_(False), "1000"])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale_factor"):
        utils.quantize([0.5], scale_factor=scale)
    with pytest.raises(ValueError, match="scale_factor"):
        utils.dequantize_distance(1, scale_factor=scale)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-np.inf],
                                    [1 + 2j], ["0.5"], [True]])
def test_invalid_coordinates(values):
    with pytest.raises(ValueError):
        utils.quantize(values)


@pytest.mark.parametrize("values", [
    [float(np.iinfo(int).max)],
    [np.nextafter(float(np.iinfo(int).min), -np.inf)],
    [np.finfo(float).max],
])
def test_integer_overflow(values):
    with pytest.raises(OverflowError):
        utils.quantize(values, scale_factor=2)


def test_integer_conversion_bounds():
    lower = float(np.iinfo(int).min)
    upper_inside = np.nextafter(-lower, -np.inf)
    result = utils.quantize([lower, upper_inside], scale_factor=1)
    assert int(result[0]) == np.iinfo(int).min
    assert int(result[1]) == int(upper_inside)


@pytest.mark.parametrize("distance, scale, expected", [
    (0, 1000, 0.0), (25, 10, 0.25), (np.int64(100), np.int64(10), 1.0),
])
def test_dequantize_squared_distance(distance, scale, expected):
    assert utils.dequantize_distance(distance, scale_factor=scale) == expected


@pytest.mark.parametrize("distance", [-1, 1.5, True, np.bool_(True), "25"])
def test_invalid_squared_distance(distance):
    with pytest.raises(ValueError, match="d_sq"):
        utils.dequantize_distance(distance)


@pytest.mark.parametrize("a, b, expected", [(6, 8, 24), (-6, 8, 24), (0, 5, 0)])
def test_lcm(a, b, expected):
    assert utils.lcm(a, b) == expected


def test_lcm_zero_pair():
    with pytest.raises(ZeroDivisionError):
        utils.lcm(0, 0)


@pytest.mark.parametrize("number, expected", [
    (-1, False), (0, False), (2, True), (97, True), (91, False), (1763, False),
])
def test_primality(number, expected, monkeypatch):
    monkeypatch.setattr(utils.secrets, "randbelow", lambda limit: 0)
    assert utils.is_probable_prime(number) is expected


def test_generate_prime_rejection_and_size(monkeypatch):
    # Zero becomes 2**31 + 1, a composite; the next candidate is prime.
    candidates = iter([0, 2**32 - 5])
    monkeypatch.setattr(utils.secrets, "randbits", lambda bits: next(candidates))
    result = utils.generate_prime(32)
    assert result == 2**32 - 5
    assert result.bit_length() == 32
    assert result % 2 == 1


def test_prime_size_validation():
    with pytest.raises(ValueError):
        utils.generate_prime(31)


@pytest.mark.parametrize("name", [
    "quantize", "dequantize_distance", "lcm", "is_probable_prime", "generate_prime",
])
def test_root_exports(name):
    assert getattr(sknnpaillier, name) is getattr(utils, name)


@pytest.mark.parametrize("name", ["lcm", "is_probable_prime", "generate_prime"])
def test_key_manager_compatibility_exports(name):
    assert getattr(key_m, name) is getattr(utils, name)






@pytest.mark.parametrize("level, bits", [
    ("test", 128), (80, 1024), (112, 2048), (128, 3072),
    ("80", 1024), ("112", 2048), ("128", 3072),
])
def test_security_level(level, bits):
    assert utils.resolve_key_bits(level) == bits


@pytest.mark.parametrize("level", [
    0, 256, 1024, 2048, 3072, -1, "standard", "legacy", "TEST", " test ",
    "", "0", "1024", "2048", "3072", "080", "80.0", 80.0, True, False, None, [], {},
])
def test_invalid_security_level(level):
    with pytest.raises(ValueError, match="SECURITY_LEVEL"):
        utils.resolve_key_bits(level)


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

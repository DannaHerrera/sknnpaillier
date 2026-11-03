"""Check fixed-point contracts and the relocated mathematical helpers."""

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

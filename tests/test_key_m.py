import pytest

from sknnpaillier import KeyManagementModule
from sknnpaillier.key_m import is_probable_prime


@pytest.mark.parametrize("value", [0, 1, 23, -1, -27])
def test_round_trip(key_manager, transformer, value):
    modulus = key_manager.public_key.n
    decrypted = key_manager.decrypt(transformer.encrypt(value))
    assert decrypted == value % modulus
    assert key_manager.decode_signed(decrypted, modulus) == value


def test_signed_boundaries(key_manager):
    modulus = key_manager.public_key.n
    assert key_manager.decode_signed(modulus // 2, modulus) == modulus // 2
    assert key_manager.decode_signed(modulus // 2 + 1, modulus) == -(modulus // 2)


def test_small_key_rejected():
    with pytest.raises(ValueError):
        KeyManagementModule(127)


@pytest.mark.parametrize("number, expected", [(0, False), (1, False),
                         (2, True), (97, True), (91, False)])
def test_primality(number, expected):
    assert is_probable_prime(number) is expected


@pytest.mark.parametrize("signed, strict, expected", [
    (True, False, True), (True, True, True), (False, False, False),
])
def test_comparison(key_manager, transformer, signed, strict, expected):
    assert key_manager.secure_compare(
        transformer.encrypt(-2), transformer.encrypt(1),
        signed=signed, less_or_equal=not strict,
    ) is expected


@pytest.mark.parametrize("less_or_equal", [True, False])
def test_equal_comparison(key_manager, transformer, less_or_equal):
    assert key_manager.secure_compare(
        transformer.encrypt(5), transformer.encrypt(5),
        signed=False, less_or_equal=less_or_equal,
    ) is less_or_equal


def test_masked_product(key_manager, transformer):
    result = key_manager.secure_multiply_masked(
        transformer.encrypt(3 + 11), transformer.encrypt(-4 + 7),
        transformer.encrypt,
    )
    assert key_manager.decrypt(result) == (3 + 11) * (-4 + 7)

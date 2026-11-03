"""Provide fixed-point conversion and Paillier number-theory helpers."""

from __future__ import annotations

import math
import secrets

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_scale_factor(scale_factor: int) -> int:
    """Require a positive integer scale and return a Python integer."""
    if (
        isinstance(scale_factor, (bool, np.bool_))
        or not isinstance(scale_factor, (int, np.integer))
        or scale_factor <= 0
    ):
        raise ValueError("scale_factor must be a positive integer.")
    return int(scale_factor)


def quantize(x: ArrayLike, scale_factor: int = 1000) -> NDArray[np.int_]:
    """Round real coordinates to fixed-point integers using ties to even.

    Signed coordinates and arbitrary array shapes are supported. Values
    are not restricted to [0, 1]. This function does not check Paillier
    modulus bounds or guarantee preservation of nearest-neighbor order.

    Args:
        x (ArrayLike): Finite real numeric coordinates.
        scale_factor (int): Positive integer multiplier. Defaults to 1000.

    Returns:
        NDArray[np.int_]: Rounded coordinates with the input shape.

    Raises:
        ValueError: If the scale is invalid or coordinates are not finite
            real numbers. Boolean and complex coordinate arrays are rejected.
        OverflowError: If rounded coordinates cannot fit NumPy's integer type.
    """
    scale = _validate_scale_factor(scale_factor)
    values = np.asarray(x)
    if values.dtype.kind not in "iuf":
        raise ValueError("Coordinates must contain real numbers.")
    values = values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Coordinates must be finite.")
    try:
        multiplier = float(scale)
    except OverflowError as error:
        raise OverflowError("scale_factor exceeds the floating-point range.") from error
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.rint(values * multiplier)
    limits = np.iinfo(int)
    # The exclusive upper bound is exact; float(limits.max) rounds upward.
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < float(limits.min))
        or np.any(rounded >= -float(limits.min))
    ):
        raise OverflowError("Quantized coordinates exceed NumPy's integer range.")
    return np.asarray(rounded).astype(int)


def dequantize_distance(d_sq: int, scale_factor: int = 1000) -> float:
    """Convert an integer squared distance to its fixed-point approximation.

    Args:
        d_sq (int): Nonnegative integer squared distance, already decrypted.
        scale_factor (int): Positive scale used for records and query.
            Defaults to 1000.

    Returns:
        float: Squared distance divided by scale_factor squared; no square
            root is taken.

    Raises:
        ValueError: If the scale is invalid or d_sq is not a nonnegative integer.
        OverflowError: If the result cannot be represented as a float.
    """
    scale = _validate_scale_factor(scale_factor)
    if (
        isinstance(d_sq, (bool, np.bool_))
        or not isinstance(d_sq, (int, np.integer))
        or d_sq < 0
    ):
        raise ValueError("d_sq must be a nonnegative integer.")
    return int(d_sq) / scale**2


def lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers.

    Args:
        a (int): First integer.
        b (int): Second integer.

    Returns:
        int: Nonnegative least common multiple.

    Raises:
        ZeroDivisionError: If both arguments are zero.
    """
    return abs(a * b) // math.gcd(a, b)

def is_probable_prime(
    number: int,
    rounds: int = 16,
) -> bool:
    """Test an integer using small primes and Miller-Rabin rounds.

    A positive round count is a caller requirement and is not validated.
    With no rounds, composites not caught by trial division can be accepted.

    Args:
        number (int): Integer to test.
        rounds (int): Number of random witness rounds. Defaults to 16.

    Returns:
        bool: False for a detected composite; True for a probable prime.
    """
    if number < 2:
        return False

    small_primes = (
        2, 3, 5, 7, 11, 13,
        17, 19, 23, 29, 31, 37,
    )

    for prime in small_primes:
        if number == prime:
            return True

        if number % prime == 0:
            return False

    d = number - 1
    s = 0

    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        base = secrets.randbelow(
            number - 3
        ) + 2

        x = pow(
            base,
            d,
            number,
        )

        if x in (
            1,
            number - 1,
        ):
            continue

        for _ in range(s - 1):
            x = pow(
                x,
                2,
                number,
            )

            if x == number - 1:
                break
        else:
            return False

    return True

def generate_prime(
    bits: int,
) -> int:
    """Generate a probable prime with the requested bit length.

    Args:
        bits (int): Prime bit length, at least 32.

    Returns:
        int: Probable prime with exactly bits bits.

    Raises:
        ValueError: If bits is less than 32.
    """
    if bits < 32:
        raise ValueError(
            "Cada primo debe tener al menos 32 bits."
        )

    while True:
        candidate = secrets.randbits(
            bits
        )

        candidate |= (
            1 << (bits - 1)
        )

        candidate |= 1

        if is_probable_prime(
            candidate
        ):
            return candidate

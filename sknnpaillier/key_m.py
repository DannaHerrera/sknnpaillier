"""Generate Paillier keys and provide trusted decryption and comparison operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .log import get_logger
from .utils import generate_prime as generate_prime
from .utils import is_probable_prime as is_probable_prime
from .utils import lcm as lcm


@dataclass(frozen=True)
class PublicKey:
    """Store the public Paillier modulus.

    Construction stores the supplied value without validating it.

    Args:
        n (int): Positive modulus, the product of two distinct primes.

    Attributes:
        n (int): Modulus used for plaintext arithmetic.
    """
    n: int

    @property
    def n2(self) -> int:
        """Return the ciphertext modulus.

        Returns:
            int: The square of n.
        """
        return self.n * self.n


@dataclass(frozen=True)
class PrivateKey:
    """Store the private parameters associated with a public key.

    Args:
        public_key (PublicKey): Associated public modulus.
        lambda_value (int): Least common multiple of p - 1 and q - 1.
        mu (int): Modular inverse of lambda_value modulo n.

    Attributes:
        public_key (PublicKey): Associated public key.
        lambda_value (int): Private decryption exponent.
        mu (int): Private decryption multiplier.
    """
    public_key: PublicKey
    lambda_value: int
    mu: int


class KeyManagementModule:
    """Manage Paillier keys and trusted nonlinear operations.

    The private key is stored on this object. Consumers in the same process
    can call decrypt; this interface does not provide process isolation.

    Attributes:
        public_key (PublicKey): Public modulus exposed to consumers.
    """

    def __init__(
        self,
        key_bits: int,
    ) -> None:
        """Generate a Paillier key pair.

        Use resolve_key_bits to translate SECURITY_LEVEL before construction.
        Logging configuration errors can propagate after key generation.

        Args:
            key_bits (int): Modulus length in bits, at least 128; not a security level.

        Returns:
            None: Initializes the key pair.

        Raises:
            ValueError: If key_bits is below 128 or the modular inverse cannot be computed.
            ImportError: If configured Rich logging is unavailable.
            OSError: If configured log files cannot be created.
        """
        inicio = time.perf_counter()
        (
            self._public_key,
            self._private_key,
        ) = self._generate_keypair(
            key_bits
        )
        logger = get_logger("sknn.keym")
        logger.debug({
            "evento": "generacion_llaves",
            "key_bits": key_bits,
            "tiempo_s": round(
                time.perf_counter() - inicio,
                6,
            ),
        })

    @property
    def public_key(
        self,
    ) -> PublicKey:
        """Return the public key for this manager.

        Returns:
            PublicKey: The public Paillier modulus.
        """
        return self._public_key

    # DESCIFRADO INTERNO

    def decrypt(
        self,
        ciphertext: int,
    ) -> int:
        """Decrypt a ciphertext to a residue modulo n.

        Ciphertext membership is not validated. Use decode_signed when the
        residue represents a signed coordinate.

        Args:
            ciphertext (int): Paillier ciphertext associated with this key.

        Returns:
            int: Plaintext residue in [0, n).
        """
        public_key = (
            self._private_key.public_key
        )

        u = pow(
            int(ciphertext),
            self._private_key.lambda_value,
            public_key.n2,
        )

        l_value = (
            (u - 1)
            // public_key.n
        )

        plaintext = (
            l_value
            * self._private_key.mu
        ) % public_key.n

        return plaintext

    @staticmethod
    def decode_signed(
        value: int,
        modulus: int,
    ) -> int:
        """Decode a residue using the centered signed representation.

        Args:
            value (int): Integer residue, reduced modulo modulus before decoding.
            modulus (int): Positive plaintext modulus.

        Returns:
            int: Centered representative; residues above modulus // 2 are negative.

        Raises:
            ZeroDivisionError: If modulus is zero.
        """
        value %= modulus

        if value > modulus // 2:
            return value - modulus

        return value

    # OPERACIÓN CONTROLADA PARA MULTIPLICACIÓN SEGURA+

    def secure_multiply_masked(
        self,
        encrypted_a_masked: int,
        encrypted_b_masked: int,
        encrypt_value: Callable[[int], int],
    ) -> int:
        """Decrypt masked operands and encrypt their modular product.

        The cloud removes mask cross-terms after this operation. Exceptions
        from the supplied encryption callback propagate to the caller.

        Args:
            encrypted_a_masked (int): Ciphertext of the first masked operand.
            encrypted_b_masked (int): Ciphertext of the second masked operand.
            encrypt_value (Callable[[int], int]): Encryption callback for this key.

        Returns:
            int: Encryption of the product of the masked plaintexts modulo n.
        """
        n = self._public_key.n

        a_masked = self.decrypt(
            encrypted_a_masked
        )

        b_masked = self.decrypt(
            encrypted_b_masked
        )

        masked_product = (
            a_masked * b_masked
        ) % n

        return encrypt_value(
            masked_product
        )

    # OPERACIÓN CONTROLADA PARA COMPARACIÓN SEGURA

    def secure_compare(
        self,
        encrypted_left: int,
        encrypted_right: int,
        *,
        signed: bool,
        less_or_equal: bool = True,
    ) -> bool:
        """Compare two decrypted values in the trusted context.

        The caller observes the comparison result, not the decrypted operands.
        Unsigned comparisons use residues in [0, n).

        Args:
            encrypted_left (int): Ciphertext of the left operand.
            encrypted_right (int): Ciphertext of the right operand.
            signed (bool): Decode centered signed values when True.
            less_or_equal (bool): Use <= when True, otherwise <. Defaults to True.

        Returns:
            bool: Result of the selected comparison.
        """
        n = self._public_key.n

        left = self.decrypt(
            encrypted_left
        )

        right = self.decrypt(
            encrypted_right
        )

        if signed:
            left = self.decode_signed(
                left,
                n,
            )

            right = self.decode_signed(
                right,
                n,
            )

        if less_or_equal:
            return left <= right

        return left < right

    # ==========================================================
    # GENERACIÓN DEL PAR DE LLAVES
    # ==========================================================

    @staticmethod
    def _generate_keypair(
        key_bits: int,
    ) -> tuple[
        PublicKey,
        PrivateKey,
    ]:
        """Generate keys with an exact modulus bit length.

        Args:
            key_bits (int): Requested modulus length, at least 128.

        Returns:
            tuple[PublicKey, PrivateKey]: Generated public and private parameters.

        Raises:
            ValueError: If the length is too small or the modular inverse is unavailable.
        """
        if key_bits < 128:
            raise ValueError(
                "La llave debe tener al menos 128 bits."
            )

        first_bits = (
            key_bits // 2
        )

        second_bits = (
            key_bits
            - first_bits
        )

        while True:
            p = generate_prime(
                first_bits
            )

            q = generate_prime(
                second_bits
            )

            if (
                p != q
                and
                (p * q).bit_length()
                == key_bits
            ):
                break

        n = p * q

        public_key = PublicKey(
            n
        )

        lambda_value = lcm(
            p - 1,
            q - 1,
        )

        mu = pow(
            lambda_value,
            -1,
            n,
        )

        private_key = PrivateKey(
            public_key,
            lambda_value,
            mu,
        )

        return (
            public_key,
            private_key,
        )

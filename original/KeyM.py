from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PublicKey:
    n: int

    @property
    def n2(self) -> int:
        return self.n * self.n


@dataclass(frozen=True)
class PrivateKey:
    public_key: PublicKey
    lambda_value: int
    mu: int


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def is_probable_prime(
    number: int,
    rounds: int = 16,
) -> bool:
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


class KeyManagementModule:
    """
    Módulo confiable de administración de llaves.

    Arquitectura de esta versión:
        - Existe UN solo cloud: CloudB.
        - CloudB nunca recibe PrivateKey.
        - Las primitivas que Paillier no puede ejecutar por sí solo
          se exponen aquí como operaciones controladas:
              * secure_multiply_masked()
              * secure_compare()

    Por tanto, SecureHelper.py deja de existir.
    """

    def __init__(
        self,
        key_bits: int,
    ) -> None:
        (
            self._public_key,
            self._private_key,
        ) = self._generate_keypair(
            key_bits
        )

    @property
    def public_key(
        self,
    ) -> PublicKey:
        return self._public_key

    # ==========================================================
    # DESCIFRADO INTERNO
    # ==========================================================

    def decrypt(
        self,
        ciphertext: int,
    ) -> int:
        """
        Descifra usando la llave privada encapsulada en KeyM.

        Se mantiene público por compatibilidad con la etapa final
        de validación del prototipo, pero CloudB no recibe ni accede
        a self._private_key.
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
        value %= modulus

        if value > modulus // 2:
            return value - modulus

        return value

    # ==========================================================
    # OPERACIÓN CONTROLADA PARA MULTIPLICACIÓN SEGURA
    # ==========================================================

    def secure_multiply_masked(
        self,
        encrypted_a_masked: int,
        encrypted_b_masked: int,
        encrypt_value: Callable[[int], int],
    ) -> int:
        """
        Paso confiable de la multiplicación segura.

        CloudB manda:
            E(a + r_a)
            E(b + r_b)

        KeyM descifra únicamente los valores ENMASCARADOS:
            a + r_a
            b + r_b

        y devuelve:
            E((a+r_a)(b+r_b))

        CloudB elimina después los términos de las máscaras.
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

    # ==========================================================
    # OPERACIÓN CONTROLADA PARA COMPARACIÓN SEGURA
    # ==========================================================

    def secure_compare(
        self,
        encrypted_left: int,
        encrypted_right: int,
        *,
        signed: bool,
        less_or_equal: bool = True,
    ) -> bool:
        """
        Compara los plaintexts y devuelve únicamente un booleano.

        signed=True:
            para coordenadas y límites del KD-tree.

        signed=False:
            para distancias y scores no negativos.

        CloudB aprende el resultado booleano de la comparación,
        pero no recibe los plaintexts comparados.
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

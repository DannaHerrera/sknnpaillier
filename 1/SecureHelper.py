from __future__ import annotations

from KeyM import KeyManagementModule
from SecureDataT import SecureDataTransformationModule


class TrustedPaillierHelper:
    """
    Helper criptográfico confiable para primitivas que Paillier
    no soporta directamente.

    QP NO recibe la llave privada.

    Participa en:
      1) multiplicación segura sobre valores enmascarados;
      2) compare-and-swap para ranking cifrado;
      3) comparaciones de control para el recorrido KD-tree.

    MODELO DE AMENAZA:
    - El helper es confiable.
    - Debe ejecutarse separado de QP en un despliegue real.
    - En modo KD-tree, el resultado booleano de ciertas
      comparaciones se devuelve a QP para permitir la poda.
      Esto revela el patrón de recorrido, pero no los valores
      originales de las coordenadas ni las distancias.
    """

    def __init__(
        self,
        key_manager: KeyManagementModule,
        transformer: SecureDataTransformationModule,
    ) -> None:

        self.__key_manager = key_manager
        self.__transformer = transformer

    # ========================================================
    # MULTIPLICACIÓN DE VALORES ENMASCARADOS
    # ========================================================

    def multiply_masked_values(
        self,
        encrypted_a_masked: int,
        encrypted_b_masked: int,
    ) -> int:
        """
        Recibe E(a+r_a), E(b+r_b), descifra solamente esos
        valores enmascarados, multiplica y vuelve a cifrar.
        """

        n = self.__transformer.public_key.n

        a_masked = self.__key_manager.decrypt(
            encrypted_a_masked
        )

        b_masked = self.__key_manager.decrypt(
            encrypted_b_masked
        )

        masked_product = (
            a_masked * b_masked
        ) % n

        return self.__transformer.encrypt(
            masked_product
        )

    # ========================================================
    # COMPARACIÓN ENMASCARADA -> BOOLEANO
    # ========================================================

    def compare_masked_values(
        self,
        encrypted_left_masked: int,
        encrypted_right_masked: int,
        less_or_equal: bool = True,
    ) -> bool:
        """
        Descifra solamente T(left) y T(right), donde T es una
        transformación afín creciente construida por QP.

        Se usa únicamente en modo KD-tree porque una estructura
        indexada necesita conocer qué rama recorrer/podar.

        Retorna una decisión booleana; por tanto este modo filtra
        el patrón de acceso a QP. No revela left ni right.
        """

        left_masked = self.__key_manager.decrypt(
            encrypted_left_masked
        )

        right_masked = self.__key_manager.decrypt(
            encrypted_right_masked
        )

        if less_or_equal:
            return left_masked <= right_masked

        return left_masked < right_masked

    # ========================================================
    # COMPARE-AND-SWAP ENMASCARADO
    # ========================================================

    def compare_and_swap_masked(
        self,
        encrypted_left_original: int,
        encrypted_right_original: int,
        encrypted_left_masked: int,
        encrypted_right_masked: int,
        ascending: bool,
    ) -> tuple[int, int]:
        """
        QP construye:

            E(s*left  + r)
            E(s*right + r)

        con s > 0 y sin wrap-around módulo n.

        El helper compara solamente esos valores transformados y
        devuelve los ciphertexts originales re-aleatorizados en
        el orden solicitado.
        """

        left_masked = self.__key_manager.decrypt(
            encrypted_left_masked
        )

        right_masked = self.__key_manager.decrypt(
            encrypted_right_masked
        )

        left_first = left_masked <= right_masked

        if not ascending:
            left_first = not left_first

        if left_first:
            first = encrypted_left_original
            second = encrypted_right_original
        else:
            first = encrypted_right_original
            second = encrypted_left_original

        return (
            self.__transformer.rerandomize(first),
            self.__transformer.rerandomize(second),
        )

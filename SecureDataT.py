from __future__ import annotations

import math
import secrets
from typing import Sequence

import numpy as np

from KDTree import build_encrypted_kd_tree


class SecureDataTransformationModule:
    """
    Zona confiable del Data Owner.

    Responsabilidades:
      - cifrar la base de datos;
      - construir el KD-tree EN CLARO;
      - cifrar lb/ub de sus hojas;
      - cifrar la consulta;
      - entregar BD + índice + consulta al único cloud: CloudB.
    """

    def __init__(
        self,
        public_key,
    ) -> None:
        self.public_key = (
            public_key
        )

        self.cloud = None

    def set_cloud(
        self,
        cloud,
    ) -> None:
        self.cloud = cloud

    def encrypt(
        self,
        value: int,
    ) -> int:
        n = self.public_key.n
        n2 = self.public_key.n2

        plaintext = (
            int(value) % n
        )

        while True:
            r = secrets.randbelow(
                n
            )

            if (
                r != 0
                and
                math.gcd(r, n) == 1
            ):
                break

        return (
            pow(
                n + 1,
                plaintext,
                n2,
            )
            *
            pow(
                r,
                n,
                n2,
            )
        ) % n2

    def rerandomize(
        self,
        ciphertext: int,
    ) -> int:
        return (
            int(ciphertext)
            * self.encrypt(0)
        ) % self.public_key.n2

    def encrypt_vector(
        self,
        vector: Sequence[int],
    ) -> list[int]:
        return [
            self.encrypt(
                int(value)
            )
            for value in vector
        ]

    def encrypt_database(
        self,
        database: np.ndarray,
    ) -> list[list[int]]:
        return [
            self.encrypt_vector(
                record
            )
            for record in database
        ]

    def send_database_to_cloud(
        self,
        database: np.ndarray,
        leaf_capacity: int = 16,
        max_depth: int | None = None,
    ) -> None:
        if self.cloud is None:
            raise RuntimeError(
                "CloudB no ha sido configurado."
            )

        encrypted_database = (
            self.encrypt_database(
                database
            )
        )

        encrypted_kd_tree = (
            build_encrypted_kd_tree(
                database=database,
                encrypt_value=self.encrypt,
                leaf_capacity=leaf_capacity,
                max_depth=max_depth,
            )
        )

        self.cloud.set_database(
            encrypted_database
        )

        self.cloud.set_kd_tree(
            encrypted_kd_tree
        )

    def send_query_to_cloud(
        self,
        query: np.ndarray,
        k: int,
    ):
        if self.cloud is None:
            raise RuntimeError(
                "CloudB no ha sido configurado."
            )

        encrypted_query = (
            self.encrypt_vector(
                query
            )
        )

        return self.cloud.process_query(
            encrypted_query,
            k,
        )

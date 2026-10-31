from __future__ import annotations

import math
import secrets
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .kd_tree import build_encrypted_kd_tree
from .key_m import PublicKey
from .log import get_logger

if TYPE_CHECKING:
    from .cloud_b import CloudBQueryProcessingModule


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
        public_key: PublicKey,
    ) -> None:
        self.public_key = (
            public_key
        )

        self.cloud: CloudBQueryProcessingModule | None = None

    def set_cloud(
        self,
        cloud: CloudBQueryProcessingModule,
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

        inicio = time.perf_counter()

        encrypted_database = (
            self.encrypt_database(
                database
            )
        )

        cifrado_bd_s = (
            time.perf_counter() - inicio
        )

        t_arbol = time.perf_counter()

        encrypted_kd_tree = (
            build_encrypted_kd_tree(
                database=database,
                encrypt_value=self.encrypt,
                leaf_capacity=leaf_capacity,
                max_depth=max_depth,
            )
        )

        kdtree_s = (
            time.perf_counter() - t_arbol
        )

        self.cloud.set_database(
            encrypted_database
        )

        self.cloud.set_kd_tree(
            encrypted_kd_tree
        )

        total_s = (
            time.perf_counter() - inicio
        )

        logger = get_logger("sknn.securedata")
        logger.debug({
            "evento": "outsourcing_desglose",
            "registros": int(len(database)),
            "dimensiones": int(database.shape[1]),
            "hojas": len(encrypted_kd_tree.leaves),
            "leaf_capacity": leaf_capacity,
            "max_depth": max_depth,
            "cifrado_bd_s": round(cifrado_bd_s, 6),
            "kdtree_construccion_y_bordes_s": round(
                kdtree_s,
                6,
            ),
        })

        logger.info({
            "evento": "outsourcing_total",
            "registros": int(len(database)),
            "hojas": len(encrypted_kd_tree.leaves),
            "tiempo_s": round(total_s, 6),
        })

    def send_query_to_cloud(
        self,
        query: np.ndarray,
        k: int,
    ) -> tuple[list[int], list[tuple[int, ...]], list[int]]:
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

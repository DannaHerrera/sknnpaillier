"""Encrypt integer records and queries for the cloud query processor.
Build and encrypt the spatial index before outsourcing the database.
"""

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
    """Encrypt owner data and delegate queries to the cloud.

    Labels remain with the client. Tree construction uses plaintext data
    inside the trusted owner context.

    Attributes:
        public_key (PublicKey): Key used for all encryption operations.
        cloud (CloudBQueryProcessingModule | None): Configured query processor.
    """

    def __init__(
        self,
        public_key: PublicKey,
    ) -> None:
        """Configure encryption with an existing public key.

        Args:
            public_key (PublicKey): Paillier modulus parameters.

        Returns:
            None: Stores the key and leaves the cloud unset.
        """
        self.public_key = (
            public_key
        )

        self.cloud: CloudBQueryProcessingModule | None = None

    def set_cloud(
        self,
        cloud: CloudBQueryProcessingModule,
    ) -> None:
        """Attach the query processor used for data and query delivery.

        Args:
            cloud (CloudBQueryProcessingModule): Processor using the same key.

        Returns:
            None: Stores the processor reference.
        """
        self.cloud = cloud

    def encrypt(
        self,
        value: int,
    ) -> int:
        """Encrypt an integer with fresh Paillier randomness.

        Input is converted with int. Floats are truncated, not encoded as
        fixed point; quantize them in the client before calling this method.

        Args:
            value (int): Plaintext integer, reduced modulo n.

        Returns:
            int: Ciphertext modulo n squared.
        """
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
        """Refresh a ciphertext by multiplying it by an encryption of zero.

        Args:
            ciphertext (int): Ciphertext under this public key.

        Returns:
            int: Ciphertext with the same plaintext and fresh randomness.
        """
        return (
            int(ciphertext)
            * self.encrypt(0)
        ) % self.public_key.n2

    def encrypt_vector(
        self,
        vector: Sequence[int],
    ) -> list[int]:
        """Encrypt each coordinate of a record or query.

        Args:
            vector (Sequence[int]): Integer coordinates.

        Returns:
            list[int]: Ciphertexts in the same coordinate order.
        """
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
        """Encrypt every row of an integer database.

        This method does not validate the shape; tree construction validates
        the shape when outsourcing a complete database.

        Args:
            database (numpy.ndarray): 2D integer array of records.

        Returns:
            list[list[int]]: Encrypted rows in their original order.
        """
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
        """Encrypt the database and its leaf index, then deliver both.

        Database encryption happens before index validation. Logging
        configuration errors may propagate after the cloud has been updated.

        Args:
            database (numpy.ndarray): Nonempty 2D integer array of shape (rows, dimensions).
            leaf_capacity (int): Positive target records per leaf. Defaults to 16.
            max_depth (int | None): Optional root-relative depth limit. Defaults to None.

        Returns:
            None: Updates the configured cloud database and index.

        Raises:
            RuntimeError: If no cloud is configured.
            ValueError: If index construction rejects the shape or leaf capacity.
            ImportError: If configured Rich logging is unavailable.
            OSError: If configured log files cannot be created.
        """
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
        """Encrypt a query and request its nearest neighbors.

        Cloud processing and logging exceptions propagate. Final decryption
        belongs to the trusted client.

        Args:
            query (numpy.ndarray): One-dimensional integer coordinate vector.
            k (int): Number of neighbors, between 1 and the database size.

        Returns:
            tuple[list[int], list[tuple[int, ...]], list[int]]: Visible IDs, encrypted
                records, and encrypted squared distances, in neighbor order.

        Raises:
            RuntimeError: If no cloud, database, or index is configured.
            ValueError: If k or the query dimension is invalid.
        """
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

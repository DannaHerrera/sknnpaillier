from __future__ import annotations

import numpy as np

from Config import Config
from SecureDataT import (
    SecureDataTransformationModule,
)


class DataUserModule:


    def __init__(
        self,
        config: Config,
        secure_data_transformer:
            SecureDataTransformationModule,
    ) -> None:

        self.config = config

        self.secure_data_transformer = (
            secure_data_transformer
        )

        self.queries: (
            np.ndarray | None
        ) = None

        self.query_indices: (
            np.ndarray | None
        ) = None

    # GENERACIÓN / SELECCIÓN DE CONSULTAS

    def prepare_queries(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Selecciona consultas distribuidas a lo largo
        de la parte del dataset que no pertenece a DB.

        DB:
            data[0 : db_rows]

        Queries:
            seleccionadas entre
            data[db_rows : len(data)]

        Esto evita que las consultas sean registros
        consecutivos y mejora la diversidad espacial.
        """

        start = (
            self.config.db_rows
        )

        total_queries = (
            self.config.queries
        )

        available_rows = (
            len(data)
            - start
        )

        if available_rows <= 0:
            raise ValueError(
                "No existen registros disponibles "
                "fuera de la base de datos para "
                "preparar consultas."
            )

        if (
            total_queries
            > available_rows
        ):
            raise ValueError(
                "Se solicitaron "
                f"{total_queries} consultas, "
                "pero solo existen "
                f"{available_rows} registros "
                "disponibles fuera de DB."
            )

        # Seleccionar índices distribuidos uniformemente

        self.query_indices = (
            np.linspace(
                start,
                len(data) - 1,
                num=total_queries,
                dtype=int,
            )
        )

        self.queries = (
            data[
                self.query_indices
            ].copy()
        )

        print()
        print(
            "Consultas seleccionadas:"
        )

        print(
            "Índices:",
            self.query_indices.tolist(),
        )

        return (
            self.queries
        )

    # REALIZAR CONSULTA

    def query(
        self,
        query_vector: np.ndarray,
        k: int | None = None,
    ):
        """
        Envía q y k a SecureDataT.

        SecureDataT cifra:
            q -> E(q)

        y posteriormente envía:
            E(q), k -> CloudB
        """

        if k is None:
            k = self.config.k

        if k <= 0:
            raise ValueError(
                "k debe ser mayor "
                "que cero."
            )

        return (
            self.secure_data_transformer
            .send_query_to_cloud(
                query=query_vector,
                k=k,
            )
        )


    def get_queries(
        self,
    ) -> np.ndarray:

        if self.queries is None:
            raise RuntimeError(
                "Las consultas todavía "
                "no han sido preparadas."
            )

        return (
            self.queries.copy()
        )

    def get_query_indices(
        self,
    ) -> np.ndarray:

        if self.query_indices is None:
            raise RuntimeError(
                "Los índices de consulta "
                "todavía no han sido preparados."
            )

        return (
            self.query_indices.copy()
        )
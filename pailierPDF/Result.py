from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from Config import Config


class ResultModule:

    @staticmethod
    def print_header(
        config: Config,
        database: np.ndarray,
        queries: np.ndarray,
    ) -> None:
        print()
        print("=" * 80)
        print(
            "SKNN-QP CIFRADO CON PAILLIER "
            "+ KD-TREE POR HOJAS"
        )
        print("=" * 80)

        print(
            f"Fuente           : {config.source}"
        )
        print(
            f"Registros DB     : {len(database)}"
        )
        print(
            f"Consultas        : {len(queries)}"
        )
        print(
            f"Atributos        : {database.shape[1]}"
        )
        print(
            f"k                : {config.k}"
        )
        print(
            f"Bits de Paillier : {config.key_bits}"
        )
        print(
            f"Capacidad hoja   : {config.leaf_capacity}"
        )
        print(
            "Cloud            : CloudB (único)"
        )

    @staticmethod
    def print_query_result(
        query_number: int,
        query: np.ndarray,
        plain_ids: Sequence[int],
        paillier_ids: Sequence[int],
        paillier_distances: Sequence[
            int | None
        ],
        recovered_neighbors: np.ndarray,
        verification: dict[str, object],
    ) -> None:
        print()
        print("-" * 80)
        print(
            f"CONSULTA {query_number}"
        )

        print(
            f"Vector consulta: "
            f"{query.tolist()}"
        )

        print(
            f"IDs esperados  : "
            f"{list(plain_ids)}"
        )

        print(
            f"IDs Paillier   : "
            f"{list(paillier_ids)}"
        )

        print()

        for position, record_id in enumerate(
            paillier_ids,
            start=1,
        ):
            distance = (
                paillier_distances[
                    record_id
                ]
            )

            record = (
                recovered_neighbors[
                    position - 1
                ].tolist()
            )

            print(
                f"Vecino {position}: "
                f"ID={record_id}, "
                f"distancia²={distance}, "
                f"registro={record}"
            )

        print()
        print(
            "Mismo orden       : "
            f"{verification['same_order']}"
        )
        print(
            "Mismos registros  : "
            f"{verification['same_records']}"
        )
        print(
            "Mismas distancias : "
            f"{verification['same_distances']}"
        )
        print(
            "Recall@k           : "
            f"{float(verification['recall_at_k']):.4f}"
        )
        print(
            "Jaccard@k          : "
            f"{float(verification['jaccard_at_k']):.4f}"
        )
        print(
            "Acuerdo ranking   : "
            f"{float(verification['rank_agreement']):.4f}"
        )

    @staticmethod
    def build_result_row(
        query_number: int,
        plain_ids: Sequence[int],
        paillier_ids: Sequence[int],
        verification: dict[str, object],
    ) -> dict[str, object]:
        return {
            "query_number": (
                query_number
            ),
            "plaintext_ids": " ".join(
                map(
                    str,
                    plain_ids,
                )
            ),
            "paillier_ids": " ".join(
                map(
                    str,
                    paillier_ids,
                )
            ),
            "same_order": (
                verification[
                    "same_order"
                ]
            ),
            "same_records": (
                verification[
                    "same_records"
                ]
            ),
            "same_distances": (
                verification[
                    "same_distances"
                ]
            ),
            "recall_at_k": (
                verification[
                    "recall_at_k"
                ]
            ),
            "jaccard_at_k": (
                verification[
                    "jaccard_at_k"
                ]
            ),
            "rank_agreement": (
                verification[
                    "rank_agreement"
                ]
            ),
        }

    @staticmethod
    def save_results(
        results: list[
            dict[str, object]
        ],
        output_path: Path,
    ) -> None:
        if not results:
            return

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    results[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                results
            )

    @staticmethod
    def print_summary(
        results: list[
            dict[str, object]
        ],
        output_path: Path,
    ) -> None:
        correct_queries = sum(
            bool(
                result[
                    "same_order"
                ]
            )
            for result in results
        )

        total_queries = len(
            results
        )

        print()
        print("=" * 80)
        print("RESUMEN")
        print("=" * 80)

        print(
            f"Consultas correctas: "
            f"{correct_queries}/{total_queries}"
        )

        print(
            f"Archivo de salida  : "
            f"{output_path.resolve()}"
        )

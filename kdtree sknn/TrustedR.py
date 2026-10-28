from __future__ import annotations

from typing import Sequence

import numpy as np


class TrustedResultProcessingModule:
    """
    Validación en plaintext.

    No participa en el query processing cifrado.
    """

    @staticmethod
    def plaintext_knn(
        database: np.ndarray,
        query: np.ndarray,
        k: int,
    ) -> tuple[
        list[int],
        np.ndarray,
        list[int],
    ]:
        if not (
            1 <= k <= len(database)
        ):
            raise ValueError(
                "k fuera de rango."
            )

        distances = [
            sum(
                (
                    int(x)
                    - int(y)
                ) ** 2
                for x, y in zip(
                    record,
                    query,
                )
            )
            for record in database
        ]

        ordered_ids = sorted(
            range(
                len(database)
            ),
            key=lambda record_id: (
                distances[
                    record_id
                ],
                record_id,
            ),
        )

        selected_ids = (
            ordered_ids[:k]
        )

        selected_records = (
            database[
                selected_ids
            ].copy()
        )

        return (
            selected_ids,
            selected_records,
            distances,
        )

    @staticmethod
    def calculate_metrics(
        expected_ids: Sequence[int],
        obtained_ids: Sequence[int],
    ) -> tuple[
        float,
        float,
        float,
    ]:
        expected_set = set(
            expected_ids
        )

        obtained_set = set(
            obtained_ids
        )

        recall = (
            len(
                expected_set
                & obtained_set
            )
            / len(expected_set)
        )

        union = (
            expected_set
            | obtained_set
        )

        jaccard = (
            len(
                expected_set
                & obtained_set
            )
            / len(union)
        )

        rank_agreement = (
            sum(
                expected == obtained
                for expected, obtained
                in zip(
                    expected_ids,
                    obtained_ids,
                )
            )
            / len(expected_ids)
        )

        return (
            recall,
            jaccard,
            rank_agreement,
        )

    @classmethod
    def verify_results(
        cls,
        expected_ids: Sequence[int],
        obtained_ids: Sequence[int],
        expected_records: np.ndarray,
        obtained_records: np.ndarray,
        expected_distances: Sequence[int],
        obtained_distances: Sequence[
            int | None
        ],
    ) -> dict[str, object]:
        (
            recall,
            jaccard,
            rank_agreement,
        ) = cls.calculate_metrics(
            expected_ids,
            obtained_ids,
        )

        same_order = (
            list(expected_ids)
            == list(obtained_ids)
        )

        same_records = (
            np.array_equal(
                expected_records,
                obtained_records,
            )
        )

        same_distances = all(
            obtained_distances[
                record_id
            ] is not None
            and
            expected_distances[
                record_id
            ]
            ==
            obtained_distances[
                record_id
            ]
            for record_id
            in obtained_ids
        )

        return {
            "same_order": same_order,
            "same_records": same_records,
            "same_distances": (
                same_distances
            ),
            "recall_at_k": recall,
            "jaccard_at_k": jaccard,
            "rank_agreement": (
                rank_agreement
            ),
        }

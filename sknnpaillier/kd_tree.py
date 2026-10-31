from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EncryptedKDLeaf:
    """
    Hoja del KD-tree entregada a CloudB.

    Cada hoja representa una región multidimensional:
        lb <= x <= ub

    y almacena los IDs de los registros pertenecientes a ella.
    """
    leaf_id: int
    encrypted_lower_bounds: tuple[int, ...]
    encrypted_upper_bounds: tuple[int, ...]
    record_ids: tuple[int, ...]


@dataclass(frozen=True)
class EncryptedKDTree:
    """
    Índice cifrado usado durante la consulta.

    CloudB trabaja directamente con las hojas; no necesita recorrer
    left/right como en un KD-tree convencional.
    """
    leaves: tuple[EncryptedKDLeaf, ...]
    dimensions: int
    database_size: int
    leaf_capacity: int


@dataclass(frozen=True)
class _PlainKDLeaf:
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    record_ids: tuple[int, ...]


def _validate_database(
    database: np.ndarray,
) -> tuple[int, int]:
    if database.ndim != 2:
        raise ValueError(
            "La base de datos debe ser una matriz 2D."
        )

    rows, dimensions = database.shape

    if rows == 0:
        raise ValueError(
            "La base de datos no puede estar vacía."
        )

    if dimensions == 0:
        raise ValueError(
            "Debe existir al menos un atributo."
        )

    return rows, dimensions


def _build_plain_kd_leaves(
    database: np.ndarray,
    leaf_capacity: int,
    stats: dict[str, int] | None = None,
    max_depth: int | None = None,
) -> list[_PlainKDLeaf]:
    """
    Construcción EN CLARO dentro de la zona confiable.

    Se divide recursivamente por mediana:
        axis = depth % dimensions

    El objetivo final no es conservar el árbol jerárquico, sino las
    regiones hoja con:
        lower_bounds
        upper_bounds
        record_ids
    """
    rows, dimensions = _validate_database(
        database
    )

    if leaf_capacity <= 0:
        raise ValueError(
            "leaf_capacity debe ser mayor que cero."
        )

    root_lb = [
        int(database[:, axis].min())
        for axis in range(dimensions)
    ]

    root_ub = [
        int(database[:, axis].max())
        for axis in range(dimensions)
    ]

    leaves: list[_PlainKDLeaf] = []

    def build(
        record_ids: list[int],
        depth: int,
        lower_bounds: list[int],
        upper_bounds: list[int],
    ) -> None:
        if stats is not None:
            stats["total_nodes"] = stats.get("total_nodes", 0) + 1
            stats["max_depth"] = max(stats.get("max_depth", 0), depth)
        if len(record_ids) <= leaf_capacity or (max_depth is not None and depth >= max_depth):
            leaves.append(
                _PlainKDLeaf(
                    lower_bounds=tuple(
                        lower_bounds
                    ),
                    upper_bounds=tuple(
                        upper_bounds
                    ),
                    record_ids=tuple(
                        record_ids
                    ),
                )
            )
            return

        axis = (
            depth % dimensions
        )

        ordered = sorted(
            record_ids,
            key=lambda record_id: (
                int(
                    database[
                        record_id,
                        axis,
                    ]
                ),
                record_id,
            ),
        )

        median = (
            len(ordered) // 2
        )

        left_ids = (
            ordered[:median]
        )

        right_ids = (
            ordered[median:]
        )

        if not left_ids or not right_ids:
            leaves.append(
                _PlainKDLeaf(
                    lower_bounds=tuple(
                        lower_bounds
                    ),
                    upper_bounds=tuple(
                        upper_bounds
                    ),
                    record_ids=tuple(
                        ordered
                    ),
                )
            )
            return

        left_max = max(
            int(
                database[
                    record_id,
                    axis,
                ]
            )
            for record_id in left_ids
        )

        right_min = min(
            int(
                database[
                    record_id,
                    axis,
                ]
            )
            for record_id in right_ids
        )

        left_lb = list(
            lower_bounds
        )
        left_ub = list(
            upper_bounds
        )
        right_lb = list(
            lower_bounds
        )
        right_ub = list(
            upper_bounds
        )

        if left_max < right_min:
            split = (
                (left_max + right_min)
                // 2
            )

            left_ub[axis] = split
            right_lb[axis] = (
                split + 1
            )

        else:
            # Si existen duplicados sobre el plano de corte,
            # ambas hojas comparten la frontera.
            split = left_max

            left_ub[axis] = split
            right_lb[axis] = split

        build(
            left_ids,
            depth + 1,
            left_lb,
            left_ub,
        )

        build(
            right_ids,
            depth + 1,
            right_lb,
            right_ub,
        )

    build(
        list(range(rows)),
        0,
        root_lb,
        root_ub,
    )

    return leaves


def build_encrypted_kd_tree(
    database: np.ndarray,
    encrypt_value: Callable[[int], int],
    leaf_capacity: int = 16,
    max_depth: int | None = None,
) -> EncryptedKDTree:
    """
    1. Construye el KD-tree en plaintext.
    2. Conserva únicamente sus hojas/regiones.
    3. Cifra lb y ub atributo por atributo.
    4. Entrega el índice cifrado a CloudB.
    """
    rows, dimensions = _validate_database(
        database
    )

    plain_leaves = _build_plain_kd_leaves(
        database,
        leaf_capacity,
        max_depth=max_depth,
    )

    encrypted_leaves: list[
        EncryptedKDLeaf
    ] = []

    for leaf_id, leaf in enumerate(
        plain_leaves
    ):
        encrypted_leaves.append(
            EncryptedKDLeaf(
                leaf_id=leaf_id,
                encrypted_lower_bounds=tuple(
                    encrypt_value(value)
                    for value
                    in leaf.lower_bounds
                ),
                encrypted_upper_bounds=tuple(
                    encrypt_value(value)
                    for value
                    in leaf.upper_bounds
                ),
                record_ids=leaf.record_ids,
            )
        )

    return EncryptedKDTree(
        leaves=tuple(
            encrypted_leaves
        ),
        dimensions=dimensions,
        database_size=rows,
        leaf_capacity=leaf_capacity,
    )

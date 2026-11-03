"""Partition integer records into KD-tree leaves and encrypt their bounds.
Tree construction runs on plaintext data in the trusted owner context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EncryptedKDLeaf:
    """Store encrypted bounds and visible record IDs for one region.

    Each region uses inclusive bounds. Different leaves may share a
    boundary when records have equal coordinates. Values are not validated
    by this dataclass.

    Args:
        leaf_id (int): Identifier of this leaf.
        encrypted_lower_bounds (tuple[int, ...]): Encrypted inclusive lower bounds.
        encrypted_upper_bounds (tuple[int, ...]): Encrypted inclusive upper bounds.
        record_ids (tuple[int, ...]): Database row IDs assigned to this region.

    Attributes:
        leaf_id (int): Identifier of this leaf.
        encrypted_lower_bounds (tuple[int, ...]): Encrypted inclusive lower bounds.
        encrypted_upper_bounds (tuple[int, ...]): Encrypted inclusive upper bounds.
        record_ids (tuple[int, ...]): Database row IDs assigned to this region.
    """
    leaf_id: int
    encrypted_lower_bounds: tuple[int, ...]
    encrypted_upper_bounds: tuple[int, ...]
    record_ids: tuple[int, ...]


@dataclass(frozen=True)
class EncryptedKDTree:
    """Store the leaf index used by encrypted queries.

    The index contains leaves, not a hierarchy of child nodes. A depth
    limit can produce leaves larger than leaf_capacity.

    Args:
        leaves (tuple[EncryptedKDLeaf, ...]): Encrypted leaf regions.
        dimensions (int): Number of coordinates per record.
        database_size (int): Number of records indexed.
        leaf_capacity (int): Requested maximum records per leaf.

    Attributes:
        leaves (tuple[EncryptedKDLeaf, ...]): Encrypted leaf regions.
        dimensions (int): Number of coordinates per record.
        database_size (int): Number of records indexed.
        leaf_capacity (int): Requested maximum records per leaf.
    """
    leaves: tuple[EncryptedKDLeaf, ...]
    dimensions: int
    database_size: int
    leaf_capacity: int


@dataclass(frozen=True)
class _PlainKDLeaf:
    """Store plaintext region bounds during trusted construction.

    Attributes:
        lower_bounds (tuple[int, ...]): Inclusive lower bounds.
        upper_bounds (tuple[int, ...]): Inclusive upper bounds.
        record_ids (tuple[int, ...]): Row IDs in the leaf.
    """
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    record_ids: tuple[int, ...]


def _validate_database(
    database: np.ndarray,
) -> tuple[int, int]:
    """Check the database shape.

    Args:
        database (numpy.ndarray): Input array.

    Returns:
        tuple[int, int]: Row and coordinate counts.

    Raises:
        ValueError: If the array is not 2D or either dimension is empty.
    """
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
    """Split records by alternating axes and median positions.

    Args:
        database (numpy.ndarray): Nonempty 2D integer array of shape (rows, dimensions).
        leaf_capacity (int): Positive target capacity.
        stats (dict[str, int] | None): Optional node/depth counters to update.
        max_depth (int | None): Optional root-relative depth limit. Defaults to None.

    Returns:
        list[_PlainKDLeaf]: Plaintext leaf regions.

    Raises:
        ValueError: If the array shape or leaf capacity is invalid.
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
        """Recursively partition IDs and accumulate leaf regions.

        Args:
            record_ids (list[int]): IDs to partition.
            depth (int): Current root-relative depth.
            lower_bounds (list[int]): Inclusive region lower bounds.
            upper_bounds (list[int]): Inclusive region upper bounds.

        Returns:
            None: Updates the leaves list and optional statistics.
        """
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
    """Build leaf regions and encrypt every coordinate bound.

    Coordinates are converted with int; integer input is a caller constraint.
    A depth limit takes precedence over capacity. Nonnegative max_depth is
    expected but not validated. Encryption callback exceptions propagate.

    Args:
        database (numpy.ndarray): Nonempty 2D integer array of shape (rows, dimensions).
        encrypt_value (Callable[[int], int]): Integer encryption callback.
        leaf_capacity (int): Positive target records per leaf. Defaults to 16.
        max_depth (int | None): Depth limit with root at zero. Defaults to None.

    Returns:
        EncryptedKDTree: Encrypted regions and visible index metadata.

    Raises:
        ValueError: If the database is not nonempty and 2D or leaf_capacity <= 0.
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

"""Partition integer records into KD-tree leaves and encrypt their bounds.
Tree construction runs on plaintext data in the trusted owner context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
class EncryptedKDNode:
    # Eje utilizado para dividir la región
    axis: int | None

    # Límites cifrados de la región
    encrypted_lower_bounds: tuple[int, ...]
    encrypted_upper_bounds: tuple[int, ...]

    # Referencias a los nodos hijos
    left: "EncryptedKDNode | None" = None
    right: "EncryptedKDNode | None" = None

    # Identificadores de registros en nodos hoja
    record_ids: tuple[int, ...] | None = None

@dataclass(frozen=True)
class EncryptedKDTree:
    root: EncryptedKDNode
    dimensions: int
    database_size: int
    leaf_capacity: int


@dataclass(frozen=True)
class _PlainKDNode:
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]

    axis: int | None = None

    left: "_PlainKDNode | None" = None
    right: "_PlainKDNode | None" = None

    record_ids: tuple[int, ...] | None = None


def _validate_database(
    database: npt.NDArray,
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
            "The database must be a 2D array."
        )

    rows, dimensions = database.shape

    if rows == 0:
        raise ValueError(
            "The database cannot be empty."
        )

    if dimensions == 0:
        raise ValueError(
            "There must be at least one attribute."
        )

    return rows, dimensions



def _build_plain_kd_tree(
    database: npt.NDArray,
    leaf_capacity: int,
    stats: dict[str, int] | None = None,
    max_depth: int | None = None,
) -> _PlainKDNode:

    rows, dimensions = _validate_database(database)

    if leaf_capacity <= 0:
        raise ValueError("leaf_capacity must be greater than zero.")

    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth cannot be negative.")

    root_lb = [
        int(database[:, axis].min())
        for axis in range(dimensions)
    ]

    root_ub = [
        int(database[:, axis].max())
        for axis in range(dimensions)
    ]

    def build(
        record_ids: list[int],
        depth: int,
        lower_bounds: list[int],
        upper_bounds: list[int],
    ) -> _PlainKDNode:

        if stats is not None:
            stats["total_nodes"] = stats.get("total_nodes", 0) + 1
            stats["max_depth"] = max(
                stats.get("max_depth", 0), depth
            )

        # Crear una hoja cuando se alcance la capacidad
        # o la profundidad máxima.
        if (
            len(record_ids) <= leaf_capacity
            or (max_depth is not None and depth >= max_depth)
        ):
            return _PlainKDNode(
                lower_bounds=tuple(lower_bounds),
                upper_bounds=tuple(upper_bounds),
                record_ids=tuple(record_ids),
            )

        # Seleccionar eje alternando dimensiones
        axis = depth % dimensions

        # Ordenar registros por el eje seleccionado
        ordered = sorted(
            record_ids,
            key=lambda rid: (int(database[rid, axis]), rid),
        )

        median = len(ordered) // 2

        left_ids = ordered[:median]
        right_ids = ordered[median:]

        left_max = max(
            int(database[rid, axis])
            for rid in left_ids
        )

        right_min = min(
            int(database[rid, axis])
            for rid in right_ids
        )

        left_lb = lower_bounds.copy()
        left_ub = upper_bounds.copy()

        right_lb = lower_bounds.copy()
        right_ub = upper_bounds.copy()

        if left_max < right_min:
            split = (left_max + right_min) // 2
            left_ub[axis] = split
            right_lb[axis] = split + 1
        else:
            # Los valores duplicados pueden producir
            # regiones con fronteras compartidas.
            split = left_max
            left_ub[axis] = split
            right_lb[axis] = split

        # Construir y conservar ambos subárboles
        left_child = build(
            left_ids,
            depth + 1,
            left_lb,
            left_ub,
        )

        right_child = build(
            right_ids,
            depth + 1,
            right_lb,
            right_ub,
        )

        # Devolver el nodo padre con sus dos hijos
        return _PlainKDNode(
            lower_bounds=tuple(lower_bounds),
            upper_bounds=tuple(upper_bounds),
            axis=axis,
            left=left_child,
            right=right_child,
        )

    return build(
        list(range(rows)),
        0,
        root_lb,
        root_ub,
    )

def build_encrypted_kd_tree(
    database: npt.NDArray,
    encrypt_value: Callable[[int], int],
    leaf_capacity: int = 16,
    max_depth: int | None = None,
) -> EncryptedKDTree:

    rows, dimensions = _validate_database(database)

    # Construir árbol en texto claro
    plain_root = _build_plain_kd_tree(
        database,
        leaf_capacity,
        max_depth=max_depth,
    )

    # Cifrar cada nodo conservando la jerarquía
    def encrypt_node(
        node: _PlainKDNode,
    ) -> EncryptedKDNode:

        encrypted_lb = tuple(
            encrypt_value(value)
            for value in node.lower_bounds
        )

        encrypted_ub = tuple(
            encrypt_value(value)
            for value in node.upper_bounds
        )

        left_child = (
            encrypt_node(node.left)
            if node.left is not None
            else None
        )

        right_child = (
            encrypt_node(node.right)
            if node.right is not None
            else None
        )

        return EncryptedKDNode(
            axis=node.axis,
            encrypted_lower_bounds=encrypted_lb,
            encrypted_upper_bounds=encrypted_ub,
            left=left_child,
            right=right_child,
            record_ids=node.record_ids,
        )

    encrypted_root = encrypt_node(plain_root)

    return EncryptedKDTree(
        root=encrypted_root,
        dimensions=dimensions,
        database_size=rows,
        leaf_capacity=leaf_capacity,
    )

def print_kd_tree(
    node: EncryptedKDNode,
    depth: int = 0,
) -> None:

    indent = "    " * depth

    if node.record_ids is not None:
        print(
            f"{indent}HOJA: "
            f"{len(node.record_ids)} registros"
        )
        return

    print(f"{indent}NODO: eje={node.axis}")

    if node.left is not None:
        print(f"{indent}IZQUIERDA:")
        print_kd_tree(node.left, depth + 1)

    if node.right is not None:
        print(f"{indent}DERECHA:")
        print_kd_tree(node.right, depth + 1)

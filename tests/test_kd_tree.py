from collections import Counter

import numpy as np
import pytest

from sknnpaillier import build_encrypted_kd_tree


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_leaf_partition_and_bounds(capacity):
    database = np.array([[-2, 0], [2, 3], [2, 3], [0, -1]])
    calls = []

    def encrypt(value):
        calls.append(value)
        return value + 100

    tree = build_encrypted_kd_tree(database, encrypt, leaf_capacity=capacity)
    assert tree.dimensions == 2
    assert tree.database_size == len(database)
    ids = [i for leaf in tree.leaves for i in leaf.record_ids]
    assert Counter(ids) == Counter(range(len(database)))
    assert len(calls) == 2 * tree.dimensions * len(tree.leaves)
    for leaf in tree.leaves:
        assert len(leaf.record_ids) <= capacity
        lower = np.array(leaf.encrypted_lower_bounds) - 100
        upper = np.array(leaf.encrypted_upper_bounds) - 100
        for record_id in leaf.record_ids:
            assert np.all(lower <= database[record_id])
            assert np.all(database[record_id] <= upper)


def test_depth_zero():
    tree = build_encrypted_kd_tree(
        np.array([[0], [1], [2]]), int, leaf_capacity=1, max_depth=0,
    )
    assert len(tree.leaves) == 1
    assert tree.leaves[0].record_ids == (0, 1, 2)


@pytest.mark.parametrize("database, capacity", [
    (np.array([1, 2]), 1), (np.empty((0, 2)), 1),
    (np.empty((2, 0)), 1), (np.array([[1]]), 0),
])
def test_invalid_database(database, capacity):
    with pytest.raises(ValueError):
        build_encrypted_kd_tree(database, int, leaf_capacity=capacity)

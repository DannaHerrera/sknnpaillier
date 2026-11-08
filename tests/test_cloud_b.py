import numpy as np
import pytest

from sknnpaillier import EncryptedKDLeaf


def test_homomorphic_operations(cloud, transformer, key_manager):
    a, b = transformer.encrypt(3), transformer.encrypt(4)
    modulus = key_manager.public_key.n
    assert key_manager.decrypt(cloud.encrypted_add(a, b)) == 7
    assert key_manager.decrypt(cloud.encrypted_difference(a, b)) == (-1) % modulus
    assert key_manager.decrypt(cloud.encrypted_scalar_multiply(a, -2)) == (-6) % modulus
    assert key_manager.decrypt(cloud.secure_multiply(a, b)) == 12
    result = cloud.secure_squared_distance(
        transformer.encrypt_vector([-2, 3]), transformer.encrypt_vector([1, -1]),
    )
    assert key_manager.decrypt(result) == 25
    with pytest.raises(ValueError):
        cloud.secure_squared_distance([a], [a, b])


@pytest.mark.parametrize("point, enclosed, distance", [
    ([1, 1], True, 0), ([0, 2], True, 0), ([-1, 3], False, 2),
])
def test_region(cloud, transformer, key_manager, point, enclosed, distance):
    leaf = EncryptedKDLeaf(
        0, tuple(transformer.encrypt_vector([0, 0])),
        tuple(transformer.encrypt_vector([2, 2])), (0,),
    )
    query = transformer.encrypt_vector(point)
    assert cloud.secure_point_enclosure(query, leaf) is enclosed
    assert key_manager.decrypt(cloud.secure_region_min_distance(query, leaf)) == distance


@pytest.mark.parametrize("capacity, depth", [(1, None), (2, None), (1, 0)])
@pytest.mark.parametrize("query", [[1, 1], [-3, 5]])
@pytest.mark.parametrize("k", [1, 5])
def test_inference(cloud, transformer, key_manager, capacity, depth, query, k):
    database = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [0, 0]])
    transformer.send_database_to_cloud(database, leaf_capacity=capacity, max_depth=depth)
    ids, records, distances = transformer.send_query_to_cloud(np.array(query), k)
    expected_distances = [sum((int(x) - q) ** 2 for x, q in zip(row, query))
                          for row in database]
    expected_ids = sorted(range(len(database)), key=lambda i: (expected_distances[i], i))[:k]
    assert ids == expected_ids
    assert [key_manager.decrypt(c) for c in distances] == [expected_distances[i] for i in ids]
    assert [[key_manager.decrypt(c) for c in row] for row in records] == database[ids].tolist()
    assert k <= cloud.last_query_stats["distance_evaluations"] <= len(database)


def test_missing_configuration(cloud, transformer):
    query = transformer.encrypt_vector([0, 0])
    with pytest.raises(RuntimeError):
        cloud.process_query(query, 1)
    cloud.set_database([query])
    with pytest.raises(RuntimeError):
        cloud.process_query(query, 1)


@pytest.mark.parametrize("k", [0, 3])
def test_invalid_k(cloud, transformer, k):
    transformer.send_database_to_cloud(np.array([[0], [1]]))
    with pytest.raises(ValueError):
        transformer.send_query_to_cloud(np.array([0]), k)


def test_invalid_query_dimension(cloud, transformer):
    transformer.send_database_to_cloud(np.array([[0, 1]]))
    with pytest.raises(ValueError):
        transformer.send_query_to_cloud(np.array([0]), 1)

from unittest.mock import Mock

import numpy as np
import pytest


def test_vector_and_database(transformer, key_manager):
    database = np.array([[-2, 0], [1, 9]])
    encrypted = transformer.encrypt_database(database)
    modulus = key_manager.public_key.n
    assert [[key_manager.decode_signed(key_manager.decrypt(c), modulus)
             for c in row] for row in encrypted] == database.tolist()
    assert [key_manager.decrypt(c) for c in transformer.encrypt_vector([0, 9])] == [0, 9]


def test_rerandomize(transformer, key_manager, monkeypatch):
    ciphertext = transformer.encrypt(7)
    # r=2 is coprime to the odd modulus and guarantees a nontrivial E(0).
    monkeypatch.setattr("sknnpaillier.secure_data_t.secrets.randbelow", lambda n: 2)
    refreshed = transformer.rerandomize(ciphertext)
    assert refreshed != ciphertext
    assert key_manager.decrypt(refreshed) == 7


def test_cloud_required(transformer):
    with pytest.raises(RuntimeError):
        transformer.send_database_to_cloud(np.array([[1]]))
    with pytest.raises(RuntimeError):
        transformer.send_query_to_cloud(np.array([1]), 1)


def test_database_delivery_and_query_delegation(transformer, key_manager):
    cloud = Mock()
    transformer.set_cloud(cloud)
    database = np.array([[1, 2], [3, 4]])
    transformer.send_database_to_cloud(database, leaf_capacity=1)
    encrypted = cloud.set_database.call_args.args[0]
    assert [[key_manager.decrypt(c) for c in row] for row in encrypted] == database.tolist()
    tree = cloud.set_kd_tree.call_args.args[0]
    assert (tree.database_size, tree.dimensions) == database.shape
    assert transformer.send_query_to_cloud(np.array([5, 6]), 1) is cloud.process_query.return_value
    query, k = cloud.process_query.call_args.args
    assert [key_manager.decrypt(c) for c in query] == [5, 6]
    assert k == 1

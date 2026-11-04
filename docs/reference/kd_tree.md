# KD-Tree Index

The owner builds leaf regions from a nonempty integer matrix and encrypts their
bounds. Record IDs and index metadata remain visible to the query processor.

::: sknnpaillier.kd_tree
    options:
      members:
        - EncryptedKDLeaf
        - EncryptedKDTree
        - build_encrypted_kd_tree

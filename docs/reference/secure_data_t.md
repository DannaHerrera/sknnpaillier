# Secure Data Transformation

The owner-side transformer encrypts integer coordinates, prepares the spatial
index and delegates encrypted queries. Final decryption belongs to the client.

::: sknnpaillier.secure_data_t
    options:
      members:
        - SecureDataTransformationModule

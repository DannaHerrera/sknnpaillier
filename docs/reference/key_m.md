# Key Management

The manager generates Paillier parameters and performs decryption and trusted
nonlinear operations. `PrivateKey` is available from this module, but is not
reexported from the package root.

::: sknnpaillier.key_m
    options:
      members:
        - PublicKey
        - PrivateKey
        - KeyManagementModule

The mathematical helpers `lcm`, `is_probable_prime`, and `generate_prime` are
implemented and documented in [Shared Utilities](utils.md). Their imports from
`sknnpaillier.key_m` remain supported through reexports.

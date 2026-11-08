# Shared Utilities

Fixed-point helpers convert coordinates and decrypted squared distances.
Quantization accepts finite signed real coordinates; normalization to `[0, 1]`
is a choice made by the examples, not a restriction of this API. Rounding uses
NumPy's nearest-even rule, and values outside the output integer range are
rejected before conversion. This check does not validate Paillier modulus bounds.

The number-theory helpers are the implementations used by key generation.
Their algorithms and existing validation are preserved. For compatibility,
`lcm`, `is_probable_prime`, and `generate_prime` remain importable from `key_m`.
All five functions are also exported from the package root.

::: sknnpaillier.utils
    options:
      members:
        - quantize
        - dequantize_distance
        - lcm
        - is_probable_prime
        - generate_prime

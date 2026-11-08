# Usage

Install the [development environment](installation.md) to run the examples.
See the [environment reference](environment.md) for local defaults and overrides.

## Integer records

```bash
poetry run python examples/basic_usage.py
```

The example uses twelve 2D integer records and local binary labels. It generates
keys, encrypts records and region bounds, requests neighbors, and decrypts the
result on the client. It verifies IDs, coordinates and squared distances against
an independent plaintext reference before computing a majority label.

With default settings, the selected IDs are `[1, 2, 3]`, squared distances are
`[1, 1, 1]`, and the predicted label is `0`. Equal distances are ordered by ID.

## Quantized floats

```bash
poetry run python examples/float_scaling_usage.py
```

The second example uses sixteen 2D points in `[0, 1]`. Both records and the
query are quantized using `Q(x) = rint(scale * x)`. Only the resulting integers
are sent to the library. Coordinates are reconstructed as `Q(x) / scale`;
decrypted squared distances are divided by `scale**2`.

Rounding introduces approximation. The example checks exact agreement with the
integer reference and compares the selected IDs and majority label with the
original float reference. With defaults, IDs are `[0, 1, 2]` and the label is `0`.
Changing the scale does not guarantee that the float neighbor order survives.

## Public API workflow

```python
from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
    resolve_key_bits,
)

keys = KeyManagementModule(resolve_key_bits("80"))  # Demonstration profile.
owner = SecureDataTransformationModule(keys.public_key)
cloud = CloudBQueryProcessingModule(owner, keys)
owner.set_cloud(cloud)
```

Call `owner.send_database_to_cloud` with an integer matrix. This single call
encrypts records, constructs leaf regions and encrypts their bounds. Then call
`owner.send_query_to_cloud(query, k)` with an integer vector. Its return value
contains three aligned lists: IDs, encrypted records, and encrypted squared
distances. Use `keys.decrypt` on ciphertexts and `keys.decode_signed` on residues
representing signed coordinates. Keep labels on the client.

`SECURITY_LEVEL` is interpreted by the examples, not automatically by the API.
For a larger demonstration key, use an explicit environment override:

```bash
SECURITY_LEVEL=112 poetry run python examples/basic_usage.py
```

See the [API reference](reference/api.md) for parameters, return values and
implemented validation. The library does not detect every domain or index
consistency error; consult the [limitations](index.md#current-limitations).


## Classification metrics

```bash
poetry run python examples/evaluation_metrics_usage.py
```

This independent example requires the development group, including scikit-learn.
It generates 60 reproducible binary-class samples, clips coordinates to `[0, 1]`,
and makes a stratified 70/30 split: 42 training samples and 18 held-out samples.
Both partitions use the same shared `quantize` function and configured scale.
The index is built once from the training partition only.

Each held-out query retrieves encrypted neighbors. The client decrypts the
records, checks their training-row identity, and votes using client-held labels.
The script prints `classification_report` with precision, recall, F1, support,
and macro/weighted averages. It then prints a 2-by-2 confusion matrix with true
classes as rows and predicted classes as columns, followed by TN/FP/FN/TP counts.
Undefined report divisions use zero. No particular accuracy is guaranteed.

The example reuses `examples/.env` settings. `SKNN_EXAMPLE_K` must be odd and
between 1 and 42; the default remains 3. The default security level 80 is for
demonstration only. A larger level increases execution time across the 18
queries. The resulting report evaluates the quantized classifier, not a separate
plaintext classifier. Data separation is deterministic; cryptographic randomness
is fresh. No plots or external data files are required.

## Shared fixed-point helpers

`quantize` and `dequantize_distance` are available from `sknnpaillier.utils` and
from the package root. The float example no longer defines separate copies.
See [Shared Utilities](reference/utils.md) for their contracts, including scale
validation, signed coordinates, integer-range checks, and squared-distance units.

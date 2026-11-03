# sknnpaillier

`sknnpaillier` implements encrypted k-nearest-neighbor search using Paillier
and spatial partitioning based on KD-tree leaves. The data owner encrypts
integer records and region bounds. A cloud query processor uses homomorphic
operations and a trusted key manager to select neighbors. The client decrypts
the returned records and squared distances; labels remain on the client.

## Requirements and installation

The library requires Python 3.10 or later and Poetry 2.x. Documentation tools
require Python 3.11 or later. NumPy is the base dependency; Rich is optional.

```bash
# Library only
poetry install --only main

# Library, tests, and examples
poetry install --with dev

# Include documentation tools (Python >=3.11)
poetry install --with dev,docs
```

Create local configuration files from the public templates if needed. These
commands preserve existing files:

```bash
cp -n tests/.env.example tests/.env
cp -n examples/.env.example examples/.env
```

Local `.env` files are ignored by Git. Tests and examples load their own file
using its directory, with shell variables taking precedence. The library itself
does not load `.env` files. Missing files use defaults from the consuming code.

## Quick start

Run either self-contained example in the development environment:

```bash
poetry run python examples/basic_usage.py
poetry run python examples/float_scaling_usage.py
```

The first uses integer records. The second quantizes floats using a common
scale, then divides decrypted squared distances by `scale**2`. Both verify
results against client-side references. Quantization can change neighbor order;
the float example checks whether the order is preserved for its dataset.

The API is available directly from the installed package:

```python
from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
    resolve_key_bits,
)

# Demonstration setting, not a production security recommendation.
keys = KeyManagementModule(resolve_key_bits("80"))
owner = SecureDataTransformationModule(keys.public_key)
cloud = CloudBQueryProcessingModule(owner, keys)
owner.set_cloud(cloud)
```

`SECURITY_LEVEL` accepts only `test`, 80, 112, or 128; numeric strings are
accepted for environment settings. These map to 128, 1024, 2048, and 3072 modulus
bits, respectively. Raw modulus sizes and alternative aliases are rejected.
The `test` setting is for unit tests; level 80 is for demonstrations only.

## Documentation

- [Introduction](docs/index.md)
- [Requirements and installation](docs/installation.md)
- [Environment variables](docs/environment.md)
- [Usage](docs/usage.md)
- [API reference](docs/reference/api.md)

Build or preview the Zensical site with the documentation group installed:

```bash
poetry run zensical build --clean
poetry run zensical serve
```

The API reference is generated from source docstrings. It covers constructors,
public methods, parameters, return values, and implemented error conditions.
The site target is <https://shanelreyes.github.io/sknnpaillier/>.

## Development and CI

```bash
poetry run pytest -q
poetry run ruff check sknnpaillier tests examples
```

CI runs pytest on Python 3.10 and 3.11 for pull requests and pushes to `main`.
Documentation builds on pull requests and deploys from `main`. GitHub Pages
requires **Settings → Pages → Build and deployment → Source: GitHub Actions**
and an allowed `github-pages` deployment environment. No generated-site branch
or personal access token is required.

Commit `pyproject.toml` and `poetry.lock` together when dependencies change.
Run `poetry lock` only when the lock needs updating. The current lock and
installation have been validated locally by the project maintainer.

## Current limitations

This is a prototype. Records must use integer coordinates; modular distance
and ranking-score overflow is not detected. The key manager and cloud run in
the same process without enforced isolation. Record IDs, access patterns and
comparison outcomes are visible. Selecting a security level does not certify
the implementation or the full protocol.

The legacy experiments under `scripts/` are outside the current library
migration. Their imports still require migration. The self-contained examples
do not require the external Letter Recognition dataset.

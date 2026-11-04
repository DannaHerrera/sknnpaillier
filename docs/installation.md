# Requirements and Installation

## Requirements

- Python 3.10 or later for the library, tests, and examples.
- Python 3.11 or later for documentation tools.
- Poetry 2.x for dependency management and virtual environments.

Run the commands below from the repository root. Poetry manages a virtual
environment; use `poetry run` to select its Python interpreter and tools.

## Dependency groups

```bash
# Base library: NumPy and the installed sknnpaillier package
poetry install --only main

# Development tools, pytest, python-dotenv, and scikit-learn
poetry install --with dev

# Development and documentation tools, using Python >=3.11
poetry install --with dev,docs

# Documentation without test tools
poetry install --only main,docs

# Optional Rich console formatting
poetry install --with dev --extras rich
```

Plain `poetry install` installs the library and non-optional groups, including
`dev`. The optional `docs` group contains Zensical and `mkdocstrings-python`.
The library does not require those tools or `python-dotenv` at runtime.

Use the committed lock file for reproducible installation. When dependency
constraints change, update it locally and commit it with the manifest:

```bash
poetry lock
poetry check --lock
```

CI checks lock consistency and installs locked versions; it does not resolve
new versions on every run. Existing Poetry metadata deprecation warnings do
not indicate a failed installation; migrating that metadata is a separate task.

## Local configuration

The templates contain public development defaults, not credentials. Copy them
only if you want local overrides; `cp -n` preserves existing files:

```bash
cp -n tests/.env.example tests/.env
cp -n examples/.env.example examples/.env
```

The resulting `.env` files are ignored by Git. Each consumer loads only the
file in its own directory, independently of the current working directory.
The environment-variable [reference](environment.md) lists defaults and valid
values. No local file is required for CI or for the default examples.

## Validation and documentation

```bash
poetry run pytest -q
poetry run ruff check sknnpaillier tests examples
poetry run zensical build --clean
poetry run zensical serve
```

The last two commands require the `docs` group and Python 3.11 or later.
Building documentation does not execute the example programs. The static
output directory is `site/` and is ignored by Git.

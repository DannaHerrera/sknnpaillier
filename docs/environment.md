# Environment Variables Reference

## Loading and precedence

Tests load `tests/.env` when the session configuration fixture is created.
Examples load `examples/.env` inside `main()`. Both use an explicit path based
on `__file__` with `load_dotenv(..., override=False)`; they do not search parent
directories. Importing the library or an example does not load a `.env` file.

The precedence is:

1. Existing terminal or CI environment.
2. The consumer's local `.env` file.
3. Defaults in the consuming code.

Local files are ignored by Git. Versioned `.env.example` files document the
settings. A shell variable, including an empty value, takes precedence; an
empty value is valid only where the table explicitly allows it.

## Security level

| `SECURITY_LEVEL` value | Paillier modulus bits | Purpose |
|---|---:|---|
| `test` | 128 | Fast unit tests only |
| `80` | 1024 | Demonstrations only |
| `112` | 2048 | Classical security-level policy |
| `128` | 3072 | Classical security-level policy |

`resolve_key_bits` accepts exactly the string `"test"`, integer levels 80,
112, 128, and their exact numeric strings. Other values raise `ValueError`,
including `0`, `1024`, `2048`, `3072`, `legacy`, `standard`, booleans, floats,
alternate capitalization, or surrounding whitespace. The value `128` always
selects a 3072-bit modulus; it never means a 128-bit test key.

`KeyManagementModule(key_bits)` still accepts a raw modulus length. The resolver
is a configuration adapter, not a change to that constructor.

The mapping uses the classical IFC/RSA size correspondence in
[NIST SP 800-57 Part 1 Rev. 5](https://doi.org/10.6028/NIST.SP.800-57pt1r5)
as a project policy. It is not a formal equivalence for Paillier or a
certification of the implementation. The test profile has no practical
security, and the historical 80-bit profile is not suitable for new protection.

## Tests: `tests/.env`

| Variable | Default | Accepted values and effect |
|---|---|---|
| `SECURITY_LEVEL` | `test` | A supported level; controls the session key fixture. |
| `SKNN_TEST_LEAF_CAPACITY` | `2` | Positive integer; configured integration case. |
| `SKNN_TEST_MAX_DEPTH` | empty | Nonnegative integer, or empty for no limit. |
| `SKNN_LOG_DISABLED` | `1` | Boolean flag; controls logging in ordinary fixtures. |

Parametrized boundary tests retain their explicit inputs. The configured
integration case uses the capacity and depth above. CI explicitly sets
`SECURITY_LEVEL=test` and disables logging. Selecting a larger level locally
increases key-generation time. Import-isolation tests manage their own logging
environment and do not inherit ordinary fixture settings.

## Examples: `examples/.env`

| Variable | Default | Accepted values and effect |
|---|---|---|
| `SECURITY_LEVEL` | `80` | Supported level, resolved before key generation. |
| `SKNN_EXAMPLE_K` | `3` | Positive odd integer within the dataset size. |
| `SKNN_EXAMPLE_LEAF_CAPACITY` | `2` | Positive integer. |
| `SKNN_EXAMPLE_MAX_DEPTH` | empty | Nonnegative integer, or empty for no limit. |
| `SKNN_EXAMPLE_SCALE` | `1000` | Positive integer; float and evaluation examples. |
| `SKNN_EXAMPLE_SEED` | `42` | Nonnegative integer; float and evaluation examples. |
| `SKNN_LOG_DISABLED` | `1` | Boolean flag; examples default to no logging output. |

The example loaders validate tree parameters before generating keys. Examples
with floats also validate their scale and seed. A different scale can change
neighbor order. The float-scaling example checks against the original float
reference; the evaluation example measures the quantized classifier. None of
the examples reads an external dataset or needs an input-file path.

## Logger: `SKNN_LOG_*`

The following defaults apply to the library's `get_logger` factory, before
any test/example defaults or explicit constructor options. Boolean flags treat
`1`, `true`, and `yes` as true, case-insensitively. The test/example loaders
accept `0`, `false`, and `no` for false and reject other disabled-flag values;
the logger itself treats other boolean strings as false.

| Variable | Factory default | Effect |
|---|---|---|
| `SKNN_LOG_DISABLED` | `0` | Return `DumbLogger` instead of constructing handlers. |
| `SKNN_LOG_LEVEL` | `DEBUG` | Global logger threshold. |
| `SKNN_LOG_CONSOLE_LEVEL` | `INFO` | Console handler threshold. Direct `Log` construction defaults to `DEBUG`. |
| `SKNN_LOG_FILE_LEVEL` | `INFO` | Normal file handler threshold. |
| `SKNN_LOG_TO_FILE` | `0` | Enable the normal rotating log file. |
| `SKNN_LOG_ERROR_FILE` | `0` | Enable a separate file for ERROR and higher. |
| `SKNN_LOG_PATH` | `.sknn/log` | Log directory; relative paths use the process working directory. |
| `SKNN_LOG_RICH` | `0` | Use Rich console output; requires the `rich` extra. |
| `SKNN_LOG_JSON_INDENT` | `0` | Positive digit string for indentation; other values use compact JSON. |
| `SKNN_LOG_ROTATION_WHEN` | `m` | Rotation unit, interpreted by `TimedRotatingFileHandler`. |
| `SKNN_LOG_ROTATION_INTERVAL` | `10` | Integer rotation interval in the selected unit. |

Common rotation units are `S`, `M`, `H`, `D`, `midnight`, and `W0` through `W6`;
unit names are case-insensitive. The default is ten minutes. Use a positive
interval: the library does not add its own positivity validation.

Recognized textual levels are resolved through Python's `logging` module;
unrecognized names fall back to DEBUG. The global threshold still filters
records even if a handler threshold is lower. The normal file excludes ERROR
and higher records, so an error-only file requires its own flag.

Files are created only when their flags are enabled. Direct `Log(disabled=True)`
attaches a `NullHandler`; the factory returns `DumbLogger` without handlers.
Invalid rotation settings can raise `ValueError`, missing Rich can raise
`ImportError`, and file failures can raise `OSError` when a logger is created.
Explicit `Log` constructor options override the corresponding environment
values; custom output filenames are constructor arguments, not environment
variables. Their parent directories are not created automatically.


The evaluation example also reads the example settings above. It uses a fixed
60-sample binary dataset and 70/30 split; its maximum valid odd k is 41 because
there are 42 training rows. Scale and seed affect this example as well as the
float-scaling example. The classification report uses the quantized predictions.

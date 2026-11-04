# Configuration and Logging

## Security-level policy

This function translates explicit settings only. It does not read environment
variables, generate keys or change the key-manager constructor.

::: sknnpaillier.config
    options:
      members:
        - resolve_key_bits

## Logging

The factory creates a new instance on every call. Its default console threshold
differs from direct `Log` construction; the docstrings and
[environment reference](../environment.md#logger-sknn_log_) describe both paths.

::: sknnpaillier.log
    options:
      members:
        - DumbLogger
        - JsonFormatter
        - Log
        - get_logger

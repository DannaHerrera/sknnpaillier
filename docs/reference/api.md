# API Reference

The package reexports its main classes, index structures, index builder and
security-level resolver from `sknnpaillier`. The pages below are generated from
module, class, constructor and method docstrings using the Python handler for
mkdocstrings. Signatures are read from source, not maintained as separate text.

| Module | Purpose |
|---|---|
| [cloud_b](cloud_b.md) | Homomorphic operations and encrypted neighbor selection. |
| [kd_tree](kd_tree.md) | Leaf-region construction and encrypted bounds. |
| [key_m](key_m.md) | Key parameters, key generation and trusted operations. |
| [secure_data_t](secure_data_t.md) | Record encryption and delivery to the cloud. |
| [utils](utils.md) | Fixed-point conversion and number-theory helpers. |
| [config and log](configuration.md) | Security-level translation and logging configuration. |

Constructor parameters appear with class documentation. Each callable describes
its arguments and return value; exception sections describe implemented checks
and relevant propagated failures. Missing validation is documented as a caller
constraint rather than an exception that the code does not raise.

The [environment reference](../environment.md) documents consumer defaults.
Creating an API object does not automatically load a `.env` file.

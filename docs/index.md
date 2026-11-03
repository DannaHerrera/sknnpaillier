# Introduction

`sknnpaillier` implements encrypted k-nearest-neighbor search using the Paillier
cryptosystem and spatial partitioning based on KD-tree leaves. It works with
integer coordinates and returns encrypted records and squared distances.

## Architecture

1. The key manager generates a Paillier key pair.
2. The data owner encrypts records, constructs leaf regions in plaintext, and
   encrypts their coordinate bounds.
3. The cloud finds and evaluates candidate regions. It uses homomorphic
   arithmetic and delegates masked multiplication and comparison to the
   trusted key manager.
4. The client decrypts the selected records and distances. Labels remain on
   the client and can be used for classification.

The exported index is a collection of leaf regions, not a hierarchy traversed
by the cloud. Record IDs break equal-distance ties deterministically.

## Documentation guide

- [Requirements and installation](installation.md): environment setup and groups.
- [Environment variables](environment.md): settings, defaults, and precedence.
- [Usage](usage.md): integer and quantized-float workflows.
- [API reference](reference/api.md): classes, constructors, and public methods.

## Current limitations

- Paillier arithmetic operates modulo `n`. Coordinates must be encoded as
  integers; squared distances and ranking scores must fit their intended
  ranges. Modular overflow is silent and is not detected by this prototype.
- Float quantization introduces rounding error and may change the nearest
  neighbors. The example verifies its result for the selected data only.
- The cloud holds a reference to a key manager that exposes decryption. The
  trusted context is a convention within one process, not an enforced boundary.
- Record IDs, access patterns, and comparison results are visible.
- Test keys provide no practical security. The 80-bit profile is historical
  and is used only for demonstrations. The level mapping does not certify
  Paillier or the complete protocol.

Legacy experiments under `scripts/` are separate from the documented library
workflow and still need their import migration.

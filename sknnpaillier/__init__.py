"""Expose the public API for Paillier-based nearest-neighbor search."""

from .cloud_b import CloudBQueryProcessingModule
from .config import resolve_key_bits
from .kd_tree import EncryptedKDLeaf, EncryptedKDTree, build_encrypted_kd_tree
from .key_m import KeyManagementModule, PublicKey
from .secure_data_t import SecureDataTransformationModule
from .utils import (
    dequantize_distance,
    generate_prime,
    is_probable_prime,
    lcm,
    quantize,
)

__all__ = [
    "CloudBQueryProcessingModule",
    "EncryptedKDLeaf",
    "EncryptedKDTree",
    "KeyManagementModule",
    "PublicKey",
    "SecureDataTransformationModule",
    "build_encrypted_kd_tree",
    "resolve_key_bits",
    "dequantize_distance",
    "generate_prime",
    "is_probable_prime",
    "lcm",
    "quantize",
]

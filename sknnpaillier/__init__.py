"""Expose the public API for Paillier-based nearest-neighbor search."""

from sknnpaillier.cloud_b import CloudBQueryProcessingModule
from sknnpaillier.kd_tree import (
    EncryptedKDNode,
    EncryptedKDTree,
    build_encrypted_kd_tree,
)
from sknnpaillier.key_m import KeyManagementModule, PublicKey
from sknnpaillier.secure_data_t import SecureDataTransformationModule
from sknnpaillier.utils import (
    dequantize_distance,
    generate_prime,
    is_probable_prime,
    lcm,
    quantize,
    resolve_key_bits
)

__all__ = [
    "CloudBQueryProcessingModule",
    "EncryptedKDNode",
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

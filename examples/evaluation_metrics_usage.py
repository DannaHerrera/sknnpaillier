"""Evaluate encrypted k-NN on synthetic binary data using scikit-learn.

Run from the repository root after installing the development dependencies:
    poetry run python examples/evaluation_metrics_usage.py
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
    resolve_key_bits,
)
from sknnpaillier.utils import quantize


@dataclass(frozen=True)
class Settings:
    """Store validated example parameters."""

    key_bits: int
    k: int
    leaf_capacity: int
    max_depth: int | None
    scale: int
    seed: int


def load_settings() -> Settings:
    """Load the local environment without overriding shell variables."""
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    key_bits = resolve_key_bits(os.getenv("SECURITY_LEVEL", "80"))
    try:
        k = int(os.getenv("SKNN_EXAMPLE_K", "3"))
        capacity = int(os.getenv("SKNN_EXAMPLE_LEAF_CAPACITY", "2"))
        raw_depth = os.getenv("SKNN_EXAMPLE_MAX_DEPTH", "").strip()
        depth = int(raw_depth) if raw_depth else None
        scale = int(os.getenv("SKNN_EXAMPLE_SCALE", "1000"))
        seed = int(os.getenv("SKNN_EXAMPLE_SEED", "42"))
    except ValueError as error:
        raise ValueError("Example parameters must be integers.") from error

    if not 1 <= k <= 42 or k % 2 == 0:
        raise ValueError("SKNN_EXAMPLE_K must be odd and between 1 and 42.")
    if capacity < 1 or (depth is not None and depth < 0):
        raise ValueError("Capacity must be positive; depth must be nonnegative.")
    if not 1 <= scale < np.iinfo(int).max:
        raise ValueError("Scale must be positive and fit NumPy's integer type.")
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    disabled = os.environ.setdefault("SKNN_LOG_DISABLED", "1").lower()
    if disabled not in ("0", "1", "true", "false", "yes", "no"):
        raise ValueError("SKNN_LOG_DISABLED must be a boolean flag.")
    return Settings(key_bits, k, capacity, depth, scale, seed)


def main() -> None:
    """Generate data, classify encrypted queries, and print metrics."""
    settings = load_settings()
    rng = np.random.default_rng(settings.seed)

    # Fixed centers avoid fitting preprocessing on the held-out data.
    centers = np.array([[0.3, 0.3], [0.7, 0.7]])
    labels = np.repeat([0, 1], 30)
    features = np.clip(
        centers[labels] + rng.normal(0.0, 0.2, size=(60, 2)), 0.0, 1.0,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=settings.seed,
        stratify=labels,
    )

    # Both partitions use the same fixed-point scale.
    train_int = quantize(x_train, scale_factor=settings.scale)
    test_int = quantize(x_test, scale_factor=settings.scale)
    print("[1/4] Data: 42 training samples and 18 test samples.")
    print(
        f"      Classes: 0, 1 | k={settings.k} | "
        f"scale={settings.scale} | key bits={settings.key_bits}"
    )

    print("[2/4] Generating keys and preparing the encrypted index...",
          flush=True)
    # SECURITY_LEVEL=80 is a demonstration profile.
    keys = KeyManagementModule(settings.key_bits)
    max_distance = 2 * settings.scale**2
    max_score = max_distance * (len(train_int) + 1) + len(train_int) - 1
    if max_score >= keys.public_key.n:
        raise ValueError("Scale is too large for the selected modulus.")

    owner = SecureDataTransformationModule(keys.public_key)
    cloud = CloudBQueryProcessingModule(owner, keys)
    owner.set_cloud(cloud)
    owner.send_database_to_cloud(
        train_int, leaf_capacity=settings.leaf_capacity,
        max_depth=settings.max_depth,
    )

    print("[3/4] Classifying 18 encrypted test queries...", flush=True)
    y_true = y_test.tolist()
    y_pred: list[int] = []
    for query in test_int:
        ids, encrypted_records, _ = owner.send_query_to_cloud(query, settings.k)
        # The client decrypts neighbors and retains the training labels.
        records = [
            [keys.decode_signed(keys.decrypt(value), keys.public_key.n)
             for value in record]
            for record in encrypted_records
        ]
        if len(ids) != settings.k or records != train_int[ids].tolist():
            raise RuntimeError("Recovered neighbors do not match training rows.")
        votes = np.bincount(y_train[ids], minlength=2)
        y_pred.append(int(np.argmax(votes)))

    print("[4/4] Classification report")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["class 0", "class 1"],
        digits=3, zero_division=0,
    ))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix: rows=true, columns=predicted")
    print("              predicted 0  predicted 1")
    for label, row in enumerate(matrix):
        print(f"true {label:<8} {int(row[0]):>11} {int(row[1]):>12}")
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()

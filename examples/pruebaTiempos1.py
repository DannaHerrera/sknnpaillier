"""Evaluate encrypted k-NN and profile its execution stages.

Run from the repository root after installing development dependencies:
    poetry run python examples/evaluation_metrics_seconds.py

Log files are written to SKNN_METRICS_DIR (default: ./sknn_metrics).
To vary ONLY k (other parameters fixed):
    SKNN_INDEPENDENT_VARIABLE=k SKNN_INDEPENDENT_VALUE=5 \
      poetry run python examples/evaluation_metrics_seconds.py
Runs overwrite output files in the metrics directory; set SKNN_METRICS_DIR
per experiment to retain results.
"""

import csv
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
)
from sknnpaillier.utils import quantize, resolve_key_bits


@dataclass(frozen=True)
class Settings:
    key_bits: int
    k: int
    leaf_capacity: int
    max_depth: int | None
    scale: int
    seed: int
    independent_variable: str
    independent_value: str


def load_settings() -> Settings:
    """Use fixed experimental controls and vary ONE independent variable.

    Set SKNN_INDEPENDENT_VARIABLE and SKNN_INDEPENDENT_VALUE to vary one
    parameter. With neither set, run the fixed baseline.
    """
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    # Diseño experimental de referencia (constantes de control).
    base = {
        "security_level": "80",  # perfil demostrativo, no producción
        "k": 3,
        "leaf_capacity": 2,
        "max_depth": None,
        "scale": 1000,
    }
    # Estas variables se mantienen fijas en todas las corridas.
    seed = 42
    variable = os.getenv("SKNN_INDEPENDENT_VARIABLE", "baseline").strip().lower()
    raw_value = os.getenv("SKNN_INDEPENDENT_VALUE", "").strip()
    allowed = set(base)
    if variable not in allowed | {"baseline"}:
        raise ValueError(f"Independent variable must be one of: {sorted(allowed)} or baseline")
    if variable == "baseline":
        if raw_value:
            raise ValueError("Do not provide SKNN_INDEPENDENT_VALUE for baseline.")
        value = "fixed"
    else:
        if not raw_value:
            raise ValueError("SKNN_INDEPENDENT_VALUE is required when changing a variable.")
        try:
            if variable == "security_level":
                # Se resuelve a bits según las reglas de sknnpaillier.
                resolve_key_bits(raw_value)
                base[variable] = raw_value
            elif variable == "max_depth":
                base[variable] = None if raw_value.lower() in ("none", "null") else int(raw_value)
            else:
                base[variable] = int(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid value for independent variable {variable}: {raw_value}") from error
        value = raw_value

    key_bits = resolve_key_bits(str(base["security_level"]))
    k = int(base["k"])
    capacity = int(base["leaf_capacity"])
    depth = base["max_depth"]
    scale = int(base["scale"])
    if not 1 <= k <= 42 or k % 2 == 0:
        raise ValueError("k must be odd and between 1 and 42.")
    if capacity < 1 or (depth is not None and depth < 0):
        raise ValueError("Capacity must be positive; depth must be nonnegative.")
    if not 1 <= scale < np.iinfo(int).max:
        raise ValueError("Scale must be positive and fit NumPy's integer type.")
    disabled = os.environ.setdefault("SKNN_LOG_DISABLED", "1").lower()
    if disabled not in ("0", "1", "true", "false", "yes", "no"):
        raise ValueError("SKNN_LOG_DISABLED must be a boolean flag.")
    return Settings(key_bits, k, capacity, depth, scale, seed, variable, value)


def configure_logger(directory: Path) -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sknn_experiment_timing")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / "execution.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def timed_stage(logger: logging.Logger, durations: dict[str, float], stage: str):
    """Log duration even when the measured stage raises an exception."""
    start = time.perf_counter_ns()
    logger.info("START | %s", stage)
    try:
        yield
    except Exception:
        logger.exception("FAILED | %s", stage)
        raise
    finally:
        seconds = (time.perf_counter_ns() - start) / 1_000_000_000
        durations[stage] = seconds
        logger.info("END   | %-24s | %.6f s", stage, seconds)


def elapsed_s(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000_000


def main() -> None:
    total_start = time.perf_counter_ns()
    settings = load_settings()
    metrics_dir = Path(os.getenv("SKNN_METRICS_DIR", "sknn_metrics"))
    logger = configure_logger(metrics_dir)
    durations: dict[str, float] = {}
    rows: list[dict[str, int | float]] = []

    logger.info("INDEPENDENT VARIABLE | %s = %s", settings.independent_variable, settings.independent_value)
    logger.info("CONTROL VARIABLES   | n=60, features=2, test_fraction=0.30, seed=%d, noise_std=0.20", settings.seed)
    logger.info("Parameters          | key_bits=%d, k=%d, leaf_capacity=%d, max_depth=%s, scale=%d",
                settings.key_bits, settings.k, settings.leaf_capacity, settings.max_depth, settings.scale)
    logger.info("Detailed library logging is controlled separately by SKNN_LOG_DISABLED=%s",
                os.environ.get("SKNN_LOG_DISABLED"))

    try:
        with timed_stage(logger, durations, "data_generation"):
            rng = np.random.default_rng(settings.seed)
            centers = np.array([[0.3, 0.3], [0.7, 0.7]])
            labels = np.repeat([0, 1], 30)
            features = np.clip(
                centers[labels] + rng.normal(0.0, 0.2, size=(60, 2)), 0.0, 1.0
            )

        with timed_stage(logger, durations, "train_test_split"):
            x_train, x_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3,
                random_state=settings.seed, stratify=labels,
            )

        with timed_stage(logger, durations, "quantize_train"):
            train_int = quantize(x_train, scale_factor=settings.scale)
        with timed_stage(logger, durations, "quantize_test"):
            test_int = quantize(x_test, scale_factor=settings.scale)
        logger.info("Data: %d training samples; %d test queries", len(train_int), len(test_int))

        with timed_stage(logger, durations, "key_generation"):
            keys = KeyManagementModule(settings.key_bits)

        max_distance = 2 * settings.scale**2
        max_score = max_distance * (len(train_int) + 1) + len(train_int) - 1
        if max_score >= keys.public_key.n:
            raise ValueError("Scale is too large for the selected modulus.")

        with timed_stage(logger, durations, "module_initialization"):
            owner = SecureDataTransformationModule(keys.public_key)
            cloud = CloudBQueryProcessingModule(owner, keys)
            owner.set_cloud(cloud)

        with timed_stage(logger, durations, "encrypted_index_setup"):
            owner.send_database_to_cloud(
                train_int,
                leaf_capacity=settings.leaf_capacity,
                max_depth=settings.max_depth,
            )

        y_true = y_test.tolist()
        y_pred: list[int] = []
        with timed_stage(logger, durations, "classify_all_queries"):
            for query_idx, query in enumerate(test_int, start=1):
                query_start = time.perf_counter_ns()

                start = time.perf_counter_ns()
                ids, encrypted_records, _ = owner.send_query_to_cloud(query, settings.k)
                query_cloud_s = elapsed_s(start)

                start = time.perf_counter_ns()
                records = [
                    [keys.decode_signed(keys.decrypt(value), keys.public_key.n)
                     for value in record]
                    for record in encrypted_records
                ]
                decrypt_s = elapsed_s(start)

                start = time.perf_counter_ns()
                if len(ids) != settings.k or records != train_int[ids].tolist():
                    raise RuntimeError("Recovered neighbors do not match training rows.")
                verify_s = elapsed_s(start)

                start = time.perf_counter_ns()
                votes = np.bincount(y_train[ids], minlength=2)
                predicted = int(np.argmax(votes))
                y_pred.append(predicted)
                vote_s = elapsed_s(start)

                query_total_s = elapsed_s(query_start)
                row = {
                    "independent_variable": settings.independent_variable,
                    "independent_value": settings.independent_value,
                    "key_bits": settings.key_bits,
                    "k": settings.k,
                    "leaf_capacity": settings.leaf_capacity,
                    "max_depth": settings.max_depth,
                    "scale": settings.scale,
                    "seed": settings.seed,
                    "query": query_idx,
                    "cloud_query_s": query_cloud_s,
                    "decrypt_s": decrypt_s,
                    "verify_s": verify_s,
                    "vote_s": vote_s,
                    "total_s": query_total_s,
                    "true_label": int(y_true[query_idx - 1]),
                    "predicted_label": predicted,
                }
                rows.append(row)
                logger.info(
                    "QUERY %02d | cloud=%.6f s | decrypt=%.6f s | verify=%.6f s | vote=%.6f s | total=%.6f s | true=%d pred=%d",
                    query_idx, query_cloud_s, decrypt_s, verify_s, vote_s,
                    query_total_s, row["true_label"], predicted,
                )

        with timed_stage(logger, durations, "classification_metrics"):
            report = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["class 0", "class 1"], digits=3, zero_division=0,
            )
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(v) for v in matrix.ravel())
            print("\nClassification report:\n" + report)
            print("Confusion matrix (rows=true, columns=predicted):")
            print(matrix)
            print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        with timed_stage(logger, durations, "write_query_csv"):
            with (metrics_dir / "query_timings.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        total_s = elapsed_s(total_start)
        with timed_stage(logger, durations, "write_metadata_json"):
            metadata = {
                "time_unit": "seconds",
                "settings": asdict(settings),
                "fixed_controls": {
                    "samples": 60,
                    "features": 2,
                    "class_balance": "30/30",
                    "train_samples": 42,
                    "test_samples": 18,
                    "test_fraction": 0.3,
                    "gaussian_noise_std": 0.2,
                    "centers": [[0.3, 0.3], [0.7, 0.7]],
                },
                "stage_times_s": durations,
                "total_wall_time_s_before_metadata": total_s,
                "query_time_summary_s": {
                    col: {
                        "mean": mean(float(r[col]) for r in rows),
                        "median": median(float(r[col]) for r in rows),
                        "min": min(float(r[col]) for r in rows),
                        "max": max(float(r[col]) for r in rows),
                    }
                    for col in ("cloud_query_s", "decrypt_s", "verify_s", "vote_s", "total_s")
                },
                "confusion_matrix": matrix.tolist(),
                "accuracy": (tn + tp) / len(y_true),
            }
            (metrics_dir / "experiment_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info("========== TIMING SUMMARY ==========")
        for stage, seconds in durations.items():
            logger.info("%-24s %.6f s", stage, seconds)
        logger.info("total_wall_time          %.6f s", total_s)
        for col in ("cloud_query_s", "decrypt_s", "verify_s", "vote_s", "total_s"):
            values = [float(row[col]) for row in rows]
            logger.info("%-24s mean=%.6f s | median=%.6f s | min=%.6f s | max=%.6f s",
                        col, mean(values), median(values), min(values), max(values))
        logger.info("Saved logs: %s", metrics_dir.resolve())
    except Exception:
        logger.exception("Experiment terminated with an error")
        raise


if __name__ == "__main__":
    main()
    
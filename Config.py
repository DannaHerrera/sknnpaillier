from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    source: str = "synthetic"
    csv_path: Path | None = None
    drop_last_column: bool = False
    key_bits: int = 512
    db_rows: int = 100
    queries: int = 5
    k: int = 3
    leaf_capacity: int = 16
    random_seed: int = 42
    output_csv: Path = Path(
        "knn_paillier_resultados.csv"
    )


def load_synthetic(
    total_rows: int,
    seed: int,
    attributes: int = 9,
) -> np.ndarray:
    rng = np.random.default_rng(
        seed
    )

    return rng.integers(
        0,
        100,
        size=(
            total_rows,
            attributes,
        ),
        dtype=np.int64,
    )


def load_shuttle() -> np.ndarray:
    try:
        from ucimlrepo import (
            fetch_ucirepo,
        )
    except ImportError as error:
        raise RuntimeError(
            "Instale ucimlrepo o use "
            "--source synthetic/csv."
        ) from error

    dataset = fetch_ucirepo(
        id=148
    )

    features = (
        dataset.data.features
    )

    if (
        features is None
        or features.empty
    ):
        raise RuntimeError(
            "El conjunto Shuttle "
            "no contiene datos."
        )

    return features.to_numpy(
        dtype=np.int64
    )


def load_csv(
    path: Path,
    drop_last_column: bool,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if drop_last_column:
        dataframe = (
            dataframe.iloc[:, :-1]
        )

    dataframe = dataframe.apply(
        pd.to_numeric,
        errors="raise",
    )

    return dataframe.to_numpy(
        dtype=np.int64
    )


def load_dataset(
    config: Config,
) -> np.ndarray:
    total_rows = (
        config.db_rows
        + config.queries
    )

    if config.source == "synthetic":
        data = load_synthetic(
            total_rows,
            config.random_seed,
        )

    elif config.source == "shuttle":
        data = load_shuttle()

    elif config.source == "csv":
        if config.csv_path is None:
            raise ValueError(
                "Indique --csv-path."
            )

        data = load_csv(
            config.csv_path,
            config.drop_last_column,
        )

    else:
        raise ValueError(
            f"Fuente no soportada: "
            f"{config.source}"
        )

    if len(data) < total_rows:
        raise ValueError(
            f"Se requieren {total_rows} filas; "
            f"solo existen {len(data)}."
        )

    return np.rint(
        data
    ).astype(
        np.int64
    )


def generate_database(
    data: np.ndarray,
    config: Config,
) -> np.ndarray:
    return data[
        :config.db_rows
    ].copy()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "SKNN-QP con Paillier, "
            "un solo CloudB y KD-tree por hojas."
        )
    )

    parser.add_argument(
        "--source",
        choices=(
            "synthetic",
            "shuttle",
            "csv",
        ),
        default="synthetic",
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
    )

    parser.add_argument(
        "--drop-last-column",
        action="store_true",
    )

    parser.add_argument(
        "--key-bits",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--db-rows",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--leaf-capacity",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "knn_paillier_resultados.csv"
        ),
    )

    args = parser.parse_args()

    if args.leaf_capacity <= 0:
        parser.error(
            "--leaf-capacity debe ser > 0."
        )

    return Config(
        source=args.source,
        csv_path=args.csv_path,
        drop_last_column=(
            args.drop_last_column
        ),
        key_bits=args.key_bits,
        db_rows=args.db_rows,
        queries=args.queries,
        k=args.k,
        leaf_capacity=(
            args.leaf_capacity
        ),
        random_seed=(
            args.random_seed
        ),
        output_csv=(
            args.output_csv
        ),
    )


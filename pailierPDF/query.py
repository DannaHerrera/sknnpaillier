from __future__ import annotations

import numpy as np

from Config import Config


def generate_queries(
    data: np.ndarray,
    config: Config,
) -> np.ndarray:
    start = config.db_rows

    end = (
        config.db_rows
        + config.queries
    )

    if len(data) < end:
        raise ValueError(
            "No existen suficientes datos "
            "para generar las consultas."
        )

    return data[
        start:end
    ].copy()

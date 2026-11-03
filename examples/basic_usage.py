"""Ejemplo con datos sintéticos y verificación en el cliente.

Desde la raíz, con sknnpaillier instalado en el entorno activo:
    python examples/basic_usage.py
    poetry run python examples/basic_usage.py

Para mostrar solo las fases y los resultados del ejemplo:
    SKNN_LOG_DISABLED=1 poetry run python examples/basic_usage.py
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
    resolve_key_bits,
)


def load_settings() -> tuple[int, int, int, int | None]:
    """Load only this example directory's .env; the shell takes precedence."""
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    key_bits = resolve_key_bits(os.getenv("SECURITY_LEVEL", "80"))
    try:
        k = int(os.getenv("SKNN_EXAMPLE_K", "3"))
        capacity = int(os.getenv("SKNN_EXAMPLE_LEAF_CAPACITY", "2"))
        raw_depth = os.getenv("SKNN_EXAMPLE_MAX_DEPTH", "").strip()
        depth = int(raw_depth) if raw_depth else None
    except ValueError as error:
        raise ValueError("Parámetros del ejemplo: se requieren enteros.") from error
    if k < 1 or k % 2 == 0:
        raise ValueError("SKNN_EXAMPLE_K debe ser positivo e impar.")
    if capacity < 1 or (depth is not None and depth < 0):
        raise ValueError("Capacidad >= 1 y profundidad >= 0 requeridas.")
    disabled = os.environ.setdefault("SKNN_LOG_DISABLED", "1").lower()
    if disabled not in ("0", "1", "true", "false", "yes", "no"):
        raise ValueError("SKNN_LOG_DISABLED debe ser una bandera booleana.")
    return key_bits, k, capacity, depth


def main() -> None:
    """Cifra, indexa, consulta y verifica vecinos de doce registros."""
    key_bits, k, leaf_capacity, max_depth = load_settings()
    # Enteros pequeños: Paillier no opera directamente sobre floats.
    database = np.array(
        [
            [-4, -3],
            [-3, -3],
            [-3, -1],
            [-2, -2],
            [-4, -1],
            [-1, -3],
            [2, 2],
            [3, 2],
            [2, 4],
            [4, 3],
            [3, 4],
            [4, 4],
        ],
        dtype=np.int64,
    )
    # Las etiquetas permanecen en el cliente; no se envían al cloud.
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    if k > len(database):
        raise ValueError("SKNN_EXAMPLE_K excede el tamaño de la base.")
    query = np.array([-3, -2], dtype=np.int64)

    print("[1/5] Datos sintéticos: 12 registros, 2 características.")
    print(f"      Módulo Paillier: {key_bits} bits")
    print(f"      Consulta: {query.tolist()} | k={k}")

    print("[2/5] Generando claves Paillier y conectando módulos...",
          flush=True)
    # SECURITY_LEVEL=80 es demostrativo, no un parámetro de producción.
    key_manager = KeyManagementModule(key_bits)
    transformer = SecureDataTransformationModule(key_manager.public_key)
    cloud = CloudBQueryProcessingModule(transformer, key_manager)
    transformer.set_cloud(cloud)

    print("[3/5] Cifrando datos y construyendo el índice espacial...",
          flush=True)
    # Esta llamada cifra la BD, construye hojas y cifra sus límites.
    transformer.send_database_to_cloud(
        database, leaf_capacity=leaf_capacity,
        max_depth=max_depth,
    )

    print("[4/5] Ejecutando la consulta k-NN segura...", flush=True)
    ids, encrypted_records, encrypted_distances = (
        transformer.send_query_to_cloud(query, k)
    )

    print("[5/5] Descifrando y verificando en el cliente confiable.")
    modulus = key_manager.public_key.n
    records = [
        [
            key_manager.decode_signed(key_manager.decrypt(value), modulus)
            for value in record
        ]
        for record in encrypted_records
    ]
    # Las distancias cuadradas son enteros no negativos.
    distances = [key_manager.decrypt(value) for value in encrypted_distances]

    # Referencia en claro solo para verificar, fuera del protocolo seguro.
    reference_distances = [
        sum((int(x) - int(q)) ** 2 for x, q in zip(record, query))
        for record in database
    ]
    expected_ids = sorted(
        range(len(database)),
        key=lambda record_id: (reference_distances[record_id], record_id),
    )[:k]
    if ids != expected_ids:
        raise RuntimeError("Los IDs no coinciden con el top-k de referencia.")
    if records != database[expected_ids].tolist():
        raise RuntimeError("Los registros descifrados no coinciden.")
    if distances != [reference_distances[i] for i in expected_ids]:
        raise RuntimeError("Las distancias cuadradas no coinciden.")

    neighbor_labels = labels[ids].tolist()
    # k impar y etiquetas binarias permiten una mayoría sin empate.
    prediction = int(sum(neighbor_labels) > k // 2)
    print("\nID  Coordenadas     Etiqueta  Distancia cuadrada")
    for record_id, record, label, distance in zip(
        ids, records, neighbor_labels, distances
    ):
        print(f"{record_id:<3} {str(record):<15} {label:<9} {distance}")
    print(f"Predicción por mayoría: {prediction}")
    print("Verificación correcta: IDs, registros y distancias coinciden.")


if __name__ == "__main__":
    main()

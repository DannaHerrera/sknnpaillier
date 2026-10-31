"""k-NN seguro sobre flotantes normalizados mediante punto fijo.

Con el paquete instalado, ejecutar desde la raíz:
    poetry run python examples/float_scaling_usage.py

Para mostrar solo las fases y los resultados:
    SKNN_LOG_DISABLED=1 poetry run python examples/float_scaling_usage.py

El redondeo aproxima las distancias; conservar el top-k no es una garantía
general. Este ejemplo lo verifica para los datos elegidos.
"""

import numpy as np

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
)


def _validate_scale(scale: int) -> None:
    if (
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, np.integer))
        or scale <= 0
    ):
        raise ValueError("La escala debe ser un entero positivo.")


def quantize(x: np.ndarray, scale: int = 1000) -> np.ndarray:
    """Redondea coordenadas normalizadas a enteros de punto fijo."""
    _validate_scale(scale)
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(
        (values < 0.0) | (values > 1.0)
    ):
        raise ValueError("Las coordenadas deben ser finitas y estar en [0, 1].")
    scaled = np.rint(values * scale)
    # Evita desbordar el entero de NumPy con escalas ajenas al ejemplo.
    if np.any(scaled >= float(np.iinfo(int).max)):
        raise ValueError("La escala excede el rango entero disponible.")
    return scaled.astype(int)


def dequantize_distance(d_sq: int, scale: int = 1000) -> float:
    """Devuelve distancia cuadrada aproximada, sin aplicar raíz."""
    _validate_scale(scale)
    return int(d_sq) / int(scale) ** 2


def main() -> None:
    """Compara el top-k seguro con referencias enteras y flotantes."""
    scale = 1000
    k = 3
    rng = np.random.default_rng(42)
    near_points = np.array(
        [[0.2012, 0.1003], [0.1004, 0.3007], [0.4002, 0.1006]]
    )
    database = np.vstack((near_points, rng.uniform(0.6, 1.0, size=(13, 2))))
    labels = np.array([0, 1, 0] + [0] * 6 + [1] * 7, dtype=int)
    query = np.array([0.1004, 0.0997])
    database_int = quantize(database, scale)
    query_int = quantize(query, scale)

    print("[1/5] Datos: 16 puntos 2D en [0, 1], etiquetas balanceadas.")
    print(f"      Escala={scale} | k={k}")
    print(f"      Registro: {database[0].tolist()} -> "
          f"{database_int[0].tolist()}")
    print(f"      Consulta: {query.tolist()} -> {query_int.tolist()}")

    print("[2/5] Generando claves Paillier y conectando módulos...",
          flush=True)
    # Tamaño demostrativo; no es un parámetro de producción.
    key_manager = KeyManagementModule(1024)
    transformer = SecureDataTransformationModule(key_manager.public_key)
    cloud = CloudBQueryProcessingModule(transformer, key_manager)
    transformer.set_cloud(cloud)

    print("[3/5] Cifrando enteros e indexando sus regiones...", flush=True)
    transformer.send_database_to_cloud(database_int, leaf_capacity=2)

    print("[4/5] Consultando con coordenadas cuantizadas...", flush=True)
    ids, encrypted_records, encrypted_distances = (
        transformer.send_query_to_cloud(query_int, k)
    )

    print("[5/5] Descifrando y verificando las dos referencias.")
    modulus = key_manager.public_key.n
    records = [
        [
            key_manager.decode_signed(key_manager.decrypt(value), modulus)
            for value in record
        ]
        for record in encrypted_records
    ]
    distances = [key_manager.decrypt(value) for value in encrypted_distances]
    approximate_distances = [
        dequantize_distance(distance, scale) for distance in distances
    ]

    # Referencias de validación en el cliente, fuera del protocolo seguro.
    float_distances = np.sum((database - query) ** 2, axis=1)
    int_distances = [
        sum((int(x) - int(q)) ** 2 for x, q in zip(record, query_int))
        for record in database_int
    ]
    float_ids = sorted(
        range(len(database)), key=lambda i: (float_distances[i], i)
    )[:k]
    int_ids = sorted(
        range(len(database)), key=lambda i: (int_distances[i], i)
    )[:k]
    if int_ids != float_ids:
        raise RuntimeError("La cuantización cambió el orden top-k flotante.")
    if ids != int_ids:
        raise RuntimeError("El top-k seguro no coincide con la referencia.")
    if records != database_int[int_ids].tolist():
        raise RuntimeError("Los registros enteros descifrados no coinciden.")
    if distances != [int_distances[i] for i in int_ids]:
        raise RuntimeError("Las distancias enteras descifradas no coinciden.")

    # Las etiquetas nunca se envían al cloud. k impar evita empate binario.
    prediction = int(int(labels[ids].sum()) > k // 2)
    float_prediction = int(int(labels[float_ids].sum()) > k // 2)
    if prediction != float_prediction:
        raise RuntimeError("La clasificación difiere de la referencia float.")

    print("\nID  Coordenadas aprox.  Clase  d² original  d² aprox.   Error abs.")
    for record_id, record, distance in zip(ids, records, approximate_distances):
        coordinates = (np.array(record, dtype=float) / scale).tolist()
        original = float(float_distances[record_id])
        error = abs(distance - original)
        print(f"{record_id:<3} {str(coordinates):<19} "
              f"{labels[record_id]:<6} {original:.8f}   "
              f"{distance:.8f}   {error:.8f}")
    print(f"IDs seguros y flotantes: {ids}")
    print(f"Predicción segura y flotante: {prediction}")
    print("Verificación correcta: top-k y mayoría coinciden exactamente.")
    print("Las distancias descuantizadas son aproximaciones por redondeo.")


if __name__ == "__main__":
    main()

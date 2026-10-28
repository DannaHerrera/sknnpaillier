from __future__ import annotations
import time
import json
from pathlib import Path
_TIMINGS = {}
_T0 = time.perf_counter()
def _mark(name):
    global _T0
    now = time.perf_counter()
    _TIMINGS[name] = now - _T0
    _T0 = now

from Config import (
    parse_args,
)

from KeyM import (
    KeyManagementModule,
)

from SecureDataT import (
    SecureDataTransformationModule,
)

from CloudB import (
    CloudBQueryProcessingModule,
)

from DataOwner import (
    DataOwnerModule,
)

from DataUser import (
    DataUserModule,
)

from TrustedR import (
    TrustedResultProcessingModule,
)

from Result import (
    ResultModule,
)


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

config = parse_args()


# ==========================================================
# KEY MANAGEMENT
# ==========================================================

print(
    "Generando llaves Paillier..."
)

key_manager = (
    KeyManagementModule(
        config.key_bits
    )
)


_mark("key_generation_s")

# ==========================================================
# SECURE DATA TRANSFORMATION
# ==========================================================

secure_data = (
    SecureDataTransformationModule(
        key_manager.public_key
    )
)


# ==========================================================
# ÚNICO CLOUD: CLOUDB
# ==========================================================

cloud_b = (
    CloudBQueryProcessingModule(
        transformer=secure_data,
        key_manager=key_manager,
    )
)

secure_data.set_cloud(
    cloud_b
)


# ==========================================================
# DATA OWNER
# ==========================================================

data_owner = (
    DataOwnerModule(
        config=config,
        secure_data_transformer=(
            secure_data
        ),
    )
)


# ==========================================================
# DATA USER
# ==========================================================

data_user = (
    DataUserModule(
        config=config,
        secure_data_transformer=(
            secure_data
        ),
    )
)


# ==========================================================
# TRUSTED RESULT PROCESSING
# SOLO PARA VALIDACIÓN EXPERIMENTAL
# ==========================================================

trusted_result = (
    TrustedResultProcessingModule()
)


# ==========================================================
# DATA OWNER: CARGAR DATASET
# ==========================================================

print(
    "Data Owner cargando dataset..."
)

full_dataset = (
    data_owner.load_dataset()
)


_mark("dataset_load_s")

#Se agrega en la nueva version con BD del paper de 1995
print()
print(
    "Dimensiones del dataset:",
    full_dataset.shape
)

if full_dataset.ndim != 2:
    raise ValueError(
        "El dataset debe ser una matriz 2D."
    )

if full_dataset.shape[1] != 2:
    raise ValueError(
        "Para TIGER se esperan "
        "exactamente 2 dimensiones: x, y."
    )

# ==========================================================
# DATA OWNER: PREPARAR BASE DE DATOS
# ==========================================================

database = (
    data_owner.prepare_database()
)


_mark("database_prepare_s")

# ==========================================================
# DATA USER: PREPARAR CONSULTAS
# ==========================================================

queries = (
    data_user.prepare_queries(
        full_dataset
    )
)


_mark("query_prepare_s")

# ==========================================================
# DATA OWNER: OUTSOURCING
# ==========================================================

print(
    "Data Owner preparando "
    "y externalizando la base de datos..."
)

print(
    "Construyendo KD-tree, "
    "cifrando DB y enviando a CloudB..."
)

data_owner.outsource_database()


_mark("outsourcing_s")

# ==========================================================
# ENCABEZADO
# ==========================================================

ResultModule.print_header(
    config,
    database,
    queries,
)


_QUERY_TIMINGS = []

results: list[
    dict[str, object]
] = []


# ==========================================================
# DATA USER: PROCESAMIENTO DE CONSULTAS
# ==========================================================

for (
    query_number,
    query_vector,
) in enumerate(
    queries,
    start=1,
):

    print()
    print(
        f"Data User enviando "
        f"consulta {query_number}..."
    )

    # ------------------------------------------------------
    # DATA USER -> SecureDataT -> CloudB
    # ------------------------------------------------------

    _query_start = time.perf_counter()
    (
        paillier_ids,
        recovered_neighbors,
        paillier_distances,
    ) = data_user.query(
        query_vector=query_vector,
        k=config.k,
    )


    encrypted_query_s = time.perf_counter() - _query_start
    _reference_start = time.perf_counter()

    # ------------------------------------------------------
    # VALIDACIÓN KNN EN TEXTO PLANO
    # ------------------------------------------------------

    (
        plain_ids,
        plain_neighbors,
        plain_distances,
    ) = (
        trusted_result.plaintext_knn(
            database=database,
            query=query_vector,
            k=config.k,
        )
    )


    reference_s = time.perf_counter() - _reference_start

    # ------------------------------------------------------
    # COMPARACIÓN DE RESULTADOS
    # ------------------------------------------------------

    verification = (
        trusted_result.verify_results(
            expected_ids=(
                plain_ids
            ),
            obtained_ids=(
                paillier_ids
            ),
            expected_records=(
                plain_neighbors
            ),
            obtained_records=(
                recovered_neighbors
            ),
            expected_distances=(
                plain_distances
            ),
            obtained_distances=(
                paillier_distances
            ),
        )
    )


    # ------------------------------------------------------
    # MOSTRAR RESULTADOS
    # ------------------------------------------------------

    ResultModule.print_query_result(
        query_number=(
            query_number
        ),
        query=(
            query_vector
        ),
        plain_ids=(
            plain_ids
        ),
        paillier_ids=(
            paillier_ids
        ),
        paillier_distances=(
            paillier_distances
        ),
        recovered_neighbors=(
            recovered_neighbors
        ),
        verification=(
            verification
        ),
    )


    # ------------------------------------------------------
    # ESTADÍSTICAS DEL KD-TREE
    # ------------------------------------------------------

    stats = (
        cloud_b.last_query_stats
    )

    print()

    print(
        "KD-tree            : "
        f"{stats['total_leaves']} "
        f"hojas totales"
    )

    print(
        "Hojas iniciales    : "
        f"{stats['initial_leaves']}"
    )

    print(
        "Hojas adicionales  : "
        f"{stats['additional_leaves']}"
    )

    print(
        "Hojas verificadas  : "
        f"{stats['verified_leaves']}"
    )

    print(
        "Distancias ESSED   : "
        f"{stats['distance_evaluations']}"
        f"/"
        f"{stats['database_size']} "
        f"registros"
    )


    # ------------------------------------------------------
    # CONSTRUIR FILA DEL CSV
    # ------------------------------------------------------

    result_row = (
        ResultModule.build_result_row(
            query_number=(
                query_number
            ),
            plain_ids=(
                plain_ids
            ),
            paillier_ids=(
                paillier_ids
            ),
            verification=(
                verification
            ),
        )
    )

    results.append(
        result_row
    )
    _QUERY_TIMINGS.append({"query": query_number, "encrypted_query_s": encrypted_query_s, "reference_s": reference_s, "correct": bool(all(verification.get(key, False) for key in ("same_order", "same_records", "same_distances"))), "distance_evaluations": stats.get("distance_evaluations"), "verified_leaves": stats.get("verified_leaves")})


# ==========================================================
# RESULT MODULE
# ==========================================================

ResultModule.save_results(
    results,
    config.output_csv,
)

ResultModule.print_summary(
    results,
    config.output_csv,
)
_mark("queries_and_validation_s")
if config.output_csv:
    path = Path(config.output_csv).with_suffix(".timings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": _TIMINGS, "queries": _QUERY_TIMINGS}, indent=2), encoding="utf-8")

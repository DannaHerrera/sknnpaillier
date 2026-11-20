"""Benchmark del tamaño de la BD: 3, 17, 33, 50 y 66 registros.

Ejecutar desde examples/ sin Poetry:
    python pruebaTiempos2.py

La BD tiene EXACTAMENTE n registros. Se generan ceil(0.30*n) consultas
independientes (la prueba es 30 % del tamaño de la BD, redondeado hacia
arriba), de modo que k=3 sigue siendo válido con n=3.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
import sys

# Permite ejecutar el archivo desde examples/ sin instalar el proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from statistics import mean, stdev

import numpy as np
from dotenv import load_dotenv

from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
)
from sknnpaillier.utils import quantize, resolve_key_bits

SIZES = (3, 17, 33, 50, 66)
TEST_FRACTION = 0.30
K = 3
LEAF_CAPACITY = 2
MAX_DEPTH = None
SCALE = 1000
SEED = 42
NOISE_STD = 0.20
CENTERS = np.array([[0.3, 0.3], [0.7, 0.7]])
QUERY_FIELDS = (
    'cloud_query_s', 'decrypt_s', 'verify_s', 'vote_s', 'total_s'
)
STAGE_FIELDS = (
    'encrypted_index_setup_s', 'key_generation_s'
)


def stopwatch(fn):
    start = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - start) / 1e9


def make_data(n: int, repetition: int):
    """Una BD de n filas y una muestra de consultas independiente."""
    rng = np.random.default_rng(SEED + repetition)
    n_queries = math.ceil(n * TEST_FRACTION)
    # Alterna clases; mantiene n exacto incluso cuando n es impar.
    y_db = np.arange(n, dtype=int) % 2
    y_query = np.arange(n_queries, dtype=int) % 2
    rng.shuffle(y_db)
    rng.shuffle(y_query)
    x_db = np.clip(CENTERS[y_db] + rng.normal(0, NOISE_STD, (n, 2)), 0, 1)
    x_query = np.clip(
        CENTERS[y_query] + rng.normal(0, NOISE_STD, (n_queries, 2)), 0, 1
    )
    return (
        quantize(x_db, scale_factor=SCALE), y_db,
        quantize(x_query, scale_factor=SCALE), y_query,
    )


def run_once(n: int, repetition: int, key_bits: int):
    db, labels, queries, true_labels = make_data(n, repetition)
    keys, key_s = stopwatch(lambda: KeyManagementModule(key_bits))

    max_distance = 2 * SCALE**2
    max_score = max_distance * (n + 1) + n - 1
    if max_score >= keys.public_key.n:
        raise ValueError('El módulo Paillier es demasiado pequeño para esta escala.')

    owner = SecureDataTransformationModule(keys.public_key)
    cloud = CloudBQueryProcessingModule(owner, keys)
    owner.set_cloud(cloud)

    # Mide la preparación del índice cifrado completa.
    # La API disponible no permite aislar el tiempo de cifrado.
    _, setup_s = stopwatch(lambda: owner.send_database_to_cloud(
        db, leaf_capacity=LEAF_CAPACITY, max_depth=MAX_DEPTH
    ))

    stage = dict(
        n_db=n, n_queries=len(queries), repetition=repetition, k=K,
        test_fraction=TEST_FRACTION, key_bits=key_bits,
        encrypted_index_setup_s=setup_s, key_generation_s=key_s,
    )
    rows = []
    for i, (query, y_true) in enumerate(zip(queries, true_labels), 1):
        qstart = time.perf_counter_ns()
        (ids, encrypted_records, _), cloud_s = stopwatch(
            lambda: owner.send_query_to_cloud(query, K)
        )
        records, dec_s = stopwatch(lambda: [
            [keys.decode_signed(keys.decrypt(value), keys.public_key.n)
             for value in record]
            for record in encrypted_records
        ])

        def verify():
            if len(ids) != K or records != db[ids].tolist():
                raise RuntimeError('Los vecinos recuperados no coinciden con la BD.')
        _, check_s = stopwatch(verify)

        def vote():
            return int(np.argmax(np.bincount(labels[ids], minlength=2)))
        pred, vote_s = stopwatch(vote)
        rows.append(dict(
            n_db=n, n_queries=len(queries), repetition=repetition,
            query=i, k=K, true_label=int(y_true), predicted_label=pred,
            cloud_query_s=cloud_s, decrypt_s=dec_s, verify_s=check_s,
            vote_s=vote_s,
            total_s=(time.perf_counter_ns() - qstart) / 1e9,
        ))
    return stage, rows


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(stages: list[dict], queries: list[dict]):
    """Promedia primero cada repetición para no sesgar por # consultas."""
    summary = []
    per_run = []
    for stage in stages:
        qrows = [r for r in queries
                 if r['n_db'] == stage['n_db']
                 and r['repetition'] == stage['repetition']]
        run = dict(stage)
        run.update({f'{col}_mean_s': mean(r[col] for r in qrows)
                    for col in QUERY_FIELDS})
        run['accuracy'] = mean(r['true_label'] == r['predicted_label']
                               for r in qrows)
        per_run.append(run)
    numeric_fields = list(STAGE_FIELDS) + [f'{c}_mean_s' for c in QUERY_FIELDS] + ['accuracy']
    for n in SIZES:
        items = [r for r in per_run if r['n_db'] == n]
        record = dict(n_db=n, n_queries=items[0]['n_queries'], repetitions=len(items), k=K)
        for field in numeric_fields:
            values = [float(r[field]) for r in items]
            record[field] = mean(values)
            record[field + '_sd'] = stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return per_run, summary


def make_plots(summary: list[dict], out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = [row['n_db'] for row in summary]

    # Gráfica 1: preparación del índice cifrado vs tamaño de BD.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, [r['encrypted_index_setup_s'] for r in summary],
            marker='o', label='Preparación del índice cifrado')
    ax.set(xlabel='Registros en la base de datos', ylabel='Tiempo promedio (s)',
           title='Escalabilidad de la preparación del índice cifrado')
    ax.set_xticks(x)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '01_indice_cifrado_vs_registros.png', dpi=180)
    plt.close(fig)

    # Gráfica 2: tiempo de procesamiento por consulta, sin coste de setup.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label in (
        ('cloud_query_s_mean_s', 'Procesamiento en la nube'),
        ('decrypt_s_mean_s', 'Descifrado de resultados'),
        ('total_s_mean_s', 'Tiempo total de consulta'),
    ):
        ax.plot(x, [r[key] for r in summary], marker='o', label=label)
    ax.set(xlabel='Registros en la base de datos',
           ylabel='Tiempo promedio por consulta (s)',
           title='Escalabilidad del procesamiento de consultas')
    ax.set_xticks(x)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '02_consultas_vs_registros.png', dpi=180)
    plt.close(fig)



def make_query_plots(queries: list[dict], out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for n in SIZES:
        nrows = [r for r in queries if r['n_db'] == n]
        order = sorted(set(r['query'] for r in nrows))
        means = {col: [mean(r[col] for r in nrows if r['query'] == i)
                       for i in order]
                 for col in ('cloud_query_s', 'decrypt_s', 'total_s')}
        fig, ax = plt.subplots(figsize=(9, 5))
        for col, name in (
            ('cloud_query_s', 'Nube'),
            ('decrypt_s', 'Descifrado'),
            ('total_s', 'Total'),
        ):
            ax.plot(order, means[col], marker='o', label=name)
        ax.set(xlabel='Número de consulta', ylabel='Tiempo medio (s)',
               title=f'BD de {n} registros: {len(order)} consultas (30 % redondeado)')
        ax.set_xticks(order)
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'consultas_bd_{n:02d}.png', dpi=180)
        plt.close(fig)


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
    repeat = int(os.environ.get('SKNN_REPETITIONS', '3'))
    if repeat < 1:
        raise ValueError('SKNN_REPETITIONS debe ser >= 1')
    key_bits = resolve_key_bits(os.environ.get('SKNN_SECURITY_LEVEL', '80'))
    out = Path(os.environ.get('SKNN_METRICS_DIR', 'sknn_metrics_database_scaling'))
    out.mkdir(parents=True, exist_ok=True)
    stage_rows, query_rows = [], []
    for n in SIZES:
        for rep in range(repeat):
            print(f'BD={n:2d} | repetición {rep + 1}/{repeat}', flush=True)
            stage, query = run_once(n, rep, key_bits)
            stage_rows.append(stage)
            query_rows.extend(query)
            # Guardado incremental para retener resultados si se interrumpe.
            save_csv(out / 'stage_timings.csv', stage_rows)
            save_csv(out / 'query_timings.csv', query_rows)
    per_run, summary = aggregate(stage_rows, query_rows)
    save_csv(out / 'repetition_summary.csv', per_run)
    save_csv(out / 'comparison_summary.csv', summary)
    make_plots(summary, out)
    make_query_plots(query_rows, out)
    (out / 'experiment_settings.json').write_text(json.dumps({
        'database_sizes': SIZES, 'test_fraction': TEST_FRACTION,
        'test_count_rule': 'ceil(0.3 * database_size)',
        'dataset_separation': 'independent synthetic database and queries',
        'k': K, 'leaf_capacity': LEAF_CAPACITY, 'max_depth': MAX_DEPTH,
        'scale': SCALE, 'seed': SEED, 'noise_std': NOISE_STD,
        'key_bits': key_bits, 'repetitions': repeat,
        'timing_unit': 'seconds',
        'note': ('No se mide KD-tree por separado. El tiempo de preparación del índice '
                 'cifrado incluye las operaciones internas de send_database_to_cloud; '
                 'no es un tiempo exclusivo de cifrado.'),
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nListo. CSV y 7 gráficas PNG: {out.resolve()}')


if __name__ == '__main__':
    main()

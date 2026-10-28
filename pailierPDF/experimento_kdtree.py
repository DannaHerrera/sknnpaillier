"""Experimento de escalabilidad para KDTree.py (particiones y hojas en claro).

Ejemplos:
 python experimento_kdtree.py --csv ruta.csv --sizes 100,500,1000,5000
 python experimento_kdtree.py --synthetic --dimensions 2 --sizes 100,1000,10000
 python experimento_kdtree.py --csv ruta.csv --sizes 1000,5000,50000 --augment

En el CSV espacial se utilizan exclusivamente las columnas x,y como DataOwner.py.
Se aplica una sola permutacion de todo el dataset antes de obtener subconjuntos
anidados; --augment crea puntos sinteticos por remuestreo con perturbacion.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from KDTree import _build_plain_kd_leaves


def load_data(args, maximum):
    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        return rng.integers(0, 100000, size=(maximum, args.dimensions), dtype=np.int64), False, 0
    df = pd.read_csv(args.csv)
    if {"x", "y"}.issubset(df.columns):
        df = df[["x", "y"]]
    elif args.drop_last_column:
        df = df.iloc[:, :-1]
    arr = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("El CSV contiene valores faltantes o no finitos")
    if np.any((arr > np.iinfo(np.int64).max) | (arr < np.iinfo(np.int64).min)):
        raise ValueError("Valores fuera del rango int64")
    arr = np.rint(arr).astype(np.int64)
    if not len(arr) or not arr.shape[1]:
        raise ValueError("El CSV debe tener registros y atributos numericos")
    arr = arr[rng.permutation(len(arr))]
    original_rows = len(arr)
    augmented = False
    if maximum > len(arr):
        if not args.augment:
            raise ValueError(f"Se solicitaron {maximum} registros, pero el CSV tiene {len(arr)}. Use --augment para datos sinteticos.")
        missing = maximum - len(arr)
        sampled = arr[rng.integers(0, len(arr), size=missing)].copy()
        jitter = rng.integers(-args.jitter, args.jitter + 1, size=sampled.shape, dtype=np.int64)
        arr = np.concatenate((arr, sampled + jitter), axis=0)
        augmented = True
    return arr, augmented, original_rows


def get_args():
    parser = argparse.ArgumentParser(description="Estudio del crecimiento del KD-tree")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV de DataOwner (columnas x,y o numericas)")
    source.add_argument("--synthetic", action="store_true", help="Generar datos totalmente sinteticos")
    parser.add_argument("--sizes", default="100,500,1000,2500,5000,10000,25000,50000,100000")
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--drop-last-column", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Permitir extension sintetica del CSV")
    parser.add_argument("--jitter", type=int, default=1, help="Perturbacion entera +/- del remuestreo")
    parser.add_argument("--repeat", type=int, default=3, help="Numero de mediciones por tamano")
    parser.add_argument("--out", type=Path, default=Path("resultados_kdtree"))
    a = parser.parse_args()
    a.sizes = sorted(set(int(v.strip()) for v in a.sizes.split(",") if v.strip()))
    if not a.sizes or min(a.sizes) < 1 or a.leaf_capacity < 1 or a.repeat < 1 or a.dimensions < 1 or a.jitter < 0:
        parser.error("sizes, leaf-capacity, repeat y dimensions positivos; jitter >= 0")
    if a.csv and not a.csv.is_file():
        parser.error(f"No existe el CSV: {a.csv}")
    return a


def run():
    args = get_args()
    data, augmented, original_rows = load_data(args, max(args.sizes))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in args.sizes:
        subset = data[:n]
        timings = []
        leaves = None
        stats = None
        peak_memory = 0
        for r in range(args.repeat):
            s = {}
            # Memoria pico Python aproximada, NO memoria total del proceso ni cifrado.
            tracemalloc.start()
            start = time.perf_counter()
            current = _build_plain_kd_leaves(subset, args.leaf_capacity, stats=s)
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            timings.append(elapsed)
            peak_memory = max(peak_memory, peak)
            if r == 0:
                leaves = current
                stats = s
        counts = [len(leaf.record_ids) for leaf in leaves]
        result = dict(
            registros=n,
            dimensiones=subset.shape[1],
            capacidad_hoja=args.leaf_capacity,
            nodos_logicos=stats["total_nodes"],
            nodos_internos=stats["total_nodes"] - len(leaves),
            hojas=len(leaves),
            profundidad_maxima=stats["max_depth"],
            registros_hoja_min=min(counts),
            registros_hoja_promedio=round(statistics.mean(counts), 4),
            registros_hoja_max=max(counts),
            tiempo_mediano_seg=round(statistics.median(timings), 6),
            tiempo_min_seg=round(min(timings), 6),
            tiempo_max_seg=round(max(timings), 6),
            pico_memoria_python_mib=round(peak_memory / 1024**2, 4),
            datos_ampliados_sinteticamente=bool(args.synthetic or (augmented and n > original_rows)),
        )
        results.append(result)
        print(f"{n:>9,} registros | {len(leaves):>7,} hojas | profundidad {stats['max_depth']:>2} | tiempo mediano {result['tiempo_mediano_seg']:.4f} s", flush=True)
    csv_path = args.out / "metricas_kdtree.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    settings = {"origen": "sintetico" if args.synthetic else str(args.csv), "capacidad_hoja": args.leaf_capacity, "semilla": args.seed, "repeticiones": args.repeat, "aumento_sintetico": augmented, "fecha": datetime.now().isoformat(timespec="seconds"), "nota": "Se mide solo la construccion del indice de hojas en claro; no el cifrado, envio a nube ni consultas."}
    (args.out / "configuracion.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    # Graficas PNG e informe HTML autocontenido en cuanto a texto y datos.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    graphics = [("hojas", "Hojas"), ("nodos_logicos", "Nodos logicos (incluye internos)"), ("profundidad_maxima", "Profundidad maxima"), ("tiempo_mediano_seg", "Tiempo mediano de construccion (s)"), ("pico_memoria_python_mib", "Pico memoria Python (MiB)")]
    images = []
    for key, label in graphics:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot([r["registros"] for r in results], [r[key] for r in results], marker="o")
        ax.set_xlabel("Cantidad de registros")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        imgname = f"grafica_{key}.png"
        fig.savefig(args.out / imgname, dpi=160)
        plt.close(fig)
        images.append((label, imgname))
    headers = list(results[0])
    rows = "\n".join("<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in headers) + "</tr>" for r in results)
    visuals = "\n".join(f'<section><h2>{html.escape(title)}</h2><img src="{img}" alt="{html.escape(title)}"></section>' for title, img in images)
    report = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Escalabilidad KD-tree</title><style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#202936}}h1{{margin-bottom:4px}}.note{{background:#f1f5f9;padding:14px;border-radius:8px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ddd;padding:8px;white-space:nowrap;text-align:right}}th{{background:#e9eef5}}img{{width:100%;max-width:850px}}section{{margin-top:28px;break-inside:avoid}}@media print{{body{{margin:12px}}section{{page-break-inside:avoid}}}}</style></head><body><h1>Crecimiento del indice basado en KD-tree</h1><p>Experimento: {html.escape(settings["fecha"])} | origen: {html.escape(settings["origen"])} | capacidad por hoja: {args.leaf_capacity} | semilla: {args.seed} | repeticiones: {args.repeat}</p><p class="note"><strong>Alcance:</strong> se miden la particion recursiva y las hojas conservadas. No se mide el cifrado. Los nodos internos se cuentan durante la construccion, pero no se almacenan. La memoria es una aproximacion del pico rastreado por tracemalloc, no el tamano serializado del indice cifrado. {"La muestra incluye ampliacion sintetica." if augmented else "Sin ampliacion sintetica del CSV."}</p><h2>Resultados</h2><div class="scroll"><table><thead><tr>{''.join('<th>'+html.escape(k)+'</th>' for k in headers)}</tr></thead><tbody>{rows}</tbody></table></div>{visuals}</body></html>'''
    (args.out / "informe_kdtree.html").write_text(report, encoding="utf-8")
    print(f"\nArchivos guardados en: {args.out.resolve()}")


if __name__ == "__main__":
    run()

import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

from sknnpaillier.log import get_logger


def leer_resumen(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError(f"{path} está vacío")
    return filas[0]


def guardar_csv(path, filas):
    if not filas:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)


def main():
    logger = get_logger("sknn.promedios")
    t_global = time.perf_counter()
    # Parámetros para reproducir la figura
    n_values = [2000, 4000, 6000, 8000, 10000]
    h_values = [5, 6, 7, 8, 9]
    key_bits_values = [512, 1024]

    k = 10
    m = 6
    queries = 1
    leaf_capacity = 1
    repeticiones = 35
    dataset = "letter+recognition.zip"

    salida_promedios = Path("resultados") / "promedios_fig4.csv"
    salida_detalle = Path("resultados") / "mediciones_fig4_detalle.csv"

    filas_promedio = []
    filas_detalle = []

    for key_bits in key_bits_values:
        for n in n_values:
            for h in h_values:
                tiempos = []

                print("=" * 70)
                print(f"Configuración: n={n}, h={h}, k={k}, m={m}, K={key_bits}")
                print("=" * 70)

                for rep in range(1, repeticiones + 1):
                    output_dir = Path("resultados") / "tmp" / f"n{n}_h{h}_K{key_bits}_rep{rep}"

                    cmd = [
                        sys.executable,
                        "Main.py",
                        "--dataset", dataset,
                        "--db-rows", str(n),
                        "--queries", str(queries),
                        "--k", str(k),
                        "--attributes", str(m),
                        "--max-depth", str(h),
                        "--key-bits", str(key_bits),
                        "--leaf-capacity", str(leaf_capacity),
                        "--output-dir", str(output_dir),
                    ]

                    print(f"  Ejecución {rep}/{repeticiones}...")
                    t_ejecucion = time.perf_counter()
                    subprocess.run(cmd, check=True)
                    logger.info({
                        "evento": "ejecucion_hija",
                        "n": n,
                        "h": h,
                        "key_bits": key_bits,
                        "repeticion": rep,
                        "tiempo_s": round(time.perf_counter() - t_ejecucion, 6),
                    })

                    resumen_path = output_dir / "resumen.csv"
                    resumen = leer_resumen(resumen_path)

                    tiempo = float(resumen["tiempo_consulta_promedio_s"])
                    tiempos.append(tiempo)

                    filas_detalle.append({
                        "n": n,
                        "h": h,
                        "k": k,
                        "m": m,
                        "key_bits": key_bits,
                        "repeticion": rep,
                        "query_time_s": tiempo,
                    })

                media = statistics.mean(tiempos)
                std = statistics.stdev(tiempos) if len(tiempos) > 1 else 0.0

                filas_promedio.append({
                    "n": n,
                    "h": h,
                    "k": k,
                    "m": m,
                    "key_bits": key_bits,
                    "mean_time_s": media,
                    "std_time_s": std,
                })

                guardar_csv(salida_detalle, filas_detalle)
                guardar_csv(salida_promedios, filas_promedio)

                print(f"Promedio: {media:.6f} s")
                print(f"Desv. estándar: {std:.6f} s")

    print("\nArchivo generado:")
    print(salida_promedios.resolve())
    logger.info({
        "evento": "promedios_total",
        "tiempo_total_s": round(time.perf_counter() - t_global, 6),
    })


main()
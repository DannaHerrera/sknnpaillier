"""Repite SKNN-QP con Paillier 35 veces y resume tiempos.
Colocar junto a Main.py y letter+recognition.zip.
"""
import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


def leer_resumen(ruta):
    with ruta.open(newline="", encoding="utf-8-sig") as archivo:
        return next(csv.DictReader(archivo))


def guardar_csv(ruta, filas):
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


parser = argparse.ArgumentParser(description="35 repeticiones de SKNN-QP + Paillier")
parser.add_argument("--repeticiones", type=int, default=35)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--h", type=int, default=7)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--m", type=int, default=6)
parser.add_argument("--bits", type=int, default=1024)
parser.add_argument("--queries", type=int, default=1)
parser.add_argument("--leaf-capacity", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.repeticiones < 2:
    parser.error("Se requieren al menos 2 repeticiones para desviación estándar")
base = Path(__file__).resolve().parent
principal = base / "Main.py"
dataset = base / "letter+recognition.zip"
if not principal.is_file() or not dataset.is_file():
    parser.error("Coloca este archivo junto a Main.py y letter+recognition.zip")

carpeta = base / "resultados" / "repeticiones_35"
carpeta.mkdir(parents=True, exist_ok=True)
filas = []
for i in range(1, args.repeticiones + 1):
    salida = carpeta / f"ejecucion_{i:02d}"
    comando = [
        sys.executable, str(principal), "--dataset", str(dataset),
        "--db-rows", str(args.n), "--queries", str(args.queries),
        "--k", str(args.k), "--attributes", str(args.m),
        "--max-depth", str(args.h), "--key-bits", str(args.bits),
        "--leaf-capacity", str(args.leaf_capacity),
        "--seed", str(args.seed), "--output-dir", str(salida)
    ]
    print(f"Ejecución {i}/{args.repeticiones}...", flush=True)
    # Si falla una ejecución, se detiene: nunca se promedian corridas incompletas.
    subprocess.run(comando, cwd=base, check=True, stdout=subprocess.DEVNULL)
    datos = leer_resumen(salida / "resumen.csv")
    fila = {
        "repeticion": i, "n": args.n, "h": args.h, "k": args.k,
        "m": args.m, "key_bits": args.bits,
        "query_time_s": float(datos["tiempo_consulta_promedio_s"]),
        "outsourcing_s": float(datos["tiempo_outsourcing_s"]),
        "decrypt_time_s": float(datos["tiempo_descifrado_promedio_s"]),
        "accuracy": float(datos["accuracy"]),
    }
    filas.append(fila)
    print(f"  Tiempo de consulta: {fila['query_time_s']:.4f} s", flush=True)
tiempos = [f["query_time_s"] for f in filas]
resumen = {
    "n": args.n, "h": args.h, "k": args.k, "m": args.m,
    "key_bits": args.bits, "repeticiones": len(filas),
    "query_time_s": statistics.mean(tiempos),
    "query_std_s": statistics.stdev(tiempos),
    "query_min_s": min(tiempos), "query_max_s": max(tiempos),
}
guardar_csv(carpeta / "mediciones_35.csv", filas)
guardar_csv(carpeta / "promedio_35.csv", [resumen])
print("\nPromedio: {:.4f} s | Desviación estándar: {:.4f} s".format(
    resumen["query_time_s"], resumen["query_std_s"]))
  
print("Archivos guardados en:", carpeta)

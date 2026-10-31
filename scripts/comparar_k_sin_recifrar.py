"""Compara k=1,2,3 usando exclusivamente vecinos ya recuperados por SKNN-QP.

No invoca Paillier, CloudB, ni realiza una nueva búsqueda kNN.
Reconstruye las etiquetas del entrenamiento confiable con la misma partición
que experimento_letter_paillier_comparacion.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

CLASES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def guardar_csv(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        raise ValueError("No hay filas para guardar")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def cargar_etiquetas(dataset: Path, semilla: int) -> np.ndarray:
    with zipfile.ZipFile(dataset) as archivo:
        datos = archivo.read("letter-recognition.data").decode("ascii")
    filas = list(csv.reader(io.StringIO(datos)))
    if len(filas) != 20000 or any(len(fila) != 17 for fila in filas):
        raise ValueError("El dataset no tiene el formato esperado (20000 x 17)")
    etiquetas = np.asarray([fila[0] for fila in filas])
    permutacion = np.random.default_rng(semilla).permutation(len(etiquetas))
    return etiquetas[permutacion]


def reconstruir_particion(etiquetas: np.ndarray, semilla: int,
                          n: int, consultas: int) -> tuple[np.ndarray, np.ndarray]:
    # Reproduce literalmente la selección del experimento de comparación.
    rng_consultas = np.random.default_rng(semilla + 1000003)
    indices_q = rng_consultas.permutation(len(etiquetas))[:consultas]
    mascara = np.ones(len(etiquetas), dtype=bool)
    mascara[indices_q] = False
    clases = np.unique(etiquetas)
    base, sobrantes = divmod(n, len(clases))
    indices_db: list[int] = []
    rng_db = np.random.default_rng(semilla + 2000003)
    for posicion, clase in enumerate(clases):
        indices_clase = np.flatnonzero((etiquetas == clase) & mascara)
        rng_db.shuffle(indices_clase)
        cantidad = base + (1 if posicion < sobrantes else 0)
        if len(indices_clase) < cantidad:
            raise ValueError(f"Sin suficientes datos de entrenamiento para {clase}")
        indices_db.extend(indices_clase[:cantidad].tolist())
    return etiquetas[np.asarray(indices_db, dtype=int)], indices_q


def predecir_mayoria(letras: list[str]) -> str:
    frecuencias = Counter(letras)
    maximo = max(frecuencias.values())
    return next(letra for letra in letras if frecuencias[letra] == maximo)


def predecir_ponderada(letras: list[str], distancias: list[int]) -> str:
    votos: dict[str, float] = {}
    for letra, distancia in zip(letras, distancias):
        votos[letra] = votos.get(letra, 0.0) + 1.0 / (1.0 + distancia)
    return max(votos, key=votos.get)


def metricas(reales: list[str], predichas: list[str]) -> dict[str, float]:
    if not reales or len(reales) != len(predichas):
        raise ValueError("Cantidad de consultas y predicciones incompatible")
    precision, recall, f1 = [], [], []
    for clase in CLASES:
        tp = sum(r == clase and p == clase for r, p in zip(reales, predichas))
        fp = sum(r != clase and p == clase for r, p in zip(reales, predichas))
        fn = sum(r == clase and p != clase for r, p in zip(reales, predichas))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "accuracy": sum(r == p for r, p in zip(reales, predichas)) / len(reales),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar k=1,2,3 sin nuevas consultas cifradas")
    parser.add_argument("--dataset", type=Path, default=Path("letter+recognition.zip"))
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Carpeta con consultas.csv y resumen.csv del experimento original")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    ruta_consultas = args.results_dir / "consultas.csv"
    ruta_resumen = args.results_dir / "resumen.csv"
    consultas = leer_csv(ruta_consultas)
    resumen_filas = leer_csv(ruta_resumen)
    if len(resumen_filas) != 1 or not consultas:
        raise ValueError("Se necesita un resumen.csv de una ejecución y consultas.csv no vacío")
    resumen = resumen_filas[0]
    n = int(resumen["registros"])
    semilla = int(resumen["semilla"])
    k_original = int(resumen["k"])
    q_total = int(resumen["consultas"])
    if k_original < 3:
        raise ValueError("Solo pueden comparar k=1,2,3 los CSV que guardaron al menos 3 vecinos")
    if len(consultas) != q_total:
        raise ValueError("El número de filas de consultas.csv no coincide con resumen.csv")

    etiquetas = cargar_etiquetas(args.dataset, semilla)
    y_db, indices_q = reconstruir_particion(etiquetas, semilla, n, q_total)
    indices_registrados = list(map(int, resumen["indices_consultas"].split()))
    if indices_registrados != indices_q.tolist():
        raise ValueError("Los índices de prueba no coinciden: comprueba dataset, semilla y CSV")

    detalle: list[dict] = []
    predicciones: dict[tuple[int, str], list[str]] = {
        (k, metodo): [] for k in (1, 2, 3) for metodo in ("mayoria", "ponderada")
    }
    reales: list[str] = []
    for posicion, fila in enumerate(consultas):
        if int(fila["consulta"]) != posicion + 1:
            raise ValueError(f"Orden de consultas inesperado en fila {posicion + 1}")
        if int(fila["indice_dataset_consulta"]) != int(indices_q[posicion]):
            raise ValueError(f"Índice del dataset distinto en consulta {posicion + 1}")
        real = str(etiquetas[indices_q[posicion]])
        if fila["etiqueta_real"] != real:
            raise ValueError(f"Etiqueta real distinta en consulta {posicion + 1}")
        ids = list(map(int, fila["ids_sknn"].split()))
        distancias = list(map(int, fila["distancias_cuadradas"].split()))
        if len(ids) < 3 or len(distancias) != len(ids):
            raise ValueError(f"No se guardaron 3 vecinos y sus distancias en consulta {posicion + 1}")
        if any(i < 0 or i >= n for i in ids) or any(d < 0 for d in distancias):
            raise ValueError(f"ID o distancia fuera de rango en consulta {posicion + 1}")
        letras = [str(y_db[i]) for i in ids]
        # Verificación fuerte de consistencia con el CSV original para k=3.
        if (predecir_mayoria(letras[:3]) != fila["prediccion_mayoria"] or
                predecir_ponderada(letras[:3], distancias[:3]) != fila["prediccion_ponderada"]):
            raise ValueError(
                f"No se reproducen las predicciones originales en consulta {posicion + 1}. "
                "Comprueba que el CSV corresponde a esta versión y al dataset correcto."
            )
        reales.append(real)
        for k in (1, 2, 3):
            for metodo in ("mayoria", "ponderada"):
                pred = (predecir_mayoria(letras[:k]) if metodo == "mayoria"
                        else predecir_ponderada(letras[:k], distancias[:k]))
                predicciones[(k, metodo)].append(pred)
                detalle.append({
                    "consulta": posicion + 1,
                    "indice_dataset_consulta": int(indices_q[posicion]),
                    "n": n,
                    "m": int(resumen["atributos_m"]),
                    "k": k,
                    "metodo": metodo,
                    "real": real,
                    "predicha": pred,
                    "correcta": int(real == pred),
                    "ids_utilizados": " ".join(map(str, ids[:k])),
                    "letras_vecinos": " ".join(letras[:k]),
                    "distancias_cuadradas": " ".join(map(str, distancias[:k])),
                })

    resumen_comparativo: list[dict] = []
    for k in (1, 2, 3):
        for metodo in ("mayoria", "ponderada"):
            valores = metricas(reales, predicciones[(k, metodo)])
            resumen_comparativo.append({
                "n": n,
                "m": int(resumen["atributos_m"]),
                "consultas": q_total,
                "seed": semilla,
                "k": k,
                "metodo": metodo,
                **valores,
            })

    salida = args.output_dir or args.results_dir
    guardar_csv(salida / "comparacion_k_resumen.csv", resumen_comparativo)
    guardar_csv(salida / "comparacion_k_detalle.csv", detalle)
    print(f"\nComparación de k=1,2,3 | n={n}, m={resumen['atributos_m']}, consultas={q_total}")
    print(f"{'k':<4} {'método':<12} {'Accuracy':>11} {'Precision':>11} {'Recall':>11} {'F1':>11}")
    for fila in resumen_comparativo:
        print(f"{fila['k']:<4} {fila['metodo']:<12} "
              f"{100*fila['accuracy']:>10.2f}% {100*fila['precision_macro']:>10.2f}% "
              f"{100*fila['recall_macro']:>10.2f}% {100*fila['f1_macro']:>10.2f}%")
    print(f"\nArchivos generados: {salida / 'comparacion_k_resumen.csv'} y "
          f"{salida / 'comparacion_k_detalle.csv'}")
    print("No se realizaron nuevas consultas cifradas. Los resultados de k=1 y k=2 "
          "son reanálisis de los vecinos guardados para k=3.")


if __name__ == "__main__":
    main()

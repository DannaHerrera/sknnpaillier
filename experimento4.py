from __future__ import annotations
import argparse
import csv
import io
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from CloudB import CloudBQueryProcessingModule
from KeyM import KeyManagementModule
from SecureDataT import SecureDataTransformationModule


def cargar_letras(ruta: Path, semilla: int):
    with zipfile.ZipFile(ruta) as archivo:
        datos = archivo.read("letter-recognition.data").decode("ascii")
    filas = list(csv.reader(io.StringIO(datos)))
    X = np.asarray([[int(v) for v in fila[1:]] for fila in filas], dtype=np.int64)
    y = np.asarray([fila[0] for fila in filas])
    if X.shape != (20000, 16):
        raise ValueError(f"Dimensiones inesperadas: {X.shape}")
    indices = np.random.default_rng(semilla).permutation(len(X))
    return X[indices], y[indices]


def cargar_preprocesado(carpeta: Path, db_rows: int, consultas: int,
                        atributos: int, semilla: int, bits_escala: int):
    """Carga el dataset_n de benchmark; conserva IDs originales y clases."""
    if db_rows not in (10, 30, 50, 100):
        raise ValueError("db-rows debe ser 10, 30, 50 o 100")
    if not 0 <= bits_escala <= 20:
        raise ValueError("fixed-point-bits fuera de rango")
    carpeta_n = carpeta / f"dataset_{db_rows}"
    X = np.load(carpeta_n / "X.npy", allow_pickle=False)
    codigos = np.load(carpeta_n / "clases.npy", allow_pickle=False)
    ids = np.load(carpeta_n / "ids.npy", allow_pickle=False)
    X_q = np.load(carpeta_n / "consultas.npy", allow_pickle=False)
    codigos_q = np.load(carpeta_n / "clases_consultas.npy", allow_pickle=False)
    ids_q = np.load(carpeta_n / "ids_consultas.npy", allow_pickle=False)
    mapa = np.load(carpeta / "mapa_etiquetas.npy", allow_pickle=True)  # Solo archivos propios
    if X.shape != (db_rows, 16) or X_q.ndim != 2 or X_q.shape[1] != 16:
        raise ValueError(f"Dimensiones inesperadas: BD={X.shape}, Q={X_q.shape}")
    if atributos < 1 or atributos > 16:
        raise ValueError("attributes debe estar entre 1 y 16")
    if not 1 <= consultas <= len(X_q):
        raise ValueError(f"queries debe estar entre 1 y {len(X_q)}")
    if any(len(a) != db_rows for a in (codigos, ids)):
        raise ValueError("Inconsistencia en los vectores de BD")
    if len(codigos_q) != len(X_q) or len(ids_q) != len(X_q):
        raise ValueError("Inconsistencia en los vectores de consulta")
    if np.intersect1d(ids, ids_q).size:
        raise ValueError("Los IDs de BD y consultas se superponen")
    if not np.isfinite(X).all() or not np.isfinite(X_q).all():
        raise ValueError("Existen valores no finitos")
    if (np.min(codigos) < 0 or np.max(codigos) >= len(mapa)
            or np.min(codigos_q) < 0 or np.max(codigos_q) >= len(mapa)):
        raise ValueError("Etiquetas fuera del mapa")
    # Los 16 atributos fueron escalados de forma compartida en preprocessing.
    # Se seleccionan los primeros m de forma idéntica para BD y consultas.
    S = 1 << bits_escala
    db_entera = np.rint(X[:, :atributos] * S).astype(np.int64)
    q_entera = np.rint(X_q[:consultas, :atributos] * S).astype(np.int64)
    normas2_enteras = np.sum(db_entera * db_entera, axis=1)
    return (db_entera, mapa[codigos].astype(str), q_entera,
            mapa[codigos_q[:consultas]].astype(str), ids_q[:consultas],
            ids, normas2_enteras)


def clasificar_por_mayoria(ids: list[int], etiquetas: np.ndarray) -> str:
    # Etiquetas conservadas solamente en el cliente confiable.
    votos = [str(etiquetas[i]) for i in ids]
    frecuencias = Counter(votos)
    maximo = max(frecuencias.values())
    return next(etiqueta for etiqueta in votos if frecuencias[etiqueta] == maximo)



def clasificar_ponderado(ids: list[int], etiquetas: np.ndarray, distancias: list[int]) -> str:
    """Clasifica los vecinos recuperados por SKNN-QP usando distancia cuadrada."""
    if not ids or len(ids) != len(distancias):
        raise ValueError("Vecinos o distancias inválidos")

    votos = {}
    for record_id, distancia in zip(ids, distancias):
        letra = str(etiquetas[record_id])
        peso = 1.0 / (1.0 + int(distancia))
        votos[letra] = votos.get(letra, 0.0) + peso

    # dict mantiene el orden de inserción: en empate gana el vecino más cercano.
    return max(votos, key=votos.get)


def metricas_clasificacion(
    reales: list[str],
    predichas: list[str]
):
    if len(reales) != len(predichas) or not reales:
        raise ValueError(
            "Error en clasificación: "
            f"{len(reales)} etiquetas reales y "
            f"{len(predichas)} predicciones."
        )

    clases = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    precision = []
    recall = []
    f1 = []

    for clase in clases:
        tp = sum(
            a == clase and b == clase
            for a, b in zip(reales, predichas)
        )

        fp = sum(
            a != clase and b == clase
            for a, b in zip(reales, predichas)
        )

        fn = sum(
            a == clase and b != clase
            for a, b in zip(reales, predichas)
        )

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    accuracy = sum(
        a == b for a, b in zip(reales, predichas)
    ) / len(reales)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


def guardar_csv(ruta: Path, filas: list[dict]):
    if not filas:
        return
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

        
def append_medicion(path, fila):
    existe = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n", "h", "k", "m",
                "key_bits", "query_time_s"
            ]
        )

        if not existe:
            writer.writeheader()

        writer.writerow(fila)



def main():
    parser = argparse.ArgumentParser(
        description="Consultas SKNN-QP con Paillier; sin busqueda kNN en texto plano"
    )
    parser.add_argument("--dataset", type=Path, default=None, help="No se usa con los archivos preprocesados; conservado por compatibilidad")
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("salida_preprocesamiento"),
                        help="Carpeta que contiene dataset_10, dataset_30, dataset_50 y dataset_100")
    parser.add_argument("--fixed-point-bits", type=int, default=12,
                        help="gamma para S=2**gamma antes de Paillier (por defecto 12)")
    parser.add_argument("--db-rows", type=int, default=10)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--key-bits", type=int, default=1024)
    parser.add_argument("--leaf-capacity", type=int, default=1)
    parser.add_argument("--attributes", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    args = parser.parse_args()
    print("\nCONFIGURACIÓN RECIBIDA:")
    print(vars(args))
    if not (1 <= args.k <= args.db_rows and args.queries >= 1 and
            (args.db_rows in (10, 30, 50, 100) and args.queries <= 10)):
        parser.error("Requiere 1 <= k <= db-rows, db-rows en {10,30,50,100}, queries entre 1 y 10")
    if args.attributes < 1 or args.attributes > 16 or args.max_depth < 0:
        parser.error("attributes debe estar entre 1 y 16, max-depth >= 0")
    if args.key_bits < 128 or args.leaf_capacity < 1:
        parser.error("key-bits >= 128 y leaf-capacity >= 1")

    if args.fixed_point_bits < 0 or args.fixed_point_bits > 20:
        parser.error("fixed-point-bits debe estar entre 0 y 20")
    try:
        (X_db, y_db, X_q, y_q, indices_q,
         ids_db_originales, normas2_enteras) = cargar_preprocesado(
            args.preprocessed_dir, args.db_rows, args.queries,
            args.attributes, args.seed, args.fixed_point_bits
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))

    print("Datos preprocesados cargados; escala entera:", 2 ** args.fixed_point_bits)
    print("BD:", X_db.shape, "Consultas:", X_q.shape)
    print("IDs originales BD (muestra):", ids_db_originales[:10].tolist())
    print("IDs originales consultas (muestra):", indices_q[:10].tolist())
    print("Normas enteras precalculadas:", normas2_enteras.shape)
    print("Distribución de etiquetas BD:", dict(Counter(y_db)))

    inicio = time.perf_counter()

    print("PASO 1: Generando claves Paillier...", flush=True)
    key_manager = KeyManagementModule(args.key_bits)
    print("PASO 1 COMPLETADO", flush=True)

    secure_data = SecureDataTransformationModule(
        key_manager.public_key
    )

    cloud = CloudBQueryProcessingModule(
        transformer=secure_data,
        key_manager=key_manager
    )

    secure_data.set_cloud(cloud)

    print("PASO 2: Cifrando base de datos...", flush=True)

    t = time.perf_counter()

    secure_data.send_database_to_cloud(
        X_db,
        leaf_capacity=args.leaf_capacity,
        max_depth=args.max_depth
    )

    tiempo_outsourcing = time.perf_counter() - t

    print(
        f"PASO 2 COMPLETADO: {tiempo_outsourcing:.2f} segundos",
        flush=True
    )

    print("PASO 3: Iniciando consultas...", flush=True)


    filas = []
    predicciones_mayoria = []
    predicciones_ponderadas = []
    for num, (q, etiqueta_real) in enumerate(zip(X_q, y_q), start=1):
        t = time.perf_counter()
        ids, registros_cifrados, distancias_cifradas = secure_data.send_query_to_cloud(q, args.k)
        tiempo_consulta = time.perf_counter() - t

        # Solo el cliente confiable descifra los resultados finales.
        t = time.perf_counter()
        registros_recuperados = [
            [key_manager.decode_signed(key_manager.decrypt(c), key_manager.public_key.n)
             for c in registro]
            for registro in registros_cifrados
        ]
        distancias_recuperadas = [key_manager.decrypt(c) for c in distancias_cifradas]
        tiempo_descifrado = time.perf_counter() - t

        print("\nVERIFICACIÓN MATEMÁTICA")
        distancias_coinciden = True
        for posicion, record_id in enumerate(ids):
            distancia_esperada = sum(
                (int(a) - int(b)) ** 2 for a, b in zip(X_db[record_id], q)
            )
            distancia_paillier = distancias_recuperadas[posicion]
            coincide = distancia_esperada == distancia_paillier
            distancias_coinciden &= coincide
            print(
                f"Vecino {posicion + 1}: ID={record_id}, "
                f"Esperada={distancia_esperada}, "
                f"Paillier={distancia_paillier}, Coincide={coincide}"
            )

        # Validación opcional fuera del protocolo: NO es una segunda búsqueda
        # usada para producir los resultados SKNN-QP.
        distancias_bd = [
            sum((int(a) - int(b)) ** 2 for a, b in zip(registro, q))
            for registro in X_db
        ]
        ids_esperados = sorted(
            range(len(X_db)), key=lambda i: (distancias_bd[i], i)
        )[:args.k]
        topk_correcto = ids_esperados == list(ids)
        print("\nVALIDACIÓN TOP-K")
        print("IDs esperados:", ids_esperados)
        print("IDs SKNN-QP:", ids)
        print("Coincidencia exacta:", topk_correcto)
        orden_correcto = distancias_recuperadas == sorted(distancias_recuperadas)
        print("Distancias ordenadas correctamente:", orden_correcto)

        predicha_mayoria = clasificar_por_mayoria(ids, y_db)
        predicha_ponderada = clasificar_ponderado(ids, y_db, distancias_recuperadas)
        predicciones_mayoria.append(str(predicha_mayoria))
        predicciones_ponderadas.append(str(predicha_ponderada))
        predicha = predicha_ponderada  # Compatibilidad: salida principal anterior.
        print("\n========= DIAGNÓSTICO =========")
        print("Letra real:", etiqueta_real)
        print("Predicción por mayoría:", predicha_mayoria)
        print("Predicción ponderada:", predicha_ponderada)
        print("IDs recuperados:", ids)
        print("Etiquetas de los vecinos:", [str(y_db[i]) for i in ids])
        print("Distribución de votos:", dict(Counter(str(y_db[i]) for i in ids)))
        print("¿Predicción correcta?:", str(predicha) == str(etiqueta_real))
        print("===============================\n")

        estadisticas = cloud.last_query_stats.copy()
        filas.append({
            "consulta": num,
            "etiqueta_real": str(etiqueta_real),
            "prediccion_paillier": str(predicha),
            "correcta": int(str(predicha) == str(etiqueta_real)),
            "prediccion_mayoria": str(predicha_mayoria),
            "correcta_mayoria": int(str(predicha_mayoria) == str(etiqueta_real)),
            "prediccion_ponderada": str(predicha_ponderada),
            "correcta_ponderada": int(str(predicha_ponderada) == str(etiqueta_real)),
            "indice_dataset_consulta": int(indices_q[num - 1] - 1),
            "id_original_consulta": int(indices_q[num - 1]),
            "ids_sknn": " ".join(map(str, ids)),
            "distancias_cuadradas": " ".join(map(str, distancias_recuperadas)),
            "tiempo_consulta_cifrada_s": tiempo_consulta,
            "tiempo_descifrado_final_s": tiempo_descifrado,
            "tiempo_consulta_total_s": tiempo_consulta + tiempo_descifrado,
            "distancias_evaluadas": estadisticas["distance_evaluations"],
            "hojas_totales": estadisticas["total_leaves"],
            "hojas_iniciales": estadisticas["initial_leaves"],
            "hojas_adicionales": estadisticas["additional_leaves"],
            "distancias_coinciden": bool(distancias_coinciden),
            "topk_correcto": bool(topk_correcto),
            "orden_correcto": bool(orden_correcto),
        })
        print(
            f"Consulta {num}/{args.queries} | IDS={ids} | "
            f"letra={predicha} | consulta cifrada={tiempo_consulta:.3f}s",
            flush=True
        )

    print("\n=== DIAGNÓSTICO DE CLASIFICACIÓN ===")
    print("Consultas esperadas:", args.queries)
    print("Etiquetas reales:", len(y_q))
    print("Predicciones mayoría:", len(predicciones_mayoria))
    print("Predicciones ponderadas:", len(predicciones_ponderadas))
    print("Valores reales:", list(y_q))
    print("Predichos mayoría:", predicciones_mayoria)
    print("Predichos ponderados:", predicciones_ponderadas)
    print("===================================")


    reales = list(map(str, y_q))
    metricas_mayoria = metricas_clasificacion(reales, predicciones_mayoria)
    metricas_ponderadas = metricas_clasificacion(reales, predicciones_ponderadas)
    # Campos originales corresponden al método ponderado.
    metricas = {
        **metricas_ponderadas,
        **{f"{nombre}_mayoria": valor for nombre, valor in metricas_mayoria.items()},
        **{f"{nombre}_ponderada": valor for nombre, valor in metricas_ponderadas.items()},
    }
    print("\nMÉTRICAS MAYORÍA:", metricas_mayoria)
    print("MÉTRICAS PONDERADAS:", metricas_ponderadas)
    resumen = {
        "dataset": "Letter Recognition", "registros": args.db_rows,
        "consultas": args.queries, "k": args.k, "bits_paillier": args.key_bits,
        "capacidad_hoja": args.leaf_capacity, "atributos_m": args.attributes,
        "altura_maxima_h": args.max_depth, "semilla": args.seed, "modo": "seguro",
        "indices_consultas": " ".join(map(str, (indices_q - 1).tolist())),
        "ids_originales_consultas": " ".join(map(str, indices_q.tolist())),
        "factor_escala": 2 ** args.fixed_point_bits,
        "tiempo_outsourcing_s": tiempo_outsourcing,
        "tiempo_consulta_promedio_s": float(np.mean([f["tiempo_consulta_cifrada_s"] for f in filas])),
        "tiempo_descifrado_promedio_s": float(np.mean([f["tiempo_descifrado_final_s"] for f in filas])),
        "tiempo_consulta_total_promedio_s": float(np.mean([f["tiempo_consulta_total_s"] for f in filas])),
        "topk_correcto_todas": all(f["topk_correcto"] for f in filas),
        "distancias_coinciden_todas": all(f["distancias_coinciden"] for f in filas),
        "distancias_evaluadas_promedio": float(np.mean([f["distancias_evaluadas"] for f in filas])),
        **metricas,
        "tiempo_total_s": time.perf_counter() - inicio,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    guardar_csv(args.output_dir / f"consultas_seguro_n{args.db_rows}.csv", filas)
    guardar_csv(args.output_dir / f"resumen_seguro_n{args.db_rows}.csv", [resumen])
    
    # Guardar medición para las gráficas de la tesis
    mediciones_path = args.output_dir / "mediciones_seguro.csv"

    append_medicion(
        mediciones_path,
        {
            "n": args.db_rows,
            "h": args.max_depth,
            "k": args.k,
            "m": args.attributes,
            "key_bits": args.key_bits,
            "query_time_s": resumen["tiempo_consulta_promedio_s"],
        }
    )

    print("Resumen:", resumen)
    print("Archivos:", args.output_dir.resolve())

    
main()
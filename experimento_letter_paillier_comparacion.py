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
    """Carga exclusivamente archivos preprocesados; no requiere el ZIP.

    BD: IDs originales impares (X_reducido).
    Consultas: IDs pares tomados de X_completo y transformados usando
    el escalador de la configuración reducida.
    """
    if atributos != 6:
        raise ValueError("La configuración reducida utiliza exactamente 6 atributos")
    if not (1 <= db_rows <= 10000 and 1 <= consultas <= 10000):
        raise ValueError("db-rows y queries deben estar entre 1 y 10000")

    X = np.load(carpeta / "X_reducido.npy", allow_pickle=False)
    codigos = np.load(carpeta / "clases_reducido.npy", allow_pickle=False)
    ids = np.load(carpeta / "ids_reducido.npy", allow_pickle=False)
    normas2 = np.load(carpeta / "normas2_reducido.npy", allow_pickle=False)
    mapa = np.load(carpeta / "mapa_etiquetas.npy", allow_pickle=True)  # Solo archivos propios/confiables

    X_total = np.load(carpeta / "X_completo.npy", allow_pickle=False)
    y_total = np.load(carpeta / "clases_completo.npy", allow_pickle=False)
    ids_total = np.load(carpeta / "ids_completo.npy", allow_pickle=False)

    if X.shape != (10000, 6) or X_total.shape != (20000, 16):
        raise ValueError("Dimensiones inesperadas para matrices reducida/completa")
    if any(len(a) != 10000 for a in (codigos, ids, normas2)):
        raise ValueError("Las matrices auxiliares reducidas no tienen 10000 filas")
    if len(y_total) != 20000 or len(ids_total) != 20000:
        raise ValueError("Los vectores completos deben tener 20000 filas")
    if not np.array_equal(ids, np.arange(1, 20001, 2)):
        raise ValueError("Los IDs reducidos deben ser 1,3,...,19999")
    if not np.array_equal(ids_total, np.arange(1, 20001)):
        raise ValueError("Los IDs completos deben ser 1,2,...,20000")
    if not np.allclose(normas2, np.sum(X ** 2, axis=1)):
        raise ValueError("normas2_reducido.npy no coincide con X_reducido.npy")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(X_total)):
        raise ValueError("Se encontraron valores no finitos")
    if not np.array_equal(codigos, y_total[::2]):
        raise ValueError("Las etiquetas reducidas no coinciden con los IDs impares")
    if not np.allclose(X, X_total[::2, :6]):
        raise ValueError("Los atributos reducidos no coinciden con la versión completa")
    if np.min(codigos) < 0 or np.max(y_total) >= len(mapa):
        raise ValueError("Etiquetas fuera del mapa de clases")

    rng_bd = np.random.default_rng(semilla + 2000003)
    posiciones_db = (np.arange(10000) if db_rows == 10000 else
                     np.sort(rng_bd.choice(10000, db_rows, replace=False)))
    X_db_float = X[posiciones_db]
    y_db = mapa[codigos[posiciones_db]].astype(str)
    ids_db_originales = ids[posiciones_db]

    posiciones_pares = np.flatnonzero(ids_total % 2 == 0)
    rng_q = np.random.default_rng(semilla + 1000003)
    posiciones_q = rng_q.choice(posiciones_pares, consultas, replace=False)

    # Invertir la normalización completa y aplicar la normalización reducida.
    # Esto evita volver a hacer fit y no requiere el archivo ZIP original.
    with np.load(carpeta / "escalador_completo.npz") as esc_completo:
        scale_completo = esc_completo["scale"][:6]
        offset_completo = esc_completo["min"][:6]
    with np.load(carpeta / "escalador_reducido.npz") as esc_reducido:
        scale_reducido = esc_reducido["scale"]
        offset_reducido = esc_reducido["min"]
    if np.any(scale_completo == 0):
        raise ValueError("Escala completa inválida para invertir normalización")
    X_q_crudo = (X_total[posiciones_q, :6] - offset_completo) / scale_completo
    X_q_float = X_q_crudo * scale_reducido + offset_reducido
    y_q = mapa[y_total[posiciones_q]].astype(str)
    ids_q_originales = ids_total[posiciones_q]

    S = 2 ** bits_escala
    X_db = np.rint(X_db_float * S).astype(np.int64)
    X_q = np.rint(X_q_float * S).astype(np.int64)
    normas2_enteras = np.sum(X_db * X_db, axis=1)
    return (X_db, y_db, X_q, y_q, ids_q_originales,
            ids_db_originales, normas2_enteras)

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
                        help="Carpeta con archivos .npy y .npz preprocesados")
    parser.add_argument("--fixed-point-bits", type=int, default=12,
                        help="gamma para S=2**gamma antes de Paillier (por defecto 12)")
    parser.add_argument("--db-rows", type=int, default=10)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--key-bits", type=int, default=1024)
    parser.add_argument("--leaf-capacity", type=int, default=1)
    parser.add_argument("--attributes", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    args = parser.parse_args()
    print("\nCONFIGURACIÓN RECIBIDA:")
    print(vars(args))
    if not (1 <= args.k <= args.db_rows and args.queries >= 1 and
            (args.db_rows <= 10000 and args.queries <= 10000)):
        parser.error("Requiere 1 <= k <= db-rows; queries >= 1; filas totales <= 20000")
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
        "altura_maxima_h": args.max_depth, "semilla": args.seed,
        "indices_consultas": " ".join(map(str, (indices_q - 1).tolist())),
        "ids_originales_consultas": " ".join(map(str, indices_q.tolist())),
        "factor_escala": 2 ** args.fixed_point_bits,
        "tiempo_outsourcing_s": tiempo_outsourcing,
        "tiempo_consulta_promedio_s": float(np.mean([f["tiempo_consulta_cifrada_s"] for f in filas])),
        "tiempo_descifrado_promedio_s": float(np.mean([f["tiempo_descifrado_final_s"] for f in filas])),
        "distancias_evaluadas_promedio": float(np.mean([f["distancias_evaluadas"] for f in filas])),
        **metricas,
        "tiempo_total_s": time.perf_counter() - inicio,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    guardar_csv(args.output_dir / "consultas.csv", filas)
    guardar_csv(args.output_dir / "resumen.csv", [resumen])
    
    # Guardar medición para las gráficas de la tesis
    mediciones_path = args.output_dir / "mediciones_fig4.csv"

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
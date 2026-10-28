# Fase 2 — Benchmark reproducible de baseline SkNN-QP

## Carpetas
- `original/`: los 12 archivos Python originales, idénticos a `sknnqp.zip`.
- `instrumentada/`: copia funcional con cronómetros agregados únicamente a `Main.py`. El resto de los módulos no se altera.
- `runs/`: 24 ejecuciones, logs, CSV de resultados y JSON de tiempos por consulta.
- `benchmark_riguroso.py`: ejecuta 8 configuraciones × 5 repeticiones por defecto.
- `mediciones.csv`, `resumen.csv`: mediciones individuales y resumen estadístico.
- `grafica_*.png`: gráficas de tiempo de consulta cifrada, no tiempo total de proceso.

## Ejecución
Desde esta carpeta, con Python 3, NumPy, Pandas y Matplotlib:

```bash
python benchmark_riguroso.py --repeats 5 --timeout 90
python resumir.py
```

## Interpretación
Se realizaron 3 repeticiones por configuración para la primera batería (24 ejecuciones, 48 consultas).
Todas las configuraciones usaron `leaf-capacity=4`, 2 dimensiones y 2 consultas por ejecución.
El conjunto `test_2d.csv` es fijo y tiene 110 registros; en cada prueba se utilizan los primeros n registros como base y los siguientes dos como consultas.
La coincidencia se verifica mediante orden de identificadores, registros y distancias contra el kNN en texto claro.
`encrypted_query_mean_s` mide la llamada a `DataUser.query()` y contiene sus protocolos de búsqueda; no incluye plaintext kNN ni CSV.
`outsourcing_s` mide conjuntamente construcción del árbol y cifrado/externalización. No los separa todavía.
`wall_s` incluye arranque del intérprete, importaciones, creación de claves, operaciones de preparación, consultas, validación, impresión y escritura a disco.
La fase es una medición de la baseline actual, no una implementación criptográficamente idéntica al esquema de dos nubes del artículo.
128 bits y 256 bits son parámetros solo de pruebas y no ofrecen seguridad criptográfica adecuada.
Las mediciones pueden variar según el hardware, la carga del sistema y el carácter aleatorio de Paillier.
Los 3 ensayos son exploratorios; para estudio cuantitativo robusto repetir >= 5 o >= 10 y registrar especificaciones de máquina.
No se extrapolan los resultados a 10 000 registros ni a seis dimensiones.

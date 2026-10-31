# SKNN-QP con Paillier — parámetros solicitados

Parámetros predeterminados: **n=10, h=7 (profundidad máxima, no profundidad alcanzada), k=10, m=6 y K=1024 bits**. Se utilizan los primeros seis atributos de Letter Recognition, con la etiqueta de clase conservada en el cliente confiable.

## Dataset

El dataset Letter Recognition (UCI) no se distribuye en el repositorio. Antes de ejecutar, descárgalo y empaquétalo como `letter+recognition.zip` en la raíz del proyecto (el zip debe contener el archivo `letter-recognition.data` en su primer nivel):

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data
zip letter+recognition.zip letter-recognition.data
```

## Biblioteca instalable con Poetry

Se requiere Python >=3.10 y Poetry 2.x. NumPy es la única dependencia base;
Rich es opcional. Las herramientas pytest y Ruff están en el grupo dev.
`pyproject.toml` es la fuente de dependencias; se eliminó requirements.txt
para evitar dos listas manuales. El lock vacío inicial debe generarse localmente:

```bash
poetry lock
poetry check --lock
poetry install --with dev --extras rich
poetry run pytest -q
poetry run ruff check sknnpaillier tests
poetry build
```

Para instalar únicamente la biblioteca: `poetry install --only main`.
El paquete construido incluye `sknnpaillier`, sin empaquetar scripts o pruebas.

```python
from sknnpaillier import (
    CloudBQueryProcessingModule,
    KeyManagementModule,
    SecureDataTransformationModule,
)

keys = KeyManagementModule(1024)
transformer = SecureDataTransformationModule(keys.public_key)
cloud = CloudBQueryProcessingModule(transformer, keys)
transformer.set_cloud(cloud)
```

Los módulos públicos usan snake_case: `cloud_b`, `kd_tree`, `key_m` y
`secure_data_t`. Los scripts se trasladaron a `scripts/` y conservan imports
antiguos: su ejecución queda pendiente de migración. Esta etapa no los modifica.
Pandas y Matplotlib son dependencias de esos scripts, no de la biblioteca.

## Logging de tiempos

El proyecto usa `sknnpaillier.log` (JSON estructurado por stdout) para los tiempos de cada etapa: generación de llaves, outsourcing (cifrado de BD vs construcción del KD-tree), fases de la consulta cifrada, descifrado y totales. Por defecto la consola muestra nivel INFO; para ver el desglose por fases:

```bash
SKNN_LOG_CONSOLE_LEVEL=DEBUG poetry run python tu_aplicacion.py
```

Variables de entorno disponibles:

| Variable | Default | Descripción |
|---|---|---|
| `SKNN_LOG_LEVEL` | `DEBUG` | Nivel global del logger |
| `SKNN_LOG_CONSOLE_LEVEL` | `INFO` | Nivel mínimo mostrado en consola |
| `SKNN_LOG_DISABLED` | `0` | `1` desactiva la salida del logging |
| `SKNN_LOG_TO_FILE` | `0` | `1` escribe logs rotativos en archivo |
| `SKNN_LOG_ERROR_FILE` | `0` | `1` escribe archivo separado de errores |
| `SKNN_LOG_PATH` | `.sknn/log` | Directorio de logs (solo si `TO_FILE`/`ERROR_FILE`) |
| `SKNN_LOG_RICH` | `0` | `1` usa `rich` en consola (requiere `poetry install --extras rich`) |
| `SKNN_LOG_JSON_INDENT` | `0` | `>0` indenta el JSON de consola |

`h=7` es una cota de profundidad. Para un KD-tree por medianas con 10 registros, la máxima profundidad efectiva será alrededor de 4 aristas, según el árbol; imponer exactamente siete niveles requeriría una política de partición diferente. `leaf-capacity=1` permite subdividir hasta hojas individuales.

Los diez vecinos devueltos representan toda la BD (k=n), por lo que las métricas de recuperación no indican capacidad discriminatoria. Este programa solo usa SKNN-QP + Paillier, no ejecuta kNN plaintext de referencia. Las etiquetas son manejadas en el cliente y el descifrado final es autorizado.

**Advertencia:** 1024 bits no es un tamaño recomendado para seguridad contemporánea; se usa aquí únicamente por el parámetro solicitado.

Los imports del paquete no crean handlers ni archivos: los loggers se obtienen
al generar llaves, enviar una base o procesar una consulta. La suite usa datos
sintéticos y llaves de 128 bits solo para pruebas; no necesita el dataset.

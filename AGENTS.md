# Directrices de colaboración

## Contexto y alcance
- Leer `README.md` y `report.md` antes de trabajar; las secciones finales del informe reflejan el estado más reciente.
- El núcleo está en `sknnpaillier/kd_tree.py`, `key_m.py`, `cloud_b.py` y `secure_data_t.py`. Paillier está implementado localmente.
- Respetar los cambios locales y el índice de Git existentes. No restaurar archivos depurados ni hacer commits, borrados o migraciones sin una solicitud que los autorice.

## Convenciones técnicas
- Aplicar PEP 8: cuatro espacios, funciones y variables en `snake_case`, clases en `PascalCase`, constantes en `UPPER_SNAKE_CASE`; limitar líneas de código nuevas a 79 caracteres cuando resulte práctico.
- Conservar nombres públicos y nombres actuales de módulos hasta planificar una migración de sus consumidores. Las clases en PascalCase son correctas.
- Añadir tipos a parámetros y retornos nuevos o modificados, con `PublicKey`, protocolos o referencias diferidas para colaboradores; evitar `Any` sin justificación. Documentar unidades, dimensiones y dominio modular.
- Mantener coordenadas enteras y distinguir residuos módulo n de enteros con signo. No cambiar operaciones criptográficas ni parámetros experimentales sin justificar y validar el efecto.
- Usar `pathlib.Path`; documentar cómo se resuelven las rutas de entrada y salida. No introducir rutas personales absolutas.
- Capturar excepciones específicas en las fronteras de entrada/salida, conservar su causa y evitar ocultar errores. Proteger ejecución de scripts con `if __name__ == "__main__":`.
- Usar `log.get_logger`; no registrar claves privadas ni datos sensibles en claro.

## Entorno y dependencias
- Usar Poetry 2.x y su entorno virtual; `pyproject.toml` es la fuente de dependencias y metadatos. No mantener un requirements.txt manual paralelo.
- Resolver con `poetry lock` e instalar con `poetry install --with dev`; usar `--extras rich` para consola enriquecida.
- Ejecutar herramientas con `poetry run`; `pytest` y `ruff` pertenecen al grupo dev. No fijar manualmente el contenido de poetry.lock.
- Si el usuario restringe instalaciones o ejecución de pruebas, entregar los comandos exactos para su ejecución local.
- Declarar dependencias directas según imports; no agregar paquetes de versiones antiguas del proyecto ni congelar el entorno global.
- No versionar `.venv`, datasets, resultados, logs ni caches; respetar `.gitignore`. `report.md` está ignorado y se conserva como informe local.

## Flujo de desarrollo
1. Inspeccionar el estado de Git y los consumidores del código afectado.
2. Presentar diagnóstico y alcance antes de renombrar módulos o reorganizar el proyecto.
3. Hacer cambios pequeños y reversibles; separar documentación de cambios de comportamiento.
4. Validar con datos sintéticos pequeños y pruebas específicas; evitar disparar lotes de experimentos durante imports o pruebas unitarias.
5. Usar `poetry run pytest -q` para las pruebas en `tests/`; comparar resultados descifrados con un cálculo independiente en claro. Las llaves pequeñas de pruebas no son parámetros de producción.
6. Registrar en `report.md` hallazgos, cambios, comandos de validación y limitaciones reales. No presentar análisis estático como ejecución verificada.

## Alcance de esta migración
- `scripts/` está fuera de alcance hasta nueva indicación. Sus imports antiguos quedan pendientes de migración; no crear puentes de compatibilidad.
- Importar la biblioteca no debe generar llaves, crear handlers ni archivos o ejecutar consultas. Obtener los loggers durante las operaciones.
- La API pública se exporta desde `sknnpaillier/__init__.py`; mantener imports internos relativos y evitar importar scripts desde la biblioteca.

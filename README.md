# SKNN-QP con Paillier — parámetros solicitados

Parámetros predeterminados: **n=10, h=7 (profundidad máxima, no profundidad alcanzada), k=10, m=6 y K=1024 bits**. Se utilizan los primeros seis atributos de Letter Recognition, con la etiqueta de clase conservada en el cliente confiable.

## Ejecutar

```bash
pip install numpy pandas
python Main.py --dataset letter+recognition.zip --db-rows 10 --queries 1 --k 10 --attributes 6 --max-depth 7 --key-bits 1024 --leaf-capacity 1
```

`h=7` es una cota de profundidad. Para un KD-tree por medianas con 10 registros, la máxima profundidad efectiva será alrededor de 4 aristas, según el árbol; imponer exactamente siete niveles requeriría una política de partición diferente. `leaf-capacity=1` permite subdividir hasta hojas individuales.

Los diez vecinos devueltos representan toda la BD (k=n), por lo que las métricas de recuperación no indican capacidad discriminatoria. Este programa solo usa SKNN-QP + Paillier, no ejecuta kNN plaintext de referencia. Las etiquetas son manejadas en el cliente y el descifrado final es autorizado.

**Advertencia:** 1024 bits no es un tamaño recomendado para seguridad contemporánea; se usa aquí únicamente por el parámetro solicitado.
# sknnpaillier

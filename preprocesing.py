
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. CARGAR EL DATASET
RUTA_ZIP = Path("letter+recognition.zip")
SALIDA = Path("salida_preprocesamiento")
SALIDA.mkdir(parents=True, exist_ok=True)

with ZipFile(RUTA_ZIP) as archivo_zip:
    with archivo_zip.open("letter-recognition.data") as archivo:
        df = pd.read_csv(archivo, header=None)

print("Dimensiones originales:", df.shape)

if df.isna().any().any():
    raise ValueError("Existen valores faltantes")

# 2. SEPARAR Y CODIFICAR ETIQUETAS
etiquetas = df.iloc[:, 0]

encoder = LabelEncoder()
y = encoder.fit_transform(etiquetas)

# Solo los 16 atributos numéricos
X = df.iloc[:, 1:].to_numpy(dtype=np.float64)

print("Etiquetas codificadas:", encoder.classes_)

# 3. GENERAR IDs DE REGISTROS

ids = np.arange(1, len(df) + 1)

# 4. CREAR DOS CONFIGURACIONES

# Configuración completa: 20,000 x 16
X_completo = X
y_completo = y
ids_completo = ids

# Configuración reducida: IDs impares, 6 atributos
mascara = (ids % 2 == 1)

X_reducido = X[mascara, :6]
y_reducido = y[mascara]
ids_reducido = ids[mascara]

# 5. NORMALIZACIÓN CON SKLEARN

scaler_completo = MinMaxScaler(feature_range=(0, 1))
X_completo_norm = scaler_completo.fit_transform(X_completo)

scaler_reducido = MinMaxScaler(feature_range=(0, 1))
X_reducido_norm = scaler_reducido.fit_transform(X_reducido)

# 6. PRECALCULAR NORMAS AL CUADRADO

normas2_completo = np.sum(X_completo_norm ** 2, axis=1)

normas2_reducido = np.sum(X_reducido_norm ** 2, axis=1)

# 7. EXPORTAR ARCHIVOS NUMPY

configuraciones = {
    "completo": (
        X_completo_norm, y_completo,
        ids_completo, normas2_completo,
        scaler_completo
    ),
    "reducido": (
        X_reducido_norm, y_reducido,
        ids_reducido, normas2_reducido,
        scaler_reducido
    )
}

for nombre, (matriz, clases, registros, normas, scaler) in configuraciones.items():

    np.save(SALIDA / f"X_{nombre}.npy", matriz)
    np.save(SALIDA / f"clases_{nombre}.npy", clases)
    np.save(SALIDA / f"ids_{nombre}.npy", registros)
    np.save(SALIDA / f"normas2_{nombre}.npy", normas)

    # Parámetros de normalización para futuras consultas
    np.savez(
        SALIDA / f"escalador_{nombre}.npz",
        data_min=scaler.data_min_,
        data_max=scaler.data_max_,
        scale=scaler.scale_,
        min=scaler.min_
    )

    print(f"{nombre}: {matriz.shape}")

np.save(SALIDA / "mapa_etiquetas.npy", encoder.classes_)

print("Preprocesamiento terminado correctamente")

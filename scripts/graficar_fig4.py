
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Leer resultados de las 35 ejecuciones
archivo = Path("resultados/promedios_fig4.csv")
df = pd.read_csv(archivo)

# Crear dos gráficas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, bits in zip(axes, [512, 1024]):

    datos = df[df["key_bits"] == bits]

    for n in sorted(datos["n"].unique()):

        serie = datos[datos["n"] == n].sort_values("h")

        ax.plot(
            serie["h"],
            serie["mean_time_s"],
            marker="o",
            linewidth=2,
            label=f"n={n}"
        )

    ax.set_xlabel("h (level of the kd-tree)")
    ax.set_ylabel("Average query time (sec)")
    ax.set_title(f"Paillier K={bits} bits")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "SKNN-QP: Average query processing time (35 runs)"
)

plt.tight_layout()

salida = Path("resultados/graficas")
salida.mkdir(parents=True, exist_ok=True)

plt.savefig(
    salida / "promedio_35_ejecuciones.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

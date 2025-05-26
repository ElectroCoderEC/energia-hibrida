import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# === 1. CARGAR EL ARCHIVO Excel (formato ancho) ===
archivo = "static/datos/consumo.xlsx"  # Ajusta al nombre real
df = pd.read_excel(archivo, engine="openpyxl")

# === 2. RENOMBRAR Y PARSEAR FECHAS ===
# Se asume que la primera columna es la fecha
df.rename(columns={df.columns[0]: "FECHA"}, inplace=True)
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

# === 3. ELIMINAR FILAS SIN FECHA VÁLIDA ===
df = df[~df["FECHA"].isna()]

# === 4. FIJAR LA FECHA COMO ÍNDICE ===
df.set_index("FECHA", inplace=True)
df = df.sort_index()

# === 5. RELLENAR VALORES FALTANTES (forward/backward) ===
df = df.ffill().bfill()

# === 6. NORMALIZAR ENTRE 0 Y 1 ===
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# === 7. GRAFICAR CONVERSIÓN A NumPy para evitar errores de multi-indexing ===
x = data_scaled.index.values  # convierte DatetimeIndex a numpy array

plt.figure(figsize=(12, 6))
for col in data_scaled.columns:
    y = data_scaled[col].values  # serie a numpy array
    plt.plot(x, y, label=col)

plt.title("Consumo mensual normalizado por institución")
plt.xlabel("Fecha")
plt.ylabel("Consumo (0–1)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. (Opcional) GUARDAR RESULTADOS ===
df.to_excel("consumo_limpio.xlsx")  # datos originales limpios
data_scaled.to_excel("consumo_normalizado.xlsx")  # datos escalados listos para ML

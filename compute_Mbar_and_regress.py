import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import re
import os

# ------------------------------------------------------------
# 1. CARGADOR GENERAL PARA SPARC_Lelli2016c.txt.txt
# ------------------------------------------------------------

def load_sparc_table(file_path):
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue
            parts = re.split(r"\s+", line.strip())
            rows.append(parts)

    max_cols = max(len(r) for r in rows)
    clean_rows = [r + [""]*(max_cols - len(r)) for r in rows]

    df = pd.DataFrame(clean_rows)

    # Primera fila = nombres reales
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # Convertir numéricas donde se pueda
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def extract_global_barionics(df):
    possible_Ldisk = ["Ldisk", "L_disk", "LDisk", "DiskLum"]
    possible_Lbul  = ["Lbul", "L_bul", "LBulge", "BulLum"]
    possible_Mgas  = ["Mgas", "M_gas", "GasMass", "MHI"]

    def find_column(possibles):
        for p in possibles:
            if p in df.columns:
                return p
        return None

    col_Ld = find_column(possible_Ldisk)
    col_Lb = find_column(possible_Lbul)
    col_Mg = find_column(possible_Mgas)

    return {
        "L_disk": df[col_Ld].iloc[0] if col_Ld else 0,
        "L_bul":  df[col_Lb].iloc[0] if col_Lb else 0,
        "M_gas":  df[col_Mg].iloc[0] if col_Mg else 0
    }


# ------------------------------------------------------------
# 2. CARGAR RESULTADOS EDR
# ------------------------------------------------------------

RESULTS_FILE = "EDR/data/sparc/sparc_results.csv"
SPARC_TABLE_FILE = "EDR/data/sparc/SPARC_Lelli2016c.txt.txt"

df_res = pd.read_csv(RESULTS_FILE)

# Mantener solo tus 10 galaxias
galaxies = df_res["Galaxy"].tolist()

print("\n=== GALAXIAS A PROCESAR ===")
print(galaxies)

# Cargar tabla SPARC
print("\n=== CARGANDO TABLA SPARC ===")
df_tab = load_sparc_table(SPARC_TABLE_FILE)
print("Columnas detectadas:", df_tab.columns.tolist())

# Extraer L_disk, L_bul, M_gas
bar = extract_global_barionics(df_tab)
print("\nDatos bariónicos extraídos:")
print(bar)


# ------------------------------------------------------------
# 3. CALCULAR Mbar PARA CADA GALAXIA
# ------------------------------------------------------------

Mbar_list = []
A_list = []
gal_list = []

for _, row in df_res.iterrows():
    galaxy = row["Galaxy"]
    A = row["A"]
    Yd = row["Yd"]
    Yb = row["Yb"]

    Ld = bar["L_disk"]
    Lb = bar["L_bul"]
    Mg = bar["M_gas"]

    Mbar = Yd * Ld + Yb * Lb + Mg

    gal_list.append(galaxy)
    A_list.append(A)
    Mbar_list.append(Mbar)

df_out = pd.DataFrame({
    "Galaxy": gal_list,
    "A": A_list,
    "Mbar": Mbar_list
})

df_out.to_csv("Mbar_results.csv", index=False)
print("\n=== Mbar_results.csv generado ===")


# ------------------------------------------------------------
# 4. REGRESIÓN log(A) vs log(Mbar)
# ------------------------------------------------------------

logA = np.log10(df_out["A"])
logM = np.log10(df_out["Mbar"])

reg = linregress(logM, logA)

slope = reg.slope
intercept = reg.intercept
rval = reg.rvalue
pval = reg.pvalue
stderr = reg.stderr

print("\n=== REGRESIÓN log(A) vs log(Mbar) ===")
print(f"Slope:     {slope}")
print(f"Intercept: {intercept}")
print(f"R value:   {rval}")
print(f"P value:   {pval}")
print(f"Std Err:   {stderr}")


# ------------------------------------------------------------
# 5. GRAFICAR RESULTADO
# ------------------------------------------------------------

plt.figure(figsize=(9,6))
plt.scatter(logM, logA, s=90)

xfit = np.linspace(min(logM), max(logM), 100)
yfit = slope * xfit + intercept
plt.plot(xfit, yfit, linewidth=2)

plt.xlabel("log(Mbar)")
plt.ylabel("log(A)")
plt.title("Relación EDR: log(A) vs log(Mbar)\n(10 galaxias)")

plt.grid(True, alpha=0.3)
plt.savefig("A_vs_Mbar_regression.png", dpi=300)

print("\n=== Gráfico guardado como A_vs_Mbar_regression.png ===\n")
print("Proceso completado con éxito 🚀")

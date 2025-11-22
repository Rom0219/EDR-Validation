import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

RESULTS_FILE = "EDR/data/sparc/results/sparc_results.csv"
TABLE2_FILE = "EDR/data/sparc/SPARC_Lelli2016_Table2.csv"
OUT_DIR = "edr_baryons_output"

import os
os.makedirs(OUT_DIR, exist_ok=True)

df_res = pd.read_csv(RESULTS_FILE)
df_tab = pd.read_csv(TABLE2_FILE)

df = pd.merge(df_res, df_tab, on="Galaxy", how="inner")

df["Mbar"] = df["Yd"] * df["Ldisk"] + df["Yb"] * df["Lbul"] + df["Mgas"]

df["logA"] = np.log10(df["A"])
df["logMbar"] = np.log10(df["Mbar"])

x = df["logMbar"]
y = df["logA"]

reg = linregress(x, y)

plt.figure(figsize=(8,6))
plt.scatter(x, y, s=60, label="Galaxias")
plt.plot(x, reg.intercept + reg.slope*x, label=f"Fit: slope={reg.slope:.3f}", linewidth=2)

plt.xlabel("log(M_bar)")
plt.ylabel("log(A)")
plt.grid(True)
plt.legend()
plt.title("Relación log(A) – log(M_bar)")

out_plot = f"{OUT_DIR}/regression_logA_logMbar.png"
plt.savefig(out_plot, dpi=200)
plt.close()

out_csv = f"{OUT_DIR}/computed_Mbar.csv"
df.to_csv(out_csv, index=False)

print(df[["Galaxy","A","Mbar","logA","logMbar"]])
print(f"\nSaved: {out_plot}")
print(f"Saved: {out_csv}")

print("\nRegression summary:")
print(f"Slope = {reg.slope}")
print(f"Intercept = {reg.intercept}")
print(f"R = {reg.rvalue}")
print(f"p-value = {reg.pvalue}")

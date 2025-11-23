#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlaciones_sparc_edr.py — Diagnóstico de Robustez del Fit Local
------------------------------------------------------------------
Calcula y plotea las matrices de correlación (Pearson, Spearman)
entre los parámetros ajustados (A, R0, Yd, Yb) y el chi2 reducido.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---- CONFIG Y RUTAS CORREGIDAS PARA PORTABILIDAD -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumimos que este script está en scripts/analysis/, subimos dos niveles a la raíz del proyecto
ROOT_DIR = os.path.join(BASE_DIR, "..", "..") 

# Ruta al archivo de resultados principal
CSV_PATH = os.path.join(ROOT_DIR, "results", "sparc_results.csv")
# Directorio de salida dentro de la carpeta 'results'
OUTDIR = os.path.join(ROOT_DIR, "results", "correlations")
os.makedirs(OUTDIR, exist_ok=True)

# ---- CARGA -------------------------------------------------
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"[ERROR] Archivo CSV no encontrado en: {CSV_PATH}. Ejecute run_selected_sparc.py primero.")
    exit()

print("COLUMNS:", df.columns.tolist())
print(df.head())

# ---- NORMALIZAR NOMBRES (busca nombres comunes) ----------
def pick(df, options):
    for o in options:
        if o in df.columns:
            return o
    return None

A_col = pick(df, ["A","A_value","A_mean"])
R0_col = pick(df, ["R0","R_0"])
Yd_col = pick(df, ["Yd","Y_d","Ydisk","Y_disk"])
Yb_col = pick(df, ["Yb","Y_b","Ybul","Y_bul"])
chi2r_col = pick(df, ["chi2_red","chi2_reduced","chi2r","chi2_red."])
sigma_col = pick(df, ["sigma_extra","sigma_jitter"])

cols_map = {"A":A_col, "R0":R0_col, "Yd":Yd_col, "Yb":Yb_col, "chi2_red":chi2r_col, "sigma_extra":sigma_col}
print("Columns used:", cols_map)

# ---- CONSTRUCCIÓN DATAFRAME DE ANÁLISIS --------------------
# Filtrar solo las columnas que se encontraron
use_cols = {k:v for k,v in cols_map.items() if v is not None}
analysis = df[list(use_cols.values())].copy()
analysis.columns = list(use_cols.keys())  # renombrar a llaves: A,R0,...
analysis = analysis.apply(pd.to_numeric, errors='coerce').dropna()
print(f"Usando {len(analysis)} filas para análisis")

if len(analysis) < 3:
    print("[SKIP] Datos insuficientes para correlación (N < 3). Finalizando.")
    exit()

# ---- MATRICES DE CORRELACIÓN ------------------------------
pearson = analysis.corr(method='pearson')
spearman = analysis.corr(method='spearman')
pearson.to_csv(os.path.join(OUTDIR,"pearson_corr.csv"))
spearman.to_csv(os.path.join(OUTDIR,"spearman_corr.csv"))
print("Saved correlation matrices")

# ---- PLOT Y REGRESIONES POR PAREJAS -------------------------
pairs = [("A","R0"), ("A","Yd"), ("R0","Yd"), ("A","chi2_red"), ("R0","chi2_red")]
summary_rows = []

for x,y in pairs:
    if x not in analysis.columns or y not in analysis.columns: 
        continue
    X = analysis[x].values
    Y = analysis[y].values
    if len(X) < 3:
        continue

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    rho, spearman_p = stats.spearmanr(X, Y)
    pearson_r, pearson_p = stats.pearsonr(X, Y)

    # Save summary
    summary_rows.append({
        "x": x, "y": y,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_rho": rho, "spearman_p": spearman_p,
        "slope": slope, "intercept": intercept,
        "r_value": r_value, "r_pvalue": p_value, "std_err": std_err,
        "N": len(X)
    })

    # Scatter + fit
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(X, Y)
    # Solo plotear la línea de regresión si hay varianza en X
    if np.std(X) > 1e-9:
        xs = np.linspace(np.min(X), np.max(X), 200)
        ax.plot(xs, intercept + slope*xs, linestyle='--')
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}\npearson r={pearson_r:.3f} (p={pearson_p:.2g}), N={len(X)}")
    ax.grid(True)
    outpng = os.path.join(OUTDIR, f"scatter_{x}_vs_{y}.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---- GUARDAR RESUMEN ---------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTDIR,"pairwise_regression_summary.csv"), index=False)
print("Saved pairwise summary")

# ---- HISTOGRAMAS Y DIAGNÓSTICOS -----------------------------
# Residuales globales (si tu CSV tiene sigma_extra)
if "sigma_extra" in analysis.columns:
    plt.figure(figsize=(6,4))
    plt.hist(analysis["sigma_extra"].dropna(), bins=20, edgecolor='k')
    plt.title("Distribución de sigma_extra")
    plt.xlabel("sigma_extra (km/s)")
    plt.savefig(os.path.join(OUTDIR,"hist_sigma_extra.png"), dpi=200)
    plt.close()

# ---- PRINT BREVE RESUMEN -----------------------------------
print("\nPairwise regression summary:")
print(summary_df.to_string(index=False))

print("\nResultados y plots guardados en", OUTDIR)

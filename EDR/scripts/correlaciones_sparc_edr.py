"""
correlaciones_sparc_edr.py — Diagnóstico de Robustez del Fit Local
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---- CONFIG & INPUT ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta corregida: asume que sparc_results.csv está en el directorio 'results'
CSV_PATH = os.path.join(BASE_DIR, "..", "results", "sparc_results.csv") 

OUTDIR = os.path.join(BASE_DIR, "..", "results", "diagnostics")
os.makedirs(OUTDIR, exist_ok=True)

# ---- CARGA Y FILTRO ----------------------------------------
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"[ERROR] Archivo CSV no encontrado en: {CSV_PATH}. Ejecute run_selected_sparc.py primero.")
    exit()

df = df.dropna(subset=["chi2_red"]).query("fit_ok == True").copy()
analysis = df[["A", "R0", "Yd", "Yb", "chi2_red", "sigma_extra"]].apply(pd.to_numeric, errors='coerce').dropna()
print(f"Usando {len(analysis)} filas para el análisis de correlación.")

if len(analysis) < 2:
    print("[SKIP] Se necesitan al menos 2 puntos para calcular correlación. Finalizando.")
    exit()

# ---- MATRICES DE CORRELACIÓN ------------------------------
pearson = analysis.corr(method='pearson')
spearman = analysis.corr(method='spearman')

pearson.to_csv(os.path.join(OUTDIR,"pearson_corr.csv"))
spearman.to_csv(os.path.join(OUTDIR,"spearman_corr.csv"))
print(f"[OK] Matrices de correlación guardadas en {OUTDIR}")

# ---- PLOT Y REGRESIONES POR PAREJAS -------------------------
pairs = [("A","R0"), ("A","Yd"), ("R0","Yd"), ("A","chi2_red"), ("R0","chi2_red")]
summary_rows = []

for x,y in pairs:
    if x not in analysis.columns or y not in analysis.columns:  
        continue
    X = analysis[x].values
    Y = analysis[y].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    rho, spearman_p = stats.spearmanr(X, Y)
    pearson_r, pearson_p = stats.pearsonr(X, Y)

    summary_rows.append({"x": x, "y": y, "pearson_r": pearson_r, "spearman_rho": rho, "N": len(X)})

    # Scatter + fit
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(X, Y)
    if np.std(X) > 1e-9:
        xs = np.linspace(np.min(X), np.max(X), 200)
        ax.plot(xs, intercept + slope*xs, linestyle='--', color='r')
    
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}\nPearson r={pearson_r:.3f}, N={len(X)}")
    outpng = os.path.join(OUTDIR, f"scatter_{x}_vs_{y}.png")
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---- GUARDAR RESUMEN ---------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTDIR,"pairwise_regression_summary.csv"), index=False)
print("[OK] Resumen de regresión guardado.")

print("\n>>> DIAGNÓSTICO COMPLETADO <<<")

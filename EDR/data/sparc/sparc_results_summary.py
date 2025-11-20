#!/usr/bin/env python3
# sparc_results_summary.py
# Resumen automático y gráficas maestras a partir de sparc_results.csv

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
CSV = os.path.join(ROOT, "results", "sparc_results.csv")
PLOTS = os.path.join(ROOT, "results", "summary_plots")
os.makedirs(PLOTS, exist_ok=True)

if not os.path.isfile(CSV):
    print("No encuentro", CSV)
    sys.exit(1)

df = pd.read_csv(CSV)
print("Loaded", CSV)
print(df.describe(include='all').T)

# columnas que esperamos (si no están, lo manejamos)
expected = ["Galaxy","A","Aerr","R0","R0err","Yd","Yderr","Yb","Yberr","sigma_extra","chi2","chi2_red","Ndata","Ndof"]
for c in expected:
    if c not in df.columns:
        df[c] = np.nan

# 1) histograma chi2_red
plt.figure(figsize=(6,4))
plt.hist(df['chi2_red'].dropna(), bins=12, edgecolor='k')
plt.axvline(1.0, color='r', linestyle='--', label='chi2_red=1')
plt.xlabel('chi2_red'); plt.ylabel('Ngal')
plt.title('Distribución de chi2_red')
plt.legend()
plt.savefig(os.path.join(PLOTS, "hist_chi2_red.png"), dpi=150)
plt.close()

# 2) A vs chi2_red
plt.figure(figsize=(6,4))
plt.scatter(df['A'], df['chi2_red'])
for i,row in df.iterrows():
    plt.text(row['A'], row['chi2_red'], row['Galaxy'], fontsize=8, alpha=0.8)
plt.xlabel('A [km/s]'); plt.ylabel('chi2_red'); plt.yscale('log')
plt.title('A vs chi2_red')
plt.savefig(os.path.join(PLOTS, "A_vs_chi2red.png"), dpi=150)
plt.close()

# 3) R0 vs A
plt.figure(figsize=(6,4))
plt.errorbar(df['R0'], df['A'], xerr=df['R0err'], yerr=df['Aerr'], fmt='o')
for i,row in df.iterrows():
    plt.text(row['R0'], row['A'], row['Galaxy'], fontsize=8, alpha=0.8)
plt.xlabel('R0 [kpc]'); plt.ylabel('A [km/s]')
plt.title('R0 vs A')
plt.savefig(os.path.join(PLOTS, "R0_vs_A.png"), dpi=150)
plt.close()

# 4) Yd,Yb summary (if present)
if df['Yd'].notna().any():
    plt.figure(figsize=(6,4))
    plt.scatter(df['Yd'], df['A'])
    plt.xlabel('Y_disk'); plt.ylabel('A [km/s]')
    plt.title('Y_disk vs A')
    plt.savefig(os.path.join(PLOTS, "Yd_vs_A.png"), dpi=150)
    plt.close()

# 5) Detailed table for inspection (sorted by chi2_red desc)
df_sorted = df.sort_values('chi2_red', ascending=False)
out_table = os.path.join(PLOTS, "sparc_results_sorted_by_chi2red.csv")
df_sorted.to_csv(out_table, index=False)
print("Saved sorted table:", out_table)
print("Saved summary plots in", PLOTS)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute M_bar per galaxy using parsed Table1 and your sparc_results.csv, then
perform regression log10(A) vs log10(M_bar) for the galaxies present.
Saves:
 - Mbar_table.csv
 - regress_logA_vs_logMbar.png
 - regress_results.txt
"""
import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt

# Paths (adjust if needed)
TABLE1_PARSED = "EDR/data/sparc/SPARC_table1_parsed.csv"
SPARC_RESULTS = "EDR/data/sparc/sparc_results.csv"   # your results with A,Yd,Yb
OUT_MBAR = "EDR/data/sparc/Mbar_table.csv"
OUT_PLOT = "EDR/data/sparc/regress_logA_vs_logMbar.png"
OUT_SUM = "EDR/data/sparc/regress_results.txt"

# Load
df_res = pd.read_csv(SPARC_RESULTS)
df_tab = pd.read_csv(TABLE1_PARSED)

# merge on Galaxy name (be robust: uppercase trimming)
df_res["Galaxy_key"] = df_res["Galaxy"].str.strip().str.upper()
df_tab["Galaxy_key"] = df_tab["Galaxy"].str.strip().str.upper()
df = pd.merge(df_res, df_tab, left_on="Galaxy_key", right_on="Galaxy_key", how="left", suffixes=("_res","_tab"))

# Compute L_disk from SBdisk and Rdisk:
# Table1: SBdisk in solLum/pc^2 and Rdisk in kpc, L_disk = 2π * I0 * Rd^2 (convert kpc->pc: 1 kpc = 1000 pc)
def compute_Ldisk(sb_pc2, Rd_kpc):
    if np.isnan(sb_pc2) or np.isnan(Rd_kpc):
        return np.nan
    Rd_pc = Rd_kpc * 1000.0
    return 2.0 * np.pi * sb_pc2 * (Rd_pc**2) / 1e9  # result in 1e9 Lsun

df["Ldisk_1e9L"] = df.apply(lambda r: compute_Ldisk(r["SBdisk_Lsol_pc2"], r["Rdisk_kpc"]), axis=1)
# Use total L3.6 (in 1e9 Lsun) and set Lbul = Ltot - Ldisk (floor at 0)
df["Ltot_1e9L"] = df["L3.6_1e9Lsun"]
df["Lbul_1e9L"] = df["Ltot_1e9L"] - df["Ldisk_1e9L"]
df.loc[df["Lbul_1e9L"] < 0, "Lbul_1e9L"] = 0.0

# M_gas: use MHI_1e9Msun from Table1 and multiply by 1.33 (helium)
df["Mgas_1e9Msun"] = df["MHI_1e9Msun"] * 1.33

# Now compute Mbar = Yd * Ldisk + Yb * Lbul + Mgas
df["Mbar_1e9Msun"] = df["Yd"] * df["Ldisk_1e9L"] + df["Yb"] * df["Lbul_1e9L"] + df["Mgas_1e9Msun"]

# Keep only galaxies with finite Mbar and A
mask = df["Mbar_1e9Msun"].notnull() & (df["Mbar_1e9Msun"] > 0) & df["A"].notnull()
df_sel = df[mask].copy()

# Regression: log10(A) vs log10(Mbar)
x = np.log10(df_sel["Mbar_1e9Msun"].astype(float))
y = np.log10(df_sel["A"].astype(float))
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# save outputs
df_sel_out = df_sel[["Galaxy","A","R0","Yd","Yb","Mbar_1e9Msun","Ldisk_1e9L","Lbul_1e9L","Mgas_1e9Msun","chi2_red"]]
df_sel_out.to_csv(OUT_MBAR, index=False)

with open(OUT_SUM, "w") as f:
    f.write(f"N_gal = {len(df_sel)}\n")
    f.write(f"slope = {slope}\nintercept = {intercept}\n")
    f.write(f"r_value = {r_value}\n p_value = {p_value}\n std_err = {std_err}\n")

# Plot
plt.figure(figsize=(6,5))
plt.scatter(df_sel["Mbar_1e9Msun"], df_sel["A"], label="galaxies")
xs = np.logspace(np.log10(df_sel["Mbar_1e9Msun"].min()*0.8), np.log10(df_sel["Mbar_1e9Msun"].max()*1.2), 100)
plt.plot(xs, 10**(intercept + slope * np.log10(xs)), label=f"fit: slope={slope:.2f}", linestyle="--")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("M_bar (1e9 Msun)")
plt.ylabel("A (km/s?)")
plt.title("log A vs log M_bar")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", OUT_MBAR, OUT_PLOT, OUT_SUM)

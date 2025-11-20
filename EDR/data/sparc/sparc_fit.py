#!/usr/bin/env python3
# sparc_fit.py
# Contiene todas las funciones de ajuste, carga y gráficos para modelos SPARC y EDR.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------------------------------------
# 1. MODELO EDR PARA CURVAS DE ROTACIÓN
# -------------------------------------------------------------
def v_edr_model(r, A, R0):
    """
    Modelo básico de la EDR:
       v(r) = sqrt( A * r / (1 + (r/R0)) )
    """
    r = np.asarray(r)
    return np.sqrt( A * r / (1.0 + (r / R0)) )

# -------------------------------------------------------------
# 2. CARGA ROBUSTA DE ARCHIVOS rotmod / SPARC
# -------------------------------------------------------------
def load_rotmod_generic(path):
    """
    Intenta leer archivos rotmod/SPARC con columnas desconocidas.
    Devuelve: r, v_obs, v_err, v_bary
    """
    try:
        df = pd.read_csv(path, comment='#', sep=None, engine='python')
    except Exception:
        df = pd.read_table(path, comment='#', sep=r"\s+", engine='python')

    def pick(colnames):
        for key in colnames:
            for c in df.columns:
                if key in str(c).lower():
                    return c
        return None

    r_col = pick(["r", "rad", "radius"])
    v_col = pick(["vobs", "v_obs", "vrot", "v"])
    e_col = pick(["e_v", "err", "error", "sigma"])
    d_col = pick(["vdisk", "disk"])
    g_col = pick(["vgas", "gas"])
    b_col = pick(["vbulge", "bulge"])

    r = df[r_col].values.astype(float)
    v_obs = df[v_col].values.astype(float)
    v_err = df[e_col].values.astype(float) if e_col else np.ones_like(v_obs)*5.0

    vdisk = df[d_col].values.astype(float) if d_col else np.zeros_like(v_obs)
    vgas = df[g_col].values.astype(float) if g_col else np.zeros_like(v_obs)
    vbulge = df[b_col].values.astype(float) if b_col else np.zeros_like(v_obs)

    v_bary = np.sqrt(vdisk**2 + vgas**2 + vbulge**2)

    return {
        "r": r,
        "v_obs": v_obs,
        "v_err": v_err,
        "v_bary": v_bary,
        "name": path
    }

# -------------------------------------------------------------
# 3. AJUSTE NO LINEAL
# -------------------------------------------------------------
def fit_galaxy(data, initial=(0.05, 5.0)):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    try:
        popt, pcov = curve_fit(
            v_edr_model,
            r,
            v_obs,
            sigma=v_err,
            p0=initial,
            absolute_sigma=True,
            maxfev=100000
        )
        A, R0 = popt
    except Exception:
        A, R0 = np.nan, np.nan
        pcov = np.zeros((2, 2))

    return {
        "A": A,
        "R0": R0,
        "cov": pcov
    }

# -------------------------------------------------------------
# 4. BOOTSTRAP DE ERRORES
# -------------------------------------------------------------
def bootstrap_errors(data, fitres, nboot=200):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    Ab, Rb = [], []

    for _ in range(nboot):
        idx = np.random.randint(0, len(r), len(r))
        r_b = r[idx]
        v_b = v_obs[idx]
        e_b = v_err[idx]

        try:
            popt, _ = curve_fit(
                v_edr_model,
                r_b,
                v_b,
                sigma=e_b,
                p0=(fitres["A"], fitres["R0"]),
                absolute_sigma=True
            )
            Ab.append(popt[0])
            Rb.append(popt[1])
        except Exception:
            continue

    return {
        "A_err": np.std(Ab) if Ab else np.nan,
        "R_err": np.std(Rb) if Rb else np.nan
    }

# -------------------------------------------------------------
# 5. GENERADOR DE PLOTS
# -------------------------------------------------------------
def plot_fit(data, fitres, fname=None):
    r = data["r"]
    v_obs = data["v_obs"]
    v_err = data["v_err"]

    A = fitres["A"]
    R0 = fitres["R0"]

    r_plot = np.linspace(min(r), max(r), 300)
    v_model = v_edr_model(r_plot, A, R0)

    plt.figure(figsize=(7, 5))
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label="Datos", alpha=0.8)
    plt.plot(r_plot, v_model, '-', label=f"EDR fit: A={A:.3f}, R0={R0:.3f}")
    plt.xlabel("r [kpc]")
    plt.ylabel("v [km/s]")
    plt.legend()
    plt.grid(True)

    if fname:
        plt.savefig(fname, dpi=150)
    plt.close()

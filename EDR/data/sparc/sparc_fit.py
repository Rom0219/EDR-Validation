#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_fit.py — Versión final con jitter condicionado

Funciones principales:
- load_rotmod_generic(path)
- fit_galaxy(data, galaxy_name="Galaxy") -> (result, Vmodel_plot, sigma_extra, residuals)
- plot_fit_with_residuals(...)
- plot_residual_histogram_single(...)
- plot_residuals_hist_global(...)
Notas:
- El pipeline usa /mnt/data/FORMULAS_V2.pdf como referencia (variable REFERENCE_PDF).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

REFERENCE_PDF = "/mnt/data/FORMULAS_V2.pdf"

# -------------------------
# 1) Lectura SPARC rotmod
# -------------------------
def load_rotmod_generic(path):
    df = pd.read_csv(path, comment="#", sep=r"\s+")
    if len(df.columns) < 8:
        raise KeyError(f"Archivo SPARC con columnas insuficientes: {len(df.columns)}")
    df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    return {
        "r": df["r"].values.astype(float),
        "Vobs": df["Vobs"].values.astype(float),
        "errV": df["errV"].values.astype(float),
        "Vgas": df["Vgas"].values.astype(float),
        "Vdisk": df["Vdisk"].values.astype(float),
        "Vbul": df["Vbul"].values.astype(float),
    }

# -------------------------
# 2) Modelo EDR + bariones
# -------------------------
def v_edr_component(r, A, R0):
    # núcleo + recorte exponencial (forma simple, ajustable)
    return A * (1.0 - np.exp(-r / R0))

def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    V_edr = v_edr_component(r, A, R0)
    return np.sqrt((Yd * Vdisk)**2 + (Yb * Vbul)**2 + (Vgas)**2 + V_edr**2)

# -------------------------
# 3) Estimador de jitter condicionado
# -------------------------
def find_sigma_extra(residuals, errV, target_chi2red=1.0):
    """
    Encuentra sigma_extra >= 0 tal que chi2_red == target_chi2red si es posible.
    Si no converge a una raíz positiva, devuelve 0.
    """
    resid = np.asarray(residuals)
    errV = np.asarray(errV)

    def chi2_minus_target(sig):
        # chi2(sig) - target*dof
        denom = errV**2 + sig**2
        val = np.sum((resid**2) / denom)
        dof = len(resid) - 4  # asumimos 4 parámetros en el ajuste (A,R0,Yd,Yb)
        if dof <= 0:
            return val - target_chi2red  # fallback
        return val / dof - target_chi2red

    # If already below target with sig=0, return 0
    try:
        if chi2_minus_target(0.0) <= 0:
            return 0.0
    except Exception:
        return 0.0

    # Try to find root in [0, sig_max]
    sig_max = np.std(resid) * 10.0 + np.median(errV)  # generous upper bound
    try:
        root = brentq(chi2_minus_target, 1e-12, max(sig_max, 1e-6), maxiter=100, disp=False)
        return float(max(0.0, root))
    except Exception:
        # If root finding fails, fallback to a positive estimate or zero
        return 0.0

# -------------------------
# 4) Fit principal
# -------------------------
def fit_galaxy(data, galaxy_name="Galaxy"):
    """
    Ajusta los parámetros [A,R0,Yd,Yb] con curve_fit.
    Calcula sigma_extra condicionado (si corresponde) para obtener chi2_red ~ 1.
    Retorna: (result_dict, Vmodel_plot, sigma_extra, residuals_obs)
    """

    r = data["r"]
    Vobs = data["Vobs"]
    errV = data["errV"]
    Vgas = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul = data["Vbul"]

    # initial guess and bounds
    p0 = [120.0, 2.0, 0.5, 0.1]
    bounds = ([10.0, 0.01, 0.0, 0.0], [400.0, 20.0, 3.0, 1.0])

    def model_obs(r_arr, A, R0, Yd, Yb):
        return v_model_total(r_arr, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model_obs, r, Vobs,
            sigma=errV,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=30000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # Interpolated smooth model (interpolate baryonic components)
    r_plot = np.linspace(np.min(r), np.max(r), 300)
    Vdisk_p = np.interp(r_plot, r, Vdisk)
    Vbul_p = np.interp(r_plot, r, Vbul)
    Vgas_p = np.interp(r_plot, r, Vgas)

    Vmodel_plot = v_model_total(r_plot, A, R0, Yd, Yb, Vdisk_p, Vbul_p, Vgas_p)
    Vmodel_obs = v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    # Residuales y chi2 (antes de jitter)
    residuals = Vobs - Vmodel_obs
    dof = max(len(r) - 4, 1)

    chi2_raw = np.sum(((residuals) / errV)**2)
    chi2_red_raw = chi2_raw / dof

    # Calcular sigma_extra condicionado: si chi2_red_raw > 1, resolver. Si <=1 -> sigma_extra=0
    if chi2_red_raw <= 1.0:
        sigma_extra = 0.0
    else:
        sigma_extra = find_sigma_extra(residuals, errV, target_chi2red=1.0)

    # Recompute chi2 with sigma_extra
    denom = errV**2 + sigma_extra**2
    chi2 = np.sum((residuals**2) / denom)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # Decide mode string
    mode = "EDR_barions_jitter_conditioned" if sigma_extra > 0 else "EDR_barions_interpolated"

    result = {
        "ok": True,
        "A": float(A),
        "R0": float(R0),
        "Yd": float(Yd),
        "Yb": float(Yb),
        "Aerr": float(perr[0]) if perr.size > 0 else np.nan,
        "R0err": float(perr[1]) if perr.size > 1 else np.nan,
        "Yderr": float(perr[2]) if perr.size > 2 else np.nan,
        "Yberr": float(perr[3]) if perr.size > 3 else np.nan,
        "chi2": float(chi2),
        "chi2_red": float(chi2_red),
        "r_plot": r_plot,
        "mode": mode
    }

    return result, Vmodel_plot, float(sigma_extra), residuals

# -------------------------
# 5) PLOTS
# -------------------------
def plot_fit_with_residuals(data, Vmodel_plot, result, fname, galaxy_name="Galaxy"):
    r = data["r"]
    Vobs = data["Vobs"]
    errV = data["errV"]

    # Interpolate model to observational points
    Vmodel_interp = np.interp(r, result["r_plot"], Vmodel_plot)
    residuals = Vobs - Vmodel_interp

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=errV, fmt="o", alpha=0.8, label="Observado")
    ax1.plot(result["r_plot"], Vmodel_plot, "-b", lw=2, label=f"EDR model ({result['mode']})")
    ax1.set_ylabel("Velocidad (km/s)")
    ax1.set_title(f"{galaxy_name}")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=errV, fmt="o", color="darkred")
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residual_histogram_single(residuals, fname, galaxy_name="Galaxy"):
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=20, alpha=0.85, edgecolor="black")
    plt.axvline(np.mean(residuals), color="red", linestyle="--", label=f"media={np.mean(residuals):.2f}")
    plt.title(f"Residuales — {galaxy_name}")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def plot_residuals_hist_global(residuals_list, fname="hist_global.png", figsize=(15, 12)):
    # residuals_list: list of 1D arrays
    all_res = np.concatenate([np.asarray(x) for x in residuals_list if len(x) > 0])
    plt.figure(figsize=figsize)
    plt.hist(all_res, bins=40, alpha=0.85, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", lw=1.4")
    plt.title("Histograma global de residuales — SPARC + EDR")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

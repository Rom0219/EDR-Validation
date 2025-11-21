#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_fit_100.py — versión completa para SPARC100
Compatible con run_selected_sparc_100.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ===========================================================
# 1. LECTURA ESTÁNDAR SPARC
# ===========================================================

def load_rotmod_generic(path):
    df = pd.read_csv(path, comment="#", sep=r"\s+")

    if df.shape[1] < 8:
        raise ValueError(f"Archivo SPARC inválido: {path}")

    df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]

    return {
        "r": df["r"].values.astype(float),
        "Vobs": df["Vobs"].values.astype(float),
        "errV": df["errV"].values.astype(float),
        "Vgas": df["Vgas"].values.astype(float),
        "Vdisk": df["Vdisk"].values.astype(float),
        "Vbul": df["Vbul"].values.astype(float)
    }


# ===========================================================
# 2. MODELO EDR + BARIONES
# ===========================================================

def v_edr(r, A, R0):
    return A * (1 - np.exp(-r / R0))


def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    return np.sqrt(
        (Yd * Vdisk)**2 +
        (Yb * Vbul)**2 +
        (Vgas)**2 +
        (v_edr(r, A, R0))**2
    )


# ===========================================================
# 3. AJUSTE PRINCIPAL
# ===========================================================

def fit_galaxy(data, galaxy_name="Galaxy"):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    Vgas = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul = data["Vbul"]

    # Parámetros iniciales
    p0 = [120, 2.0, 0.6, 0.1]
    bounds = (
        [10, 0.05, 0.0, 0.0],
        [400, 20.0, 2.0, 1.0]
    )

    def model(r_arr, A, R0, Yd, Yb):
        return v_model_total(r_arr, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model, r, Vobs,
            sigma=eV,
            absolute_sigma=True,
            p0=p0, bounds=bounds,
            maxfev=25000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # Modelo interpolado
    r_plot = np.linspace(r.min(), r.max(), 300)
    Vmodel_plot = model(r_plot, *popt)

    # Chi2 + jitter condicionado
    V_obs_model = model(r, *popt)
    residuals = Vobs - V_obs_model

    chi2 = np.sum((residuals / eV)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    sigma_extra = np.std(residuals) if chi2_red > 1.0 else 0.0

    result = {
        "ok": True,
        "A": A, "R0": R0, "Yd": Yd, "Yb": Yb,
        "Aerr": perr[0], "R0err": perr[1],
        "Yderr": perr[2], "Yberr": perr[3],
        "chi2": chi2, "chi2_red": chi2_red,
        "sigma_extra": sigma_extra,
        "Ndata": len(r), "Ndof": dof,
        "mode": "EDR_barions_jitter_conditioned",
        "r_plot": r_plot
    }

    return result, Vmodel_plot, sigma_extra, residuals


# ===========================================================
# 4. PLOT CON RESIDUALES
# ===========================================================

def plot_fit_with_residuals(data, Vmodel_plot, result, fname, galaxy_name):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    # Interpolación
    model_interp = np.interp(r, result["r_plot"], Vmodel_plot)
    residuals = Vobs - model_interp

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

    # --- Panel superior ---
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=eV, fmt="o", alpha=0.8)
    ax1.plot(result["r_plot"], Vmodel_plot, "-b", lw=2)
    ax1.set_title(f"{galaxy_name} — Ajuste EDR")
    ax1.set_ylabel("Velocidad (km/s)")
    ax1.grid(True)

    # --- Residuales ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=eV, fmt="o", color="red")
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close()


# ===========================================================
# 5. HISTOGRAMA POR GALAXIA
# ===========================================================

def plot_residual_histogram(residuals, fname, galaxy_name):

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=12, color="gray", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", lw=1.4)
    plt.title(f"Histograma de residuales — {galaxy_name}")
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(fname, dpi=220, bbox_inches="tight")
    plt.close()

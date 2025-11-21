#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_fit.py — versión final estable

Incluye:

✔ Lectura estandarizada SPARC rotmod
✔ Modelo total: gas + disco + bulbo + EDR
✔ Ajuste robusto + jitter
✔ Plot clásico
✔ Plot con residuales
✔ Histograma por galaxia
✔ Compatible 100% con run_selected_sparc.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ===========================================================
# 1. LECTURA SPARC
# ===========================================================

def load_rotmod_generic(path):
    df = pd.read_csv(path, comment='#', sep=r'\s+')

    if len(df.columns) < 8:
        raise KeyError("Formato SPARC desconocido")

    df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]

    return {
        "r": df["r"].values.astype(float),
        "Vobs": df["Vobs"].values.astype(float),
        "errV": df["errV"].values.astype(float),
        "Vgas": df["Vgas"].values.astype(float),
        "Vdisk": df["Vdisk"].values.astype(float),
        "Vbul": df["Vbul"].values.astype(float),
    }


# ===========================================================
# 2. MODELO EDR + BARIONES
# ===========================================================

def v_edr_component(r, A, R0):
    return A * (1 - np.exp(-r / R0))


def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    """Velocidad total cuadrática"""
    return np.sqrt(
        (Yd * Vdisk)**2 +
        (Yb * Vbul)**2 +
        (Vgas)**2 +
        v_edr_component(r, A, R0)**2
    )


# ===========================================================
# 3. AJUSTE fit_galaxy()
# ===========================================================

def fit_galaxy(data, galaxy_name="Galaxy"):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    Vdisk = data["Vdisk"]
    Vbul = data["Vbul"]
    Vgas = data["Vgas"]

    # Parámetros iniciales
    p0 = [120, 2.0, 0.5, 0.1]  # A, R0, Yd, Yb

    bounds = (
        [10, 0.01, 0.0, 0.0],
        [400, 10.0, 2.0, 1.0]
    )

    def model_obs(r_array, A, R0, Yd, Yb):
        return v_model_total(r_array, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    # Intentar ajustarlo
    try:
        popt, pcov = curve_fit(
            model_obs, r, Vobs,
            sigma=eV,
            p0=p0,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=25000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # GRID suave para el plot
    r_plot = np.linspace(min(r), max(r), 300)
    # Interpolar bariones al r_plot
    Vdisk_p = np.interp(r_plot, r, Vdisk)
    Vbul_p  = np.interp(r_plot, r, Vbul)
    Vgas_p  = np.interp(r_plot, r, Vgas)

    def model_smooth(r_arr):
        return v_model_total(r_arr, A, R0, Yd, Yb, Vdisk_p, Vbul_p, Vgas_p)

    Vmodel_plot = model_smooth(r_plot)

    # Modelo sobre puntos observados
    Vmodel_obs = model_obs(r, *popt)

    # Chi2
    chi2 = np.sum(((Vobs - Vmodel_obs) / eV)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # Residuos observacionales
    residuals = Vobs - Vmodel_obs

    # sigma extra
    sigma_extra = np.std(residuals)

    # Resultado
    result = {
        "ok": True,
        "A": A,
        "R0": R0,
        "Yd": Yd,
        "Yb": Yb,
        "Aerr": perr[0],
        "R0err": perr[1],
        "Yderr": perr[2],
        "Yberr": perr[3],
        "chi2": chi2,
        "chi2_red": chi2_red,
        "r_plot": r_plot,
        "mode": "EDR_barions_interpolated"
    }

    return result, Vmodel_plot, sigma_extra, residuals


# ===========================================================
# 4. PLOT BÁSICO
# ===========================================================

def plot_fit(data, Vmodel_plot, result, fname, galaxy_name="Galaxy"):
    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    plt.figure(figsize=(7, 5))
    plt.errorbar(r, Vobs, yerr=eV, fmt="o", label="Observado")
    plt.plot(result["r_plot"], Vmodel_plot, "-b", lw=2, label="Modelo")
    plt.grid()
    plt.xlabel("Radio (kpc)")
    plt.ylabel("Velocidad (km/s)")
    plt.legend()
    plt.title(f"{galaxy_name} — SPARC+EDR")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


# ===========================================================
# 5. PLOT CON RESIDUALES
# ===========================================================

def plot_fit_with_residuals(data, Vmodel_plot, result, fname, galaxy_name="Galaxy"):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    # Interpolar modelo al r observado
    model_interp = np.interp(r, result["r_plot"], Vmodel_plot)
    residuals = Vobs - model_interp

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

    # PANEL SUPERIOR
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=eV, fmt="o", alpha=0.8)
    ax1.plot(result["r_plot"], Vmodel_plot, "-b", lw=2)
    ax1.set_ylabel("Velocidad (km/s)")
    ax1.set_title(f"{galaxy_name}")
    ax1.grid(True)

    # PANEL INFERIOR
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=eV, fmt="o", color="darkred")
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


# ===========================================================
# 6. HISTOGRAMA POR GALAXIA
# ===========================================================

def plot_residual_histogram_single(residuals, fname, galaxy_name="Galaxy"):

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=15, alpha=0.8)
    plt.xlabel("Residual (km/s)")
    plt.ylabel("Frecuencia")
    plt.title(f"Histograma — {galaxy_name}")
    plt.grid(True)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

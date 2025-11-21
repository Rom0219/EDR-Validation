#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_fit.py — versión completa
----------------------------------------
Incluye:

✔ Lectura estandarizada de archivos SPARC rotmod
✔ Modelo EDR + bariones
✔ Función v_model_total()
✔ Ajuste fit_galaxy() con retorno triple:
      result, Vmodel, sigma_extra
✔ plot_fit()
✔ plot_fit_with_residuals() (nuevo)

Todo funciona con run_selected_sparc.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ===========================================================
# 1. LECTURA DE ARCHIVO SPARC
# ===========================================================

def load_rotmod_generic(path):
    df = pd.read_csv(path, comment='#', sep=r'\s+')

    # SPARC: columnas esperadas en orden estándar
    cols = df.columns
    if len(cols) < 8:
        raise KeyError(f"Formato SPARC desconocido ({len(cols)} columnas)")

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
    """Componente EDR: núcleo + cola logarítmica"""
    return A * (1 - np.exp(-r / R0))


def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    """Velocidad total: gas + disco + bulbo + EDR"""
    Vb = np.sqrt(
        (Yd * Vdisk)**2
        + (Yb * Vbul)**2
        + (Vgas)**2
        + (v_edr_component(r, A, R0))**2
    )
    return Vb


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
    p0 = [120, 2.0, 0.5, 0.1]   # [A, R0, Yd, Yb]

    bounds = (
        [10, 0.01, 0.0, 0.0],    # lower
        [400, 10.0, 2.0, 1.0]     # upper
    )

    def model(r, A, R0, Yd, Yb):
        return v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model, r, Vobs,
            sigma=eV,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # Modelo en grilla suave
    r_plot = np.linspace(min(r), max(r), 300)
    Vmodel_plot = model(r_plot, *popt)

    # Chi cuadrado
    Vmodel_obs = model(r, *popt)
    chi2 = np.sum(((Vobs - Vmodel_obs) / eV)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # sigma extra (jitter)
    residuals = (Vobs - Vmodel_obs)
    sigma_extra = np.std(residuals)

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
        "mode": "EDR_barions_jitter"
    }

    return result, Vmodel_plot, sigma_extra


# ===========================================================
# 4. PLOT CLÁSICO
# ===========================================================

def plot_fit(data, Vmodel, result, fname="plot.png", galaxy_name="Galaxy"):
    plt.figure(figsize=(7, 5))

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    plt.errorbar(r, Vobs, yerr=eV, fmt="o", label="Observado", alpha=0.8)
    plt.plot(result["r_plot"], Vmodel, "-b", lw=2, label="Modelo EDR")

    plt.xlabel("Radio (kpc)")
    plt.ylabel("Velocidad (km/s)")
    plt.title(f"{galaxy_name} — SPARC + EDR")
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


# ===========================================================
# 5. PLOT CON RESIDUALES (NUEVO)
# ===========================================================

def plot_fit_with_residuals(data, Vmodel, result, fname="plot.png", galaxy_name="Galaxy"):
    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    model_interp = np.interp(r, result["r_plot"], Vmodel)
    residuals = Vobs - model_interp

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

    # ----- PANEL SUPERIOR -----
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=eV, fmt="o", label="Observado", alpha=0.8)
    ax1.plot(result["r_plot"], Vmodel, "-b", lw=2, label="Modelo EDR")
    ax1.set_ylabel("Velocidad (km/s)")
    ax1.set_title(f"{galaxy_name} — SPARC + EDR")
    ax1.grid(True)
    ax1.legend()

    # ----- PANEL INFERIOR -----
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=eV, fmt="o", color="darkred")
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

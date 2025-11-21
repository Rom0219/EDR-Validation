#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_fit.py — versión científica (Opción A)
-------------------------------------------
✔ Lectura SPARC
✔ Modelo EDR + bariones (solo para puntos observados)
✔ Modelo suave SOLO EDR (sin bariones)
✔ Ajuste estable
✔ Residuales y sigma_extra
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

    if len(df.columns) < 8:
        raise ValueError("Archivo SPARC con columnas insuficientes")

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
# 2. MODELO EDR
# ===========================================================

def v_edr(r, A, R0):
    return A * (1 - np.exp(-r / R0))


# Modelo TOTAL para puntos observados (con bariones)
def model_obs(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    Vtot = np.sqrt((Yd * Vdisk)**2 +
                   (Yb * Vbul)**2 +
                   Vgas**2 +
                   v_edr(r, A, R0)**2)
    return Vtot


# Modelo SUAVE (solo EDR)
def model_smooth(r, A, R0):
    return v_edr(r, A, R0)


# ===========================================================
# 3. AJUSTE COMPLETO
# ===========================================================

def fit_galaxy(data, galaxy_name="Galaxy"):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    Vgas = data["Vgas"]
    Vdisk = data["Vdisk"]
    Vbul = data["Vbul"]

    # Parámetros iniciales
    p0 = [120, 2.0, 0.5, 0.1]
    bounds = ([10, 0.01, 0.0, 0.0],
              [400, 10.0, 2.0, 1.0])

    def model_fit(r, A, R0, Yd, Yb):
        return model_obs(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model_fit,
            r, Vobs,
            sigma=eV,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # Modelo observado
    Vmodel_obs = model_fit(r, *popt)

    # Modelo suave SOLO EDR
    r_plot = np.linspace(min(r), max(r), 300)
    Vmodel_plot = model_smooth(r_plot, A, R0)

    # Chi2
    chi2 = np.sum(((Vobs - Vmodel_obs) / eV)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # Residuales
    residuals = Vobs - Vmodel_obs
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
        "mode": "EDR_scientific"
    }

    return result, Vmodel_plot, sigma_extra, residuals


# ===========================================================
# 4. PLOT + RESIDUALES
# ===========================================================

def plot_fit_with_residuals(data, Vmodel, result,
                            fname="plot.png",
                            galaxy_name="Galaxy"):

    r = data["r"]
    Vobs = data["Vobs"]
    eV = data["errV"]

    # Interpolo SOLO curva suave EDR → residuales no cambian (porque usamos modelo_obs para residuales)
    model_interp = np.interp(r, result["r_plot"], Vmodel)
    residuals = Vobs - model_interp

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)

    # Panel principal
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(r, Vobs, yerr=eV, fmt="o", label="Observado", alpha=0.8)
    ax1.plot(result["r_plot"], Vmodel, "-b", lw=2, label="EDR (suave)")

    ax1.set_ylabel("Velocidad (km/s)")
    ax1.set_title(f"{galaxy_name} — SPARC + EDR")
    ax1.grid(True)
    ax1.legend()

    # Residuales
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=eV, fmt="o", color="darkred")

    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

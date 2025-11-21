#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ===========================================================
# 1. LECTURA SPARC
# ===========================================================

def load_rotmod_generic(path):
    df = pd.read_csv(path, comment='#', sep=r'\s+')
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
    return A * (1 - np.exp(-r / R0))

def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    return np.sqrt(
        (Yd * Vdisk)**2
        + (Yb * Vbul)**2
        + (Vgas)**2
        + (v_edr_component(r, A, R0))**2
    )

# ===========================================================
# 3. AJUSTE
# ===========================================================

def fit_galaxy(data, galaxy_name="Galaxy"):

    r      = data["r"]
    Vobs   = data["Vobs"]
    eV     = data["errV"]
    Vgas   = data["Vgas"]
    Vdisk  = data["Vdisk"]
    Vbul   = data["Vbul"]

    p0 = [120, 2.0, 0.5, 0.1]
    bounds = ([10, 0.01, 0.0, 0.0],
              [400, 10.0, 2.0, 1.0])

    def model(r, A, R0, Yd, Yb):
        return v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    try:
        popt, pcov = curve_fit(
            model, r, Vobs,
            sigma=eV, absolute_sigma=True,
            p0=p0, bounds=bounds, maxfev=25000
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}, None, None

    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.diag(pcov))

    # grilla suave para plot
    r_plot = np.linspace(min(r), max(r), 300)
    Vmodel_plot = model(r_plot, *popt)

    # chi²
    Vmodel_obs = model(r, *popt)
    chi2 = np.sum(((Vobs - Vmodel_obs) / eV)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # residuales → sigma_extra
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
        "mode": "EDR_barions_jitter"
    }

    return result, Vmodel_plot, sigma_extra

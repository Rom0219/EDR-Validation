#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sparc_fit.py — versión con salida de residuales
-----------------------------------------------
Funciones principales:
- load_rotmod_generic(path)
- fit_galaxy(data, galaxy_name="Galaxy")
- plot_fit(data, Vmodel, result, fname=..., galaxy_name=...)
- plot_fit_with_residuals(data, Vmodel, result, fname=..., galaxy_name=...)

Cambios claves:
- fit_galaxy(...) ahora devuelve 4 valores:
    (result_dict, Vmodel_plot, sigma_extra, residuals_obs)
  donde `residuals_obs` es un array con Vobs - Vmodel evaluated at observed radii.
- Ninguna función escribe en disco (salvo los plot_* que guardan figuras).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ===========================================================
# 1. LECTURA DE ARCHIVO SPARC
# ===========================================================
def load_rotmod_generic(path):
    """
    Carga un archivo *_rotmod.dat de SPARC.
    Se espera un archivo con (al menos) estas columnas:
    r, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    (comentarios con '#' son ignorados).
    Devuelve un dict con arrays numpy:
    { "r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul" }
    """
    df = pd.read_csv(path, comment='#', sep=r'\s+', engine='python')
    cols = list(df.columns)
    if len(cols) < 6:
        raise KeyError(f"Formato SPARC desconocido ({len(cols)} columnas) en {path}")

    # Normalizar nombres en caso de que el archivo tuviera columnas sin encabezado
    # Asumimos el orden estándar si no vienen nombres descriptivos.
    # Si el archivo incluye encabezados correctos, esto seguirá funcionando.
    if len(cols) >= 8:
        df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    else:
        # mínimo: r Vobs errV Vgas Vdisk Vbul
        df.columns = ["r", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"] + cols[6:]

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
    """Componente EDR simple (función de ejemplo, flexible)."""
    # proteger contra division by zero
    r = np.asarray(r)
    R0_safe = np.maximum(R0, 1e-12)
    return A * (1.0 - np.exp(-r / R0_safe))


def v_model_total(r, A, R0, Yd, Yb, Vdisk, Vbul, Vgas):
    """
    Velocidad total (en km/s) en cada r (vectorizable).
    Modelo:   V^2 = Vgas^2 + (Yd*Vdisk)^2 + (Yb*Vbul)^2 + v_edr^2
    """
    V_edr = v_edr_component(r, A, R0)
    # Asegurarse que todos los inputs tengan la misma forma broadcasting-friendly
    Vdisk = np.asarray(Vdisk)
    Vbul = np.asarray(Vbul)
    Vgas = np.asarray(Vgas)
    return np.sqrt((Yd * Vdisk)**2 + (Yb * Vbul)**2 + Vgas**2 + V_edr**2)


# ===========================================================
# 3. AJUSTE PRINCIPAL
# ===========================================================
def fit_galaxy(data, galaxy_name="Galaxy"):
    """
    Ajusta el modelo EDR+bariones a los datos provistos en `data` (dict).
    Devuelve:
      result (dict), Vmodel_plot (array sobre r_plot), sigma_extra (float), residuals_obs (array sobre puntos observados)
    Result dict contiene A,R0,Yd,Yb, errores (Aerr,R0err,Yderr,Yberr), chi2, chi2_red, r_plot, mode, ok
    - residuals_obs = Vobs - Vmodel_at_observed_r
    - sigma_extra = std(residuals_obs) (estimación empírica de jitter)
    """
    # Extraer datos
    r = np.asarray(data["r"], dtype=float)
    Vobs = np.asarray(data["Vobs"], dtype=float)
    eV = np.asarray(data["errV"], dtype=float)
    Vgas = np.asarray(data["Vgas"], dtype=float)
    Vdisk = np.asarray(data["Vdisk"], dtype=float)
    Vbul = np.asarray(data["Vbul"], dtype=float)

    # Parámetros iniciales y límites
    p0 = [120.0, 2.0, 0.5, 0.1]   # [A, R0, Yd, Yb]
    bounds = ([1e-6, 1e-6, 0.0, 0.0], [1e3, 1e2, 5.0, 5.0])

    # Definir la función para curve_fit (toma r como primer argumento)
    def model_func(r_array, A, R0, Yd, Yb):
        return v_model_total(r_array, A, R0, Yd, Yb, Vdisk, Vbul, Vgas)

    # Ejecutar fit (capturamos excepciones)
    try:
        popt, pcov = curve_fit(
            model_func, r, Vobs,
            sigma=eV,
            absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            maxfev=200000
        )
    except Exception as exc:
        # Devolver estructura de fallo (no se lanza excepción para mantener pipeline)
        return {"ok": False, "error": str(exc)}, None, None, None

    # Parámetros ajustados
    A, R0, Yd, Yb = popt
    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else np.full_like(popt, np.nan)

    # Grilla fina para plots
    r_plot = np.linspace(np.min(r), np.max(r), 300)
    Vmodel_plot = model_func(r_plot, *popt)

    # Modelo evaluado en los radios observados (para residuales)
    Vmodel_obs = model_func(r, *popt)

    # Estadísticos: chi2 y chi2_red
    # proteger contra errV==0
    eV_safe = np.where(eV <= 0, np.std(Vobs - Vmodel_obs) + 1e-6, eV)
    chi2 = np.sum(((Vobs - Vmodel_obs) / eV_safe)**2)
    dof = max(1, len(r) - len(popt))
    chi2_red = chi2 / dof

    # Residuales y sigma_extra (estimador empírico)
    residuals_obs = Vobs - Vmodel_obs
    sigma_extra = float(np.std(residuals_obs))

    # Preparar resultado
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
        "mode": "EDR_barions_with_residuals"
    }

    return result, Vmodel_plot, sigma_extra, residuals_obs


# ===========================================================
# 4. PLOT CLÁSICO
# ===========================================================
def plot_fit(data, Vmodel, result, fname="plot.png", galaxy_name="Galaxy"):
    r = np.asarray(data["r"])
    Vobs = np.asarray(data["Vobs"])
    eV = np.asarray(data["errV"])

    plt.figure(figsize=(7, 5))
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
# 5. PLOT CON RESIDUALES
# ===========================================================
def plot_fit_with_residuals(data, Vmodel, result, fname="plot.png", galaxy_name="Galaxy"):
    """
    Crea una figura con dos paneles:
      - Panel superior: datos + modelo
      - Panel inferior: residuales (Vobs - Vmodel_at_obs)
    """
    r = np.asarray(data["r"])
    Vobs = np.asarray(data["Vobs"])
    eV = np.asarray(data["errV"])

    # Interpolar el modelo fino sobre los puntos observados para calcular residuales
    model_on_obs = np.interp(r, result["r_plot"], Vmodel)
    residuals = Vobs - model_on_obs

    fig = plt.figure(figsize=(7, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)

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
    ax2.axhline(0.0, color="k", lw=1)
    ax2.errorbar(r, residuals, yerr=eV, fmt="o", color="darkred", alpha=0.85)
    ax2.set_xlabel("Radio (kpc)")
    ax2.set_ylabel("Res.")
    ax2.grid(True)

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

# sparc_fit.py  (ACTUALIZADO: jitter automático solo para NGC2403 e IC2574)
# Referencia: /mnt/data/FORMULAS_V2.pdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ---------------------------
# CONFIG: Galaxias con jitter automático
# ---------------------------
AUTO_JITTER_GALS = ["NGC2403", "IC2574"]
MAX_JITTER = 100.0  # km/s, límite superior razonable
CHI2_TARGET = 1.0
CHI2_TOL = 0.02     # tolerancia en chi2_red
BISECTION_ITERS = 40

# ---------------------------
# UTILIDADES DE I/O
# ---------------------------

def load_rotmod_generic(path):
    """
    Carga un archivo *_rotmod.dat de SPARC.
    Columnas esperadas (en orden): R, Vobs, errV, Vgas, Vdisk, Vbul (opcional).
    """
    df = pd.read_csv(path, comment='#', sep=r'\s+', engine='python')
    cols = list(df.columns)
    if len(cols) < 5:
        raise ValueError(f"Archivo {path} no tiene columnas suficientes.")
    R = df[cols[0]].values.astype(float)
    Vobs = df[cols[1]].values.astype(float)
    errV = df[cols[2]].values.astype(float)
    Vgas = df[cols[3]].values.astype(float)
    Vdisk = df[cols[4]].values.astype(float)
    if len(cols) >= 6:
        Vbul = df[cols[5]].values.astype(float)
    else:
        Vbul = np.zeros_like(R)
    return {"R": R, "Vobs": Vobs, "errV": errV, "Vgas": Vgas, "Vdisk": Vdisk, "Vbul": Vbul}

# ---------------------------
# MODELOS
# ---------------------------

def v_edr(R, A, R0):
    # Modelo EDR simple (velocidad aditiva en cuadratura)
    return A * (1.0 - np.exp(-R / R0))

def model_no_bulge(R, A, R0, Yd, Vgas, Vdisk):
    return np.sqrt(Vgas**2 + (Yd * Vdisk)**2 + v_edr(R, A, R0)**2)

def model_with_bulge(R, A, R0, Yd, Yb, Vgas, Vdisk, Vbul):
    return np.sqrt(Vgas**2 + (Yd * Vdisk)**2 + (Yb * Vbul)**2 + v_edr(R, A, R0)**2)

# ---------------------------
# AJUSTE (con opción jitter automático específica por galaxia)
# ---------------------------

def _compute_chi2(Vobs, Vmodel, err):
    chi2 = np.sum(((Vobs - Vmodel) / err)**2)
    return chi2

def _find_sigma_extra_bisect(Vobs, Vmodel_func, params, err_obs, dof):
    """
    Busca sigma_extra >=0 por bisección tal que chi2_red ~ 1.
    Vmodel_func(params) debe devolver Vmodel array.
    """
    lo = 0.0
    hi = MAX_JITTER
    # check extremes
    Vmodel = Vmodel_func(params)
    chi2_lo = _compute_chi2(Vobs, Vmodel, err_obs) / dof
    if abs(chi2_lo - CHI2_TARGET) <= CHI2_TOL:
        return 0.0, chi2_lo
    # if even with huge jitter it's still > target, return hi
    chi2_hi = _compute_chi2(Vobs, Vmodel, np.sqrt(err_obs**2 + hi**2)) / dof
    if chi2_hi > CHI2_TARGET + CHI2_TOL:
        return hi, chi2_hi
    # bisection
    for _ in range(BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        chi2_mid = _compute_chi2(Vobs, Vmodel, np.sqrt(err_obs**2 + mid**2)) / dof
        if chi2_mid > CHI2_TARGET:
            lo = mid
        else:
            hi = mid
        if abs(chi2_mid - CHI2_TARGET) <= CHI2_TOL:
            return mid, chi2_mid
    # return midpoint if not converged
    mid = 0.5 * (lo + hi)
    chi2_mid = _compute_chi2(Vobs, Vmodel, np.sqrt(err_obs**2 + mid**2)) / dof
    return mid, chi2_mid

def fit_galaxy(data, galaxy_name=None):
    """
    Ajusta la galaxia con límites SPARC (opción B).
    Si galaxy_name en AUTO_JITTER_GALS, se estima sigma_extra por bisección.
    Devuelve (result_dict, modelV, sigma_extra)
    """
    R = data["R"]; Vobs = data["Vobs"]; errV = data["errV"]
    Vgas = data["Vgas"]; Vdisk = data["Vdisk"]; Vbul = data["Vbul"]

    has_bulge = not np.allclose(Vbul, 0.0)

    if has_bulge:
        def fit_fun(R_arr, A, R0, Yd, Yb):
            return model_with_bulge(R_arr, A, R0, Yd, Yb, Vgas, Vdisk, Vbul)
        p0 = [100.0, 1.5, 0.6, 1.0]
        bounds = ([0.01, 0.01, 0.3, 0.5], [500.0, 20.0, 1.1, 2.0])
    else:
        def fit_fun(R_arr, A, R0, Yd):
            return model_no_bulge(R_arr, A, R0, Yd, Vgas, Vdisk)
        p0 = [100.0, 1.5, 0.6]
        bounds = ([0.01, 0.01, 0.3], [500.0, 20.0, 1.1])

    # initial fit (no jitter)
    try:
        popt, pcov = curve_fit(fit_fun, R, Vobs, sigma=errV, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=200000)
    except Exception as e:
        return {"ok": False, "error": f"curve_fit failed: {e}"}, None, None

    # compute model and chi2
    Vmodel = fit_fun(R, *popt)
    nparams = len(popt)
    dof = max(1, len(R) - nparams)
    chi2 = _compute_chi2(Vobs, Vmodel, errV)
    chi2_red = chi2 / dof

    sigma_extra = 0.0
    # if galaxy requires auto-jitter and chi2_red >> 1, estimate sigma_extra
    if (galaxy_name in AUTO_JITTER_GALS) and (chi2_red > 1.2):
        # define Vmodel_func closure for bisection
        Vmodel_func = lambda params: fit_fun(R, *params)
        sigma_extra, chi2_after = _find_sigma_extra_bisect(Vobs, Vmodel_func, popt, errV, dof)
        # recompute chi2 with sigma_extra
        Verr = np.sqrt(errV**2 + sigma_extra**2)
        chi2 = _compute_chi2(Vobs, Vmodel, Verr)
        chi2_red = chi2 / dof

    # package results
    if has_bulge:
        A, R0, Yd, Yb = popt
    else:
        A, R0, Yd = popt
        Yb = 0.0

    result = {
        "ok": True,
        "A": float(A),
        "R0": float(R0),
        "Yd": float(Yd),
        "Yb": float(Yb),
        "chi2": float(chi2),
        "chi2_red": float(chi2_red),
        "dof": int(dof),
        "Ndata": int(len(R)),
        "has_bulge": bool(has_bulge),
        "sigma_extra": float(sigma_extra)
    }

    return result, Vmodel, sigma_extra

# ---------------------------
# PLOT
# ---------------------------

def plot_fit(data, modelV, result, fname=None, galaxy_name=""):
    R = data["R"]; Vobs = data["Vobs"]; errV = data["errV"]
    Vgas = data["Vgas"]; Vdisk = data["Vdisk"]; Vbul = data["Vbul"]

    A = result["A"]; R0 = result["R0"]; Yd = result["Yd"]; Yb = result["Yb"]
    has_bulge = result["has_bulge"]
    sigma_extra = result.get("sigma_extra", 0.0)

    plt.figure(figsize=(8,7))
    gs = plt.GridSpec(2,1, height_ratios=[3,1], hspace=0.08)
    ax = plt.subplot(gs[0])
    axr = plt.subplot(gs[1], sharex=ax)

    # plot data with original err in top panel
    ax.errorbar(R, Vobs, yerr=errV, fmt='k.', label='Vobs (err_obs)')
    # plot data with total error if sigma_extra>0 (dashed markers)
    if sigma_extra > 0:
        ax.errorbar(R, Vobs, yerr=np.sqrt(errV**2 + sigma_extra**2), fmt='none', ecolor='0.6', alpha=0.7, label=f'err_total (sigma_extra={sigma_extra:.2f} km/s)')

    # components
    ax.plot(R, Vgas, 'c--', label='Gas')
    ax.plot(R, Yd * Vdisk, 'g-', label=f'Disk (Yd={Yd:.2f})')
    if has_bulge:
        ax.plot(R, Yb * Vbul, 'm-', label=f'Bulge (Yb={Yb:.2f})')
    else:
        ax.plot(R, np.zeros_like(R), 'm--', label='Bulge: No bulge')

    ax.plot(R, v_edr(R, A, R0), 'b-', label='EDR (v_edr)')
    ax.plot(R, modelV, 'r-', linewidth=2.2, label='Total model')

    ax.set_ylabel('V [km/s]')
    ax.set_title(f"{galaxy_name}  —  χ²_red={result['chi2_red']:.3g}")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True)

    # residuals
    resid = Vobs - modelV
    axr.errorbar(R, resid, yerr=errV, fmt='k.')
    axr.axhline(0, color='0.3', lw=1)
    axr.set_xlabel('R [kpc]')
    axr.set_ylabel('Obs - Model [km/s]')
    axr.grid(True)

    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()

# ---------------------------
# CSV APPEND
# ---------------------------

def append_result(csv_path, galaxy, result):
    exists = os.path.exists(csv_path)
    row = {
        "Galaxy": galaxy,
        "A": result["A"],
        "R0": result["R0"],
        "Yd": result["Yd"],
        "Yb": result["Yb"],
        "chi2": result["chi2"],
        "chi2_red": result["chi2_red"],
        "sigma_extra": result.get("sigma_extra", 0.0),
        "fit_ok": True,
        "mode": "EDR_barions_jitter_conditioned"
    }
    df_row = pd.DataFrame([row])
    if exists:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, index=False)

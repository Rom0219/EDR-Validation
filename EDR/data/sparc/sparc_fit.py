#!/usr/bin/env python3
# sparc_fit.py
# Ajuste automático EDR a curvas SPARC (por galaxia)
# Requiere: numpy, scipy, matplotlib, pandas, astropy (opcional)
# Guarda: results/sparc_fits.csv y plots en results/fits_plots/

import os, glob, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import simps
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 120

# ---------------------
# PARÁMETROS (ajusta según quieras)
# ---------------------
DATA_DIR = "data/sparc"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "fits_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Modelo EDR: parámetros a ajustar por galaxia (A, R_Omega)
# A = kflow * eta * Omega0^2  (amplitud)
# R_Omega = escala radial de decaimiento (en las mismas unidades que r)

def Omega_flow(r, R_Omega, Omega0=1.0):
    return Omega0 * np.exp(-r / R_Omega)

def a_EDR_from_A(r, A, R_Omega):
    # interpretamos A = kflow * eta * Omega0^2, asumimos Omega0=1 en A
    # para tener a_EDR(r) = A * exp(-2 r / R_Omega)
    return A * np.exp(-2*r / R_Omega)

def v_EDR_from_A(r, A, R_Omega):
    a = a_EDR_from_A(r, A, R_Omega)
    # evitar negativos numéricos
    a = np.maximum(a, 0.0)
    return np.sqrt(r * a)

# ---------------------
# Funciones para leer SPARC
# ---------------------
# NOTA: SPARC viene en varios formatos. Aquí asumimos:
# - un CSV por galaxia con columnas: r [kpc], v_obs [km/s], v_err [km/s],
#   v_disk, v_gas, v_bulge (componentes predichas por baryons)
# Si tu SPARC está en otro formato, modifica load_sparc_galaxy() para adaptarlo.

def load_sparc_galaxy(path):
    # intenta leer CSV/TSV - si falla, lanza excepción para que adaptes
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_table(path)
    # columnas esperadas (intenta detectar comunes)
    cols = df.columns.str.lower()
    # heurística
    r_col = next((c for c in df.columns if 'r' in c.lower()), df.columns[0])
    v_col = next((c for c in df.columns if 'v_obs' in c.lower() or 'v'==c.lower()), None)
    if v_col is None:
        # intenta 'vobs' o 'v_circ'
        v_col = next((c for c in df.columns if 'vobs' in c.lower() or 'vcirc' in c.lower()), df.columns[1])
    # componentes baryonicas
    def pick(colnames):
        for name in colnames:
            for c in df.columns:
                if name in c.lower():
                    return c
        return None
    v_disk = pick(['disk','vdisk'])
    v_gas = pick(['gas','vgas'])
    v_bulge = pick(['bulge','vbulge'])
    v_err = pick(['err','error','e_','sigma'])
    # build standardized frame
    r = np.array(df[r_col], dtype=float)
    v = np.array(df[v_col], dtype=float)
    verr = np.array(df[v_err], dtype=float) if v_err is not None else np.ones_like(v)*5.0
    vdisk = np.array(df[v_disk], dtype=float) if v_disk is not None else np.zeros_like(v)
    vgas = np.array(df[v_gas], dtype=float) if v_gas is not None else np.zeros_like(v)
    vbulge = np.array(df[v_bulge], dtype=float) if v_bulge is not None else np.zeros_like(v)
    # baryonic total
    v_bary = np.sqrt(np.clip(vdisk**2 + vgas**2 + vbulge**2, 0.0, None))
    return {'r': r, 'v_obs': v, 'v_err': verr, 'v_bary': v_bary, 'name': os.path.basename(path)}

# ---------------------
# Fitting: chi2
# ---------------------
def chi2_params(params, r, v_obs, v_err, v_bary):
    A, R_Omega = params
    # penaliza R_Omega fuera de rango razonable
    if R_Omega <= 0 or A < 0:
        return 1e9 + 1e6*((R_Omega<=0) + (A<0))
    v_model = np.sqrt(v_bary**2 + v_EDR_from_A(r, A, R_Omega)**2)
    chi2 = np.sum(((v_obs - v_model)/v_err)**2)
    return chi2

def fit_galaxy(data, initial=(0.01, 5.0)):
    r = data['r']; v_obs = data['v_obs']; v_err = data['v_err']; v_bary = data['v_bary']
    res = minimize(lambda x: chi2_params(x, r, v_obs, v_err, v_bary),
                   x0=np.array(initial),
                   method='Nelder-Mead',
                   options={'maxiter':2000, 'fatol':1e-8})
    A_best, R_best = float(res.x[0]), float(res.x[1])
    chi2_best = chi2_params((A_best, R_best), r, v_obs, v_err, v_bary)
    dof = len(r) - 2
    return {'A':A_best, 'R_Omega':R_best, 'chi2':chi2_best, 'dof':dof, 'success':res.success, 'message':res.message}

# ---------------------
# Bootstrap for errors (resampling)
# ---------------------
def bootstrap_errors(data, fit_res, nboot=200):
    r = data['r']; v_obs = data['v_obs']; v_err = data['v_err']; v_bary = data['v_bary']
    A_samples = []
    R_samples = []
    N = len(r)
    rng = np.random.default_rng(12345)
    for _ in range(nboot):
        # resample indices with replacement
        idx = rng.integers(0, N, N)
        r_s, v_s, verr_s, vb_s = r[idx], v_obs[idx], v_err[idx], v_bary[idx]
        try:
            res = minimize(lambda x: chi2_params(x, r_s, v_s, verr_s, vb_s),
                           x0=np.array([fit_res['A'], fit_res['R_Omega']]),
                           method='Nelder-Mead',
                           options={'maxiter':800})
            A_samples.append(res.x[0]); R_samples.append(res.x[1])
        except Exception:
            continue
    if len(A_samples)==0:
        return {'A_err': np.nan, 'R_err': np.nan}
    return {'A_err': float(np.std(A_samples)), 'R_err': float(np.std(R_samples))}

# ---------------------
# Runner: procesa todos los archivos SPARC en data/sparc
# ---------------------
def run_all_sparc(glob_pattern="*.csv"):
    files = glob.glob(os.path.join(DATA_DIR, glob_pattern))
    if len(files)==0:
        print("No se encontraron archivos SPARC en", DATA_DIR, "-> coloca tus CSV ahí o cambia el patrón.")
        return
    rows = []
    for f in files:
        print("Procesando:", f)
        try:
            data = load_sparc_galaxy(f)
        except Exception as e:
            print("Error leyendo", f, ":", e)
            continue
        # fit
        fitres = fit_galaxy(data, initial=(0.01, 10.0))
        errres = bootstrap_errors(data, fitres, nboot=200)
        fitres.update(errres)
        # guardar row
        name = data.get('name', os.path.basename(f))
        rows.append({'galaxy': name, **fitres})
        # plot
        try:
            plot_fit(data, fitres, fname=os.path.join(PLOTS_DIR, name + ".png"))
        except Exception as e:
            print("Error plot", name, e)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "sparc_fits.csv"), index=False)
    print("Guardado:", os.path.join(RESULTS_DIR, "sparc_fits.csv"))

# ---------------------
# Plot helper
# ---------------------
def plot_fit(data, fitres, fname=None):
    r = data['r']; v_obs = data['v_obs']; v_err = data['v_err']; v_bary = data['v_bary']
    A, R = fitres['A'], fitres['R_Omega']
    r_plot = np.linspace(np.min(r), np.max(r), 300)
    v_model = np.sqrt(v_bary**2 + v_EDR_from_A(r_plot, A, R)**2)
    plt.figure(figsize=(6,4))
    plt.errorbar(r, v_obs, yerr=v_err, fmt='o', ms=4, label='Obs')
    plt.plot(r_plot, v_model, '-', label='GR + EDR fit')
    plt.plot(r_plot, v_bary, '--', label='Baryonic')
    plt.xlabel('r [kpc]'); plt.ylabel('v [km/s]')
    plt.legend()
    plt.title(f"{data.get('name','galaxy')}\nA={A:.3e}, R={R:.2f}, chi2red={fitres['chi2']/fitres['dof']:.2f}")
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    run_all_sparc(glob_pattern="*.csv")

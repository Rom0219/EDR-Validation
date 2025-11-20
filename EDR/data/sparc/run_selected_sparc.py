#!/usr/bin/env python3
# run_selected_sparc.py
# Busca archivos rotmod/dat por nombre de galaxia, carga y corre el fit para la lista seleccionada.

import os, glob
import numpy as np
import pandas as pd

# IMPORTA tus funciones del script anterior (ajusta si están en otro módulo)
# from sparc_fit import load_sparc_galaxy, fit_galaxy, bootstrap_errors, plot_fit
# Si no están en módulo, asegúrate de pegar las funciones fit_galaxy, bootstrap_errors y plot_fit en este archivo.

# ---------- Lista final de galaxias ----------
GALAXIES = [
    "NGC3198","NGC2403","NGC2841","NGC6503","NGC3521",
    "DDO154","NGC3741","IC2574","NGC3109","NGC2976"
]

DATA_DIR = "data/sparc"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "fits_plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Loader genérico para archivos rotmod/.dat ----------
def load_rotmod_generic(path):
    """
    Intenta leer archivos rotmod .dat/.txt/.csv robustamente.
    Intenta detectar columnas: r [kpc], v_obs [km/s], v_err, v_disk, v_gas, v_bulge.
    Devuelve dict {'r','v_obs','v_err','v_bary','name'} compatible con fit_galaxy.
    """
    name = os.path.basename(path)
    # leer como tabla de texto, permitir comentarios
    try:
        # intenta lectura simple con pandas (sep any whitespace or comma)
        df = pd.read_csv(path, comment='#', sep=None, engine='python')
    except Exception:
        try:
            df = pd.read_table(path, comment='#', sep='\s+', engine='python')
        except Exception as e:
            raise RuntimeError(f"Cannot parse {path}: {e}")

    cols = [c.lower() for c in df.columns.astype(str)]
    # heurísticas para elegir columnas
    def find(col_keys, default=None):
        for k in col_keys:
            for c in df.columns:
                if k in str(c).lower():
                    return c
        return default

    r_col = find(['r','rad','radius'], df.columns[0])
    v_col = find(['vobs','v_obs','v_obs_kms','v','vcirc','vrot'], None)
    verr_col = find(['e_v','err','error','v_err','sigma'], None)
    vdisk_col = find(['vdisk','disk'], None)
    vgas_col = find(['vgas','gas'], None)
    vbulge_col = find(['vbulge','bulge'], None)

    # fallback si no hay v explicito
    if v_col is None:
        # a menudo los rotmod files dan columnas: r vdisk vbulge vgas vtotal
        # intentamos detectar una columna con nombre 'v' o la última columna como v_obs
        v_col = df.columns[-1]

    r = np.array(df[r_col], dtype=float)
    v_obs = np.array(df[v_col], dtype=float)
    v_err = np.array(df[verr_col], dtype=float) if verr_col is not None else np.ones_like(v_obs)*5.0

    # construir v_bary si hay componentes, si no usar zeros
    vdisk = np.array(df[vdisk_col], dtype=float) if vdisk_col is not None else np.zeros_like(v_obs)
    vgas = np.array(df[vgas_col], dtype=float) if vgas_col is not None else np.zeros_like(v_obs)
    vbulge = np.array(df[vbulge_col], dtype=float) if vbulge_col is not None else np.zeros_like(v_obs)
    v_bary = np.sqrt(np.clip(vdisk**2 + vgas**2 + vbulge**2, 0.0, None))

    return {'r': r, 'v_obs': v_obs, 'v_err': v_err, 'v_bary': v_bary, 'name': name}

# ---------- Utilidades: encontrar archivo para una galaxia ----------
def find_file_for_galaxy(galname):
    """
    Busca en DATA_DIR archivos que contengan el substring galname (case-insensitive).
    Retorna el primer match o None.
    """
    pattern = os.path.join(DATA_DIR, '**', f'*{galname}*')
    matches = glob.glob(pattern, recursive=True)
    # filtrar por extensiones de interés
    matches = [m for m in matches if os.path.splitext(m)[1].lower() in ('.dat','.txt','.csv','.asc')]
    return matches[0] if matches else None

# ---------- Stub de fit (si ya tienes sparc_fit.py, importalas) ----------
# Para funcionar aquí copia/pega las funciones fit_galaxy, bootstrap_errors, plot_fit desde tu sparc_fit.py
# AQUI se asume que existen; si no, pega las funciones completas.

# Intentemos importar desde sparc_fit.py
try:
    from sparc_fit import fit_galaxy, bootstrap_errors, plot_fit
    print("Imported fit functions from sparc_fit.py")
except Exception as e:
    print("No se pudo importar funciones desde sparc_fit.py -> pega fit_galaxy, bootstrap_errors, plot_fit en este archivo.")
    # raise e

# ---------- Runner ----------
results = []
for gal in GALAXIES:
    fpath = find_file_for_galaxy(gal)
    if fpath is None:
        print(f"[SKIP] No se encontró archivo para {gal} en {DATA_DIR}")
        continue
    print("Cargando", fpath)
    try:
        data = load_rotmod_generic(fpath)
    except Exception as e:
        print("Error cargando", fpath, ":", e)
        continue

    # fit
    try:
        fitres = fit_galaxy(data, initial=(0.01, 10.0))
    except Exception as e:
        print("Error en fit_galaxy for", gal, ":", e)
        continue

    # errors bootstrap (intenta, pero no fatal)
    try:
        errres = bootstrap_errors(data, fitres, nboot=200)
    except Exception as e:
        print("Bootstrap failed for", gal, ":", e)
        errres = {'A_err': np.nan, 'R_err': np.nan}

    fitres.update(errres)
    fitres['galaxy'] = gal
    results.append(fitres)

    # plot
    try:
        plot_fname = os.path.join(PLOTS_DIR, f"{gal}.png")
        plot_fit(data, fitres, fname=plot_fname)
        print("Saved plot:", plot_fname)
    except Exception as e:
        print("Plot failed for", gal, ":", e)

# Guardar CSV resumen
if results:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "sparc_fits_selected.csv"), index=False)
    print("Resultados guardados en", os.path.join(RESULTS_DIR, "sparc_fits_selected.csv"))
else:
    print("No hay resultados para guardar.")

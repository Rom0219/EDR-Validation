import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS DE DATOS ---
TABLE1_FILENAME = "SPARC_Lelli2016_Table1.txt"
TABLE2_FILENAME = "SPARC_Lelli2016_Table2.txt"

# --- FUNCIONES DE PARSEO DE DATOS (Tabla 1: Global) ---

def parse_table1(txt_path):
    """
    Parse Table1 (Galaxy Sample) para extraer datos globales (Luminosidad, MHI, Vflat).
    """
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    rows = []
    for ln in lines:
        s = ln.strip()
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+\d+\s+[-+]?\d", s.strip()):
            parts = re.split(r"\s+", s)
            rows.append(parts)

    if not rows:
        raise ValueError(f"No se encontraron filas tipo Table1 en el archivo: {txt_path}")

    cols = ["ID","T","D","e_D","f_D","Inc","e_Inc","L[3.6]","e_L[3.6]","Reff","SBeff","Rdisk","SBdisk","MHI","RHI","Vflat","e_Vflat","Q","Ref"]
    maxcols = max(len(r) for r in rows)
    if maxcols > len(cols):
        cols = cols + [f"extra_{i}" for i in range(len(cols), maxcols)]

    norm = [r + [""]*(len(cols)-len(r)) for r in rows]
    df = pd.DataFrame(norm, columns=cols[:maxcols])
    
    for c in df.columns:
        if c not in ["ID", "Ref"] and not c.startswith("extra"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    return df

# --- FUNCIONES DE PARSEO DE DATOS (Tabla 2: Radial) ---

def parse_table2(txt_path):
    """
    Parse Table2 (Mass Models) para extraer perfiles radiales (R, Vobs, Vgas, Vdisk, Vbul).
    """
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    start_idx = 0
    for i, ln in enumerate(lines):
        if "Mass Models" in ln or "file: datafile2" in ln:
            start_idx = i+1
            break
            
    data_lines = []
    for ln in lines[start_idx:]:
        s = ln.strip()
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", s):
            parts = re.split(r"\s+", s)
            data_lines.append(parts)

    if not data_lines:
        raise ValueError(f"No se detectaron líneas de datos tipo Table2 en el archivo: {txt_path}")

    maxcols = max(len(r) for r in data_lines)
    cols = ["ID","D","R","Vobs","e_Vobs","Vgas","Vdisk","Vbul","SBdisk","SBbul"]
    if maxcols > len(cols):
        cols = cols + [f"extra_{i}" for i in range(len(cols), maxcols)]

    norm = [r + [""]*(len(cols)-len(r)) for r in data_lines]
    df = pd.DataFrame(norm, columns=cols[:maxcols])
    
    for c in df.columns:
        if c != "ID" and not c.startswith("extra"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# --- FUNCIÓN DE CARGA PARA EL FIT LOCAL (ASUMIDA) ---

def load_rotmod_generic(fpath):
    """
    Carga los datos pre-procesados de una sola galaxia (*_rotmod.dat) para el fit.
    Asume un formato simple de columnas (R, Vobs, e_Vobs, Vgas, Vdisk, Vbul).
    """
    data = np.loadtxt(fpath, comments="#", usecols=(0, 1, 2, 3, 4, 5), unpack=True)
    r, Vobs, e_Vobs, Vgas, Vdisk, Vbul = data
    
    mask = (~np.isnan(Vobs)) & (e_Vobs > 0)
    
    if not np.any(mask):
        raise ValueError("No hay puntos de datos válidos con incertidumbre Vobs > 0.")
        
    return {
        "r": r[mask],
        "Vobs": Vobs[mask],
        "e_Vobs": e_Vobs[mask],
        "Vgas": Vgas[mask],
        "Vdisk": Vdisk[mask],
        "Vbul": Vbul[mask],
        "N_valid": len(r[mask])
    }

# --- FUNCIÓN DE FIT Y PLOTTING (SIMULADAS, debe reemplazar con tu lógica real) ---

def fit_galaxy(data, galaxy_name):
    # Lógica de fitting real aquí... (simulada con valores aleatorios para este ejemplo)
    r, Vobs, e_Vobs = data["r"], data["Vobs"], data["e_Vobs"]
    # ...
    # Simulación de resultado exitoso
    A_fit, R0_fit, Yd_fit, Yb_fit = 30.0, 5.0, 0.5, 0.7 
    Vmodel_plot = Vobs * 1.01 # Simula un fit cercano
    residuals_raw = Vobs - Vmodel_plot
    chi2_red = 1.01 
    sigma_extra = 0.0
    residuals = residuals_raw / e_Vobs
    
    result = {"ok": True, "A": A_fit, "R0": R0_fit, "Yd": Yd_fit, "Yb": Yb_fit,
              "chi2": chi2_red * (len(r)-4), "chi2_red": chi2_red, "mode": "EDR_Phenom"}
    
    return result, Vmodel_plot, sigma_extra, residuals

# Funciones de plotting simuladas (dejadas en sparc_utils para importación)
def plot_fit_with_residuals(data, Vmodel_plot, result, out_path, galaxy_name):
    print(f"Generando plot de fit para {galaxy_name} en {out_path}")
def plot_residual_histogram_single(residuals, out_path, galaxy_name):
    print(f"Generando histograma de residuales para {galaxy_name} en {out_path}")
def plot_residuals_hist_global(residuals_for_global, fname, figsize):
    print(f"Generando histograma global en {fname}")
def plot_btfr(log_M, log_V, slope, intercept, r_squared, out_path):
    print(f"Generando plot BTFR en {out_path}")

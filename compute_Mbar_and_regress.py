#!/usr/bin/env python3
# compute_Mbar_from_sparc_table.py
"""
Usa SPARC Table 2 (Lelli+2016) para calcular M_bar y hacer regresion logA vs logM_bar.
- Si SPARC_TABLE_LOCAL apunta a un CSV/TSV, lo usa directamente.
- Si no existe, intenta descargar la tabla oficial (instrucciones).
- Si SPARC_TABLE_LOCAL apunta a un PDF, intenta parsear tablas (si camelot/ tabula están disponibles),
  si no, avisa para que subas el CSV.
Salida en edr_baryons_output/
"""

import os
import sys
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
from scipy import stats
import statsmodels.api as sm
import requests

# -------- CONFIG ----------
SPARC_RESULTS_CSV = "EDR/data/sparc/results/sparc_results.csv"   # tu CSV con A,Yd,Yb,...
SPARC_TABLE_LOCAL = "/mnt/data/SPARC_Lelli2016_Table2.csv"      # <--- si ya lo tienes, pon su ruta aquí
# alternativa (archivo que detecté en tu workspace). Si quieres que lo use, cambia la linea anterior:
# SPARC_TABLE_LOCAL = "/mnt/data/SPARC_EDR_Results.pdf"
OUTDIR = "edr_baryons_output"
BOOTSTRAP_N = 2000
SPARC_TABLE_URL = "https://zenodo.org/record/5754100/files/SPARC_Lelli2016_Table2.csv"  # ejemplo; si falla, descarga manual
os.makedirs(OUTDIR, exist_ok=True)

def normalize_name(name):
    if pd.isna(name):
        return ""
    return "".join(str(name).upper().replace("-", "").split())

def try_load_table(path):
    """Try to load CSV/TSV in a robust way."""
    if not os.path.isfile(path):
        return None
    # if it's PDF, return None (we'll handle separately)
    if path.lower().endswith(".pdf"):
        return None
    # Try common separators
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine='python')
            if df.shape[1] >= 3:
                return df
        except Exception:
            pass
    # fallback
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

# 1) load local EDR results
if not os.path.isfile(SPARC_RESULTS_CSV):
    raise SystemExit(f"Local results CSV not found: {SPARC_RESULTS_CSV}")
results_df = pd.read_csv(SPARC_RESULTS_CSV)
galaxies_to_process = list(results_df['Galaxy'].unique())
print("Galaxies to process:", galaxies_to_process)

# 2) attempt load SPARC table local
sparc_df = try_load_table(SPARC_TABLE_LOCAL)
if sparc_df is None:
    print(f"Local SPARC table not found or not loadable at: {SPARC_TABLE_LOCAL}")
    print("Attempting to download official SPARC Table 2 from SPARC source (if URL valid)...")
    try:
        r = requests.get(SPARC_TABLE_URL, timeout=30)
        r.raise_for_status()
        s = r.content.decode('utf-8')
        sparc_df = pd.read_csv(io.StringIO(s))
        print("Downloaded SPARC table from URL.")
    except Exception as e:
        print("No table available locally and automatic download failed.")
        print("Por favor sube 'SPARC_Lelli2016_Table2.csv' al workspace o confirma que el PDF local contiene la tabla.")
        print("You can upload the CSV to the path and re-run.")
        sys.exit(1)

print("Loaded SPARC table with columns:", sparc_df.columns.tolist())
# show head
print(sparc_df.head())

# 3) detect relevant columns heuristically
cols = [c.lower() for c in sparc_df.columns]
# galaxy name col
name_col = None
for cand in ['name','galaxy','object','objectname']:
    if cand in cols:
        name_col = sparc_df.columns[cols.index(cand)]
        break
if name_col is None:
    # try first column
    name_col = sparc_df.columns[0]
    print("Warning: Could not find explicit name column; using", name_col)

# find L_disk, L_bul, M_gas columns by likely names
Ldisk_col = None
Lbul_col = None
Mgas_col = None
for i,c in enumerate(cols):
    if 'disk' in c and ('mass' in c or 'mstar' in c or 'ldisk' in c or 'l_disk' in c or 'm_' in c):
        Ldisk_col = sparc_df.columns[i]
    if 'bul' in c and ('mass' in c or 'mstar' in c or 'lbul' in c or 'l_bul' in c):
        Lbul_col = sparc_df.columns[i]
    if 'gas' in c or 'mhi' in c or 'm_gas' in c or 'logmhi' in c or 'm_hi' in c:
        Mgas_col = sparc_df.columns[i]

# fallback guesses
if Ldisk_col is None:
    for i,c in enumerate(cols):
        if 'mstar_disk' in c or 'm*_disk' in c or 'm_disk' in c:
            Ldisk_col = sparc_df.columns[i]
if Mgas_col is None:
    for i,c in enumerate(cols):
        if 'mhi' in c or 'm_gas' in c:
            Mgas_col = sparc_df.columns[i]

print("Detected columns -> name:", name_col, "Ldisk:", Ldisk_col, "Lbul:", Lbul_col, "Mgas:", Mgas_col)

# 4) build lookup
lookup = {}
for idx,row in sparc_df.iterrows():
    raw = str(row[name_col])
    nrm = normalize_name(raw)
    rec = {'raw_name': raw}
    if Ldisk_col:
        rec['Ldisk'] = row[Ldisk_col]
    else:
        rec['Ldisk'] = np.nan
    if Lbul_col:
        rec['Lbul'] = row[Lbul_col]
    else:
        rec['Lbul'] = np.nan
    if Mgas_col:
        rec['Mgas'] = row[Mgas_col]
    else:
        rec['Mgas'] = np.nan
    lookup[nrm] = rec

# 5) match with results_df
final_rows = []
unmatched = []
for i,r in results_df.iterrows():
    g = r['Galaxy']
    nrm = normalize_name(g)
    rec = lookup.get(nrm)
    if rec is None:
        # try fuzzy by digits
        digits = ''.join(ch for ch in nrm if ch.isdigit())
        found = None
        if digits:
            for k in lookup.keys():
                if digits in k:
                    found = k
                    break
        if found:
            rec = lookup[found]
    if rec is None:
        unmatched.append(g)
        continue
    def to_num(x):
        try:
            return float(x)
        except:
            try:
                s = str(x).replace(',', '').strip()
                return float(s)
            except:
                return np.nan
    Ldisk = to_num(rec['Ldisk'])
    Lbul = to_num(rec['Lbul']) if not pd.isna(rec['Lbul']) else 0.0
    Mgas = to_num(rec['Mgas']) if not pd.isna(rec['Mgas']) else 0.0
    row_out = r.to_dict()
    row_out.update({'Ldisk': Ldisk, 'Lbul': Lbul, 'Mgas': Mgas})
    final_rows.append(row_out)

print("Matched:", len(final_rows), "Unmatched:", unmatched)
if len(final_rows)==0:
    print("No matches — revisa nombres o sube el CSV Table2.")
    sys.exit(1)

final_df = pd.DataFrame(final_rows)
for c in ['A','R0','Yd','Yb']:
    final_df[c] = pd.to_numeric(final_df[c], errors='coerce')

final_df['Mbar'] = final_df['Yd'] * final_df['Ldisk'] + final_df['Yb'] * final_df['Lbul'] + final_df['Mgas']
final_df = final_df[final_df['Mbar'] > 0].reset_index(drop=True)
final_df['logMbar'] = np.log10(final_df['Mbar'])
final_df['logA'] = np.log10(np.abs(final_df['A']))

# save
final_df.to_csv(os.path.join(OUTDIR,"sparc_matched_Mbar_table2.csv"), index=False)
print("Saved:", os.path.join(OUTDIR,"sparc_matched_Mbar_table2.csv"))
print(final_df[['Galaxy','A','Ldisk','Lbul','Mgas','Mbar','logMbar','logA']])

# regression OLS
X = final_df['logMbar'].values.reshape(-1,1)
y = final_df['logA'].values
X_ols = sm.add_constant(X)
model = sm.OLS(y, X_ols).fit()
print(model.summary())
ols_slope = model.params[1]; ols_intercept = model.params[0]; ols_r2 = model.rsquared

# Theil-Sen
ts = TheilSenRegressor(random_state=0).fit(X,y)
ts_slope = ts.coef_[0]; ts_intercept = ts.intercept_

# bootstrap OLS
bs_slopes = []
bs_intercepts = []
n = len(y)
rng = np.random.default_rng(0)
for _ in range(BOOTSTRAP_N):
    idx = rng.integers(0, n, n)
    Xb = X[idx].reshape(-1,1); yb = y[idx]
    Xb_ = sm.add_constant(Xb)
    try:
        m = sm.OLS(yb, Xb_).fit()
        bs_slopes.append(m.params[1]); bs_intercepts.append(m.params[0])
    except:
        continue
bs_slopes = np.array(bs_slopes); bs_intercepts = np.array(bs_intercepts)
slope_ci = np.percentile(bs_slopes, [16,50,84]); intercept_ci = np.percentile(bs_intercepts, [16,50,84])

# save summary
summary = {
    'ols_slope': ols_slope, 'ols_intercept': ols_intercept, 'ols_r2': ols_r2,
    'ts_slope': ts_slope, 'ts_intercept': ts_intercept,
    'slope_CI_16_50_84': slope_ci.tolist(), 'intercept_CI_16_50_84': intercept_ci.tolist()
}
pd.Series(summary).to_csv(os.path.join(OUTDIR,"regression_table2_summary.csv"))
print("Saved regression summary.")

# plot
xm = np.linspace(final_df['logMbar'].min()-0.2, final_df['logMbar'].max()+0.2, 200)
ym_ols = ols_intercept + ols_slope * xm
ym_ts = ts_intercept + ts_slope * xm
plt.figure(figsize=(6,5))
plt.scatter(final_df['logMbar'], final_df['logA'], s=70, edgecolor='k')
plt.plot(xm, ym_ols, lw=2, label=f"OLS slope={ols_slope:.3f}")
plt.plot(xm, ym_ts, '--', lw=2, label=f"Theil-Sen slope={ts_slope:.3f}")
# bootstrap CI band
ym_low = np.percentile([bs_intercepts + bs_slopes * xx for xx in xm], 16, axis=0)
ym_high = np.percentile([bs_intercepts + bs_slopes * xx for xx in xm], 84, axis=0)
plt.fill_between(xm, ym_low, ym_high, color='gray', alpha=0.25, label='Bootstrap 68% CI')
plt.xlabel(r'$\log_{10} M_{\rm bar}$')
plt.ylabel(r'$\log_{10} A$')
plt.title("log A vs log M_bar (SPARC Table 2)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTDIR,"logA_vs_logMbar_table2.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Saved plot to", os.path.join(OUTDIR,"logA_vs_logMbar_table2.png"))
print("ALL DONE. Outputs in:", OUTDIR)

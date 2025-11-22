#!/usr/bin/env python3
# compute_Mbar_and_regress.py
"""
Descarga SPARC, extrae L_disk, L_bul, M_gas para las galaxias del CSV local,
calcula M_bar = Yd*Ldisk + Yb*Lbul + Mgas, y hace regresion log(A) vs log(M_bar).
Guarda outputs en edr_baryons_output/.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm
import statsmodels.api as sm

# -------- CONFIG --------
ZENODO_ZIP_URL = "https://zenodo.org/record/16284118/files/sparc_database.zip"  # Zenodo SPARC package
SPARC_RESULTS_CSV = "EDR/data/sparc/results/sparc_results.csv"  # tu CSV con A, R0, Yd, Yb...
GALAXY_LIST = None  # None -> use all galaxies in SPARC_RESULTS_CSV (you said 10; it'll use whatever is there)
OUTDIR = "edr_baryons_output"
BOOTSTRAP_N = 2000
np.random.seed(0)

os.makedirs(OUTDIR, exist_ok=True)

# -------- helper utilities --------
def normalize_name(name):
    # Normalize e.g. "NGC 3198" -> "NGC3198", lower-case for matching
    if pd.isna(name):
        return ""
    return "".join(str(name).upper().replace("-", "").split())

def download_sparc_zip(url, target_path):
    print("Downloading SPARC dataset from Zenodo...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Saved SPARC zip to", target_path)

def find_metadata_table_in_zip(zf):
    # Try to locate a file with global metadata (MassModels or SPARC_Lelli2016c)
    candidates = [name for name in zf.namelist() if name.lower().endswith((".mrt", ".csv", ".txt", ".dat"))]
    # Prefer massmodels or sparc database
    for name in candidates:
        ln = name.lower()
        if "massmodel" in ln or "massmodels" in ln or "sparc_l" in ln or "sparc_database" in ln:
            return name
    # fallback: return first candidate
    return candidates[0] if candidates else None

def load_table_from_zip(zf, entry_name):
    with zf.open(entry_name) as fh:
        # try read as csv with pandas (sep auto)
        try:
            s = fh.read().decode("utf-8")
        except:
            s = fh.read().decode("latin1")
        # Attempt to parse with pandas (several fallbacks)
        for sep in [",", r"\s+", "\t", ";"]:
            try:
                df = pd.read_csv(io.StringIO(s), sep=sep, engine="python")
                if len(df.columns) >= 3:
                    return df
            except Exception:
                pass
        # last resort: try whitespace
        df = pd.read_csv(io.StringIO(s), sep=r"\s+", engine="python")
        return df

# -------- 1) Load local results CSV --------
if not os.path.isfile(SPARC_RESULTS_CSV):
    raise SystemExit(f"Local results CSV not found: {SPARC_RESULTS_CSV}")

results_df = pd.read_csv(SPARC_RESULTS_CSV)
print("Loaded local results:", SPARC_RESULTS_CSV)
print("Rows:", len(results_df))

if GALAXY_LIST is None:
    galaxy_names = list(results_df['Galaxy'].unique())
else:
    galaxy_names = GALAXY_LIST

gal_norm = {normalize_name(g): g for g in galaxy_names}

# -------- 2) Download SPARC dataset (if not already downloaded) --------
zip_path = os.path.join(OUTDIR, "sparc_database.zip")
if not os.path.isfile(zip_path):
    download_sparc_zip(ZENODO_ZIP_URL, zip_path)
else:
    print("Using cached SPARC zip:", zip_path)

# -------- 3) Extract and parse metadata table --------
zf = zipfile.ZipFile(zip_path, "r")
meta_entry = find_metadata_table_in_zip(zf)
if meta_entry is None:
    raise SystemExit("Could not find metadata table inside SPARC zip")
print("Found metadata in zip:", meta_entry)

meta_df = load_table_from_zip(zf, meta_entry)
print("Loaded metadata table with shape:", meta_df.shape)
# Show head
print(meta_df.head())

# -------- 4) Identify useful columns (heuristic) --------
# Common SPARC fields in literature: 'Name' or 'Name' or 'galaxy', L[3.6] total, Mgas or M_HI, L_disk, L_bulge
cols_lower = [c.lower() for c in meta_df.columns]

# Try to find columns
c_name = None
for cand in ['name','galaxy','object','objectname']:
    if cand in cols_lower:
        c_name = meta_df.columns[cols_lower.index(cand)]
        break
if c_name is None:
    raise SystemExit("Could not find galaxy name column in SPARC metadata")

# candidates for disk luminosity, bulge luminosity and gas mass
c_ldisk = None
c_lbul = None
c_mgas = None

# look for 3.6 micron total/disk/bulge or L_disk/Lbul
for i,c in enumerate(cols_lower):
    if 'disk' in c and ('lum' in c or 'ldisk' in c or 'l[3.6]' in c or 'ldisk' in c):
        c_ldisk = meta_df.columns[i]
    if 'bul' in c and ('lum' in c or 'lbul' in c or 'l[3.6]' in c):
        c_lbul = meta_df.columns[i]
    if 'm_gas' in c or 'mhi' in c or 'm_gas' in c or 'log_mhi' in c or 'mhi' in c:
        c_mgas = meta_df.columns[i]

# fallback guesses by column name patterns
if c_ldisk is None:
    for i,c in enumerate(cols_lower):
        if 'ldisk' in c or 'l_disk' in c or 'disklum' in c or 'ld[3.6]' in c:
            c_ldisk = meta_df.columns[i]
if c_mgas is None:
    for i,c in enumerate(cols_lower):
        if 'm_gas' in c or 'm_gas' in c or 'hi mass' in c or 'mhi' in c:
            c_mgas = meta_df.columns[i]

print("Detected columns (may be None): name=", c_name, "L_disk=", c_ldisk, "L_bul=", c_lbul, "M_gas=", c_mgas)

# If the table uses log quantities, try to detect and convert later.
# For safety, we'll inspect a sample now.
sample = meta_df[[c_name] + ([c_ldisk] if c_ldisk else []) + ([c_lbul] if c_lbul else []) + ([c_mgas] if c_mgas else [])].head(10)
print("Sample of relevant columns:\n", sample)

# -------- 5) Build a SPARC lookup dict keyed by normalized name --------
lookup = {}
for idx, row in meta_df.iterrows():
    name_raw = str(row[c_name])
    nrm = normalize_name(name_raw)
    rec = {'raw_name': name_raw}
    if c_ldisk:
        rec['Ldisk'] = row[c_ldisk]
    else:
        rec['Ldisk'] = np.nan
    if c_lbul:
        rec['Lbul'] = row[c_lbul]
    else:
        rec['Lbul'] = np.nan
    if c_mgas:
        rec['Mgas'] = row[c_mgas]
    else:
        rec['Mgas'] = np.nan
    lookup[nrm] = rec

# Also create alternative lookups (with/without space)
# Now try match
matches = []
unmatched = []
final_rows = []
for i, r in results_df.iterrows():
    g = r['Galaxy']
    nrm = normalize_name(g)
    if nrm in lookup:
        rec = lookup[nrm]
    else:
        # try fuzzy: try to find a key that contains the numeric part
        digits = ''.join(ch for ch in nrm if ch.isdigit())
        found = None
        if digits:
            for k in lookup.keys():
                if digits in k:
                    found = k
                    break
        if found:
            rec = lookup[found]
        else:
            rec = None

    if rec is None:
        unmatched.append(g)
        continue

    # parse numeric values robustly
    def to_num(x):
        try:
            return float(x)
        except:
            # try remove non-numeric
            s = str(x).replace(',', '').strip()
            try:
                return float(s)
            except:
                return np.nan

    Ldisk = to_num(rec.get('Ldisk', np.nan))
    Lbul = to_num(rec.get('Lbul', np.nan))
    Mgas = to_num(rec.get('Mgas', np.nan))

    # Heuristics: if log values detected (e.g., typical SPARC column logMHI), convert
    # If the numbers look like logs (e.g., values ~7-11) and column name suggests log, treat as log10.
    if c_mgas and 'log' in c_mgas.lower() and not np.isnan(Mgas):
        Mgas = 10**Mgas
    if c_ldisk and 'log' in str(c_ldisk).lower() and not np.isnan(Ldisk):
        Ldisk = 10**Ldisk
    if c_lbul and c_lbul and 'log' in str(c_lbul).lower() and not np.isnan(Lbul):
        Lbul = 10**Lbul

    # Build result row: combine with our local Yd,Yb and A
    row_out = r.to_dict()
    row_out.update({
        "Ldisk": Ldisk,
        "Lbul": Lbul if not np.isnan(Lbul) else 0.0,
        "Mgas": Mgas if not np.isnan(Mgas) else 0.0,
    })
    final_rows.append(row_out)
    matches.append(g)

print("Matched galaxies:", len(matches), "Unmatched:", len(unmatched))
if unmatched:
    print("Unmatched list (check naming):", unmatched)

# Create dataframe for matched galaxies
final_df = pd.DataFrame(final_rows)
if final_df.empty:
    raise SystemExit("No galaxies matched. Check naming conventions.")

# -------- 6) Compute M_bar using local Yd, Yb (from results_df) --------
# Ensure numeric
for col in ['A','R0','Yd','Yb']:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

final_df['Mbar'] = final_df['Yd'] * final_df['Ldisk'] + final_df['Yb'] * final_df['Lbul'] + final_df['Mgas']

# Sanity: remove non-positive Mbar
final_df = final_df[final_df['Mbar'] > 0].reset_index(drop=True)
final_df['logMbar'] = np.log10(final_df['Mbar'])
final_df['logA'] = np.log10(np.abs(final_df['A']))  # A should be positive

# Save table
csv_out = os.path.join(OUTDIR, "sparc_matched_Mbar.csv")
final_df.to_csv(csv_out, index=False)
print("Saved matched table to", csv_out)
print(final_df[['Galaxy','A','Yd','Yb','Ldisk','Lbul','Mgas','Mbar','logMbar','logA']])

# -------- 7) Regression: OLS (statsmodels) and Theil-Sen -----------
X = final_df['logMbar'].values.reshape(-1,1)
y = final_df['logA'].values

# OLS
X_ols = sm.add_constant(X)
model = sm.OLS(y, X_ols).fit()
print("OLS summary:\n", model.summary())
ols_slope = model.params[1]
ols_intercept = model.params[0]
ols_p = model.pvalues[1]
ols_r2 = model.rsquared

# Theil-Sen
ts = TheilSenRegressor(random_state=0)
ts.fit(X, y)
ts_slope = ts.coef_[0]
ts_intercept = ts.intercept_

# Bootstrap for slope/intercept uncertainties
bs_slopes = []
bs_intercepts = []
n = len(y)
rng = np.random.default_rng(0)
for _ in tqdm(range(BOOTSTRAP_N), desc="Bootstrap"):
    idx = rng.integers(0, n, n)
    Xb = X[idx].reshape(-1,1)
    yb = y[idx]
    # OLS on bootstrap sample
    Xb_ols = sm.add_constant(Xb)
    try:
        m = sm.OLS(yb, Xb_ols).fit()
        bs_slopes.append(m.params[1])
        bs_intercepts.append(m.params[0])
    except:
        continue

bs_slopes = np.array(bs_slopes)
bs_intercepts = np.array(bs_intercepts)
slope_ci = np.percentile(bs_slopes, [16,50,84])
intercept_ci = np.percentile(bs_intercepts, [16,50,84])

# Save regression summary
reg_summary = {
    "ols_slope": ols_slope, "ols_intercept": ols_intercept, "ols_p": ols_p, "ols_r2": ols_r2,
    "ts_slope": ts_slope, "ts_intercept": ts_intercept,
    "slope_CI_16_50_84": slope_ci.tolist(),
    "intercept_CI_16_50_84": intercept_ci.tolist()
}
pd.Series(reg_summary).to_csv(os.path.join(OUTDIR, "regression_summary.csv"))
print("Saved regression_summary.csv")

# -------- 8) Plot: logA vs logMbar with fits and bootstrap band --------
plt.figure(figsize=(6,5))
plt.scatter(final_df['logMbar'], final_df['logA'], s=70, edgecolor='k', alpha=0.9)
# OLS line
xm = np.linspace(final_df['logMbar'].min()-0.2, final_df['logMbar'].max()+0.2, 200)
ym_ols = ols_intercept + ols_slope * xm
plt.plot(xm, ym_ols, label=f"OLS slope={ols_slope:.3f}", lw=2)

# Theil-Sen line
ym_ts = ts_intercept + ts_slope * xm
plt.plot(xm, ym_ts, '--', label=f"Theil-Sen slope={ts_slope:.3f}", lw=2)

# Bootstrap band (use percentiles)
ym_bs_low = np.percentile([bs_intercepts + bs_slopes * xx for xx in xm], 16, axis=0)
ym_bs_high = np.percentile([bs_intercepts + bs_slopes * xx for xx in xm], 84, axis=0)
plt.fill_between(xm, ym_bs_low, ym_bs_high, color='gray', alpha=0.25, label="Bootstrap 68% CI")

plt.xlabel(r'$\log_{10} M_{\rm bar}\ (M_\odot\ or\ L_\odot\ scale)$')
plt.ylabel(r'$\log_{10} A\ ( {\rm km/s} )$')
plt.title("EDR: log A vs log M_bar")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTDIR, "logA_vs_logMbar.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved plot ->", os.path.join(OUTDIR, "logA_vs_logMbar.png"))

# -------- 9) Print final results summary ----------
print("Regression results:")
print("OLS slope = {:.4f}, intercept = {:.4f}, p = {:.3e}, R2 = {:.3f}".format(ols_slope, ols_intercept, ols_p, ols_r2))
print("Theil-Sen slope = {:.4f}, intercept = {:.4f}".format(ts_slope, ts_intercept))
print("Bootstrap slope 16/50/84%:", slope_ci)
print("Bootstrap intercept 16/50/84%:", intercept_ci)

print("All outputs saved in:", OUTDIR)

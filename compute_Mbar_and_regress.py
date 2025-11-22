"""
Carga:
 - EDR/data/sparc/sparc_results.csv  (resultados de tus ajustes: A, R0, Yd, Yb, sigma_extra...)
 - EDR/data/sparc/SPARC_Table1_parsed.csv (output de parse_table1)
 - EDR/data/sparc/SPARC_Table2_parsed.csv (output de parse_table2)  <- opcional en este script
Calcula M_bar para cada galaxia:
 M_bar = Yd * L_disk + Yb * L_bul + M_gas
NOTA: Revisar las unidades:
 - SPARC L[3.6] viene dado en 10^9 L_sun (según Table1)
 - MHI viene dado en 10^9 M_sun (Table1 indica la unidad; confirmar)
Ajusta conversiones si fuese necesario (ej: multiplicar por 1e9).
Hace regresión log10(A) vs log10(M_bar) y guarda plots y CSV de salida.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths (ajusta si tus paths son distintos)
RESULTS_FILE = Path("EDR/data/sparc/sparc_results.csv")
TABLE1_FILE = Path("EDR/data/sparc/SPARC_Table1_parsed.csv")   # salida parse_table1
TABLE2_FILE = Path("EDR/data/sparc/SPARC_Table2_parsed.csv")   # opcional
OUT_DIR = Path("EDR/data/sparc/edr_mbar_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carga
df_res = pd.read_csv(RESULTS_FILE)
df_t1 = pd.read_csv(TABLE1_FILE)

# Normalizar nombres de galaxias (ID)
def normalize(name):
    return str(name).strip()

df_res['Galaxy_norm'] = df_res['Galaxy'].apply(normalize)
df_t1['ID_norm'] = df_t1['ID'].apply(normalize)

# Construir mapeo rápido desde Table1: L_disk (usamos L[3.6] como luminosidad total).
# Table1 may not explicitly split disk and bulge luminosities; SPARC Table1 has total L[3.6].
# If you have separate L_disk and L_bul columns, use them. Otherwise, approximate:
#  - If SBdisk and SBbul etc present, user could compute integrals, but here we try to find columns directly.
# Buscamos columnas posibles
Lcol_candidates = [c for c in df_t1.columns if "L[3.6]" in c or "L[3.6]"==c or "L" in c and "3.6" in c]
if not Lcol_candidates:
    # posible que en parse resultante la columna sea 'col08' etc; imprimimos y pedimos revisión
    print("Warning: no se encontró columna obvia L[3.6] en Table1. Columnas disponibles:", df_t1.columns.tolist())

# Strategy: try several candidate names
def get_Ldisk_row(row):
    # Prefer explicit Ldisk if exists, else fall back to L[3.6]
    for cand in ("Ldisk","L_disk","L[3.6]","L[3.6]", "col08", "col09"):
        if cand in row.index:
            return row[cand]
    # try scanning numeric columns that look like luminosity (heurística: values ~ 0.01-100)
    for c in row.index:
        try:
            v = float(row[c])
        except:
            continue
        if 0.001 < abs(v) < 1e5:
            # conservative: pick the first plausible numeric that is not distance (we risk mistakes)
            return v
    return np.nan

# Build dictionary Galaxy -> (L_disk, L_bul, M_gas)
table1_map = {}
for _, r in df_t1.iterrows():
    gid = r.get("ID") if "ID" in r else r.get("col01", None)
    if pd.isna(gid):
        continue
    Ltot = None
    for name in ["L[3.6]","L[3.6] ","L[3.6]"]:
        if name in r:
            Ltot = r[name]
    if Ltot is None:
        # common parse name 'col08' maybe corresponds to L[3.6]
        if "col08" in r:
            Ltot = r["col08"]
    # MHI column: try candidates
    MHI = None
    for c in ("MHI","M_HI","col14","MHI "):
        if c in r:
            MHI = r[c]
    # store
    table1_map[str(gid).strip()] = {"L_3p6": Ltot, "MHI": MHI}

# Now compute Mbar for galaxies in results
out_rows = []
for _, rr in df_res.iterrows():
    g = str(rr['Galaxy']).strip()
    Yd = float(rr['Yd'])
    Yb = float(rr['Yb']) if 'Yb' in rr and not pd.isna(rr['Yb']) else 0.0
    # find L_disk, L_bul (we may only have L_total -> as approximation assume L_disk = L_total, L_bul = 0 if missing)
    t = table1_map.get(g, None)
    if t is None:
        # try match by startswith
        cand = None
        for k in table1_map:
            if k.upper().startswith(g.upper()) or g.upper().startswith(k.upper()):
                cand = table1_map[k]; break
        t = cand
    Ldisk = None
    Lbul = 0.0
    Mgas = 0.0
    if t:
        Ltot = t.get("L_3p6", np.nan)
        if not pd.isna(Ltot):
            # units: SPARC L[3.6] is in 10^9 L_sun (check Table1 header). Here we keep that unit but note it.
            Ldisk = float(Ltot)
        Mhi = t.get("MHI", np.nan)
        if not pd.isna(Mhi):
            Mgas = float(Mhi)  # units per table (likely 10^9 M_sun)
    # Fallbacks
    if Ldisk is None or pd.isna(Ldisk):
        Ldisk = 0.0

    # Compute Mbar (in the same units as L and Mgas) — user must confirm units; below we assume:
    #  - L columns are in 1e9 L_sun
    #  - Mgas (MHI) is in 1e9 M_sun
    # So Mbar (in 1e9 M_sun) = Yd * Ldisk + Yb * Lbul + Mgas
    Mbar = Yd * Ldisk + Yb * Lbul + Mgas

    out_rows.append({
        "Galaxy": g,
        "A": rr.get("A"),
        "R0": rr.get("R0"),
        "Yd": Yd,
        "Yb": Yb,
        "Ldisk_1e9Lsun": Ldisk,
        "Lbul_1e9Lsun": Lbul,
        "Mgas_1e9Msun": Mgas,
        "Mbar_1e9Msun": Mbar,
        "chi2_red": rr.get("chi2_red"),
        "sigma_extra": rr.get("sigma_extra")
    })

df_out = pd.DataFrame(out_rows)
df_out = df_out.replace([np.inf, -np.inf], np.nan)
df_out.to_csv(OUT_DIR / "Mbar_results.csv", index=False)
print("Saved Mbar table to:", OUT_DIR / "Mbar_results.csv")

# Regression log10(A) vs log10(Mbar)
# Keep only rows with positive Mbar and positive A
mask = (df_out["Mbar_1e9Msun"] > 0) & (df_out["A"].notna()) & (df_out["A"]>0)
df_fit = df_out[mask].copy()
if df_fit.empty:
    print("No data available for regression (check Mbar or A values).")
else:
    x = np.log10(df_fit["Mbar_1e9Msun"].astype(float))
    y = np.log10(df_fit["A"].astype(float))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print("Regression results: slope, intercept, r, p, std_err:", slope, intercept, r_value, p_value, std_err)
    # Save regression summary
    with open(OUT_DIR / "regression_summary.txt", "w") as f:
        f.write(f"slope {slope}\nintercept {intercept}\nr_value {r_value}\np_value {p_value}\nstd_err {std_err}\nN {len(x)}\n")

    # Plot scatter + fit
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=40)
    xx = np.linspace(x.min(), x.max(), 100)
    yy = slope * xx + intercept
    plt.plot(xx, yy, linestyle="--")
    plt.xlabel("log10(Mbar) [log10(1e9 Msun)]")
    plt.ylabel("log10(A)")
    plt.title("Regression log A vs log Mbar")
    plt.grid(True)
    plt.savefig(OUT_DIR / "regression_logA_vs_logMbar.png", dpi=200)
    plt.close()
    print("Saved regression plot to:", OUT_DIR / "regression_logA_vs_logMbar.png")

    # Create a combined PNG with histograms (one row: hist(A), hist(Mbar), scatter)
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    axes[0].hist(df_out["A"].dropna(), bins=20)
    axes[0].set_title("Histogram A")
    axes[1].hist(df_out["Mbar_1e9Msun"].dropna(), bins=20)
    axes[1].set_title("Histogram Mbar (1e9 Msun)")
    axes[2].scatter(df_out["Mbar_1e9Msun"], df_out["A"])
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel("Mbar (1e9 Msun)")
    axes[2].set_ylabel("A")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_histograms_and_scatter.png", dpi=200)
    plt.close()
    print("Saved combined PNG:", OUT_DIR / "global_histograms_and_scatter.png")

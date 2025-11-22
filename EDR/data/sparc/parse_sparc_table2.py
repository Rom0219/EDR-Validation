#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse SPARC Table2 (mass models per radius) into a DataFrame and per-galaxy aggregates.
Input: same TXT (it contains many repeated galaxy blocks)
Output: CSV with per-radius rows and an optional per-galaxy stats CSV.
"""
import re
import pandas as pd
INFILE = "/workspaces/EDR-Validation/EDR/data/sparc/SPARC_Lelli2016c.txt.txt"
OUT_RADIAL = "/workspaces/EDR-Validation/EDR/data/sparc/SPARC_table2_radial.csv"
OUT_GAL_SUM = "/workspaces/EDR-Validation/EDR/data/sparc/SPARC_table2_per_galaxy.csv"

def parse_table2(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            # detect radial data lines: start with ID then distance then radius etc.
            if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", ln):
                parts = re.sub(r"\s+", " ", ln).strip().split(" ")
                # expected: ID, D, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul
                try:
                    ID = parts[0]
                    D = float(parts[1])
                    R = float(parts[2])
                    Vobs = float(parts[3])
                    eV = float(parts[4])
                    Vgas = float(parts[5])
                    Vdisk = float(parts[6])
                    Vbul = float(parts[7])
                    SBdisk = float(parts[8]) if len(parts) > 8 else None
                    SBbul = float(parts[9]) if len(parts) > 9 else None
                    rows.append({
                        "Galaxy": ID,
                        "D_Mpc": D,
                        "R_kpc": R,
                        "Vobs": Vobs,
                        "eV": eV,
                        "Vgas": Vgas,
                        "Vdisk": Vdisk,
                        "Vbul": Vbul,
                        "SBdisk": SBdisk,
                        "SBbul": SBbul
                    })
                except Exception:
                    continue
    df = pd.DataFrame(rows)
    df.to_csv(OUT_RADIAL, index=False)
    # per-galaxy summary
    g = df.groupby("Galaxy").agg({
        "R_kpc":"max",
        "Vobs":"count",
        "Vgas":"mean",
        "Vdisk":"mean",
        "Vbul":"mean"
    }).rename(columns={"Vobs":"Nradial"})
    g.to_csv(OUT_GAL_SUM)
    return df, g

if __name__ == "__main__":
    df, g = parse_table2(INFILE)
    print(f"Radial rows: {len(df)}, Galaxies: {len(g)}")

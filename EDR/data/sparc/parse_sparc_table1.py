#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse SPARC Table1-like TXT (the file you uploaded) and produce a CSV with
one row per galaxy including L3.6, Rdisk, SBdisk, MHI, etc.
Input: /workspaces/EDR-Validation/EDR/data/sparc/SPARC_Lelli2016c.txt.txt
Output: /workspaces/EDR-Validation/EDR/data/sparc/SPARC_table1_parsed.csv
"""
import re
import pandas as pd
from collections import OrderedDict
FILE_IN = "/workspaces/EDR-Validation/EDR/data/sparc/SPARC_Lelli2016c.txt.txt"
OUT_CSV = "/workspaces/EDR-Validation/EDR/data/sparc/SPARC_table1_parsed.csv"

def parse_table1(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            # skip header lines until real data (we detect lines that start with a galaxy id)
            # Galaxy IDs are left-aligned names (e.g. "NGC3198", "F571-8", "CamB", etc.)
            # Data lines typically start with a name followed by numbers, so match that:
            if not ln.strip(): 
                continue
            # We consider data line if it begins with a word (letters/digits/-) followed by spaces then a number
            if re.match(r"^[A-Za-z0-9\-\_\.]+(\s+){1,}\d", ln):
                # collapse multiple spaces -> single space, then split
                parts = re.sub(r"\s+", " ", ln).strip().split(" ")
                # Based on the Table1 byte layout we expect columns like:
                # ID, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L[3.6], Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
                # We'll be conservative and map by positions (some files may lack trailing fields)
                # Ensure list long enough:
                # Convert numeric fields when possible
                try:
                    ID = parts[0]
                    # try to map positions safely
                    D = float(parts[2]) if len(parts) > 2 else None
                    L36 = float(parts[8]) if len(parts) > 8 else None
                    Rdisk = float(parts[12]) if len(parts) > 12 else None
                    SBdisk = float(parts[13]) if len(parts) > 13 else None
                    MHI = float(parts[14]) if len(parts) > 14 else None
                except Exception:
                    # fallback parse robustly by trying to extract floats from the line after the name
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", " ".join(parts[1:]))
                    nums = [float(x) for x in nums]
                    D = nums[0] if len(nums) > 0 else None
                    L36 = nums[7] if len(nums) > 7 else None
                    Rdisk = nums[11] if len(nums) > 11 else None
                    SBdisk = nums[12] if len(nums) > 12 else None
                    MHI = nums[13] if len(nums) > 13 else None
                rows.append(OrderedDict([
                    ("Galaxy", ID),
                    ("D_Mpc", D),
                    ("L3.6_1e9Lsun", L36),
                    ("Rdisk_kpc", Rdisk),
                    ("SBdisk_Lsol_pc2", SBdisk),
                    ("MHI_1e9Msun", MHI),
                    ("raw_parts_len", len(parts))
                ]))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="Galaxy").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    return df

if __name__ == "__main__":
    df = parse_table1(FILE_IN)
    print(f"Parsed {len(df)} galaxies -> {OUT_CSV}")

"""
Parse Table2 (Mass Models) desde el TXT descargado.
Salida: CSV con columnas:
ID, D, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul
"""
import re
import pandas as pd
from pathlib import Path

def parse_table2(txt_path):
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    # buscar la sección datafile2 o "Table: Mass Models" y tomar las líneas siguientes que correspondan a datos
    start_idx = None
    for i, ln in enumerate(lines):
        if "Table:" in ln and "Mass Models" in ln:
            start_idx = i+1
            break
        if "Byte-by-byte Description of file: datafile2" in ln:
            start_idx = i+1
            break

    if start_idx is None:
        # si no se encuentra, parsear globalmente buscando líneas tipo "CamB 3.36 0.16 1.99 ..."
        start_idx = 0

    data_lines = []
    for ln in lines[start_idx:]:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", s):
            parts = re.split(r"\s+", s)
            data_lines.append(parts)

    if not data_lines:
        raise ValueError("No se detectaron líneas de datos tipo Table2 en el archivo.")

    maxcols = max(len(r) for r in data_lines)
    # Standard columns:
    cols = ["ID","D","R","Vobs","e_Vobs","Vgas","Vdisk","Vbul","SBdisk","SBbul"]
    if maxcols > len(cols):
        cols = cols + [f"extra_{i}" for i in range(maxcols - len(cols))]

    norm = [r + [""]*(len(cols)-len(r)) for r in data_lines]
    df = pd.DataFrame(norm, columns=cols)
    # Convert numeric columns where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

if __name__ == "__main__":
    import sys
    infile = sys.argv[1] if len(sys.argv)>1 else "EDR/data/sparc/SPARC_Lelli2016c.txt.txt"
    out_csv = Path(infile).with_name("SPARC_Table2_parsed.csv")
    df = parse_table2(infile)
    df.to_csv(out_csv, index=False)
    print("Saved Table2 parsed to:", out_csv)
    print(df.groupby("ID").size().head())

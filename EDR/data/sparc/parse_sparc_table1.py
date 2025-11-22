"""
Parse Table1-like text (Galaxy Sample) para extraer la tabla global por galaxia.
Salida: CSV con columnas (Galaxy, D, L_3p6, e_L_3p6, Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, ...)
Usar cuando tengas SPARC Table1 en TXT.
"""
import re
import pandas as pd
from pathlib import Path

def parse_table1(txt_path):
    txt = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    # Buscamos el bloque que contiene las filas tipo:
    # CamB 10   3.36  0.26  2 65.0  5.0   0.075   0.003  1.21 ...
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s: 
            continue
        # Detectar líneas que empiezan por ID y luego un número (tipo "CamB 10")
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+\d+\s+[-+]?\d", s):
            parts = re.split(r"\s+", s)
            rows.append(parts)

    if not rows:
        raise ValueError("No se encontraron filas tipo Table1 en el archivo")

    # El número de columnas puede variar; seleccionar esquema común de Table1:
    # Asumimos orden: ID, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L[3.6], Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
    # Si hay menos columnas, se adapta automáticamente
    maxcols = max(len(r) for r in rows)
    cols = ["ID","T","D","e_D","f_D","Inc","e_Inc","L[3.6]","e_L[3.6]","Reff","SBeff","Rdisk","SBdisk","MHI","RHI","Vflat","e_Vflat","Q","Ref"]
    if maxcols > len(cols):
        cols = cols + [f"extra_{i}" for i in range(maxcols - len(cols))]

    norm = [r + [""]*(len(cols)-len(r)) for r in rows]
    df = pd.DataFrame(norm, columns=cols)
    # Convertir numéricas
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

if __name__ == "__main__":
    import sys
    infile = sys.argv[1] if len(sys.argv)>1 else "EDR/data/sparc/SPARC_Lelli2016c.txt.txt"
    out = Path(infile).with_name("SPARC_Table1_parsed.csv")
    df = parse_table1(infile)
    df.to_csv(out, index=False)
    print("Saved Table1 parsed to:", out)
    print(df[["ID","D","L[3.6]","MHI"]].head())

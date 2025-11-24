import pandas as pd
import sys
import re

def parse_table2(file_path):
    """
    Parser para SPARC Table 2 (Lelli+2016)
    Extrae columnas:
       ID, D, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul
    """

    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)

            # Table 2 tiene ≥ 10 columnas
            if len(parts) >= 10:
                rows.append(parts[:10])

    if len(rows) == 0:
        raise ValueError("No se encontraron filas tipo Table2 en el archivo")

    df = pd.DataFrame(rows, columns=[
        "ID", "D", "R", "Vobs", "e_Vobs",
        "Vgas", "Vdisk", "Vbul",
        "SBdisk", "SBbul"
    ])

    # intentar convertir a numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python parse_sparc_table2.py <archivo>")
        sys.exit(1)

    infile = sys.argv[1]

    df = parse_table2(infile)

    outfile = "EDR/data/sparc/SPARC_Table2_parsed.csv"
    df.to_csv(outfile, index=False)

    print(f"Saved Table2 parsed to: {outfile}")
    print(df.head())

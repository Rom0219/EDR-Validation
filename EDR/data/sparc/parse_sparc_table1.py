import pandas as pd
import sys
import re

def parse_table1(file_path):
    """
    Parser para SPARC Table 1 (Lelli+2016)
    Extrae:
      - ID
      - Distancia D
      - L[3.6]
      - MHI
    """

    rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # ignora líneas vacías o comentarios
            if len(line) == 0 or line.startswith("#"):
                continue

            # detecta columnas separadas por varios espacios
            parts = re.split(r"\s+", line)

            # Table1 siempre tiene al menos 4 columnas útiles
            if len(parts) >= 4:
                rows.append(parts[:4])

    if len(rows) == 0:
        raise ValueError("No se encontraron filas tipo Table1 en el archivo")

    df = pd.DataFrame(rows, columns=["ID", "D", "L36", "MHI"])

    # convierte numéricas
    for col in ["D", "L36", "MHI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python parse_sparc_table1.py <archivo>")
        sys.exit(1)

    infile = sys.argv[1]

    df = parse_table1(infile)

    outfile = "EDR/data/sparc/SPARC_Table1_parsed.csv"
    df.to_csv(outfile, index=False)

    print(f"Saved Table1 parsed to: {outfile}")
    print(df.head())

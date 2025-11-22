"""
Lectura flexible de archivos SPARC (Table1/Table2 en formato txt tal cual se descargan).
Devuelve DataFrame pandas (o CSV) con detección automática de columnas.
"""
import pandas as pd
import re
from pathlib import Path

def load_free_table(file_path):
    """
    Lee un txt con columnas separadas por espacios múltiples o tabs.
    Omite bloques de cabeceras (líneas que empiezan por 'Title:' o 'Byte-by-byte' o 'Table:')
    Devuelve pandas.DataFrame con la primera fila de datos consistente como header si existe,
    o devuelve DataFrame con columnas genéricas.
    """
    p = Path(file_path)
    text_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Eliminar encabezados largos: buscamos la primera línea que parezca un dato (ID o nombre de galaxia)
    data_lines = []
    for ln in text_lines:
        s = ln.strip()
        if not s:
            continue
        # salto comentarios
        if s.startswith('#'):
            continue
        # si la línea empieza por una etiqueta como "CamB" o "NGC" o "F" seguida de números -> línea de datos
        if re.match(r"^[A-Za-z0-9\-\_\.]+\s+[-+]?\d", s):
            data_lines.append(s)
        else:
            # también incluir si la línea parece contener columnas separadas por espacios y la primera "palabra" no es texto largo explicativo
            # para Table1 (hay algunas líneas con preámbulo - ignoramos)
            continue

    if not data_lines:
        raise ValueError(f"No se detectaron líneas de datos en {file_path} - revisa el archivo.")

    # split y normalizar
    rows = [re.split(r"\s+", dl) for dl in data_lines]
    maxcols = max(len(r) for r in rows)
    norm = [r + [""]*(maxcols - len(r)) for r in rows]

    # construir dataframe con header generico
    headers = ["col{:02d}".format(i+1) for i in range(maxcols)]
    df = pd.DataFrame(norm, columns=headers)

    return df

if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "EDR/data/sparc/SPARC_Lelli2016c.txt.txt"
    df = load_free_table(fp)
    print("Preview:")
    print(df.head())
    out_csv = Path(fp).with_suffix(".parsed.csv")
    df.to_csv(out_csv, index=False)
    print("Saved parsed CSV to:", out_csv)

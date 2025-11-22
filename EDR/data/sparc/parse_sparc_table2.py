import pandas as pd
import re

INPUT_FILE = "/mnt/data/SPARC_Lelli2016c.txt.txt"
OUTPUT_FILE = "EDR/data/sparc/SPARC_Lelli2016_Table2.csv"

def parse_table2(file_path):
    rows = []
    pat = re.compile(r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            m = pat.match(line)
            if not m:
                continue

            galaxy, dist, inc, m_d, m_b, m_g = m.groups()
            rows.append({
                "Galaxy": galaxy,
                "Distance_Mpc": float(dist),
                "Incl_deg": float(inc),
                "Ldisk": float(m_d),
                "Lbul": float(m_b),
                "Mgas": float(m_g)
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = parse_table2(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SAVED: {OUTPUT_FILE}")
    print(df.head())

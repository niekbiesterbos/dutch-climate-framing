"""
Summarises motions per CAP topic category.

Usage: python3 summary.py
"""
import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

INPUT_FILE        = "data/unformatted_data/historical_motions_with_urls.csv"
OUTPUT_DIR        = "data"
OUTPUT_FILE_FULL  = os.path.join(OUTPUT_DIR, "motions_per_topic.csv")
OUTPUT_FILE_SUMMARY = os.path.join(OUTPUT_DIR, "motions_summary.csv")

CAP_TO_TOPIC = {
    1: "Economie & Financiën",
    15: "Economie & Financiën",
    18: "Economie & Financiën",
    5: "Arbeid & Sociale Zekerheid",
    13: "Arbeid & Sociale Zekerheid",
    3: "Gezondheid & Zorg",
    6: "Onderwijs, Wetenschap & Cultuur",
    17: "Onderwijs, Wetenschap & Cultuur",
    23: "Onderwijs, Wetenschap & Cultuur",
    4: "Klimaat, Landbouw & Energie",
    7: "Klimaat, Landbouw & Energie",
    8: "Klimaat, Landbouw & Energie",
    21: "Klimaat, Landbouw & Energie",
    10: "Wonen & Infrastructuur",
    14: "Wonen & Infrastructuur",
    12: "Veiligheid, Justitie & Defensie",
    16: "Veiligheid, Justitie & Defensie",
    2: "Rechten & Democratie",
    20: "Rechten & Democratie",
    9: "Migratie & Integratie",
    19: "Buitenlands & Europees Beleid",
    22: "Buitenlands & Europees Beleid",
}


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} niet gevonden.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    if "majortopic" not in df.columns:
        raise ValueError("Kolom 'majortopic' ontbreekt in de input CSV.")

    df = df[df["majortopic"].notna()].copy()
    df["majortopic"] = df["majortopic"].astype(int)
    df["category"] = df["majortopic"].map(CAP_TO_TOPIC).fillna("Onbekend")
    df_out = df.drop(columns=["majortopic"]).copy()

    df_out.to_csv(OUTPUT_FILE_FULL, index=False, encoding="utf-8")
    print(f"Alle moties met categorie opgeslagen in {OUTPUT_FILE_FULL}")

    summary = (
        df_out.groupby("category")
        .size()
        .reset_index(name="aantal_moties")
        .sort_values("aantal_moties", ascending=False)
    )
    summary.to_csv(OUTPUT_FILE_SUMMARY, index=False, encoding="utf-8")
    print(f"Samenvatting opgeslagen in {OUTPUT_FILE_SUMMARY}")
    print(summary)


if __name__ == "__main__":
    main()

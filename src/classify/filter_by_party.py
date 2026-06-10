import pandas as pd

# ----------------------------------------
# Configuration
# ----------------------------------------

INPUT_FILE = "data/unformatted_data/all_motions_2008_2025.csv"
OUTPUT_FILE_TARGETS = "data/formatted_data/all_motions_poi_without_text.csv"
OUTPUT_FILE_NON_TARGETS

# Target parties (lowercase, no spaces)
TARGET_PARTIES = {"vvd", "cda", "d66", "pvda", "groenlinks-pvda"}


# ----------------------------------------
# Main logic
# ----------------------------------------

def normalize_party_name(name: str) -> str:
    """Normalize a single party name to lowercase and no spaces."""
    return str(name).strip().lower().replace(" ", "")


def party_matches_any(targets: set, cell_value: str) -> bool:
    """Return True if any of the semicolon-separated parties in cell_value match."""
    if not cell_value or pd.isna(cell_value):
        return False
    parties = [normalize_party_name(x) for x in str(cell_value).split(";")]
    return any(p in targets for p in parties)


def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Inputbestand niet gevonden: {INPUT_FILE}")
        return

    # Controle op kolomnaam
    if "Fracties" not in df.columns:
        print("Kolom 'Fracties' ontbreekt in de CSV.")
        return

    # Filter op ten minste één van de doelpartijen
    mask = df["Fracties"].apply(lambda x: party_matches_any(TARGET_PARTIES, x))
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        print("Geen moties gevonden voor de opgegeven partijen.")
        return

    # Sorteer chronologisch indien aanwezig
    if "Datum" in df_filtered.columns:
        df_filtered["Datum"] = pd.to_datetime(
            df_filtered["Datum"], errors="coerce")
        df_filtered.sort_values("Datum", inplace=True)

    # Opslaan
    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"{len(df_filtered)} moties opgeslagen in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

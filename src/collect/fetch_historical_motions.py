import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import urllib.parse
import csv
import os

# ----------------------------------------
# Config
# ----------------------------------------

INPUT_FILE = "historical_annotated_motions.xlsx"
OUTPUT_FILE = "historical_motions_with_urls.csv"

BASE_URL = "https://zoek.officielebekendmakingen.nl/resultaten"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HistoricalMotionScraper/1.0; +https://example.com)"
}


# ----------------------------------------
# Helper functions
# ----------------------------------------

def build_query_url(lawnr: str) -> str:
    """Build the full search URL for the historical archive (search only in title/description)."""
    # Limit search to 'motie' in title or description, not in the full text
    query = (
        f'(c.product-area=="sgd")'
        f'and((dt.type=="Kamerstuk")and(w.dossiernummer=="{lawnr}"))'
        f'and((dt.title="motie" or dt.description="motie"))'
    )
    params = {
        "q": query,
        "zv": "motie",
        "col": "Kamerstuk",
        "hist": "1",
        "svel": "Publicatiedatum",
        "svol": "Aflopend",
        "pg": "1"
    }
    return BASE_URL + "?" + urllib.parse.urlencode(params, safe="(),':=")


def fetch_first_pdf_url(lawnr: str) -> str | None:
    """Return the first valid PDF URL for a motion within this dossier, if any."""
    url = build_query_url(lawnr)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️ Fout bij {lawnr}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    container = soup.find("div", id="Publicaties")
    if not container:
        return None

    # Iterate over all result items and find the first one where the subtitle contains 'motie'
    for item in container.find_all("li"):
        subtitle = item.select_one("p > a.result--subtitle")
        if not subtitle:
            continue
        if "motie" not in subtitle.get_text(strip=True).lower():
            continue

        pdf_link = item.select_one("a.icon--download[href$='.pdf']")
        if pdf_link:
            return pdf_link["href"]

    return None


def write_row_to_csv(row_dict: dict):
    """Append a single record to the CSV output file."""
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ----------------------------------------
# Main logic
# ----------------------------------------

def main():
    df = pd.read_excel(INPUT_FILE)
    df = df[df["typebill"].str.lower() == "gewoon wetsvoorstel"].copy()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scrapen"):
        lawnr = str(row["lawnr"]).strip()
        if not lawnr or lawnr.lower() == "nan":
            continue

        pdf_url = fetch_first_pdf_url(lawnr)
        print(f"🔍 Checking {lawnr} → pdf_url={pdf_url}")

        if not pdf_url:
            continue  # niets gevonden, skippen

        record = {
            "id": row["id"],
            "text": row["text"],
            "lawnr": lawnr,
            "datesubmission": row.get("datesubmission"),
            "year": row.get("year"),
            "majortopic": row.get("majortopic"),
            "subtopic": row.get("subtopic"),
            "typebill": row.get("typebill"),
            "pdf_url": pdf_url
        }

        write_row_to_csv(record)
        print(f"✅ Opgeslagen voor law {lawnr}")

        time.sleep(0.5)  # polite scraping delay

    print("\n🏁 Klaar met scrapen!")


if __name__ == "__main__":
    main()

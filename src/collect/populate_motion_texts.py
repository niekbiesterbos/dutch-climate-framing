"""
populate_text_in_csv_streamed_cleanup.py
------------------------------------
Sequentially downloads Tweede Kamer motion PDFs and extracts text directly to CSV.
- Uses the `pdf_url` column directly (no doc_id parsing).
- Writes each processed row immediately to disk (streaming, resumable).
- Automatically skips already processed rows.
- Deletes downloaded PDFs after successful text extraction to save disk space.
------------------------------------
"""

import os
import time
import requests
import pandas as pd
from pypdfium2 import PdfDocument
from src.normalize_texts import clean_for_roberta

# -----------------------------
# Configuration
# -----------------------------

INPUT_CSV = "data/unformatted_data/motions_per_topic.csv"
OUTPUT_CSV = "data/formatted_data/motions_per_topic_with_text.csv"
PDF_DIR = "pdf_cache"

SLEEP_BETWEEN_REQUESTS = 0.3  # polite crawling delay

os.makedirs(PDF_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------


def download_pdf_from_url(pdf_url: str) -> str | None:
    """Download a PDF from a direct URL."""
    if not pdf_url or not pdf_url.lower().endswith(".pdf"):
        return None
    filename = os.path.basename(pdf_url).split("?")[0]
    local_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    try:
        r = requests.get(pdf_url, timeout=60)
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
            with open(local_path, "wb") as f:
                f.write(r.content)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return local_path
        else:
            print(f"[WARN] Unexpected response for {pdf_url}: {r.status_code}")
            return None
    except requests.RequestException as e:
        print(f"[ERROR] Failed to download {pdf_url}: {e}")
        return None


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extract text from a PDF file."""
    try:
        pdf = PdfDocument(pdf_path)
        text = "\n".join(page.get_textpage().get_text_bounded()
                         for page in pdf)
        pdf.close()
        return text.strip() if text else None
    except Exception as e:
        print(f"[ERROR] Could not read {pdf_path}: {e}")
        return None

# -----------------------------
# I/O Helpers
# -----------------------------


def write_row_to_csv(row_dict: dict, output_file: str, header_written: bool):
    """Append one processed row to CSV (create header only once)."""
    mode = "a" if header_written else "w"
    df_single = pd.DataFrame([row_dict])
    with open(output_file, mode, encoding="utf-8", newline="") as f:
        df_single.to_csv(f, header=not header_written, index=False)
    return True

# -----------------------------
# Main routine
# -----------------------------


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Prepare processed tracker
    processed_urls = set()
    header_written = False

    # Load output if it exists and skip rows already processed
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        # Only consider rows that actually have a normalized_text
        processed_urls = set(
            df_existing.loc[df_existing["normalized_text"].notna(),
                            "pdf_url"].astype(str)
        )
        header_written = True
        print(
            f"Resuming from previous run: {len(processed_urls)} rows already processed.")

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        pdf_url = str(row.get("pdf_url")).strip()
        if pdf_url in processed_urls or not pdf_url or pdf_url.lower() == "nan":
            continue

        pdf_path = download_pdf_from_url(pdf_url)
        if not pdf_path:
            text = None
        else:
            text = extract_text_from_pdf(pdf_path)
            try:
                os.remove(pdf_path)
            except OSError:
                pass

        normalized = clean_for_roberta(text) if text else None

        row_out = row.to_dict()
        row_out["text_extracted"] = text
        # Add the motion title (text column) to the normalized text for context
        row_out["normalized_text"] = f"{row['text']} {normalized}" if normalized else None

        write_row_to_csv(row_out, OUTPUT_CSV, header_written)
        header_written = True

        print(f"[{i}/{total}] {pdf_url} → extracted={'OK' if text else 'EMPTY'}, normalized={'OK' if normalized else 'SKIP'}")

    print(f"\n✅ All rows processed. Output saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

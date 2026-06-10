"""
populate_text_in_batches.py
-------------------------------------------
Processes Tweede Kamer motions in batches.
- Downloads PDFs in parallel (batch_size at a time)
- Extracts text sequentially per batch
- Normalizes and writes results immediately
- Resumable and memory safe
-------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import requests
import pandas as pd
import csv
from pypdfium2 import PdfDocument
from src.normalize_texts import clean_for_roberta


INPUT_CSV = "data/unformatted_data/all_motions_2008_2025.csv"
OUTPUT_CSV = "data/formatted_data/all_motions_2008_2025_with_text_and_normalized.csv"
PDF_DIR = "pdf_cache"

MAX_WORKERS = 8
BATCH_SIZE = 40
SLEEP_BETWEEN_REQUESTS = 0.05

os.makedirs(PDF_DIR, exist_ok=True)


def extract_document_id(url: str):
    """Extract the document ID from a Tweedekamer URL."""
    if not isinstance(url, str):
        return None
    parts = url.split("did=")
    return parts[1].split("&")[0] if len(parts) > 1 else None


def download_pdf(doc_id: str):
    """Download a PDF and return its local path or None."""
    if not doc_id:
        return None
    url = f"https://www.tweedekamer.nl/downloads/document?id={doc_id}"
    path = os.path.join(PDF_DIR, f"{doc_id}.pdf")

    if os.path.exists(path):
        return path

    try:
        r = requests.get(url, timeout=40)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        print(f"[WARN] {doc_id} returned {r.status_code}")
    except Exception as e:
        print(f"[ERROR] download {doc_id}: {e}")
    return None


def extract_text(pdf_path: str):
    """Extract text safely from a PDF file."""
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        pdf = PdfDocument(pdf_path)
        text = "\n".join(page.get_textpage().get_text_bounded()
                         for page in pdf)
        pdf.close()
        return text.strip() or None
    except Exception:
        return None


def process_batch(batch_rows, header_written):
    """Process one batch of motions: download, extract, normalize, write."""
    results = []
    downloaded_paths = []

    # Parallel download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_pdf, extract_document_id(
            r["url"])): r for r in batch_rows}
        for future in as_completed(futures):
            row = futures[future]
            path = future.result()
            row["_pdf_path"] = path  # temporary, not written to CSV
            results.append(row)
            if path:
                downloaded_paths.append(path)

    # Sequential text extraction + normalization
    for row in results:
        path = row.get("_pdf_path")
        text = extract_text(path) if path else None
        case_subject = str(row.get("case_subject") or "").strip()
        norm = clean_for_roberta(text) if text else None

        row["text"] = text
        row["normalized_text"] = f"{case_subject} {norm}".strip(
        ) if norm else None

        # Ensure temp key isn't written to CSV
        if "_pdf_path" in row:
            del row["_pdf_path"]

        # Safe CSV write
        df = pd.DataFrame([row])
        with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
            df.to_csv(
                f,
                index=False,
                header=not header_written,
                quoting=csv.QUOTE_ALL,
            )
        header_written = True

    # Remove all PDFs at the end of this batch
    for path in downloaded_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return header_written


def main():
    df = pd.read_csv(INPUT_CSV)
    processed = set()
    header_written = False

    # Resume support
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        processed = set(existing["id"].astype(str))
        header_written = True
        print(
            f"Resuming from previous run: {len(processed)} already processed.")

    df = df[~df["id"].astype(str).isin(processed)]
    total = len(df)
    print(f"Processing {total} remaining motions...")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end].to_dict(orient="records")
        print(f"\n--- Batch {start+1} to {end} ---")
        header_written = process_batch(batch, header_written)
        print(f"Completed batch {start+1}-{end}")

    print(f"\nAll {total} motions processed. Output saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

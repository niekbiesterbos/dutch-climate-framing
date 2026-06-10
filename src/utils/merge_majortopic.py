"""
Merges the 'majortopic' column from historical_motions_with_urls.csv
into historical_motions_annotated.csv on 'id'.

Usage: python3 format.py
"""

import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

SOURCE_CSV = "data/unformatted_data/historical_motions_with_urls.csv"
TARGET_CSV = "data/formatted_data/historical_motions_annotated.csv"
OUTPUT_CSV = "data/results/historical_motions_annotated.csv"


def main():
    source_df = pd.read_csv(SOURCE_CSV)
    target_df = pd.read_csv(TARGET_CSV)

    # Ensure consistent ID typing
    source_df["id"] = source_df["id"].astype(str)
    target_df["id"] = target_df["id"].astype(str)

    if "majortopic" not in source_df.columns:
        raise ValueError("Source CSV does not contain 'majortopic' column")

    # Reduce source to necessary columns only
    source_df = source_df[["id", "majortopic"]]

    # Drop existing majortopic to avoid duplicate columns
    if "majortopic" in target_df.columns:
        target_df = target_df.drop(columns=["majortopic"])

    # Merge
    merged_df = target_df.merge(
        source_df,
        on="id",
        how="left",
        validate="one_to_one",
    )

    # Write back (in-place update as requested)
    merged_df.to_csv(OUTPUT_CSV, index=False)

    print(
        f"Merged 'majortopic' into annotated file. "
        f"Rows: {len(merged_df)}, "
        f"Majortopic filled: {merged_df['majortopic'].notna().sum()}"
    )


if __name__ == "__main__":
    main()

"""
Macro-Frame Gold Standard Annotation — Motions
===============================================
Interactive terminal tool for manually annotating parliamentary motions
on 7 macro-frames using a Likert 1–5 scale.

Reads:  results/motions/gold_sample.csv
Writes: results/motions/gold_macro.csv  (updated after each annotation)

Usage:
  python3 src/annotate/gold_motions_macro.py           # full run
  python3 src/annotate/gold_motions_macro.py --debug   # first 5 only, no file written
  python3 src/annotate/gold_motions_macro.py --max 20  # annotate at most 20 items

Resumable: already-annotated IDs are skipped automatically.
"""

import os
import sys
import csv
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(Path(__file__).parent))
from frames import FRAMES, FRAME_KEYS, LIKERT_SCALE

SAMPLE_PATH = Path("results/motions/gold_sample.csv")
OUT_PATH    = Path("results/motions/gold_macro.csv")
TEXT_COL    = "normalized_text"
MAX_CHARS   = 800
DEBUG       = "--debug" in sys.argv
MAX_N       = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None


def load_existing(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["id"].astype(str).tolist())


def append_row(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prompt_score(frame_key):
    frame = FRAMES[frame_key]
    print(f"\n  {frame['label']}")
    print(f"  {frame['decision_rule']}")
    print()
    for score, anchor in frame["anchors"].items():
        print(f"    {score} — {anchor}")
    print()
    while True:
        val = input(f"  Score [1–5]: ").strip()
        if val.isdigit() and 1 <= int(val) <= 5:
            return int(val)
        print("  Enter a number 1–5.")


def main():
    df = pd.read_csv(SAMPLE_PATH)
    already_done = load_existing(OUT_PATH)

    todo = df[~df["id"].astype(str).isin(already_done)].reset_index(drop=True)
    if MAX_N:
        todo = todo.head(MAX_N)

    print(f"\nMacro-frame annotation — motions")
    print(f"Scale: {LIKERT_SCALE}")
    print(f"Total in sample: {len(df)}  |  Already done: {len(already_done)}  |  Remaining: {len(todo)}")
    print(f"\nPress q at the continue prompt to quit and save progress.\n")

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        text = str(row.get(TEXT_COL, ""))
        motion_id = str(row["id"])

        print(f"\n{'='*70}")
        print(f"[{i}/{len(todo)}]  id={motion_id}  year={row.get('year', '?')}")
        print(f"\n{text[:MAX_CHARS]}{'...' if len(text) > MAX_CHARS else ''}\n")
        print(f"Scale: {LIKERT_SCALE}")

        scores = {}
        for key in FRAME_KEYS:
            try:
                scores[key] = prompt_score(key)
            except (EOFError, KeyboardInterrupt):
                print("\nAnnotation interrupted. Progress saved.")
                return

        if not DEBUG:
            append_row(OUT_PATH, {
                "id": motion_id,
                "year": row.get("year", ""),
                TEXT_COL: text,
                **scores,
            })

        cont = input("\n  Continue? (Enter=yes / q=quit): ").strip().lower()
        if cont == "q":
            print(f"\nPaused after {i} items. Run again to continue.")
            return

    print(f"\nDone. {len(todo)} motions annotated.")
    if not DEBUG:
        print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()

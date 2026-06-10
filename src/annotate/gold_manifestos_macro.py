"""
Macro-Frame Gold Standard Annotation — Manifestos
==================================================
Interactive terminal tool for manually annotating manifesto climate
phrases on 7 macro-frames using a Likert 1–5 scale.

Step 1: Draw a stratified sample (party × year) from relevant_phrases.csv
Step 2: Annotate interactively in the terminal

Reads:  data/manifestos/relevant_phrases.csv
Writes: results/manifestos/gold_sample.csv   (sample, drawn once)
        results/manifestos/gold_macro.csv    (updated after each annotation)

Usage:
  python3 src/annotate/gold_manifestos_macro.py            # full run
  python3 src/annotate/gold_manifestos_macro.py --debug    # first 5, no file written
  python3 src/annotate/gold_manifestos_macro.py --max 20   # at most 20 items

Resumable: already-annotated rows are skipped.
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

INPUT_CSV   = Path("data/manifestos/relevant_phrases.csv")
SAMPLE_PATH = Path("results/manifestos/gold_sample.csv")
OUT_PATH    = Path("results/manifestos/gold_macro.csv")
TEXT_COL    = "text"
MAX_CHARS   = 800
SAMPLE_FRAC = 0.15
RANDOM_SEED = 42
DEBUG       = "--debug" in sys.argv
MAX_N       = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None


def draw_sample():
    df = pd.read_csv(INPUT_CSV)
    sample = (
        df.groupby(["party", "year"], group_keys=False)
        .apply(lambda x: x.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Sample drawn: {len(sample)} phrases → {SAMPLE_PATH}")
    return sample


def load_existing(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["sample_idx"].astype(str).tolist())


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
        val = input("  Score [1–5]: ").strip()
        if val.isdigit() and 1 <= int(val) <= 5:
            return int(val)
        print("  Enter a number 1–5.")


def main():
    if not SAMPLE_PATH.exists():
        sample = draw_sample()
    else:
        sample = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded existing sample: {len(sample)} phrases")

    already_done = load_existing(OUT_PATH)
    todo = sample[~sample.index.astype(str).isin(already_done)].reset_index(drop=True)
    if MAX_N:
        todo = todo.head(MAX_N)

    print(f"\nMacro-frame annotation — manifestos")
    print(f"Scale: {LIKERT_SCALE}")
    print(f"Total: {len(sample)}  |  Done: {len(already_done)}  |  Remaining: {len(todo)}")
    print(f"\nPress q to quit and save progress.\n")

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        text = str(row.get(TEXT_COL, ""))
        idx = str(i - 1)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(todo)}]  party={row.get('party','?')}  year={row.get('year','?')}")
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
                "sample_idx": idx,
                "party": row.get("party", ""),
                "year": row.get("year", ""),
                TEXT_COL: text,
                **scores,
            })

        cont = input("\n  Continue? (Enter=yes / q=quit): ").strip().lower()
        if cont == "q":
            print(f"\nPaused after {i} items. Run again to continue.")
            return

    print(f"\nDone. {len(todo)} phrases annotated.")
    if not DEBUG:
        print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()

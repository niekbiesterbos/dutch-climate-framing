"""
Micro Gold Standard Annotation — Motions
=========================================
Interactive terminal tool for manually annotating LOME clauses with
actor and patient spans + stakeholder categories.

Step 1: Draw a stratified sample (party × frame type) from actor_patient_classified.csv
Step 2: Annotate interactively in the terminal

Reads:  results/motions/micro_scores/actor_patient_classified.csv
Writes: results/motions/gold_micro_sample.csv   (sample, drawn once)
        results/motions/gold_micro.csv          (updated after each annotation)

Usage:
  python3 src/annotate/gold_motions_micro.py            # full run
  python3 src/annotate/gold_motions_micro.py --debug    # first 5, no file written
  python3 src/annotate/gold_motions_micro.py --max 20   # at most 20 items

Resumable: already-annotated sample_idx rows are skipped.
"""

import os
import sys
import csv
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

INPUT_CSV   = Path("results/motions/micro_scores/actor_patient_classified.csv")
SAMPLE_PATH = Path("results/motions/gold_micro_sample.csv")
OUT_PATH    = Path("results/motions/gold_micro.csv")
SAMPLE_N    = 200
RANDOM_SEED = 42
DEBUG       = "--debug" in sys.argv
MAX_N       = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else None

TAXONOMY = [
    "Government",
    "Industry",
    "Agriculture",
    "Citizens",
    "Nature",
    "Science",
    "International",
    "Energy",
    "Other",
]

TAXONOMY_STR = "  " + "\n  ".join(f"{i+1}. {t}" for i, t in enumerate(TAXONOMY))

TARGET_PARTIES = ["GroenLinks", "PvdA", "GroenLinks-PvdA", "D66", "CDA",
                  "VVD", "PVV", "FvD", "BBB", "PvdD"]


def draw_sample(df):
    df["party"] = df["fractions"].str.split(";").str[0].str.strip()
    df["year"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.year
    df = df[df["filler"].notna() & (df["filler"].str.strip().str.len() > 10)].copy()
    df_target = df[df["party"].isin(TARGET_PARTIES)].copy()

    sample = (
        df_target.groupby(["party", "frame"], group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), max(1, SAMPLE_N // (len(TARGET_PARTIES) * df_target["frame"].nunique()))),
            random_state=RANDOM_SEED,
        ))
        .reset_index(drop=True)
        .head(SAMPLE_N)
        .reset_index(drop=True)
    )
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Sample drawn: {len(sample)} clauses → {SAMPLE_PATH}")
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


def prompt_span(label):
    print(f"\n  {label}")
    print("  (Paste exact Dutch text from the clause, or press Enter for none)")
    return input("  > ").strip() or None


def prompt_category(label):
    print(f"\n  {label}")
    print(TAXONOMY_STR)
    while True:
        val = input("  Category number (or Enter for none): ").strip()
        if val == "":
            return None
        if val.isdigit() and 1 <= int(val) <= len(TAXONOMY):
            return TAXONOMY[int(val) - 1]
        print(f"  Enter a number 1–{len(TAXONOMY)}, or press Enter to skip.")


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    if not SAMPLE_PATH.exists():
        df = pd.read_csv(INPUT_CSV)
        sample = draw_sample(df)
    else:
        sample = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded existing sample: {len(sample)} clauses")

    already_done = load_existing(OUT_PATH)
    todo = sample[~sample.index.astype(str).isin(already_done)].reset_index(drop=True)
    if MAX_N:
        todo = todo.head(MAX_N)

    print(f"\nMicro gold annotation — motions")
    print(f"Total: {len(sample)}  |  Done: {len(already_done)}  |  Remaining: {len(todo)}")
    print(f"\nFor each clause, identify the actor and patient (or leave blank).")
    print(f"Press q at the continue prompt to quit and save progress.\n")

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        idx = str(i - 1)
        filler = str(row.get("filler", ""))
        trigger = str(row.get("trigger", ""))
        frame = str(row.get("frame", ""))
        party = str(row.get("party", row.get("fractions", "?")))
        year = str(row.get("year", "?"))

        print(f"\n{'='*70}")
        print(f"[{i}/{len(todo)}]  party={party}  year={year}")
        print(f"  Frame: {frame}  |  Trigger: {trigger}")
        print(f"\n  Clause:\n  {filler}\n")

        pred_actor_spans = str(row.get("actor_spans", ""))
        pred_patient_spans = str(row.get("patient_spans", ""))
        if pred_actor_spans:
            print(f"  Model actor spans:   {pred_actor_spans}")
        if pred_patient_spans:
            print(f"  Model patient spans: {pred_patient_spans}")

        print(f"\n  Stakeholder categories:\n{TAXONOMY_STR}")

        try:
            actor_span = prompt_span("Actor span")
            actor_cat  = prompt_category("Actor category") if actor_span else None
            patient_span = prompt_span("Patient span")
            patient_cat  = prompt_category("Patient category") if patient_span else None
        except (EOFError, KeyboardInterrupt):
            print("\nAnnotation interrupted. Progress saved.")
            return

        if not DEBUG:
            append_row(OUT_PATH, {
                "sample_idx":        idx,
                "motie_id":          row.get("motie_id", ""),
                "frame":             frame,
                "trigger":           trigger,
                "filler":            filler,
                "party":             party,
                "year":              year,
                "gold_actor_span":   actor_span,
                "gold_actor_cat":    actor_cat,
                "gold_patient_span": patient_span,
                "gold_patient_cat":  patient_cat,
                "pred_actor_spans":  pred_actor_spans,
                "pred_actor_cat":    row.get("actor_category", ""),
                "pred_patient_spans": pred_patient_spans,
                "pred_patient_cat":  row.get("patient_category", ""),
            })

        cont = input("\n  Continue? (Enter=yes / q=quit): ").strip().lower()
        if cont == "q":
            print(f"\nPaused after {i} items. Run again to continue.")
            return

    print(f"\nDone. {len(todo)} clauses annotated.")
    if not DEBUG:
        print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()

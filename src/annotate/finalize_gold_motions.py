"""
Finalize Gold Standard — Motions Macro
========================================
Reads the interactive annotation output and writes the final gold CSV,
keeping only the required columns.

Input:  results/motions/gold_macro.csv   (from gold_motions_macro.py)
Output: results/motions/gold_macro.csv   (in-place cleanup — same file)

Run once after annotation is complete to drop any stray columns.
"""

import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

GOLD_PATH = Path("results/motions/gold_macro.csv")

FRAMES = ["economic", "moral", "scientific", "security",
          "health_environment", "crisis_urgency", "weaponization"]

df = pd.read_csv(GOLD_PATH)
print(f"Loaded {len(df)} annotations.")

keep_cols = ["id", "normalized_text", "year"] + FRAMES
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols]

df.to_csv(GOLD_PATH, index=False)
print(f"Gold standard saved: {GOLD_PATH}  ({len(df)} motions)")

print("\nFrame score means:")
for k in FRAMES:
    if k in df.columns:
        valid = df[k][df[k] >= 1]
        print(f"  {k:<25} mean={valid.mean():.2f}  N={len(valid)}")

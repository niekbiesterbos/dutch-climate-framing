"""
Gold Standard Sample for Macro-Frame Annotation
==========================================================
Draws a stratified sample of ~5% of climate motions for gold standard
annotation. Stratification is by year to ensure temporal representativeness.

Output: results/motions/gold_sample.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

DATA_PATH   = 'results/motions/climate_motions.csv'
SAMPLE_PATH = Path('results/motions/gold_sample.csv')

SAMPLE_FRAC = 0.05
RANDOM_SEED = 42

SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Parse year from date column
df['year'] = pd.to_datetime(df['date'], utc=True).dt.year

print(f"Total climate motions: {len(df)}")
print(f"\nDistribution per year:")
print(df['year'].value_counts().sort_index().to_string())

# Stratified sample per year
sample = (
    df.groupby('year', group_keys=False)
    .apply(lambda x: x.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED), include_groups=False)
    .reset_index(drop=True)
)

# year kolom opnieuw aanmaken na groupby
sample['year'] = pd.to_datetime(sample['date'], utc=True).dt.year

print(f"\nSample size: {len(sample)} motions ({len(sample)/len(df)*100:.1f}%)")
print(f"\nSample distribution per year:")
print(sample['year'].value_counts().sort_index().to_string())

sample.to_csv(SAMPLE_PATH, index=False)
print(f"\nSaved to: {SAMPLE_PATH}")
print("Run src/annotate/gold_motions_macro.py to annotate the sample.")
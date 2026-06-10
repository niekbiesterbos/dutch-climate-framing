"""
Contrastive Substitution Table
========================================
Generates a Table 5-style table (Hoeken et al. 2023) showing the most
distinctive substitutions per party per target word, computed as the
highest relative frequency difference between focal party and comparison party.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

SUB_DIR = Path('results/analysis/substitutions')
OUT_DIR = Path('results/analysis')

TARGET_WORDS = [
    'klimaat', 'energie', 'uitstoot', 'natuur', 'transitie',
    'biodiversiteit', 'kernenergie', 'subsidie', 'industrie',
    'landbouw', 'stikstof', 'duurzaam',
]

# Party pairs to compare — ideologically most opposed
PAIRS = [
    ('PVV',      'PvdD'),
    ('FvD',      'GroenLinks'),
    ('BBB',      'PvdD'),
    ('VVD',      'GroenLinks'),
]

TOP_N = 5  # distinctive substitutions per cell


def load_counter(target: str, party: str) -> dict:
    path = SUB_DIR / f'{target}_{party}.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df['token'], df['count']))


def get_distinctive(c_focal: dict, c_other: dict, top_n: int = 5) -> list:
    """
    Return top_n tokens with highest relative frequency in focal vs other.
    Relative frequency difference: freq_focal - freq_other.
    """
    total_focal = sum(c_focal.values())
    total_other = sum(c_other.values())
    if total_focal == 0 or total_other == 0:
        return []

    vocab = set(c_focal.keys()) | set(c_other.keys())
    diffs = {
        t: (c_focal.get(t, 0) / total_focal) -
           (c_other.get(t, 0) / total_other)
        for t in vocab
    }
    top = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    return [t for t, d in top if d > 0][:top_n]


def main():
    for party_a, party_b in PAIRS:
        print(f'\n{"="*60}')
        print(f'  {party_a} vs {party_b}')
        print(f'{"="*60}')
        print(f'{"Term":<18} {party_a:<30} {party_b}')
        print('-' * 70)

        rows = []
        for target in TARGET_WORDS:
            c_a = load_counter(target, party_a)
            c_b = load_counter(target, party_b)

            if not c_a or not c_b:
                continue

            words_a = get_distinctive(c_a, c_b, TOP_N)
            words_b = get_distinctive(c_b, c_a, TOP_N)

            cell_a = ', '.join(words_a) if words_a else '—'
            cell_b = ', '.join(words_b) if words_b else '—'

            print(f'{target:<18} {cell_a:<30} {cell_b}')
            rows.append({
                'target':  target,
                'party_a': party_a,
                'party_b': party_b,
                'words_a': cell_a,
                'words_b': cell_b,
            })

        # Save per pair
        out = pd.DataFrame(rows)
        fname = f'contrastive_table_{party_a}_vs_{party_b}.csv'
        out.to_csv(OUT_DIR / fname, index=False)
        print(f'\nSaved: {fname}')


if __name__ == '__main__':
    main()
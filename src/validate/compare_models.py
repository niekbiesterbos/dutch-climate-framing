"""
EXP_9 model comparison against EXP_15 gold standard.

Usage: python3 compare.py
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

GOLD_PATH = 'results/motions/gold_macro.csv'
MODELS = {
    'gemma-3-12b-it': 'results/motions/macro_scores/gemma-3-12b-it.csv',
    'gemma-3-27b-it': 'results/motions/macro_scores/gemma-3-27b-it.csv',
    'qwen2.5-14b':    'results/motions/macro_scores/qwen2.5-14b.csv',
    'qwen2.5-32b':    'results/motions/macro_scores/qwen2.5-32b.csv',
}
FRAMES = ['economic', 'moral', 'scientific', 'security',
          'health_environment', 'crisis_urgency', 'weaponization']

gold = pd.read_csv(GOLD_PATH)

for model_name, path in MODELS.items():
    if not Path(path).exists():
        print(f'Skipping {model_name}: file not found')
        continue
    df = pd.read_csv(path)
    # EXP_9 stores scores as frame_<name>; gold uses plain <name>
    df = df.rename(columns={f'frame_{k}': k for k in FRAMES if f'frame_{k}' in df.columns})

    shared_cols = [k for k in FRAMES if k in df.columns]
    merged = gold[['id'] + FRAMES].merge(df[['id'] + shared_cols], on='id', suffixes=('_gold', '_pred'))
    if merged.empty:
        print(f'No overlapping IDs for {model_name}')
        continue

    print(f'\n{"="*60}')
    print(f'Model: {model_name}  |  Matched motions: {len(merged)}')
    print(f'{"="*60}')
    print(f'  {"Frame":<25} {"Gold":>6} {"Model":>6} {"r":>6} {"MAE":>6}')
    print(f'  {"-"*53}')

    for k in FRAMES:
        gv = merged[f'{k}_gold']
        mv = merged[f'{k}_pred']
        valid = (gv != -1) & (mv != -1)
        gv, mv = gv[valid], mv[valid]
        r   = float(np.corrcoef(gv, mv)[0, 1]) if valid.sum() > 2 else float('nan')
        mae = float((gv - mv).abs().mean())
        print(f'  {k:<25} {gv.mean():>6.2f} {mv.mean():>6.2f} {r:>6.3f} {mae:>6.2f}')

    all_gold  = pd.concat([merged[f'{k}_gold'] for k in FRAMES])
    all_model = pd.concat([merged[f'{k}_pred'] for k in FRAMES])
    valid_all = (all_gold != -1) & (all_model != -1)
    print(f'\n  Overall MAE: {(all_gold[valid_all] - all_model[valid_all]).abs().mean():.3f}')
    print(f'  Overall r:   {np.corrcoef(all_gold[valid_all], all_model[valid_all])[0, 1]:.3f}')

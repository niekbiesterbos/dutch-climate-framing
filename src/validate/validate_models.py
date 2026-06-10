"""
Comprehensive macro-frame validation
=============================================
Two validation layers, all four models, all three text types.

(1) Model-selection metrics (per-text, min-max normalised):
    Pearson, Spearman, Cosine, Dominant-frame accuracy.

(2) Per-frame absolute-agreement validation (Qwen 2.5 32B):
    Pearson, Spearman, mean bias, MAE, ICC(2,1).

Merge keys:
    motions    — id      (UUID)
    speeches   — doc_id  (ParlaMint id)
    manifestos — text    (phrase text, whitespace-normalised)
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cos_dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

FRAMES = ['economic', 'moral', 'scientific', 'security',
          'health_environment', 'crisis_urgency', 'weaponization']

# gold_key  : column in the gold file used as merge key
# pred_key  : preferred column in the prediction file
# pred_alts : fallbacks if pred_key absent
CONFIGS = {
    'motions': dict(
        gold='results/motions/gold_macro.csv',
        pred_glob='results/motions/macro_scores/*.csv',
        gold_key='id', pred_key='id',
        pred_alts=()),
    'manifestos': dict(
        gold='results/manifestos/gold_macro.csv',
        pred_glob='results/manifestos/macro_scores/*.csv',
        gold_key='text', pred_key='text',
        pred_alts=('normalized_text',)),
    'speeches': dict(
        gold='results/speeches/gold_macro.csv',
        pred_glob='results/speeches/macro_scores/*.csv',
        gold_key='doc_id', pred_key='doc_id',
        pred_alts=('id',)),
}
PRIMARY = 'qwen2.5-32b'
OUT = 'results/validation'
os.makedirs(OUT, exist_ok=True)


def fcol(df, frame):
    for c in (frame, f'frame_{frame}', f'{frame}_score', f'pred_{frame}'):
        if c in df.columns:
            return c
    raise KeyError(f'{frame} not found')


def clean_key(s):
    return s.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)


def model_name(path):
    return os.path.basename(path).split('_scores_')[-1].replace('.csv', '')


def load_scored(path, prefix, key, alts=()):
    df = pd.read_csv(path)
    try:
        frame_cols = [fcol(df, f) for f in FRAMES]
    except KeyError:
        return None, None
    # find merge key
    found_key = key if key in df.columns else next(
        (a for a in alts if a in df.columns), None)
    if found_key is None:
        return None, None
    sub = df[[found_key] + frame_cols].copy()
    sub.columns = ['id'] + [f'{prefix}_{f}' for f in FRAMES]
    sub['id'] = clean_key(sub['id'])
    return sub, found_key


def minmax(v):
    rng = v.max() - v.min()
    return None if rng == 0 else (v - v.min()) / rng


def icc_2_1(g, p):
    X = np.column_stack([g, p]).astype(float)
    n, k = X.shape
    grand = X.mean()
    SST = ((X - grand) ** 2).sum()
    SSR = k * ((X.mean(axis=1) - grand) ** 2).sum()
    SSC = n * ((X.mean(axis=0) - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    return (MSR - MSE) / denom if denom != 0 else np.nan


def per_text_metrics(d):
    pears, spear, coss, dom = [], [], [], []
    for _, row in d.iterrows():
        g = np.array([row[f'g_{f}'] for f in FRAMES], float)
        p = np.array([row[f'p_{f}'] for f in FRAMES], float)
        dom.append(int(np.argmax(g) == np.argmax(p)))
        gn, pn = minmax(g), minmax(p)
        if gn is None or pn is None:
            continue
        coss.append(1 - cos_dist(gn, pn))
        if gn.std() > 0 and pn.std() > 0:
            pears.append(pearsonr(gn, pn)[0])
            spear.append(spearmanr(gn, pn)[0])
    def s(lst): return np.mean(lst) if lst else np.nan
    return s(pears), s(spear), s(coss), s(dom)


def main():
    for tt, cfg in CONFIGS.items():
        print(f'\n{"="*60}')
        if not os.path.exists(cfg['gold']):
            print(f'[skip] {tt}: gold not found at {cfg["gold"]}')
            continue

        gold, gkey = load_scored(
            cfg['gold'], 'g', cfg['gold_key'])
        if gold is None:
            print(f'[skip] {tt}: gold could not be loaded')
            continue
        print(f'{tt}: gold loaded  (key={gkey!r}, n={len(gold)})')

        pred_files = sorted(
            f for f in glob.glob(cfg['pred_glob'])
            if 'interim' not in os.path.basename(f))
        if not pred_files:
            print(f'[skip] {tt}: no pred files')
            continue

        preds, pred_keys = {}, {}
        for pf in pred_files:
            s, pk = load_scored(pf, 'p', cfg['pred_key'], cfg['pred_alts'])
            if s is None:
                print(f'  [skip] {os.path.basename(pf)}: '
                      f'no frame cols or key col')
                continue
            mn = model_name(pf)
            preds[mn] = s
            pred_keys[mn] = pk
        if not preds:
            print(f'[skip] {tt}: no usable pred files')
            continue

        prim = next((m for m in preds if PRIMARY in m), list(preds)[0])
        print(f'  primary model: {prim} (pred key={pred_keys[prim]!r})')

        gp = gold.merge(preds[prim], on='id', how='inner')
        if len(gp) == 0:
            print(f'[warn] {tt}: merge yielded 0 rows. '
                  f'Gold key={cfg["gold_key"]!r}, '
                  f'pred key={pred_keys[prim]!r}. '
                  f'Check that values match.')
            continue
        allones = (gp[[f'p_{f}' for f in FRAMES]] == 1).all(axis=1)
        eval_ids = set(gp.loc[~allones, 'id'])

        print(f'\n--- Model selection: {tt} '
              f'(merged={len(gp)}, eval n={len(eval_ids)}) ---')
        print(f'{"Model":<22}{"Pearson":>9}{"Spearman":>10}'
              f'{"Cosine":>9}{"DomAcc":>9}')
        sel_rows = []
        for m in sorted(preds):
            d = gold.merge(preds[m], on='id', how='inner')
            d = d[d['id'].isin(eval_ids)]
            pe, sp, co, da = per_text_metrics(d)
            sel_rows.append((m, pe, sp, co, da))
            print(f'{m:<22}{pe:>9.3f}{sp:>10.3f}{co:>9.3f}{da:>9.3f}')
        pd.DataFrame(sel_rows,
                     columns=['model','pearson','spearman','cosine','dom_acc']
                     ).to_csv(f'{OUT}/model_selection_{tt}.csv', index=False)

        # Per-frame validation (primary model only)
        d = gold.merge(preds[prim], on='id', how='inner')
        d = d[d['id'].isin(eval_ids)]
        print(f'\n--- Per-frame validation: {tt}, {prim} (n={len(d)}) ---')
        print(f'{"Frame":<20}{"Pearson":>9}{"Spearman":>10}'
              f'{"Bias":>8}{"MAE":>7}{"ICC(2,1)":>10}')
        frame_rows = []
        for f in FRAMES:
            gv = d[f'g_{f}'].to_numpy(float)
            pv = d[f'p_{f}'].to_numpy(float)
            ok = gv.std() > 0 and pv.std() > 0
            r   = pearsonr(gv, pv)[0] if ok else np.nan
            rho = spearmanr(gv, pv)[0] if ok else np.nan
            bias, mae, icc = (pv-gv).mean(), np.abs(pv-gv).mean(), icc_2_1(gv,pv)
            frame_rows.append((f, r, rho, bias, mae, icc))
            print(f'{f:<20}{r:>9.3f}{rho:>10.3f}{bias:>+8.2f}'
                  f'{mae:>7.2f}{icc:>10.3f}')
        pd.DataFrame(frame_rows,
                     columns=['frame','pearson','spearman','bias','mae','icc_2_1']
                     ).to_csv(f'{OUT}/per_frame_{tt}.csv', index=False)


if __name__ == '__main__':
    main()
"""
Lexical Party Similarity Analysis
===========================================
Computes cosine similarity between party TF-IDF vectors derived from the
combined climate discourse corpus (motions + manifestos + speeches).
Visualises as a similarity heatmap and hierarchical dendrogram to show
which parties share vocabulary and whether clustering aligns with
ideological blocs.

Input:
    results/analysis/motions_linguistic.csv
    results/analysis/manifestos_linguistic.csv
    results/analysis/speeches_linguistic.csv

Output:
    results/analysis/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import spacy
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

OUT_DIR   = Path('results/analysis')
INPUT_DIR = Path('results/analysis')
FIG_DIR   = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

nlp = spacy.load('nl_core_news_lg', disable=['ner', 'parser'])

TARGET = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'D66', 'CDA',
          'VVD', 'PVV', 'FvD', 'BBB', 'PvdD']

LEFT   = ['GroenLinks', 'PvdA', 'GroenLinks-PvdA', 'PvdD']
CENTER = ['D66', 'CDA']
RIGHT  = ['VVD', 'PVV', 'FvD', 'BBB']

BLOC_MAP = ({p: 'Left'   for p in LEFT}  |
            {p: 'Center' for p in CENTER} |
            {p: 'Right'  for p in RIGHT})

BLOC_COLORS = {
    'Left': '#2e8b57', 'Center': '#f4a261', 'Right': '#e63946'
}

DUTCH_STOPS = set(stopwords.words('dutch'))
EXTRA_STOPS = {
    'constaterende', 'overwegende', 'verzoekt', 'motie', 'kamer',
    'regering', 'minister', 'staatssecretaris', 'tweede', 'eerste',
    'lid', 'leden', 'dag', 'kst', 'vergaderjaar', 'voorgesteld',
    'vergadering', 'voorzitter', 'spreekt', 'tevens', 'beraadslaging',
    'gehoord', 'gelet', 'mening', 'besluit', 'gegeven', 'verklaring',
    'indiener', 'ondertekend', 'voorstellen', 'aangenomen', 'stemming',
    'debat', 'behandeling', 'agenda', 'vergaderstuk', 'kamerstuk',
    'groenlinks', 'pvda', 'pvv', 'cda', 'd66', 'vvd', 'fvd', 'bbb',
    'pvdd', 'partij', 'fractie', 'forum', 'democratie', 'volkspartij',
    'boerburgerbeweging', 'dieren', 'arbeid', 'vrijheid',
    'jaar', 'wel', 'ook', 'nog', 'dus', 'want', 'omdat', 'tenzij',
    'reeds', 'steeds', 'altijd', 'nooit', 'soms', 'vaak', 'zeker',
    'echter', 'daarbij', 'daarmee', 'daarvoor', 'daarna', 'daarin',
    'hierbij', 'hiermee', 'hiervoor', 'hierna', 'hierin', 'hierdoor',
    'waarbij', 'waarmee', 'waarvoor', 'waarna', 'waarin', 'waardoor',
    'gaan', 'gaat', 'komen', 'komt', 'maken', 'maakt', 'moeten',
    'moet', 'kunnen', 'kan', 'willen', 'wil', 'zijn', 'wij', 'hen',
    'heel', 'zeer', 'echt', 'juist', 'snel', 'groot', 'goed', 'nieuw',
    'heer', 'mevrouw', 'nederland', 'nederlands', 'kabinet',
    'vendrik', 'halsema', 'gent', 'ouwehand', 'thieme', 'wassenberg',
    'tongeren', 'grashoff', 'moorlag', 'jacobi', 'veldhoven',
    'kops', 'klever', 'madlener', 'dijck', 'lodders', 'neppérus',
    'ziengs', 'grinwis', 'koopmans', 'mulder', 'bontenbal', 'vedder',
    'geurts', 'plas', 'eppink', 'pierik', 'eerdmans', 'jansen',
    'haga', 'blanke', 'raan', 'teunissen', 'kröger', 'thijssen',
    'gabriëls', 'kostic', 'bushoff', 'beckerman', 'tjeerd', 'remco',
    'dijkstra', 'veltman', 'eijs', 'ham', 'mos', 'tony', 'bromet',
    'klaver', 'bouchallikh', 'ouchallikh', 'lee', 'smeulders',
    'bamenga', 'paternotte', 'hagen', 'boucke', 'podt', 'schouw',
    'hachchi', 'verhoeven', 'kaya', 'boswijk', 'werf', 'sienot',
    'martels', 'harbers', 'koerhuis', 'wijngaarden', 'peter',
    'graaf', 'graus', 'wilders', 'bemmel', 'heutink', 'blank',
    'arts', 'exirel', 'omtzigt', 'bisschop', 'flach', 'vestering',
    'esch', 'nijboer', 'jan', 'vos', 'albert', 'cegerek', 'smaling',
    'eigenlijk', 'vraag', 'smaak', 'schrijver', 'cdafractie',
    'pvdafractie', 'afrikaans',
}
ALL_STOPS = DUTCH_STOPS | EXTRA_STOPS

CONTENT_POS = {'NOUN', 'ADJ'}


# ── Text processing ───────────────────────────────────────────────────────────

def clean_motion_text(text: str) -> str:
    """Strip motion headers and closing formulas to remove member names."""
    if not isinstance(text, str):
        return ''
    text = re.sub(
        r'(?i)^.*?gehoord\s+de\s+beraadslaging[,\s]*',
        '', text, flags=re.DOTALL)
    text = re.sub(
        r'(?i)en\s+gaat\s+over\s+tot\s+de\s+orde\s+van\s+de\s+dag\.?',
        '', text)
    return ' '.join(text.split()).strip()


def lemmatize_content_words(text: str) -> str:
    """Lemmatize and retain only content nouns and adjectives."""
    if not isinstance(text, str) or not text.strip():
        return ''
    doc = nlp(text[:8000])
    return ' '.join(
        t.lemma_.lower() for t in doc
        if t.pos_ in CONTENT_POS
        and not t.is_stop
        and len(t.lemma_) > 4
        and t.lemma_.lower() not in ALL_STOPS
        and t.lemma_.isalpha()
    )


def load_and_lemmatize(source: str) -> pd.DataFrame:
    path = INPUT_DIR / f'{source}_linguistic.csv'
    df   = pd.read_csv(path)
    print(f'  {source}: {len(df)} rows')
    df['text_clean'] = (df['text'].apply(clean_motion_text)
                        if source == 'motions' else df['text'])
    df['text_lemma'] = df['text_clean'].apply(lemmatize_content_words)
    return df[['party', 'year', 'text_lemma']].copy()


# ── TF-IDF and similarity ─────────────────────────────────────────────────────

def build_party_corpus(df: pd.DataFrame) -> tuple:
    """
    Concatenate all lemmatized texts per party into one document.
    Returns ordered list of parties and corresponding documents.
    """
    parties = [p for p in TARGET if p in df['party'].unique()]
    docs    = []
    for party in parties:
        texts = df[df['party'] == party]['text_lemma'].dropna().tolist()
        docs.append(' '.join(texts))
    return parties, docs


def compute_similarity(parties: list, docs: list) -> pd.DataFrame:
    """
    Fit TF-IDF on all party documents and compute pairwise cosine similarity.
    Returns a party × party similarity DataFrame.
    """
    vectorizer = TfidfVectorizer(
        max_features=15000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=list(ALL_STOPS),
        sublinear_tf=True,
        token_pattern=r'\b[a-zA-Zàáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ]{4,}\b'
    )
    matrix = vectorizer.fit_transform(docs)
    sim    = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=parties, columns=parties)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_similarity_heatmap(sim_df: pd.DataFrame):
    """
    Heatmap of pairwise cosine similarity between party TF-IDF vectors.
    Diagonal masked. Parties ordered by ideological bloc.
    """
    order = [p for p in TARGET if p in sim_df.index]
    sub   = sim_df.loc[order, order]

    mask = np.eye(len(order), dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sub, ax=ax, cmap='RdYlGn', vmin=0, vmax=0.6,
        annot=True, fmt='.2f', linewidths=0.4, mask=mask,
        cbar_kws={'label': 'Cosine similarity'}
    )

    # Colour axis labels by party
    for lbl in ax.get_xticklabels():
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')
    for lbl in ax.get_yticklabels():
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')

    ax.set_title(
        'Lexical Similarity Between Parties in Climate Discourse\n'
        '(TF-IDF cosine similarity, all text types combined)',
        fontsize=11)
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'similarity_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: similarity_heatmap.pdf')


def plot_dendrogram(sim_df: pd.DataFrame):
    """
    Hierarchical clustering dendrogram of parties based on lexical similarity.
    Ward linkage on cosine distance. Party labels coloured by party colour,
    with a coloured bloc background box to indicate ideological bloc membership.
    """
    order     = [p for p in TARGET if p in sim_df.index]
    sub       = sim_df.loc[order, order].values
    dist      = 1 - sub
    np.fill_diagonal(dist, 0)
    dist      = (dist + dist.T) / 2
    condensed = squareform(dist)
    linked    = linkage(condensed, method='ward')

    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(
        linked,
        labels=order,
        ax=ax,
        orientation='top',
        leaf_font_size=12,
        color_threshold=0,
        above_threshold_color='#cccccc',
    )

    ax.set_title(
        'Hierarchical Clustering of Parties by Lexical Similarity\n'
        '(Ward linkage, cosine distance on TF-IDF, all text types combined)',
        fontsize=11)
    ax.set_ylabel('Distance (1 − cosine similarity)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [mpatches.Patch(color=c, label=b, alpha=0.5)
               for b, c in BLOC_COLORS.items()]
    ax.legend(handles=handles, title='Ideological bloc',
              loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.draw()

    for lbl in ax.get_xmajorticklabels():
        party = lbl.get_text()
        bloc  = BLOC_MAP.get(party, '')
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')
        lbl.set_bbox(dict(boxstyle='round,pad=0.25',
                          facecolor=BLOC_COLORS.get(bloc, '#ffffff'),
                          alpha=0.5, edgecolor='none'))

    plt.savefig(FIG_DIR / 'party_dendrogram.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: party_dendrogram.pdf')



def plot_combined(sim_df: pd.DataFrame):
    """
    Side-by-side figure: heatmap left, dendrogram right.
    Suitable for inclusion as a single figure in the thesis.
    """
    order     = [p for p in TARGET if p in sim_df.index]
    sub       = sim_df.loc[order, order]
    dist      = 1 - sub.values
    np.fill_diagonal(dist, 0)
    dist      = (dist + dist.T) / 2
    condensed = squareform(dist)
    linked    = linkage(condensed, method='ward')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                              gridspec_kw={'width_ratios': [1, 1]})

    # Left: heatmap
    mask = np.eye(len(order), dtype=bool)
    sns.heatmap(
        sub, ax=axes[0], cmap='RdYlGn', vmin=0, vmax=0.6,
        annot=True, fmt='.2f', linewidths=0.4, mask=mask,
        cbar_kws={'label': 'Cosine similarity'}
    )
    for lbl in axes[0].get_xticklabels():
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')
    for lbl in axes[0].get_yticklabels():
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')
    axes[0].set_title('Pairwise cosine similarity', fontsize=11)
    axes[0].tick_params(axis='x', rotation=40)

    # Right: dendrogram
    dendrogram(
        linked, labels=order, ax=axes[1],
        orientation='top', leaf_font_size=11,
        color_threshold=0, above_threshold_color='#cccccc',
    )
    axes[1].set_title('Hierarchical clustering (Ward linkage)', fontsize=11)
    axes[1].set_ylabel('Distance (1 − cosine similarity)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    handles = [mpatches.Patch(color=c, label=b, alpha=0.5)
               for b, c in BLOC_COLORS.items()]
    axes[1].legend(handles=handles, title='Bloc', loc='upper right', fontsize=9)

    fig.suptitle(
        'Lexical Similarity Between Parties in Climate Discourse\n'
        '(TF-IDF cosine similarity on all text types combined)',
        fontsize=12)

    plt.tight_layout()
    plt.draw()

    for lbl in axes[1].get_xmajorticklabels():
        party = lbl.get_text()
        bloc  = BLOC_MAP.get(party, '')
        lbl.set_color('#000000')
        lbl.set_fontweight('bold')
        lbl.set_bbox(dict(boxstyle='round,pad=0.25',
                          facecolor=BLOC_COLORS.get(bloc, '#ffffff'),
                          alpha=0.5, edgecolor='none'))

    plt.savefig(FIG_DIR / 'similarity_combined.pdf', bbox_inches='tight')
    plt.close()
    print('Saved: similarity_combined.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading and lemmatizing...')
    dfs = []
    for source in ['motions', 'manifestos', 'speeches']:
        dfs.append(load_and_lemmatize(source))

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined['party'].isin(TARGET)].copy()
    combined = combined[combined['party'] != 'GroenLinks-PvdA'].copy()
    print(f'Combined: {len(combined)} rows')
    print(combined.groupby('party').size().sort_values(ascending=False).to_string())

    print('\nBuilding party corpus and computing similarity...')
    parties, docs = build_party_corpus(combined)
    sim_df        = compute_similarity(parties, docs)

    sim_df.to_csv(OUT_DIR / 'party_similarity.csv')
    print('\nSimilarity matrix:')
    print(sim_df.round(3).to_string())

    print('\nGenerating figures...')
    plot_similarity_heatmap(sim_df)
    plot_dendrogram(sim_df)
    plot_combined(sim_df)

    print(f'\nAll outputs saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
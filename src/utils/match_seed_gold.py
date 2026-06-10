"""
Match ParlaMint seed motions against EXP_10 gold set using TF-IDF cosine similarity.
Both texts are cleaned with the same pipeline before matching.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

SIMILARITY_THRESHOLD = 0.85


def clean_motion_text(text):
    """Identical cleaning pipeline to EXP_6/EXP_8."""
    if not isinstance(text, str):
        return ""
    noise_patterns = [
        r"De Kamer,",
        r"gehoord de beraadslaging[,]?\s*",
        r"constaterende[,]? dat\s*",
        r"overwegende[,]? dat\s*",
        r"verzoekt de regering[,]?\s*",
        r"en gaat over tot de orde van de dag\.?",
        r"Motie van (het lid|de leden)[^,]+,",
        r"Motie [A-Z][^,]+,",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


print("Loading data...")
seed = pd.read_csv('results/motions/parlamint_seed_motions.csv')
gold = pd.read_csv('results/classifier/binary_annotations.csv')

seed = seed[seed['hoofd_thema'] != 'other'].copy()
seed['text_clean'] = seed['text'].apply(clean_motion_text)
gold['text_clean'] = gold['normalized_text'].apply(clean_motion_text)

# Filter empty
seed = seed[seed['text_clean'].str.len() > 50].reset_index(drop=True)
gold = gold[gold['text_clean'].str.len() > 50].reset_index(drop=True)

print(f"Seed motions : {len(seed)}")
print(f"Gold motions : {len(gold)}")

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
all_texts = pd.concat([seed['text_clean'], gold['text_clean']], ignore_index=True)
vectorizer.fit(all_texts)

seed_vectors = vectorizer.transform(seed['text_clean'])
gold_vectors = vectorizer.transform(gold['text_clean'])

print("Computing cosine similarity (batched)...")
matches = []
batch_size = 500

for i in range(0, len(seed), batch_size):
    batch = seed_vectors[i:i + batch_size]
    sims = cosine_similarity(batch, gold_vectors)
    best_idx = np.argmax(sims, axis=1)
    best_sim = np.max(sims, axis=1)

    for j, (idx, sim) in enumerate(zip(best_idx, best_sim)):
        if sim >= SIMILARITY_THRESHOLD:
            matches.append({
                'seed_idx':       i + j,
                'seed_doc_id':    seed.iloc[i + j]['doc_id'],
                'seed_label':     seed.iloc[i + j]['hoofd_thema'],
                'gold_idx':       idx,
                'gold_label':     gold.iloc[idx]['gold_label'],
                'predicted_label': gold.iloc[idx]['predicted_topic'],
                'similarity':     round(sim, 4),
            })

matches_df = pd.DataFrame(matches)
print(f"\nMatches found (similarity >= {SIMILARITY_THRESHOLD}): {len(matches_df)}")

if len(matches_df) > 0:
    disagreements = matches_df[matches_df['seed_label'] != matches_df['gold_label'].map(
        lambda x: x if x == 'climate_agriculture_energy' else 'not_relevant'
    )]
    print(f"Label disagreements (seed vs gold): {len(disagreements)}")
    print("\nDisagreement breakdown:")
    print(matches_df.groupby(['seed_label', 'gold_label']).size().to_string())

    matches_df.to_csv('results/classifier/seed_gold_matches.csv', index=False)
    print("\nSaved to: results/classifier/seed_gold_matches.csv")

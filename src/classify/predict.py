"""
apply_model_poi_check.py
--------------------------------------------------------------------------------
Thesis Project: Automated Theme Classification
Step: Final Inference (POI Validation)

Description:
    Applies the trained model (Round 1) ONLY to the 'Parties of Interest' (POI).
    Saves a specific CSV for manual inspection to validate data quality 
    before the final analysis phase.
--------------------------------------------------------------------------------
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
BEST_MODEL_PATH = "models/self_training_rounds/round_1" 

INPUT_CSV = "data/formatted_data/all_motions_2008_2025_with_text_and_normalized.csv"
OUTPUT_CSV = "data/formatted_data/poi_motions_annotated_CHECK.csv" # Voor handmatige controle

TEXT_COL = "normalized_text"
FRACTIONS_COL = "fractions"
BATCH_SIZE = 64 

# Definitie van POI (Zoals besproken: Centrum-Links tot Centrum-Rechts)
# Historisch GroenLinks zit hier NIET in.
POI_PARTIES = {
    'CDA', 
    'VVD', 
    'D66', 
    'PvdA', 
    'GroenLinks-PvdA'
}

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = [str(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

def main():
    print("=== STEP: POI INFERENCE FOR MANUAL CHECK ===")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data & Filter POI
    print(f"Loading raw data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print("Filtering for Parties of Interest (POI) only...")
    
    def is_poi_motion(fractions_str):
        if not isinstance(fractions_str, str):
            return False
        # Split "SP; VVD" -> {"SP", "VVD"}
        parties = set(p.strip() for p in fractions_str.split(';'))
        # Check overlap: Return True if ANY party is in the POI list
        return not parties.isdisjoint(POI_PARTIES)

    # Apply filter
    poi_mask = df[FRACTIONS_COL].apply(is_poi_motion)
    df_poi = df[poi_mask].copy()
    
    # Drop empty texts
    df_poi = df_poi.dropna(subset=[TEXT_COL])
    
    print(f"Total motions in corpus: {len(df)}")
    print(f"POI motions to classify: {len(df_poi)}")
    
    # 3. Load Model
    print(f"Loading model from: {BEST_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
    model.to(device)
    model.eval()

    # 4. Prepare Loader
    dataset = InferenceDataset(df_poi[TEXT_COL].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds = []

    # 5. Inference Loop
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Classifying POI", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=mask)
            
            probs = F.softmax(outputs.logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 6. Process Results
    id2label = model.config.id2label
    
    df_poi["predicted_category"] = [id2label[p] for p in all_preds]
    df_poi["confidence_score"] = all_probs

    # 7. Reorder columns for easier manual check in Excel/CSV
    # We put the prediction and text first
    cols = ['predicted_category', 'confidence_score', FRACTIONS_COL, TEXT_COL]
    # Add other columns that exist (like date, title) to the end
    remaining_cols = [c for c in df_poi.columns if c not in cols]
    df_final = df_poi[cols + remaining_cols]

    print("\n--- Prediction Distribution on POI Data ---")
    print(df_final["predicted_category"].value_counts())

    print(f"\nSaving check-file to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("Done. Open this file to verify if the labels make sense.")

if __name__ == "__main__":
    main()
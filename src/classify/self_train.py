"""
self_train_classifier.py
--------------------------------------------------------------------------------
Thesis Project: Automated Theme Classification of Parliamentary Motions
Module: Weak Supervision / Self-Training Loop

Description:
    This script implements an iterative self-training strategy to expand the 
    training dataset beyond the initial manually annotated seed set.
    
    Methodology:
    1. Data Separation: The unlabeled corpus is strictly split into:
       - POI (Parties of Interest): Used ONLY for analysis/validation (Never training).
       - NON-POI (Other Parties): Used as the pool for weak supervision.
    2. Inference: The current model predicts labels for the NON-POI pool.
    3. Filtering: High-confidence predictions (> threshold) are selected ('Silver Data').
    4. Retraining: The model is retrained on the union of Seed Data + Silver Data.
    5. Evaluation: Performance is monitored on the held-out Gold Standard Dev Set.

Usage:
    Run with '--only_split' to verify data separation logic before training.
--------------------------------------------------------------------------------
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm  # <--- Added for progress bars
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ------------------------------------------------------------------------------
# Configuration & Hyperparameters
# ------------------------------------------------------------------------------

# Paths
BASE_MODEL_DIR = "models/theme_classifier_initial" 
OUTPUT_DIR_ROOT = "models/self_training_rounds"
DEBUG_SPLIT_DIR = "data/debug_splits"

# Input Data
SEED_TRAIN_CSV = "data/formatted_data/splits/historical_motions/train.csv"
SEED_DEV_CSV =   "data/formatted_data/splits/historical_motions/dev.csv"
UNLABELED_CSV =  "data/formatted_data/all_motions_2008_2025_with_text_and_normalized.csv" 

# Column Names
TEXT_COL = "normalized_text"
LABEL_COL = "category"
FRACTIONS_COL = "fractions" # Contains semicolon-separated parties (e.g., "SP; VVD")

# Methodology: Parties of Interest (POI)
# If a motion involves ANY of these parties, it is contaminated for training purposes.
# It must be reserved for the final analysis or validation.
POI_PARTIES = {
    'CDA', 'VVD', 'D66', 'PvdA', 'GroenLinks-PvdA',
}

# Training Settings
CONFIDENCE_THRESHOLD = 0.95  # Strict threshold to minimize label noise propagation
MAX_ROUNDS = 3               # Maximum number of self-training iterations
MAX_LENGTH = 256
BATCH_SIZE = 32

# ------------------------------------------------------------------------------
# Helper Classes
# ------------------------------------------------------------------------------

class MotionDataset(Dataset):
    """Standard PyTorch Dataset for training/evaluation."""
    def __init__(self, texts, labels, tokenizer, label2id):
        self.texts = texts
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

class InferenceDataset(Dataset):
    """Dataset optimized for inference (no labels required)."""
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

class WeightedTrainer(Trainer):
    """
    Custom HuggingFace Trainer that incorporates Class Weights into the Loss function.
    Essential for preventing model laziness (predicting only majority class) during
    early stages of self-training with imbalanced data.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Inject custom class weights if they are set on the instance
        if hasattr(self, "class_weights"):
            weight_tensor = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Calculates Accuracy and Weighted F1-Score."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

def get_data_splits(df):
    """
    Separates the dataset into POI (Analysis/Validation) and NON-POI (Training Pool).
    
    Logic:
    - Parses the 'fractions' column (e.g., "SP; VVD").
    - If intersection({SP, VVD}, {POI_PARTIES}) is NOT empty -> It counts as POI.
    - We strictly remove any motion containing a POI party from the training pool.
    """
    print("Filtering Training Pool (Removing any POI influence)...")
    
    def contains_poi_influence(fractions_str):
        if not isinstance(fractions_str, str):
            return False 
        
        # Split string and cleanup whitespace
        parties_in_motion = set(p.strip() for p in fractions_str.split(';'))
        
        # Check intersection with the forbidden POI set
        return not parties_in_motion.isdisjoint(POI_PARTIES)

    # Create boolean mask
    poi_mask = df[FRACTIONS_COL].apply(contains_poi_influence)
    
    poi_df = df[poi_mask].copy()
    non_poi_df = df[~poi_mask].copy()
    
    print(f"  - Total motions: {len(df)}")
    print(f"  - POI motions (Excluded from training): {len(poi_df)}")
    print(f"  - Other motions (Available for training): {len(non_poi_df)}")
    
    return poi_df, non_poi_df

def predict_and_filter(model, tokenizer, df, label_list):
    """
    Runs inference on unlabeled data and filters for high-confidence predictions.
    Returns a dataframe containing the 'Silver Data'.
    """
    print(f"Running inference on {len(df)} unlabeled examples...")
    device = model.device
    dataset = InferenceDataset(df[TEXT_COL].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_probs = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        # Added TQDM here for progress bar on inference
        for batch in tqdm(loader, desc="  Generating Pseudo-labels", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=mask)
            # Convert logits to probabilities
            probs = F.softmax(outputs.logits, dim=1)
            
            # Get highest probability and corresponding index
            max_probs, preds = torch.max(probs, dim=1)
            
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Assign results to DataFrame
    df["confidence"] = all_probs
    df["pred_id"] = all_preds
    df[LABEL_COL] = df["pred_id"].apply(lambda x: label_list[x])
    
    # Filter based on threshold
    high_conf_df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].copy()
    return high_conf_df

# ------------------------------------------------------------------------------
# Main Execution Flow
# ------------------------------------------------------------------------------

def main():
    # 0. Argument Parsing
    parser = argparse.ArgumentParser(description="Run iterative self-training for theme classification.")
    parser.add_argument(
        "--only_split", 
        action="store_true", 
        help="If set, only performs POI/Non-POI splitting, saves debug files, and exits. Use this to verify data integrity."
    )
    args = parser.parse_args()

    # 1. Load Initial Data
    print(f"Loading seed data from {SEED_TRAIN_CSV}...")
    seed_train_df = pd.read_csv(SEED_TRAIN_CSV)
    seed_dev_df = pd.read_csv(SEED_DEV_CSV) 
    
    print(f"Loading unlabeled data from {UNLABELED_CSV}...")
    unlabeled_df = pd.read_csv(UNLABELED_CSV)
    unlabeled_df = unlabeled_df.dropna(subset=[TEXT_COL])
    
    # 2. Perform Data Split (POI vs. Non-POI)
    poi_df, training_pool_df = get_data_splits(unlabeled_df)
    
    # 3. Handle Flag: --only_split
    if args.only_split:
        print("\n[FLAG DETECTED] --only_split is active.")
        print("Saving debug split files for verification...")
        
        os.makedirs(DEBUG_SPLIT_DIR, exist_ok=True)
        
        # Save POI file (Should contain VVD, CDA, etc.)
        poi_path = os.path.join(DEBUG_SPLIT_DIR, "debug_poi_excluded.csv")
        poi_df.to_csv(poi_path, index=False)
        
        # Save Non-POI file (Should ONLY contain SP, SGP, PvdD, etc.)
        non_poi_path = os.path.join(DEBUG_SPLIT_DIR, "debug_non_poi_training_pool.csv")
        training_pool_df.to_csv(non_poi_path, index=False)
        
        print(f"1. Check '{poi_path}' -> Should contain VVD, CDA, PvdA, etc.")
        print(f"2. Check '{non_poi_path}' -> Should contain only SP, SGP, PvdD, etc.")
        print("Exiting script now.")
        return # STOP HERE

    # 4. Prepare for Training Loop
    # Setup label mappings based on the definitive seed set
    labels = sorted(seed_train_df[LABEL_COL].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    
    current_model_path = BASE_MODEL_DIR
    
    # 5. Iterative Self-Training Loop
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"SELF-TRAINING ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'='*60}")
        
        # A. Load Current Model State
        print(f"Loading model from: {current_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(current_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(current_model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model moved to: {device}")
        # ----------------------------------------
        
        # B. Generate Silver Data (Predict on Training Pool)
        # TQDM progress bar is now inside this function
        silver_df = predict_and_filter(model, tokenizer, training_pool_df, labels)
        
        print(f"High-confidence examples found (> {CONFIDENCE_THRESHOLD}): {len(silver_df)}")
        if not silver_df.empty:
            print(silver_df[LABEL_COL].value_counts())
        
        # Early stopping if saturation is reached
        if len(silver_df) < 50:
            print("Stopping: Too few new high-confidence examples to justify retraining.")
            break
            
        # C. Combine Datasets (Gold Seed + Silver Pseudo-labeled)
        combined_train_df = pd.concat([seed_train_df, silver_df[[TEXT_COL, LABEL_COL]]])
        print(f"New combined training set size: {len(combined_train_df)}")
        
        # D. Recalculate Class Weights (Critical for imbalanced loops)
        y_train_ids = [label2id[l] for l in combined_train_df[LABEL_COL]]
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train_ids), y=y_train_ids)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        # E. Configure Training
        output_dir = os.path.join(OUTPUT_DIR_ROOT, f"round_{round_num}")
        
        train_dataset = MotionDataset(
            combined_train_df[TEXT_COL].tolist(), 
            combined_train_df[LABEL_COL].tolist(), 
            tokenizer, label2id
        )
        # Validate ONLY on the Gold Standard Dev Set (Annotated POI data)
        eval_dataset = MotionDataset(
            seed_dev_df[TEXT_COL].tolist(), 
            seed_dev_df[LABEL_COL].tolist(), 
            tokenizer, label2id
        )
        
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            learning_rate=2e-5, 
            num_train_epochs=3, 
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            fp16=torch.cuda.is_available(), 
            # Trainer has its own tqdm, make sure it's not disabled
            disable_tqdm=False 
        )
        
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            tokenizer=tokenizer, compute_metrics=compute_metrics
        )
        trainer.class_weights = class_weights_tensor
        
        # F. Train
        trainer.train()
        
        # G. Save & Update Pointers
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        current_model_path = output_dir
        
        # H. Detailed Evaluation
        print("\nValidating on Golden Dev Set...")
        metrics = trainer.evaluate()
        print(f"Round {round_num} Metrics: {metrics}")
        
        # Generate Classification Report
        preds = trainer.predict(eval_dataset)
        pred_labels = np.argmax(preds.predictions, axis=1)
        true_labels = preds.label_ids
        
        print(classification_report(
            true_labels, 
            pred_labels, 
            target_names=labels, 
            labels=list(range(len(labels))), 
            zero_division=0
        ))

    print(f"\nSelf-training complete. Final model saved at: {current_model_path}")

if __name__ == "__main__":
    main()
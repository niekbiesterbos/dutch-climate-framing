"""
train_initial_classifier.py
-------------------------------------------------
Trains an initial Dutch policy-domain classifier (RobBERT)
using manually annotated seed motions.

Methodology: Step 1 (Theme Classification)
- Uses Weighted Loss to handle class imbalance in seed data.
- Saves model for downstream 'High Confidence' filtering.
-------------------------------------------------
"""

import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- CONFIGURATION ---
MODEL_NAME = "DTAI-KULeuven/robbert-2023-dutch-large"
INPUT_CSV = "data/formatted_data/historical_motions_annotated.csv"
SPLIT_DIR = "data/formatted_data/splits/historical_motions"
OUTPUT_DIR = "models/theme_classifier_initial"

LABEL_COL = "category"
TEXT_COL = "normalized_text"
MAX_LENGTH = 256


class MotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}


class WeightedTrainer(Trainer):
    """
    Custom Trainer that uses Class Weights in the Loss function.
    Essential for small/imbalanced seed datasets to prevent mode collapse.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Retrieve the weights calculated in main()
        # Ensure they are on the same device as the model
        if hasattr(self, "class_weights"):
            weight_tensor = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Computes accuracy and F1-score for the evaluation loop."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    # Weighted F1 is standard for multiclass imbalance
    f1 = f1_score(labels, preds, average="weighted")
    
    return {"accuracy": acc, "f1": f1}


def main():
    # 1. SETUP DATA
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError("Input CSV must contain 'normalized_text' and 'category' columns")

    os.makedirs(SPLIT_DIR, exist_ok=True)

    # --- FILTER RARE CLASSES (Prevents Stratification Crash) ---
    class_counts = df[LABEL_COL].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    
    if len(rare_classes) > 0:
        print(f"Warning: Dropping {len(rare_classes)} classes with < 2 instances: {list(rare_classes)}")
        df = df[~df[LABEL_COL].isin(rare_classes)]

    # Create label mappings
    labels = sorted(df[LABEL_COL].dropna().unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"Active classes ({len(labels)}): {labels}")

    # --- SPLITTING STRATEGY (80% Train, 10% Dev, 10% Test) ---
    print("Splitting data...")
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL_COL]
    )
    
    try:
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df[LABEL_COL]
        )
    except ValueError:
        print("Warning: Could not stratify Dev/Test split. Falling back to random split.")
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=None
        )

    # Save Splits
    train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)
    print(f"Sizes: Train={len(train_df)}, Dev={len(val_df)}, Test={len(test_df)}")

    # --- CALCULATE CLASS WEIGHTS ---
    # This is the key fix for the "laziness" of the model
    train_labels_ids = [label2id[l] for l in train_df[LABEL_COL]]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels_ids),
        y=train_labels_ids
    )
    # Convert to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class Weights calculated: {class_weights}")

    # 2. INITIALIZE MODEL & TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    # 3. CREATE DATASETS
    train_dataset = MotionDataset(train_df[TEXT_COL].tolist(), [label2id[l] for l in train_df[LABEL_COL]], tokenizer)
    val_dataset = MotionDataset(val_df[TEXT_COL].tolist(), [label2id[l] for l in val_df[LABEL_COL]], tokenizer)
    test_dataset = MotionDataset(test_df[TEXT_COL].tolist(), [label2id[l] for l in test_df[LABEL_COL]], tokenizer)

    # 4. TRAINING ARGUMENTS
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=10,            # Increased to allow convergence with weights
        logging_steps=10,
        eval_strategy="epoch",          # Fixed deprecated arg
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        weight_decay=0.01,
        warmup_ratio=0.1,               # Added stability
        fp16=False,                     # Set to True if you have a GPU supporting it
    )

    # 5. INITIALIZE CUSTOM TRAINER
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Inject weights into trainer
    trainer.class_weights = class_weights_tensor

    # 6. TRAIN
    print("Starting training with Class Weights...")
    trainer.train()

    # 7. SAVE & EVALUATE
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n--- Final Evaluation on Test Set ---")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # Detailed Report
    print("\n--- Detailed Classification Report ---")
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids
    
    # Use 'labels' param to ensure all classes are reported even if missing in test set
    print(classification_report(
        true_labels, 
        pred_labels, 
        target_names=labels, 
        labels=list(range(len(labels))),
        zero_division=0
    ))

if __name__ == "__main__":
    main()
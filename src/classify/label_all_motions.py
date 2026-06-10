import pandas as pd
import torch
import os
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- HABROK CONFIG ---
os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.environ['HF_HOME'])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

exp_dir = 'results/classifier'
os.makedirs(exp_dir, exist_ok=True)
output_dir = 'results/motions'
os.makedirs(output_dir, exist_ok=True)

def clean_motion_text(text):
    """Removes parliamentary boilerplate, identical to EXP_6 training pipeline."""
    if not isinstance(text, str): return ""
    noise_patterns = [
        r"De Kamer,",
        r"gehoord de beraadslaging[,]?\s*",
        r"constaterende[,]? dat\s*",
        r"overwegende[,]? dat\s*",
        r"verzoekt de regering[,]?\s*",
        r"en gaat over tot de orde van de dag\.?"
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()

print("--- FASE 1: Inladen en schoonmaken ---")
file_path = 'data/motions/all_motions_2008_2025.csv'
df_full = pd.read_csv(file_path)
df_full = df_full[df_full['normalized_text'].notna()].copy()
df_full['text_clean'] = df_full['normalized_text'].apply(clean_motion_text)
print(f"Totaal aantal moties klaar voor model predictie: {len(df_full)}")

print("\n--- FASE 2: Model en Tokenizer laden ---")
model_path = "models/robbert_binary"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")

def tokenize_function(examples):
    return tokenizer(examples["text_clean"], padding="max_length", truncation=True, max_length=512)

full_tokenized = Dataset.from_pandas(df_full[['text_clean']]).map(tokenize_function, batched=True)

print("\n--- FASE 3: Start Inference ---")
training_args = TrainingArguments(
    output_dir=f"{exp_dir}/tmp_inference",
    per_device_eval_batch_size=32,
    fp16=True,
    dataloader_num_workers=4
)
trainer = Trainer(model=model, args=training_args)

predictions = trainer.predict(full_tokenized)
preds = np.argmax(predictions.predictions, axis=-1)

label_map = {0: "not_relevant", 1: "climate_agriculture_energy"}
df_full['predicted_label_id'] = preds
df_full['predicted_topic'] = df_full['predicted_label_id'].map(label_map)

print("\n--- FASE 4: Resultaten opslaan ---")
df_climate = df_full[df_full['predicted_label_id'] == 1].copy()
print(f"Klaar! {len(df_climate)} klimaatgerelateerde moties geïdentificeerd.")

df_full.to_csv(f'{exp_dir}/all_motions_2008_2025_with_predictions.csv', index=False)

output_file = 'results/motions/climate_motions.csv'
df_climate.to_csv(output_file, index=False)
print(f"Klimaat-dataset opgeslagen in: {output_file}")
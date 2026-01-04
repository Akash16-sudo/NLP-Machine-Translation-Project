import os
import sys
import numpy as np
import torch
import evaluate
import nltk
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cnn_dailymail_processed"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_CHECKPOINT = "facebook/bart-large-cnn"

# ---------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------
metric = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

def main():
    # ---------------------------------------------------------
    # 1. SETUP
    # ---------------------------------------------------------
    print(f"Step 1: Setting up training...")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Create directories
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. LOAD DATA & MODEL
    # ---------------------------------------------------------
    print("\nStep 2: Loading processed data and model...")
    if not PROCESSED_DATA_PATH.exists():
        print(f"❌ Error: processed data not found at {PROCESSED_DATA_PATH}")
        sys.exit(1)
        
    tokenized_datasets = load_from_disk(str(PROCESSED_DATA_PATH))
    print(f"   Train samples: {len(tokenized_datasets['train'])}")
    print(f"   Validation samples: {len(tokenized_datasets['validation'])}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---------------------------------------------------------
    # 3. TRAINING ARGUMENTS
    # ---------------------------------------------------------
    print("\nStep 3: Configuring training arguments...")
    
    # Check if this is a "Smoke Test" (quick run to verify code)
    is_smoke_test = "--smoke-test" in sys.argv
    if is_smoke_test:
        print("!! SMOKE TEST MODE: Training on 20 samples only.")
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(20))
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(10))
        num_epochs = 1
    else:
        num_epochs = 3

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",  # Sync with eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, # Effective batch size = 32
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use Mixed Precision if GPU available
        logging_dir=str(LOG_DIR),
        logging_steps=50,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    # ---------------------------------------------------------
    # 4. TRAINER
    # ---------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # ---------------------------------------------------------
    # 5. TRAIN
    # ---------------------------------------------------------
    print("\nStep 4: Starting training...")
    import nltk
    nltk.download("punkt", quiet=True)
    
    train_result = trainer.train()
    
    print("\nStep 5: Training complete!")
    print(f"   Training Loss: {train_result.training_loss}")
    
    # Save Final Model
    final_save_path = MODEL_OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_save_path))
    print(f"   Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()

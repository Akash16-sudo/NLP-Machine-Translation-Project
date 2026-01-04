import sys
import re
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cnn_dailymail_datasets"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cnn_dailymail_processed"

MODEL_CHECKPOINT = "facebook/bart-large-cnn"
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

def clean_text(text):
    """
    Cleans text by normalizing whitespace and removing common artifacts.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize whitespace (tabs, newlines -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common artifacts like "(CNN) -- " at the start
    text = re.sub(r'^\(CNN\)\s*--\s*', '', text, flags=re.IGNORECASE)
    
    # Remove "By [Name]" patterns if they appear at the very start (simplified)
    # text = re.sub(r'^By\s+[\w\s]+$', '', text, flags=re.IGNORECASE) 
    
    return text

def preprocess_function(examples, tokenizer):
    """
    Cleans and tokenizes inputs (articles) and targets (highlights).
    """
    # 1. Clean the text
    inputs = [clean_text(doc) for doc in examples["article"]]
    targets = [clean_text(doc) for doc in examples["highlights"]]

    # 2. Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True
    )

    # 3. Tokenize targets
    labels = tokenizer(
        text_target=targets, 
        max_length=MAX_TARGET_LENGTH, 
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def filter_examples(example):
    """
    Filters out examples that are empty or too short.
    """
    article = example.get("article")
    highlights = example.get("highlights")
    
    # Check for None or empty string
    if not article or not highlights:
        return False
        
    # Check word count (simple whitespace split)
    if len(article.split()) < 5:
        return False
        
    if len(highlights.split()) < 3:
        return False
        
    return True

def main():
    # ---------------------------------------------------------
    # 1. SETUP & VALIDATION
    # ---------------------------------------------------------
    print("Step 1: Checking environment...")
    
    if not RAW_DATA_PATH.exists():
        print(f"❌ Error: Could not find raw data directory at:\n   {RAW_DATA_PATH}")
        sys.exit(1)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 2. LOAD DATA
    # ---------------------------------------------------------
    print("\nStep 2: Loading raw datasets from disk...")
    try:
        raw_datasets = load_from_disk(str(RAW_DATA_PATH))
        print(f"   Loaded: {raw_datasets}")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. FILTER DATA
    # ---------------------------------------------------------
    print("\nStep 3: Filtering invalid or short samples...")
    # Using .filter() to remove bad samples before processing
    filtered_datasets = raw_datasets.filter(filter_examples, desc="Filtering")
    print(f"   Filtered dataset: {filtered_datasets}")

    # ---------------------------------------------------------
    # 4. INITIALIZE TOKENIZER
    # ---------------------------------------------------------
    print(f"\nStep 4: Loading tokenizer ({MODEL_CHECKPOINT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # ---------------------------------------------------------
    # 5. TOKENIZE DATASETS
    # ---------------------------------------------------------
    print("\nStep 5: Cleaning and Tokenizing data...")
    
    tokenized_datasets = filtered_datasets.map(
        preprocess_function, 
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing"
    )

    # ---------------------------------------------------------
    # 6. SAVE PROCESSED DATA
    # ---------------------------------------------------------
    print(f"\nStep 6: Saving processed data to:\n   {PROCESSED_DATA_PATH}")
    tokenized_datasets.save_to_disk(str(PROCESSED_DATA_PATH))
    
    print("\n✨ Success! Data preprocessing complete.")

if __name__ == "__main__":
    main()
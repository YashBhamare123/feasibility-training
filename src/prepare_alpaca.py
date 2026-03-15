"""
Alpaca dataset preparation
Downloads and tokenizes the "tatsu-lab/alpaca" dataset, saving to a single train and val .npy file.
Run simply as:
$ python prepare_alpaca.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Dataset
    DATASET_REPO = 'tatsu-lab/alpaca'
    SPLIT = 'train'
    
    # Tokenizer
    TOKENIZER = 'gpt2'
    
    # Output Directory
    LOCAL_DIR = "../data/alpaca"

# ==========================================

def tokenize(doc, enc, eot):
    """ 
    tokenizes a single document and returns a list of tokens
    """
    if 'text' in doc and len(doc['text']) > 0:
        text = doc['text']
    else:
        # Fallback manual formatting just in case
        instruction = doc.get("instruction", "")
        input_text = doc.get("input", "")
        output = doc.get("output", "")
        if input_text:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"

    tokens = [eot]    # special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    return tokens


def main():
    script_dir = os.path.dirname(__file__)

    # create cache and local dir if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(script_dir, Config.LOCAL_DIR)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(f"Loading {Config.DATASET_REPO} dataset...")
    # Load dataset
    ds = load_dataset(Config.DATASET_REPO, split=Config.SPLIT)
    print(f"Loaded {len(ds)} examples.")

    # init the tokenizer
    enc = tiktoken.get_encoding(Config.TOKENIZER)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token

    all_tokens = []
    
    print("Tokenizing dataset...")
    for doc in ds:
        tokens = tokenize(doc, enc, eot)
        all_tokens.extend(tokens)
        
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens_np):,}")
    
    # Split into 90% train, 10% val
    val_split_idx = int(len(all_tokens_np) * 0.9)
    train_tokens = all_tokens_np[:val_split_idx]
    val_tokens = all_tokens_np[val_split_idx:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save files (following the naming convention DataLoaderLite expects: e.g. contains 'train' or 'val')
    train_filepath = os.path.join(DATA_CACHE_DIR, 'alpaca_train.npy')
    val_filepath = os.path.join(DATA_CACHE_DIR, 'alpaca_val.npy')
    
    print(f"Saving to {train_filepath}...")
    np.save(train_filepath, train_tokens)
    
    print(f"Saving to {val_filepath}...")
    np.save(val_filepath, val_tokens)
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()

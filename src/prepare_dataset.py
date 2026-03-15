"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python prepare_dataset.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Dataset
    DATASET_REPO = 'HuggingFaceFW/fineweb-edu'
    REMOTE_NAME = "sample-10BT"
    SPLIT = 'train'
    STREAMING = True
    
    # Target Sizing
    SHARD_SIZE = int(1e7)       # 10M tokens per shard
    TARGET_SHARDS = 10          # Number of shards to download before stopping (10 * 10M = 100M total)
    
    # Tokenizer
    TOKENIZER = 'gpt2'
    
    # Output Directory
    LOCAL_DIR = "../data/edu_fineweb10B"

# ==========================================

script_dir = os.path.dirname(__file__)

# create cache and local dir if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(script_dir, Config.LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(Config.DATASET_REPO, name=Config.REMOTE_NAME, split=Config.SPLIT, streaming=Config.STREAMING)

# init the tokenizer
enc = tiktoken.get_encoding(Config.TOKENIZER)
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    """ tokenizes a single document and returns a np array of uint16 tokens """
    tokens = [eot]    # special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), 'token dict too large for uint16'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

shard_idx = 0
# preallocate buffer to hold current shard
all_tokens_np = np.empty((Config.SHARD_SIZE,), dtype=np.uint16)
token_count = 0
progress_bar = None

for doc in fw:
    tokens = tokenize(doc)
    # check if there is enough space in current shard for new tokens
    if token_count + len(tokens) < Config.SHARD_SIZE:
        # simply append tokens to current shard
        all_tokens_np[token_count : token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=Config.SHARD_SIZE, unit='tokens', desc=f'shard {shard_idx}')
        progress_bar.update(len(tokens))
    else:
        # write current shard and start a new one
        split = 'val' if shard_idx == 0 else 'train'
        filepath = os.path.join(DATA_CACHE_DIR, f'edufineweb_{split}_{shard_idx:06d}')
        # split the document into whatever fits in this shard, remainder goes to next one
        remainder = Config.SHARD_SIZE - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
        np.save(filepath, all_tokens_np)
        shard_idx += 1
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = None
        
        if shard_idx >= Config.TARGET_SHARDS:
            break
            
        all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

if token_count != 0 and shard_idx < Config.TARGET_SHARDS:
    split = 'val' if shard_idx == 0 else 'train'
    filepath = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_idx:06d}")
    np.save(filepath, all_tokens_np[:token_count])
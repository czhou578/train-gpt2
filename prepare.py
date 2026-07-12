"""
Prepare the TinyStories 10M dataset for nanoGPT.

Expected input:
    datasets/tinystories_10M_1/

Produces:
    train.bin
    val.bin
    test.bin
"""

import os
import numpy as np
import tiktoken
from datasets import load_from_disk

# -----------------------------------------------------------------------------

# Change this path if your dataset lives elsewhere
DATASET_PATH = "datasets/tinystories_10M_1"

# Random seed for deterministic splits
SEED = 42

# -----------------------------------------------------------------------------

print("Loading dataset...")
ds = load_from_disk(DATASET_PATH)

# If the dataset is a DatasetDict with only "train"
if hasattr(ds, "keys") and "train" in ds:
    ds = ds["train"]

print(ds)

# -----------------------------------------------------------------------------
# Create train / validation / test splits
# -----------------------------------------------------------------------------

print("Creating dataset splits...")

ds = ds.shuffle(seed=SEED)

train_test = ds.train_test_split(
    test_size=0.10,
    seed=SEED,
)

train = train_test["train"]
temp = train_test["test"]

val_test = temp.train_test_split(
    test_size=0.5,
    seed=SEED,
)

val = val_test["train"]
test = val_test["test"]

print(f"Train: {len(train):,}")
print(f"Validation: {len(val):,}")
print(f"Test: {len(test):,}")

# -----------------------------------------------------------------------------
# GPT-2 tokenizer
# -----------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

def tokenize(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(eot)
    return {
        "ids": ids,
        "len": len(ids),
    }

print("Tokenizing...")

train = train.map(
    tokenize,
    remove_columns=train.column_names,
    desc="Tokenizing train",
    num_proc=os.cpu_count(),
)

val = val.map(
    tokenize,
    remove_columns=val.column_names,
    desc="Tokenizing validation",
    num_proc=os.cpu_count(),
)

test = test.map(
    tokenize,
    remove_columns=test.column_names,
    desc="Tokenizing test",
    num_proc=os.cpu_count(),
)

# -----------------------------------------------------------------------------
# Write binary files
# -----------------------------------------------------------------------------

def write_bin(dataset, filename):
    print(f"Writing {filename}...")

    arr_len = np.sum(dataset["len"], dtype=np.uint64)

    arr = np.empty(arr_len, dtype=np.uint16)

    idx = 0
    for ids in dataset["ids"]:
        arr[idx:idx + len(ids)] = ids
        idx += len(ids)

    arr.tofile(filename)

    print(f"Saved {filename}")
    print(f"Tokens: {len(arr):,}")

write_bin(train, "train.bin")
write_bin(val, "val.bin")
write_bin(test, "test.bin")

print("Done!")
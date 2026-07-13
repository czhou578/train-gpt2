"""
Prepare the WikiText-103 dataset for nanoGPT.

Expected input:
    datasets/wikitext103/

Produces:
    train_overfit.bin
    val_overfit.bin
    test_overfit.bin
"""

import os
import numpy as np
import tiktoken
from datasets import load_from_disk

# -----------------------------------------------------------------------------

DATASET_PATH = "datasets/wikitext103"

MAX_TRAIN_TOKENS = 20_000
MAX_VAL_TOKENS = 5_000
MAX_TEST_TOKENS = 5_000

# -----------------------------------------------------------------------------

print("Loading dataset...")
ds = load_from_disk(DATASET_PATH)

print(ds)

# -----------------------------------------------------------------------------
# Load official splits
# -----------------------------------------------------------------------------

train = ds["train"]
val = ds["validation"]
test = ds["test"]

print(f"Train examples: {len(train):,}")
print(f"Validation examples: {len(val):,}")
print(f"Test examples: {len(test):,}")

# -----------------------------------------------------------------------------
# Remove empty lines
# -----------------------------------------------------------------------------

def non_empty(example):
    return example["text"].strip() != ""

print("Filtering empty examples...")

train = train.filter(non_empty, num_proc=os.cpu_count())
val = val.filter(non_empty, num_proc=os.cpu_count())
test = test.filter(non_empty, num_proc=os.cpu_count())

print(f"Train after filtering: {len(train):,}")
print(f"Validation after filtering: {len(val):,}")
print(f"Test after filtering: {len(test):,}")

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

print(f"Train tokens: {sum(train['len']):,}")
print(f"Validation tokens: {sum(val['len']):,}")
print(f"Test tokens: {sum(test['len']):,}")

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
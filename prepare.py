"""
Tokenize a raw dataset into the binary files that train.py reads.

Usage:
    python prepare.py tinystories
    python prepare.py wikitext-103
    python prepare.py tinystories --max_examples 20000 --out_dir data/tinystories/overfit

Input:
    The Arrow dataset that download_datasets.py saved under datasets/.

Output:
    data/<dataset>/train.bin, val.bin, and test.bin. Each file is a flat array of
    uint16 GPT-2 token ids. Every document ends with the GPT-2 end-of-text token.

Splits:
    wikitext-103  keeps the official train, validation, and test splits and drops
                  empty lines. --max_examples keeps the first N examples of each split.
    tinystories   has one split only. The script shuffles it with a fixed seed, keeps
                  the first --max_examples examples if given, then splits it 90/5/5.
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_from_disk

DATASETS = {
    "tinystories": {"path": "datasets/tinystories_10M_1", "out_dir": "data/tinystories"},
    "wikitext-103": {"path": "datasets/wikitext103", "out_dir": "data/wikitext-103"},
}
SEED = 42


def non_empty(example) -> bool:
    return example["text"].strip() != ""


def load_splits(name: str, max_examples: int | None) -> dict:
    """Return a dict with the keys train, val, and test."""
    ds = load_from_disk(DATASETS[name]["path"])
    print(ds)

    if name == "wikitext-103":
        splits = {"train": ds["train"], "val": ds["validation"], "test": ds["test"]}
        for key, split in splits.items():
            split = split.filter(non_empty, num_proc=os.cpu_count(), desc=f"Filtering {key}")
            if max_examples:
                split = split.select(range(min(max_examples, len(split))))
            splits[key] = split
        return splits

    # TinyStories ships one split. Shuffle it and cut it 90 / 5 / 5.
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    ds = ds.shuffle(seed=SEED)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    train_rest = ds.train_test_split(test_size=0.10, seed=SEED)
    val_test = train_rest["test"].train_test_split(test_size=0.5, seed=SEED)
    return {"train": train_rest["train"], "val": val_test["train"], "test": val_test["test"]}


def tokenize(dataset, enc, desc: str):
    """Encode every example with the GPT-2 tokenizer and append the end-of-text token."""
    eot = enc.eot_token

    def encode(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    return dataset.map(encode, remove_columns=dataset.column_names, desc=desc, num_proc=os.cpu_count())


def write_bin(dataset, path: str) -> None:
    """Concatenate all token ids of `dataset` into one uint16 file at `path`."""
    arr_len = int(np.sum(dataset["len"], dtype=np.uint64))
    arr = np.empty(arr_len, dtype=np.uint16)
    idx = 0
    for ids in dataset["ids"]:
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    arr.tofile(path)
    print(f"Saved {path}: {len(arr):,} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a dataset into data/<dataset>/*.bin.")
    parser.add_argument("dataset", choices=sorted(DATASETS), help="which dataset to tokenize")
    parser.add_argument("--out_dir", default=None, help="output directory (default: data/<dataset>)")
    parser.add_argument("--max_examples", type=int, default=None, help="keep only the first N examples")
    args = parser.parse_args()

    out_dir = args.out_dir or DATASETS[args.dataset]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    splits = load_splits(args.dataset, args.max_examples)
    for key, split in splits.items():
        print(f"{key}: {len(split):,} examples")

    enc = tiktoken.get_encoding("gpt2")
    for key, split in splits.items():
        tokens = tokenize(split, enc, desc=f"Tokenizing {key}")
        write_bin(tokens, os.path.join(out_dir, f"{key}.bin"))

    print("Done!")


if __name__ == "__main__":
    main()

"""
Download a raw dataset from the Hugging Face Hub and save it to disk.

Usage:
    python download_datasets.py tinystories
    python download_datasets.py wikitext-103

The script writes an Arrow copy of the dataset into datasets/<name>/.
prepare.py reads that copy and tokenizes it into data/<dataset>/*.bin.

Sources:
    tinystories   eminorhan/tinystories, config 10M_1  (about 11M GPT-2 tokens)
    wikitext-103  Salesforce/wikitext, config wikitext-103-raw-v1
"""

import argparse

from datasets import load_dataset

SOURCES = {
    "tinystories": ("eminorhan/tinystories", "10M_1", "datasets/tinystories_10M_1"),
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "datasets/wikitext103"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a raw dataset and save it under datasets/.",
    )
    parser.add_argument("dataset", choices=sorted(SOURCES), help="which dataset to download")
    args = parser.parse_args()

    repo, config, out_dir = SOURCES[args.dataset]
    print(f"Downloading {repo} ({config}) ...")
    ds = load_dataset(repo, config)
    print(ds)
    ds.save_to_disk(out_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

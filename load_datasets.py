from datasets import load_from_disk

ds = load_from_disk("datasets/wikitext103")
print(ds)
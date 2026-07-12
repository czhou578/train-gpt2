from datasets import load_dataset

ds = load_dataset("eminorhan/tinystories", "10M_1")
ds.save_to_disk("datasets/tinystories_10M_1") # ds = load_from_disk("datasets/tinystories_100M_1")

# ds = load_dataset(
#     "Salesforce/wikitext",
#     "wikitext-103-raw-v1"
# )
# ds.save_to_disk("datasets/wikitext103")

# for example in ds["train"].select(range(5)):
#     print(example["text"])
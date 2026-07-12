from datasets import load_from_disk

ds = load_from_disk("datasets/tinystories_10M_1")

train = ds["train"]
val = ds["validation"]

# Create a 5% held-out test set from the training data
train_test = train.train_test_split(
    test_size=0.05,
    seed=42,
    shuffle=True,
)

train = train_test["train"]
test = train_test["test"]

print(len(train))
print(len(val))
print(len(test))
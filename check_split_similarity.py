
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

folder = Path(__file__).resolve().parent
data_file = folder / "aggregation_dataset.csv"

df = pd.read_csv(data_file)
df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

# Use the same random split settings as the model.
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["aggregation_label"],
)

train_sequences = set(train_df["sequence"])
test_sequences = set(test_df["sequence"])

print("NEUROFOLD AI - TRAIN/TEST SIMILARITY CHECK")
print("-" * 45)
print("Training records:", len(train_df))
print("Test records:", len(test_df))

# Count test sequences with a training sequence that differs
# at no more than one amino-acid position.
close_test_sequences = 0
exact_matches = 0

for test_seq in test_sequences:
    for train_seq in train_sequences:
        if len(test_seq) != len(train_seq):
            continue

        differences = sum(
            a != b for a, b in zip(test_seq, train_seq)
        )

        if differences == 0:
            exact_matches += 1
            close_test_sequences += 1
            break

        if differences == 1:
            close_test_sequences += 1
            break

print("Exact matches across splits:", exact_matches)
print(
    "Test sequences with a training sequence differing "
    "at 0 or 1 positions:",
    close_test_sequences,
)
print("Total unique test sequences:", len(test_sequences))
print("\nCheck completed.")

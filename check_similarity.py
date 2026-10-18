
from pathlib import Path
from itertools import combinations
import pandas as pd

folder = Path(__file__).resolve().parent
data_file = folder / "aggregation_dataset.csv"

if not data_file.exists():
    raise FileNotFoundError(
        "aggregation_dataset.csv was not found."
    )

df = pd.read_csv(data_file)
df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

sequences = df["sequence"].dropna().unique().tolist()

print("NEUROFOLD AI - SEQUENCE SIMILARITY CHECK")
print("-" * 45)
print("Dataset records:", len(df))
print("Unique sequences:", len(sequences))

# For equal-length sequences, count pairs that differ
# at no more than one amino-acid position.
if not sequences:
    raise ValueError("No sequences found.")

lengths = {len(seq) for seq in sequences}

if len(lengths) != 1:
    print(
        "Sequences have different lengths. "
        "This simple comparison needs equal-length sequences."
    )
else:
    close_pairs = 0
    total_pairs = 0

    for seq_a, seq_b in combinations(sequences, 2):
        differences = sum(
            a != b for a, b in zip(seq_a, seq_b)
        )
        total_pairs += 1

        if differences <= 1:
            close_pairs += 1

    print("Pairs compared:", total_pairs)
    print(
        "Pairs differing at 0 or 1 positions:",
        close_pairs,
    )
    print(
        "This is a dataset-level check, not yet a "
        "train/test leakage measurement."
    )

print("\nCheck completed.")

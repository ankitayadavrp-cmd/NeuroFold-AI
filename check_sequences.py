
import pandas as pd
from pathlib import Path

file_path = Path("aggregation_dataset.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"{file_path} not found. Current folder: {Path.cwd()}"
    )

df = pd.read_csv(file_path)

print("DATASET CHECK")
print("=" * 40)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head().to_string(index=False))

# Detect the sequence column
possible_columns = [
    col for col in df.columns
    if "sequence" in col.lower() or "peptide" in col.lower()
]

if not possible_columns:
    print("\nSequence column not detected.")
    print("Please share the column names shown above.")
    raise SystemExit

sequence_col = possible_columns[0]
sequences = df[sequence_col].astype("string").str.strip().str.upper()

print("\nSEQUENCE QUALITY")
print("=" * 40)
print("Missing or empty:", int(sequences.isna().sum() + sequences.eq("").sum()))
print("Unique sequences:", sequences.nunique())
print("Duplicate rows:", int(sequences.duplicated().sum()))
print("\nSequence lengths:")
print(sequences.dropna().str.len().value_counts().sort_index())

allowed = set("ACDEFGHIKLMNPQRSTVWY")
valid_sequences = sequences.dropna()
invalid = valid_sequences[
    valid_sequences.apply(lambda s: not set(s).issubset(allowed))
]

print("\nSequences with non-standard amino-acid symbols:", len(invalid))
print("\nCheck complete.")

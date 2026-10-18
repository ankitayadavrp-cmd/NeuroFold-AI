
from pathlib import Path
import csv
from Bio.SeqUtils.ProtParam import ProteinAnalysis

folder = Path(__file__).resolve().parent
input_file = folder / "aggregation_dataset.csv"
output_file = folder / "aggregation_features.csv"

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

if not input_file.exists():
    raise FileNotFoundError(
        "aggregation_dataset.csv not found. "
        "Run prepare_dataset.py first."
    )


def validate_sequence(sequence):
    """Return whether a sequence contains only standard amino acids."""
    if not isinstance(sequence, str):
        return False, "Missing sequence"

    sequence = sequence.strip().upper()

    if not sequence:
        return False, "Empty sequence"

    invalid = sorted(set(sequence) - STANDARD_AMINO_ACIDS)

    if invalid:
        return False, f"Unsupported symbols: {', '.join(invalid)}"

    return True, "Valid"


with input_file.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "record_id", "sequence", "aggregation_label"
    }

    if not reader.fieldnames or not required_columns.issubset(
        set(reader.fieldnames)
    ):
        raise ValueError(
            "Input CSV must contain these columns: "
            "record_id, sequence, aggregation_label"
        )

    rows = list(reader)

feature_rows = []
rejected_rows = []
seen_sequences = set()

for row_number, row in enumerate(rows, start=2):
    record_id = row.get("record_id") or f"row_{row_number}"
    sequence = (row.get("sequence") or "").strip().upper()
    label = (row.get("aggregation_label") or "").strip()

    valid, reason = validate_sequence(sequence)

    if not valid:
        rejected_rows.append((record_id, reason))
        continue

    if label not in {"True", "False", "true", "false", "1", "0"}:
        rejected_rows.append(
            (record_id, f"Invalid or missing label: {label!r}")
        )
        continue

    # Avoid silently treating duplicate records as independent examples.
    # Record duplicates in the report; do not automatically discard them.
    if sequence in seen_sequences:
        print(f"NOTICE: repeated sequence found: {record_id}")
    seen_sequences.add(sequence)

    analysis = ProteinAnalysis(sequence)

    features = {
        "record_id": record_id,
        "sequence": sequence,
        "length": len(sequence),
        "aggregation_label": label,
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy(),
        "aromaticity": analysis.aromaticity(),
    }

    for aa in AMINO_ACIDS:
        features[f"frac_{aa}"] = sequence.count(aa) / len(sequence)

    feature_rows.append(features)

if not feature_rows:
    raise ValueError(
        "No valid records available. Check sequences and labels."
    )

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=list(feature_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(feature_rows)

print("\nNEUROFOLD AI - FEATURE EXTRACTION")
print("-" * 40)
print("Input records:", len(rows))
print("Accepted records:", len(feature_rows))
print("Rejected records:", len(rejected_rows))
print("Features per sequence:", len(feature_rows[0]) - 2)
print("Unique accepted sequences:", len(seen_sequences))
print("Saved to:", output_file)

if rejected_rows:
    print("\nRejected records:")
    for record_id, reason in rejected_rows:
        print(f"  {record_id}: {reason}")

print("\nFeature extraction complete.")


from pathlib import Path
from Bio import SeqIO
import csv

folder = Path(__file__).resolve().parent
fasta_file = folder / "hex1421 (2).fasta"
output_file = folder / "aggregation_dataset.csv"

if not fasta_file.exists():
    print("FASTA file not found.")
    print("Check the filename in your NeuroFold_AI folder.")
    raise SystemExit

rows = []

for record in SeqIO.parse(str(fasta_file), "fasta"):
    header = record.id
    sequence = str(record.seq).upper()

    # Extract the label encoded at the end of the header.
    label_text = header.rsplit("_", 1)[-1].lower()

    if label_text not in {"true", "false"}:
        print("Unexpected label format:", header)
        raise SystemExit

    label = 1 if label_text == "true" else 0

    rows.append({
        "record_id": header,
        "sequence": sequence,
        "length": len(sequence),
        "aggregation_label": label
    })

if not rows:
    print("No sequences were found in the FASTA file.")
    raise SystemExit

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "record_id",
            "sequence",
            "length",
            "aggregation_label"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

positive = sum(row["aggregation_label"] == 1 for row in rows)
negative = sum(row["aggregation_label"] == 0 for row in rows)

print("NEUROFOLD AI - DATASET PREPARATION")
print("-" * 38)
print("Total sequences:", len(rows))
print("Label 1 (header says True):", positive)
print("Label 0 (header says False):", negative)
print("Unique sequences:", len(set(row["sequence"] for row in rows)))
print("CSV saved to:", output_file)
print("\nDataset conversion complete.")
print("Confirm label meanings against the original source before training.")

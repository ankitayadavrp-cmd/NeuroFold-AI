
import csv
from pathlib import Path

file_path = Path(__file__).resolve().parent / "protein_features.csv"

if not file_path.exists():
    print("CSV file not found. Run main.py first.")
else:
    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print("NEUROFOLD AI - DATASET CHECK")
    print("-" * 35)
    print("Number of protein records:", len(rows))

    if rows:
        print("\nAvailable columns:")
        for column in rows[0].keys():
            print("-", column)

        print("\nProtein records:")
        for row in rows:
            print(
                f"\nName: {row['name']}"
                f"\nLength: {row['length']} amino acids"
                f"\nMolecular weight: {row['molecular_weight_Da']} Da"
                f"\nGRAVY: {row['gravy_hydrophobicity']}"
            )

        print("\nDataset inspection complete.")
        print("Note: This file contains features, not")
        print("experimentally verified aggregation labels.")
    else:
        print("The CSV contains no protein records.")

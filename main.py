
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
from pathlib import Path

# NeuroFold AI - Version 0.4
# Protein Feature Extraction

REFERENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def extract_features(sequence, name):
    sequence = sequence.strip().upper()

    if not sequence:
        raise ValueError("Sequence cannot be empty.")

    invalid = set(sequence) - VALID_AMINO_ACIDS
    if invalid:
        raise ValueError(
            f"Invalid amino acid codes: {', '.join(sorted(invalid))}"
        )

    protein = ProteinAnalysis(sequence)
    helix, turn, sheet = protein.secondary_structure_fraction()

    return {
        "name": name,
        "sequence": sequence,
        "length": len(sequence),
        "molecular_weight_Da": round(protein.molecular_weight(), 3),
        "isoelectric_point": round(protein.isoelectric_point(), 3),
        "gravy_hydrophobicity": round(protein.gravy(), 4),
        "aromaticity": round(protein.aromaticity(), 4),
        "instability_index": round(protein.instability_index(), 3),
        "helix_fraction": round(helix, 4),
        "turn_fraction": round(turn, 4),
        "sheet_fraction": round(sheet, 4),
    }


def main():
    print("=" * 46)
    print("       NEUROFOLD AI - VERSION 0.4")
    print("       PROTEIN FEATURE EXTRACTOR")
    print("=" * 46)

    results = [
        extract_features(REFERENCE, "Amyloid_beta_42_reference")
    ]

    print("\nReference Aβ42 features calculated.")

    choice = input(
        "\nWould you like to analyze another sequence? (y/n): "
    ).strip().lower()

    if choice == "y":
        sequence = input("Paste the protein sequence: ")
        name = input("Enter a short sample name: ").strip()

        if not name:
            name = "Custom_protein"

        try:
            results.append(extract_features(sequence, name))
        except ValueError as error:
            print(f"\nInput error: {error}")
            return

    output_file = Path(__file__).resolve().parent / "protein_features.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n--- FEATURE SUMMARY ---")

    for result in results:
        print(f"\nSample: {result['name']}")
        print(f"Length: {result['length']} amino acids")
        print(f"Molecular weight: {result['molecular_weight_Da']} Da")
        print(f"GRAVY: {result['gravy_hydrophobicity']}")
        print(f"Isoelectric point: {result['isoelectric_point']}")

    print(f"\nCSV saved to: {output_file}")
    print("\nScientific note:")
    print("These are calculated sequence features.")
    print("They do not directly measure aggregation or toxicity.")
    print("Secondary-structure fractions are estimates, not")
    print("experimental measurements of the folded protein.")


if __name__ == "__main__":
    main()

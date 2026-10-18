
from pathlib import Path
import joblib
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

folder = Path(__file__).resolve().parent
model_file = folder / "neurofold_model.joblib"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def extract_features(sequence):
    sequence = sequence.strip().upper()

    if len(sequence) != 6:
        raise ValueError(
            "Please enter exactly 6 amino acids."
        )

    if any(aa not in AMINO_ACIDS for aa in sequence):
        raise ValueError(
            "Use only standard amino-acid letters: "
            "ACDEFGHIKLMNPQRSTVWY"
        )

    analysis = ProteinAnalysis(sequence)

    features = {
        "length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy(),
        "aromaticity": analysis.aromaticity(),
    }

    for aa in AMINO_ACIDS:
        features[f"frac_{aa}"] = sequence.count(aa) / len(sequence)

    return features


if not model_file.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

saved = joblib.load(model_file)
model = saved["model"]
feature_columns = saved["feature_columns"]
model_name = saved["model_name"]

print("NEUROFOLD AI - PEPTIDE PREDICTION")
print("-" * 38)
print("Model:", model_name)
print("Research tool for six-amino-acid peptides.")
print("This is an experimental prediction, not a diagnosis.")

while True:
    sequence = input(
        "\nEnter a 6-amino-acid sequence "
        "(or type 'quit' to exit): "
    ).strip().upper()

    if sequence == "QUIT":
        print("NeuroFold AI closed.")
        break

    try:
        features = extract_features(sequence)

        # Use exactly the same feature columns as training.
        input_data = pd.DataFrame(
            [[features.get(col, float("nan"))
              for col in feature_columns]],
            columns=feature_columns,
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        predicted_probability = probabilities[
            classes.index(prediction)
        ]

        print("\nSequence:", sequence)
        print("Predicted class:", prediction)
        print(
            "Model probability for predicted class:",
            f"{predicted_probability:.1%}",
        )

        if prediction == 1:
            print("Dataset label: True")
        else:
            print("Dataset label: False")

        print(
            "\nInterpretation: this is a model prediction. "
            "Confirm the dataset's biological label meanings "
            "and validate the model before drawing conclusions."
        )

    except ValueError as error:
        print("Input error:", error)

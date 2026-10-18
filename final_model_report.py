
import pandas as pd
from pathlib import Path

FILES = {
    "random_cv": "validation_results.csv",
    "similarity_holdout": "similarity_validation_results.csv",
    "feature_importance": "neurofold_feature_importance_comparison.csv",
}

def load_csv(filename):
    path = Path(filename)
    if not path.exists():
        print(f"Not found: {filename}")
        return None
    return pd.read_csv(path)

def main():
    print("=== NeuroFold AI: Research Summary Report ===")

    cv = load_csv(FILES["random_cv"])
    similarity = load_csv(FILES["similarity_holdout"])
    features = load_csv(FILES["feature_importance"])

    lines = [
        "NEUROFOLD AI — PRELIMINARY MODEL REPORT",
        "=" * 48,
        "",
        "Purpose:",
        "Exploratory classification of peptide sequences using",
        "sequence-derived physicochemical features.",
        "",
        "IMPORTANT: Verify dataset provenance and label meaning",
        "before interpreting class 1 as biological aggregation.",
        "",
    ]

    if cv is not None:
        lines += [
            "RANDOM 5-FOLD CROSS-VALIDATION",
            "-" * 35,
            cv.to_string(index=False),
            "",
        ]

    if similarity is not None:
        lines += [
            "SIMILARITY-FILTERED HOLDOUT",
            "-" * 35,
            similarity.to_string(index=False),
            "",
            "This is one filtered holdout, not a definitive estimate.",
            "Its test-set composition differs from random CV.",
            "",
        ]

    if features is not None:
        ranked = features.sort_values("random_forest_rank").head(10)
        lines += [
            "TOP RANDOM FOREST FEATURES",
            "-" * 35,
            ranked[
                [
                    "random_forest_importance",
                    "random_forest_rank",
                ]
            ].to_string(),
            "",
            "Feature importance describes model behavior, not causation.",
            "",
        ]

    lines += [
        "LIMITATIONS",
        "-" * 35,
        "1. Confirm the original dataset source and license.",
        "2. Verify what each class label represents.",
        "3. Similar sequences may lead to optimistic estimates.",
        "4. Feature importance does not establish causation.",
        "5. Predictions require independent experimental validation.",
        "6. This software is a research prototype, not a diagnostic tool.",
        "",
        "Report generated from existing project CSV files.",
    ]

    report = "\n".join(lines)
    Path("neurofold_research_report.txt").write_text(
        report, encoding="utf-8"
    )

    print(report)
    print("\nSaved: neurofold_research_report.txt")

if __name__ == "__main__":
    main()

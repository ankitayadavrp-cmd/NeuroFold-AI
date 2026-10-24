
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# NeuroFold AI: preliminary cross-validation
# This evaluates the existing feature table, not independent
# generalization to unseen sequence-similarity groups.

DATA_FILE = "aggregation_features.csv"
LABEL_COLUMN = "aggregation_label"
SEQUENCE_COLUMN = "sequence"

df = pd.read_csv(DATA_FILE)

print("\n=== NeuroFold AI: Dataset Check ===")
print("Rows:", len(df))
print("Columns:", list(df.columns))

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Could not find '{LABEL_COLUMN}'. "
        "Open aggregation_features.csv and check the label column name."
    )

# Convert labels safely.
def encode_label(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "positive"}:
        return 1
    if text in {"false", "0", "no", "negative"}:
        return 0

    raise ValueError(f"Unrecognized label: {value!r}")

y = df[LABEL_COLUMN].map(encode_label)

# Remove identifiers and non-numeric columns from model features.
excluded = {LABEL_COLUMN, SEQUENCE_COLUMN}
X = df.drop(
    columns=[c for c in excluded if c in df.columns]
)

X = X.select_dtypes(include=[np.number]).copy()
X = X.replace([np.inf, -np.inf], np.nan)

if X.shape[1] == 0:
    raise ValueError(
        "No numeric feature columns found. "
        "Check aggregation_features.csv."
    )

# Remove rows with missing labels and align features.
valid = y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)

if y.nunique() != 2:
    raise ValueError("The dataset must contain two classes.")

if y.value_counts().min() < 5:
    raise ValueError(
        "Too few examples in one class for 5-fold cross-validation."
    )

print("Usable rows:", len(X))
print("Numeric features:", X.shape[1])
print("Class counts:")
print(y.value_counts().sort_index())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),
    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),
    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),
    "roc_auc": "roc_auc"
}

results = []

print("\n=== 5-Fold Cross-Validation ===")

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise"
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

    print(f"\n{name}")
    for metric in scoring:
        mean = row[f"{metric}_mean"]
        std = row[f"{metric}_std"]
        print(f"{metric}: {mean:.3f} +/- {std:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv(
    "validation_results.csv",
    index=False
)

print("\nSaved results to validation_results.csv")
print(
    "\nIMPORTANT: These are random-fold results. "
    "They do not yet address similarity between peptide sequences "
    "in different folds. Do not interpret them as final "
    "independent validation."
)

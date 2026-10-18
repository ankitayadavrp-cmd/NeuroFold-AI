
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_FILE = "aggregation_features.csv"
LABEL_COLUMN = "aggregation_label"
SEQUENCE_COLUMN = "sequence"
RANDOM_STATE = 42
MAX_ALLOWED_DISTANCE = 1

df = pd.read_csv(DATA_FILE)

required = {LABEL_COLUMN, SEQUENCE_COLUMN}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.dropna(subset=[LABEL_COLUMN, SEQUENCE_COLUMN]).copy()
df[SEQUENCE_COLUMN] = (
    df[SEQUENCE_COLUMN].astype(str).str.strip().str.upper()
)

def encode_label(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "positive"}:
        return 1
    if value in {"false", "0", "no", "negative"}:
        return 0

    raise ValueError(f"Unknown label: {value!r}")

df[LABEL_COLUMN] = df[LABEL_COLUMN].map(encode_label)

lengths = df[SEQUENCE_COLUMN].str.len().unique()
if len(lengths) != 1:
    raise ValueError("All sequences must have equal length.")

conflicts = df.groupby(SEQUENCE_COLUMN)[LABEL_COLUMN].nunique()
if (conflicts > 1).any():
    raise ValueError("Duplicate sequences have conflicting labels.")

df = df.drop_duplicates(subset=[SEQUENCE_COLUMN]).reset_index(drop=True)

excluded = {LABEL_COLUMN, SEQUENCE_COLUMN, "record_id"}
feature_columns = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

if not feature_columns:
    raise ValueError("No numeric feature columns were found.")

X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
y = df[LABEL_COLUMN].astype(int)
sequences = df[SEQUENCE_COLUMN].tolist()

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

train_sequences = [sequences[i] for i in train_idx]

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

retained_test = []
nearest_distances = {}

for idx in test_idx:
    seq = sequences[idx]
    nearest = min(
        hamming_distance(seq, train_seq)
        for train_seq in train_sequences
    )

    if nearest > MAX_ALLOWED_DISTANCE:
        retained_test.append(idx)
        nearest_distances[idx] = nearest

if len(retained_test) < 20:
    raise ValueError(
        f"Only {len(retained_test)} test sequences passed the "
        "similarity filter. The holdout may be too small."
    )

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)

X_test = X.iloc[retained_test].reset_index(drop=True)
y_test = y.iloc[retained_test].reset_index(drop=True)

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training and retained test sets must each contain both classes."
    )

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
}

true_labels = y.iloc[retained_test].to_numpy()
prediction_rows = []
metric_rows = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    metric_rows.append(metrics)

    for pos, idx in enumerate(retained_test):
        prediction_rows.append({
            "model": name,
            "sequence": sequences[idx],
            "true_label": int(true_labels[pos]),
            "predicted_label": int(predictions[pos]),
            "predicted_probability_class_1": float(probabilities[pos]),
            "nearest_training_distance": nearest_distances[idx],
        })

    print(f"\n{name}")
    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.3f}")

predictions_df = pd.DataFrame(prediction_rows)
predictions_df.to_csv(
    "similarity_test_predictions.csv",
    index=False,
)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(
    "similarity_validation_results.csv",
    index=False,
)

# Class balance is calculated once per unique retained test sequence.
test_labels_df = pd.DataFrame({
    "sequence": [sequences[i] for i in retained_test],
    "true_label": true_labels,
})

class_counts = test_labels_df["true_label"].value_counts().sort_index()

print("\n=== Test-set diagnostics ===")
print("Unique sequences in dataset:", len(df))
print("Training sequences:", len(train_idx))
print("Initial test sequences:", len(test_idx))
print("Retained test sequences:", len(retained_test))
print("Excluded due to similarity:", len(test_idx) - len(retained_test))
print("Training class counts:")
print(y_train.value_counts().sort_index().to_string())
print("Retained test class counts (0=negative, 1=positive):")
print(class_counts.to_string())

print("\nSaved: similarity_test_predictions.csv")
print("Saved: similarity_validation_results.csv")
print(
    "\nImportant: class 1 is the dataset's positive label. "
    "Verify what the source dataset labels mean before interpreting "
    "these results as biological aggregation predictions."
)

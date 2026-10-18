
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

folder = Path(__file__).resolve().parent
data_file = folder / "aggregation_features.csv"
results_file = folder / "model_results.txt"
model_file = folder / "neurofold_model.joblib"
metadata_file = folder / "neurofold_model_info.txt"

if not data_file.exists():
    raise FileNotFoundError(
        "aggregation_features.csv not found. "
        "Run extract_features.py first."
    )

df = pd.read_csv(data_file)
target = "aggregation_label"

if target not in df.columns:
    raise ValueError("Missing aggregation_label column.")

# Convert True/False or 0/1 labels safely.
label_mapping = {
    "false": 0,
    "true": 1,
    "0": 0,
    "1": 1,
}

df[target] = (
    df[target].astype("string").str.strip().str.lower()
    .map(label_mapping)
)

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(int)

if df[target].nunique() != 2:
    raise ValueError("Expected both label classes: 0 and 1.")

excluded = {target, "record_id", "sequence"}
feature_columns = [
    col for col in df.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_columns].replace(
    [float("inf"), float("-inf")], float("nan")
)
y = df[target]

# Remove features that contain no usable values.
feature_columns = [
    col for col in X.columns if X[col].notna().any()
]
X = X[feature_columns]

if X.shape[1] == 0:
    raise ValueError("No usable numeric features found.")

usable_rows = ~X.isna().all(axis=1)
X = X.loc[usable_rows]
y = y.loc[usable_rows]

if y.value_counts().min() < 2:
    raise ValueError("Not enough examples in each class.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    ),
}

report_lines = [
    "NEUROFOLD AI - MODEL EVALUATION",
    "=" * 42,
    f"Dataset rows used: {len(X)}",
    f"Numeric features: {len(feature_columns)}",
    f"Training rows: {len(X_train)}",
    f"Test rows: {len(X_test)}",
    "",
    "WARNING: Random-split evaluation is preliminary.",
    "Related peptide sequences may occur across both sets.",
    "The selected model is not clinically validated.",
    "",
]

model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    model_scores[name] = auc

    report_lines.extend([
        "-" * 42,
        name,
        f"Accuracy: {accuracy_score(y_test, predictions):.3f}",
        (
            "Balanced accuracy: "
            f"{balanced_accuracy_score(y_test, predictions):.3f}"
        ),
        f"ROC-AUC: {auc:.3f}",
        "Confusion matrix (actual rows, predicted columns):",
        str(confusion_matrix(y_test, predictions, labels=[0, 1])),
        "Classification report:",
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        ),
        "",
    ])

# Select the model with the higher ROC-AUC on this test split.
best_name = max(model_scores, key=model_scores.get)
best_model = models[best_name]

# Save the full fitted pipeline and feature order.
joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_name,
        "positive_label": 1,
        "negative_label": 0,
    },
    model_file,
)

metadata_file.write_text(
    "NeuroFold AI model information\n"
    f"Selected model: {best_name}\n"
    f"Test ROC-AUC: {model_scores[best_name]:.3f}\n"
    f"Number of features: {len(feature_columns)}\n"
    "Labels: 1 = header True; 0 = header False.\n"
    "Biological label meanings must be confirmed against the source.\n"
    "Evaluation uses a random split and is preliminary.\n",
    encoding="utf-8",
)

report_lines.extend([
    "-" * 42,
    f"Selected model: {best_name}",
    f"Selected test ROC-AUC: {model_scores[best_name]:.3f}",
    f"Saved model to: {model_file}",
    f"Saved model information to: {metadata_file}",
])

report_text = "\n".join(report_lines)
results_file.write_text(report_text, encoding="utf-8")

print(report_text)
print("\nTraining and model saving completed.")

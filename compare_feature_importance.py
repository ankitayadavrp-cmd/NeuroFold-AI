
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

DATA_FILE = "aggregation_features.csv"
MODEL_FILE = "neurofold_model.joblib"

def main():
    print("=== NeuroFold AI: Feature Importance Comparison ===")

    df = pd.read_csv(DATA_FILE)
    saved = joblib.load(MODEL_FILE)

    if not isinstance(saved, dict) or "model" not in saved:
        raise ValueError("Unexpected saved model format.")

    features = saved["feature_columns"]
    label_column = "aggregation_label"

    df = df.dropna(subset=[label_column]).copy()
    X = df[features].replace([float("inf"), float("-inf")], float("nan"))
    y = df[label_column].astype(int)

    # Logistic Regression: absolute coefficient magnitude.
    lr_pipeline = saved["model"]
    lr_estimator = lr_pipeline.steps[-1][1]

    if not hasattr(lr_estimator, "coef_"):
        raise ValueError("Saved model is not a fitted linear model.")

    lr_importance = pd.Series(
        abs(lr_estimator.coef_[0]),
        index=features,
        name="logistic_absolute_coefficient",
    )

    # Fit a separate Random Forest for comparison.
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )

    rf.fit(X, y)

    rf_estimator = rf.steps[-1][1]

    rf_importance = pd.Series(
        rf_estimator.feature_importances_,
        index=features,
        name="random_forest_importance",
    )

    comparison = pd.concat(
        [lr_importance, rf_importance], axis=1
    )

    comparison["logistic_rank"] = (
        comparison["logistic_absolute_coefficient"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["random_forest_rank"] = (
        comparison["random_forest_importance"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["rank_difference"] = (
        comparison["logistic_rank"] - comparison["random_forest_rank"]
    )

    comparison = comparison.sort_values("logistic_rank")
    comparison.to_csv("neurofold_feature_importance_comparison.csv")

    print("\nTop 10 by Logistic Regression:")
    print(comparison.head(10).to_string())

    print("\nTop 10 by Random Forest:")
    print(
        comparison.sort_values("random_forest_rank")
        .head(10)
        .to_string()
    )

    print("\nSaved: neurofold_feature_importance_comparison.csv")
    print(
        "\nCaution: this Random Forest was fitted on the full dataset "
        "for exploratory interpretation, not independent evaluation. "
        "Its importance scores are not directly comparable in scale "
        "to Logistic Regression coefficients."
    )

if __name__ == "__main__":
    main()

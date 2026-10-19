
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed.cleveland.data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Place processed.cleveland.data in the data folder."
        )

    df = pd.read_csv(DATA_FILE, names=COLUMNS, na_values="?")
    df = df.dropna(subset=["num"]).copy()

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target"] = (df["num"] > 0).astype(int)
    X = df[NUMERIC + CATEGORICAL]
    y = df["target"]

    print("Dataset shape:", df.shape)
    print("\nMissing values before imputation:")
    print(X.isna().sum())
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())
    return X, y


def make_pipeline(classifier):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, class_weight="balanced"
        ),
    }

    tracking_db = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )
    mlflow.set_experiment("heart-disease-classification")

    results = []
    best_auc = -1
    best_name = None
    best_model = None

    for name, classifier in candidates.items():
        model = make_pipeline(classifier)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("preprocessing", "impute_scale_onehot")
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

        results.append({"model": name, **metrics})
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    joblib.dump(best_model, MODEL_DIR / "heart_model.joblib")
    (MODEL_DIR / "feature_schema.txt").write_text(
        "\n".join(NUMERIC + CATEGORICAL), encoding="utf-8"
    )

    print("\n=== Model comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nSelected model by test ROC-AUC: {best_name}")
    print(f"Selected test ROC-AUC: {best_auc:.3f}")
    print(f"Saved model: {MODEL_DIR / 'heart_model.joblib'}")
    print(f"Saved comparison: {ROOT / 'reports' / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()

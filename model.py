import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

DATA_PATH = "patient_data.csv"
MODEL_OUT = "diagnosis_model.joblib"
SCALER_OUT = "scaler.joblib"
REPORT_DIR = "reports"

NUMERIC_COLS = ["age", "bmi", "bp_systolic", "cholesterol", "glucose", "heart_rate"]
BINARY_COLS = ["sex", "fever", "cough", "fatigue", "chest_pain",
               "shortness_of_breath", "headache", "nausea", "joint_pain"]
TARGET_COL = "diagnosis"


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run generate_synthetic_data.py to create a "
            f"test file, or provide a real (de-identified, consented) dataset."
        )
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df[TARGET_COL].value_counts())
    return df


def preprocess(df):
    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[NUMERIC_COLS]),
        columns=[f"{c}_scaled" for c in NUMERIC_COLS],
    )

    X = pd.concat([numeric_scaled, df[BINARY_COLS].reset_index(drop=True)], axis=1)
    feature_cols = list(X.columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])

    return X, y, feature_cols, scaler, label_encoder


def train_and_evaluate(X, y, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    os.makedirs(REPORT_DIR, exist_ok=True)
    class_names = label_encoder.classes_

    for name, model in models.items():
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(classification_report(y_test, preds, target_names=class_names, digits=4))
        print(f"Accuracy: {acc:.4f}  |  Macro F1: {f1:.4f}")

        results[name] = {"model": model, "accuracy": acc, "f1": f1, "preds": preds}

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (Macro F1 = {results[best_name]['f1']:.4f})")

    if "Random Forest" in results:
        rf = results["Random Forest"]["model"]
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(7, 6))
        importances.head(15).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/feature_importance.png")
        plt.close()

    return best_model, best_name, results


def main():
    df = load_data()
    X, y, feature_cols, scaler, label_encoder = preprocess(df)
    best_model, best_name, results = train_and_evaluate(X, y, label_encoder)

    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(
        {"scaler": scaler, "feature_cols": feature_cols,
         "label_encoder": label_encoder,
         "numeric_cols": NUMERIC_COLS, "binary_cols": BINARY_COLS},
        SCALER_OUT,
    )
    print(f"\nSaved best model ({best_name}) to '{MODEL_OUT}'")
    print(f"Saved scaler/label metadata to '{SCALER_OUT}'")
    print(f"Plots saved in '{REPORT_DIR}/'")


if __name__ == "__main__":
    main()
"""
Train a compact Random Forest for maternal pregnancy health risk prediction.
Saves model + label encoder + feature names for Streamlit deployment.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "maternal_health_risk.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET_COL = "RiskLevel"


def main():
    df = pd.read_csv(DATA)
    # Clean rare heart-rate outlier sometimes present in this dataset
    df = df[(df["HeartRate"] >= 40) & (df["HeartRate"] <= 120)].copy()

    X = df[FEATURE_COLS]
    y_raw = df[TARGET_COL].str.strip().str.lower()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compact RF: enough trees for stable SHAP, small enough for GitHub
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("Classes:", list(le.classes_))
    print(f"Test accuracy: {acc:.4f}")
    print(f"5-fold CV accuracy: {cv.mean():.4f} ± {cv.std():.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "model": model,
        "label_encoder": le,
        "feature_names": FEATURE_COLS,
        "metrics": {
            "test_accuracy": float(acc),
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
        },
    }
    out = MODEL_DIR / "pregnancy_risk_model.joblib"
    joblib.dump(artifact, out)
    size_kb = out.stat().st_size / 1024
    print(f"\nSaved {out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

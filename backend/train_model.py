"""
train_model.py

Trains a logistic regression model to predict dismissal probability
on a given ball, based on:
  - bowling_type, bowling_arm, bowling_detail, phase (categorical)
  - the batter's historical dismissal tendencies (computed from
    TRAIN data only, to avoid leakage)

This lets the app estimate risk for matchups even when a specific
batter-vs-bowler pairing is rare or has never happened, by falling
back to the batter's broader tendencies against that bowling style.

Split strategy: time-based (train on all seasons except the most
recent, test on the most recent season). This is more honest than a
random split since it mimics predicting "future" matches.

Input:  data/processed/deliveries_with_styles.csv
Output: backend/models/dismissal_model.pkl
        backend/models/batter_lookup.pkl   (stats needed at inference time)

Run from project root:
    python3 backend/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

INPUT_PATH = Path("data/processed/deliveries_with_styles.csv")
MODEL_DIR = Path("backend/models")
MODEL_PATH = MODEL_DIR / "dismissal_model.pkl"
LOOKUP_PATH = MODEL_DIR / "batter_lookup.pkl"

NON_BOWLER_DISMISSALS = {
    "run out",
    "retired hurt",
    "retired out",
    "retired not out",
    "obstructing the field",
    "timed out",
}

CATEGORICAL_FEATURES = ["bowling_type", "bowling_arm", "bowling_detail", "phase"]
NUMERIC_FEATURES = ["batter_overall_dismissal_rate", "batter_vs_type_dismissal_rate"]
GLOBAL_FALLBACK_RATE = None  # computed at runtime from train data


def load_data() -> pd.DataFrame:
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["is_legal_delivery"] = df["wides"].fillna(0) == 0
    wicket_kind_clean = df["wicket_kind"].fillna("").str.strip().str.lower()
    df["bowler_dismissal"] = (
        (df["is_wicket"] == 1) & (~wicket_kind_clean.isin(NON_BOWLER_DISMISSALS))
    ).astype(int)
    df = df[df["is_legal_delivery"]].copy()
    print(f"  {len(df):,} legal deliveries loaded")
    return df


def time_based_split(df: pd.DataFrame):
    seasons = sorted(df["season"].astype(str).unique())
    test_season = seasons[-1]
    print(f"\nSeasons available: {seasons}")
    print(f"Using {test_season} as the held-out test season")

    train = df[df["season"].astype(str) != test_season].copy()
    test = df[df["season"].astype(str) == test_season].copy()
    print(f"Train: {len(train):,} balls | Test: {len(test):,} balls")
    return train, test


def add_batter_history_features(train: pd.DataFrame, test: pd.DataFrame):
    """Compute batter dismissal-rate features from TRAIN data only, then
    map onto both train and test (test batters/combos unseen in train
    fall back to global / batter-level averages)."""

    global_rate = train["bowler_dismissal"].mean()
    print(f"\nGlobal dismissal rate (train): {global_rate:.4f}")

    # Overall dismissal rate per batter
    batter_overall = train.groupby("batter")["bowler_dismissal"].mean()

    # Dismissal rate per batter vs bowling_type
    batter_vs_type = train.groupby(["batter", "bowling_type"])["bowler_dismissal"].mean()

    def apply_features(part: pd.DataFrame) -> pd.DataFrame:
        part = part.copy()
        part["batter_overall_dismissal_rate"] = (
            part["batter"].map(batter_overall).fillna(global_rate)
        )
        key = list(zip(part["batter"], part["bowling_type"]))
        part["batter_vs_type_dismissal_rate"] = (
            pd.Series(key, index=part.index)
            .map(batter_vs_type)
            .fillna(part["batter_overall_dismissal_rate"])
        )
        return part

    train_feat = apply_features(train)
    test_feat = apply_features(test)

    lookup = {
        "global_rate": global_rate,
        "batter_overall": batter_overall,
        "batter_vs_type": batter_vs_type,
    }
    return train_feat, test_feat, lookup


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )
    return pipeline


def evaluate(pipeline: Pipeline, test: pd.DataFrame):
    X_test = test[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test["bowler_dismissal"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n--- Evaluation (test season) ---")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["not out", "dismissed"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    df = load_data()
    train, test = time_based_split(df)
    train, test, lookup = add_batter_history_features(train, test)

    pipeline = build_pipeline()

    X_train = train[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train["bowler_dismissal"]

    print("\nTraining logistic regression ...")
    pipeline.fit(X_train, y_train)
    print("Done.")

    evaluate(pipeline, test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(lookup, LOOKUP_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved batter lookup tables to {LOOKUP_PATH}")


if __name__ == "__main__":
    main()

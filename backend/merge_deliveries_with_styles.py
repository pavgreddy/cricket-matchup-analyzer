"""
merge_deliveries_with_styles.py

Joins bowling style (arm, type, detail) onto every row of deliveries.csv,
producing the final dataset ready for feature engineering.

Run from project root:
    python backend/merge_deliveries_with_styles.py
"""

import pandas as pd

DELIVERIES_PATH = "data/processed/deliveries.csv"
STYLES_PATH = "data/processed/bowler_styles_normalized.csv"
OUT_PATH = "data/processed/deliveries_with_styles.csv"


def main():
    deliveries = pd.read_csv(DELIVERIES_PATH, low_memory=False)
    styles = pd.read_csv(STYLES_PATH)

    styles_slim = styles[["bowler", "bowling_arm", "bowling_type", "bowling_detail"]]

    merged = deliveries.merge(styles_slim, on="bowler", how="left")

    # sanity check: every row should have gotten a style (even if "Unknown")
    missing = merged["bowling_type"].isna().sum()

    merged.to_csv(OUT_PATH, index=False)

    print(f"Merged deliveries: {len(merged)} rows")
    print(f"Rows with no style match at all: {missing}")
    print(f"Saved to {OUT_PATH}")
    print("\nDeliveries by bowling_type:")
    print(merged["bowling_type"].value_counts())
    print("\nSample:")
    print(merged[["bowler", "bowling_arm", "bowling_type", "bowling_detail", "phase"]].head(5).to_string())


if __name__ == "__main__":
    main()

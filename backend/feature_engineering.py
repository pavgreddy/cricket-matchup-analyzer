"""
feature_engineering.py

Builds the core matchup stats table that powers the app:
  For every (batter, bowling_type, phase) combination, computes
  strike rate, dismissal rate, and boundary % from ball-by-ball data.

Input:  data/processed/deliveries_with_styles.csv
Output: data/processed/batter_matchup_stats.csv

Run from project root:
    python3 backend/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/deliveries_with_styles.csv")
OUTPUT_PATH = Path("data/processed/batter_matchup_stats.csv")

# Wicket kinds that should NOT count against the batter in a bowler matchup
# (these are not "the bowler got the batter out" events)
NON_BOWLER_DISMISSALS = {
    "run out",
    "retired hurt",
    "retired out",
    "retired not out",
    "obstructing the field",
    "timed out",
}

# Minimum balls faced for a matchup to be considered statistically meaningful
MIN_BALLS_FOR_CONFIDENCE = 30


def load_data() -> pd.DataFrame:
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # A "legal delivery" (ball actually faced by the batter) excludes wides.
    # No-balls are still faced by the batter, so they count.
    df["is_legal_delivery"] = df["wides"].fillna(0) == 0

    df["is_boundary"] = df["runs_batter"].isin([4, 6])

    wicket_kind_clean = df["wicket_kind"].fillna("").str.strip().str.lower()
    df["bowler_dismissal"] = (df["is_wicket"] == 1) & (
        ~wicket_kind_clean.isin(NON_BOWLER_DISMISSALS)
    )

    return df


def build_matchup_table(df: pd.DataFrame) -> pd.DataFrame:
    legal = df[df["is_legal_delivery"]].copy()
    print(f"  {len(legal):,} legal deliveries (excl. wides) used for stats")

    grouped = legal.groupby(["batter", "bowling_type", "phase"], as_index=False).agg(
        balls_faced=("batter", "size"),
        runs_scored=("runs_batter", "sum"),
        dismissals=("bowler_dismissal", "sum"),
        boundaries=("is_boundary", "sum"),
    )

    grouped["strike_rate"] = (
        grouped["runs_scored"] / grouped["balls_faced"] * 100
    ).round(2)

    # balls per dismissal (classic "average" style stat) -- inf if 0 dismissals
    grouped["balls_per_dismissal"] = np.where(
        grouped["dismissals"] > 0,
        (grouped["balls_faced"] / grouped["dismissals"]).round(1),
        np.nan,
    )

    grouped["dismissal_pct"] = (
        grouped["dismissals"] / grouped["balls_faced"] * 100
    ).round(2)

    grouped["boundary_pct"] = (
        grouped["boundaries"] / grouped["balls_faced"] * 100
    ).round(2)

    grouped["sufficient_sample"] = grouped["balls_faced"] >= MIN_BALLS_FOR_CONFIDENCE

    grouped = grouped.sort_values(
        ["batter", "bowling_type", "phase"]
    ).reset_index(drop=True)

    return grouped


def print_summary(table: pd.DataFrame, raw: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Unique batters: {table['batter'].nunique():,}")
    print(f"Total matchup rows: {len(table):,}")
    print(
        f"Rows with sufficient sample (>= {MIN_BALLS_FOR_CONFIDENCE} balls): "
        f"{table['sufficient_sample'].sum():,} "
        f"({table['sufficient_sample'].mean()*100:.1f}%)"
    )

    print("\nBreakdown by phase:")
    print(table.groupby("phase")["balls_faced"].sum().to_string())

    print("\nBreakdown by bowling_type:")
    print(table.groupby("bowling_type")["balls_faced"].sum().to_string())

    print("\nTop 5 toughest matchups (min 50 balls, lowest strike rate):")
    tough = table[table["balls_faced"] >= 50].nsmallest(5, "strike_rate")
    print(
        tough[
            ["batter", "bowling_type", "phase", "balls_faced", "strike_rate", "dismissal_pct"]
        ].to_string(index=False)
    )


def main():
    df = load_data()
    df = add_derived_columns(df)
    table = build_matchup_table(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved matchup table to {OUTPUT_PATH} ({len(table):,} rows)")

    print_summary(table, df)


if __name__ == "__main__":
    main()

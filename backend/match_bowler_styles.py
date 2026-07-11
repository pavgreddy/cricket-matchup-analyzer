"""
match_bowler_styles.py

Joins the unique list of bowlers in deliveries.csv against player_meta.csv
(bowling style lookup) on the 'unique_name' column, and reports the match
rate. Unmatched bowlers are saved to a separate CSV for manual review.

Run from project root:
    python backend/match_bowler_styles.py
"""

import pandas as pd

DELIVERIES_PATH = "data/processed/deliveries.csv"
META_PATH = "data/processed/player_meta.csv"
OUT_MATCHED = "data/processed/bowler_styles.csv"
OUT_UNMATCHED = "data/processed/bowlers_unmatched.csv"


def main():
    deliveries = pd.read_csv(DELIVERIES_PATH, low_memory=False)
    meta = pd.read_csv(META_PATH)

    # how many balls each bowler has bowled — used to prioritize which
    # unmatched names are worth fixing manually
    bowler_counts = deliveries["bowler"].value_counts().reset_index()
    bowler_counts.columns = ["bowler", "balls_bowled"]

    # keep only the columns we actually need from player_meta
    meta_slim = meta[["unique_name", "bowling_style", "batting_style", "playing_role"]].drop_duplicates(
        subset="unique_name"
    )

    merged = bowler_counts.merge(
        meta_slim, left_on="bowler", right_on="unique_name", how="left"
    )

    matched = merged[merged["bowling_style"].notna()]
    unmatched = merged[merged["bowling_style"].isna()].sort_values(
        "balls_bowled", ascending=False
    )

    total_balls = bowler_counts["balls_bowled"].sum()
    matched_balls = matched["balls_bowled"].sum()

    print(f"Total unique bowlers: {len(bowler_counts)}")
    print(f"Matched: {len(matched)} ({len(matched) / len(bowler_counts) * 100:.1f}%)")
    print(f"Unmatched: {len(unmatched)} ({len(unmatched) / len(bowler_counts) * 100:.1f}%)")
    print()
    print(f"Total deliveries: {total_balls}")
    print(f"Deliveries with a matched bowler style: {matched_balls} "
          f"({matched_balls / total_balls * 100:.1f}%)")

    matched[["bowler", "bowling_style", "batting_style", "playing_role", "balls_bowled"]].to_csv(
        OUT_MATCHED, index=False
    )
    unmatched[["bowler", "balls_bowled"]].to_csv(OUT_UNMATCHED, index=False)

    print(f"\nSaved matched bowlers to {OUT_MATCHED}")
    print(f"Saved unmatched bowlers to {OUT_UNMATCHED}")

    print("\nTop 20 unmatched bowlers by balls bowled (worth fixing manually):")
    print(unmatched[["bowler", "balls_bowled"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

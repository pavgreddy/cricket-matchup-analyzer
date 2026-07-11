"""
finalize_bowler_styles.py

Combines the auto-matched bowler styles (bowler_styles.csv) with your
manually-filled overrides (manual_overrides.csv) into one final lookup
table. Any bowler still missing a style after this gets bucketed as
"Unknown" — fine, since these are all low-ball-count players.

Run from project root:
    python backend/finalize_bowler_styles.py
"""

import pandas as pd

MATCHED_PATH = "data/processed/bowler_styles.csv"
UNMATCHED_PATH = "data/processed/bowlers_unmatched.csv"
OVERRIDES_PATH = "data/manual_overrides.csv"  # you fill this in by hand
OUT_PATH = "data/processed/bowler_styles_final.csv"


def main():
    matched = pd.read_csv(MATCHED_PATH)
    unmatched = pd.read_csv(UNMATCHED_PATH)
    overrides = pd.read_csv(OVERRIDES_PATH)

    # only keep override rows that were actually filled in
    overrides_filled = overrides[overrides["bowling_style"].notna()].copy()
    overrides_filled["balls_bowled"] = overrides_filled["bowler"].map(
        unmatched.set_index("bowler")["balls_bowled"]
    )

    combined = pd.concat([matched, overrides_filled], ignore_index=True)

    # anyone still missing (unmatched AND not overridden) becomes "Unknown"
    still_missing = unmatched[~unmatched["bowler"].isin(combined["bowler"])].copy()
    still_missing["bowling_style"] = "Unknown"
    still_missing["batting_style"] = "Unknown"
    still_missing["playing_role"] = "Unknown"

    final = pd.concat([combined, still_missing], ignore_index=True)
    final = final.sort_values("balls_bowled", ascending=False)

    final.to_csv(OUT_PATH, index=False)

    print(f"Final bowler style table: {len(final)} bowlers")
    print(f"  - Auto-matched: {len(matched)}")
    print(f"  - Manually filled: {len(overrides_filled)}")
    print(f"  - Marked Unknown: {len(still_missing)}")
    print(f"\nSaved to {OUT_PATH}")
    print("\nBowling style distribution:")
    print(final["bowling_style"].value_counts())


if __name__ == "__main__":
    main()

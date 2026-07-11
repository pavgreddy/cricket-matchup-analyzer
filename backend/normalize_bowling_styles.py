"""
normalize_bowling_styles.py

Cleans up the raw bowling_style strings (inconsistent capitalization,
some bowlers listing multiple styles) into three simple, ML-ready columns:
  - bowling_arm:   Right / Left / Unknown
  - bowling_type:  Pace / Spin / Unknown
  - bowling_detail: Fast / Fast-medium / Medium-fast / Medium
                     (for pace) or Offbreak / Legbreak / Left-arm orthodox /
                     Left-arm wrist spin (for spin)

Where a bowler has multiple styles listed (e.g. an all-rounder who bowls
both pace and off-spin), we take the FIRST style listed — Cricinfo lists
a player's primary/most-used style first.

Run from project root:
    python backend/normalize_bowling_styles.py
"""

import re
import pandas as pd

IN_PATH = "data/processed/bowler_styles_final.csv"
OUT_PATH = "data/processed/bowler_styles_normalized.csv"


def normalize_one_style(raw: str):
    """Takes one raw style string, returns (arm, type, detail)."""
    if not isinstance(raw, str) or raw.strip() == "" or raw == "Unknown":
        return "Unknown", "Unknown", "Unknown"

    # if multiple styles listed, take the first one
    primary = raw.split(",")[0].strip()
    text = primary.lower()

    # normalize dashes/spaces: "right-arm" and "right arm" both become "right arm"
    text = re.sub(r"[-\s]+", " ", text)

    # --- arm ---
    if "left" in text:
        arm = "Left"
    elif "right" in text:
        arm = "Right"
    else:
        arm = "Unknown"

    # --- type + detail ---
    # Note: plain "offbreak" and "legbreak/googly" are conventionally
    # right-arm deliveries (the left-arm equivalents have their own names:
    # "orthodox" for left-arm off-spin, "wrist spin"/"chinaman" for
    # left-arm leg-spin) — so default arm to Right if not otherwise stated.
    if "orthodox" in text:
        return arm, "Spin", "Left-arm orthodox"
    if "wrist" in text or "chinaman" in text:
        return arm, "Spin", "Left-arm wrist spin"
    if "offbreak" in text or "off break" in text:
        return (arm if arm != "Unknown" else "Right"), "Spin", "Offbreak"
    if "legbreak" in text or "leg break" in text or "googly" in text:
        return (arm if arm != "Unknown" else "Right"), "Spin", "Legbreak"

    if "fast medium" in text or "fast-medium" in text:
        return arm, "Pace", "Fast-medium"
    if "medium fast" in text:
        return arm, "Pace", "Medium-fast"
    if "fast" in text:
        return arm, "Pace", "Fast"
    if "medium" in text:
        return arm, "Pace", "Medium"

    return arm, "Unknown", "Unknown"


def main():
    df = pd.read_csv(IN_PATH)

    normalized = df["bowling_style"].apply(normalize_one_style)
    df["bowling_arm"] = normalized.apply(lambda x: x[0])
    df["bowling_type"] = normalized.apply(lambda x: x[1])
    df["bowling_detail"] = normalized.apply(lambda x: x[2])

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved to {OUT_PATH}\n")
    print("bowling_arm distribution:")
    print(df["bowling_arm"].value_counts())
    print("\nbowling_type distribution:")
    print(df["bowling_type"].value_counts())
    print("\nbowling_detail distribution:")
    print(df["bowling_detail"].value_counts())


if __name__ == "__main__":
    main()

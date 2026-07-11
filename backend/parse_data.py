"""
parse_data.py

Reads all Cricsheet IPL JSON files from data/raw/, flattens them into
one row per delivery (ball), and saves the result as a single CSV in
data/processed/deliveries.csv

Run this from the project root:
    python backend/parse_data.py
(adjust the RAW_DIR / OUT_PATH below if your folders differ)
"""

import json
import os
import glob
import pandas as pd

# ---- CONFIG: adjust these paths if your folder names are different ----
RAW_DIR = "data/raw"
OUT_PATH = "data/processed/deliveries.csv"


def get_phase(over_number: int) -> str:
    """T20 phase of innings, based on over number (0-indexed)."""
    if over_number <= 5:
        return "powerplay"
    elif over_number <= 14:
        return "middle"
    else:
        return "death"


def parse_match(filepath: str) -> list[dict]:
    """
    Reads a single Cricsheet JSON file and returns a list of dicts,
    one dict per delivery (ball bowled).
    """
    with open(filepath, "r") as f:
        match = json.load(f)

    info = match.get("info", {})
    match_id = os.path.splitext(os.path.basename(filepath))[0]

    # Match-level fields that are the same for every row from this match
    season = info.get("season")
    date = info.get("dates", [None])[0]
    venue = info.get("venue")
    city = info.get("city")
    match_type = info.get("match_type")
    teams = info.get("teams", [])

    rows = []

    for innings_index, innings in enumerate(match.get("innings", []), start=1):
        batting_team = innings.get("team")
        # the other team in `teams` is the bowling team
        bowling_team = next((t for t in teams if t != batting_team), None)

        for over in innings.get("overs", []):
            over_number = over.get("over")
            phase = get_phase(over_number)

            for ball_index, delivery in enumerate(over.get("deliveries", []), start=1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                row = {
                    "match_id": match_id,
                    "season": season,
                    "date": date,
                    "venue": venue,
                    "city": city,
                    "match_type": match_type,
                    "innings": innings_index,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_number,
                    "ball_in_over": ball_index,
                    "phase": phase,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "non_striker": delivery.get("non_striker"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "wides": extras.get("wides", 0),
                    "noballs": extras.get("noballs", 0),
                    "byes": extras.get("byes", 0),
                    "legbyes": extras.get("legbyes", 0),
                    "is_wicket": 1 if wickets else 0,
                    "wicket_kind": wickets[0].get("kind") if wickets else None,
                    "player_out": wickets[0].get("player_out") if wickets else None,
                }
                rows.append(row)

    return rows


def main():
    json_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files in {RAW_DIR}")

    all_rows = []
    failed_files = []

    for i, filepath in enumerate(json_files, start=1):
        try:
            all_rows.extend(parse_match(filepath))
        except Exception as e:
            failed_files.append((filepath, str(e)))

        if i % 200 == 0:
            print(f"  processed {i}/{len(json_files)} files...")

    df = pd.DataFrame(all_rows)
    print(f"\nParsed {len(df)} deliveries from {len(json_files) - len(failed_files)} matches.")

    if failed_files:
        print(f"\n{len(failed_files)} files failed to parse:")
        for fp, err in failed_files[:10]:
            print(f"  {fp}: {err}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")

    # quick sanity check preview
    print("\nSample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()

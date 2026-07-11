"""
main.py

FastAPI backend for the Cricket Matchup Analyzer.

Endpoints:
  GET /                          -> health check
  GET /batters                   -> list of all batter names (for dropdown)
  GET /matchup/{batter}          -> matchup stats table for one batter
  GET /risk                      -> predicted dismissal risk for a given
                                     bowling_type/arm/detail/phase combo

Run from project root:
    uvicorn backend.main:app --reload
"""

from pathlib import Path
from typing import Optional
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/processed/batter_matchup_stats.csv")
MODEL_PATH = Path("backend/models/dismissal_model.pkl")
LOOKUP_PATH = Path("backend/models/batter_lookup.pkl")

VALID_BOWLING_TYPES = {"Pace", "Spin", "Unknown"}
VALID_PHASES = {"powerplay", "middle", "death"}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Cricket Matchup Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit frontend runs on a different origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data + model once at startup
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    raise RuntimeError(
        f"{DATA_PATH} not found. Run backend/feature_engineering.py first."
    )
matchup_df = pd.read_csv(DATA_PATH)

model_pipeline = None
batter_lookup = None
if MODEL_PATH.exists() and LOOKUP_PATH.exists():
    model_pipeline = joblib.load(MODEL_PATH)
    batter_lookup = joblib.load(LOOKUP_PATH)
else:
    print(f"WARNING: model artifacts not found at {MODEL_PATH}. /risk will be disabled.")

ALL_BATTERS = sorted(matchup_df["batter"].unique().tolist())


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class MatchupRow(BaseModel):
    bowling_type: str
    phase: str
    balls_faced: int
    runs_scored: int
    dismissals: int
    strike_rate: float
    balls_per_dismissal: Optional[float]
    dismissal_pct: float
    boundary_pct: float
    sufficient_sample: bool


class RiskResponse(BaseModel):
    batter: str
    bowling_type: str
    bowling_arm: str
    bowling_detail: str
    phase: str
    dismissal_risk_pct: float
    note: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "batters_loaded": len(ALL_BATTERS),
        "model_loaded": model_pipeline is not None,
    }


@app.get("/batters", response_model=list[str])
def get_batters(search: Optional[str] = Query(None, description="Optional substring filter")):
    if search:
        # Cricsheet names are "Initial Surname" (e.g. "V Kohli"), but people
        # naturally search by full name ("Virat Kohli"). Match if ANY word
        # in the search matches, so typing the full name still finds them
        # via the surname.
        tokens = [t.lower() for t in search.split() if t]
        return [b for b in ALL_BATTERS if any(t in b.lower() for t in tokens)]
    return ALL_BATTERS


@app.get("/matchup/{batter}", response_model=list[MatchupRow])
def get_matchup(batter: str):
    rows = matchup_df[matchup_df["batter"] == batter]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for batter '{batter}'. Check /batters for exact spelling.",
        )
    records = rows.drop(columns=["batter"]).to_dict(orient="records")

    # NaN (e.g. balls_per_dismissal when a batter has 0 dismissals in a
    # bucket) isn't valid JSON. Note: doing this via DataFrame.where(...)
    # doesn't work -- pandas silently coerces None back to NaN on float64
    # columns. Has to be done on the plain Python dicts instead.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None

    return records


@app.get("/risk", response_model=RiskResponse)
def get_risk(
    batter: str = Query(..., description="Exact batter name, see /batters"),
    bowling_type: str = Query(..., description="Pace or Spin"),
    bowling_arm: str = Query(..., description="Right or Left"),
    bowling_detail: str = Query(..., description="e.g. Fast, Medium, Offbreak, Legbreak"),
    phase: str = Query(..., description="powerplay, middle, or death"),
):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    if bowling_type not in VALID_BOWLING_TYPES:
        raise HTTPException(status_code=400, detail=f"bowling_type must be one of {VALID_BOWLING_TYPES}")
    if phase not in VALID_PHASES:
        raise HTTPException(status_code=400, detail=f"phase must be one of {VALID_PHASES}")

    global_rate = batter_lookup["global_rate"]
    batter_overall = batter_lookup["batter_overall"]
    batter_vs_type = batter_lookup["batter_vs_type"]

    batter_overall_rate = batter_overall.get(batter, global_rate)
    batter_vs_type_rate = batter_vs_type.get((batter, bowling_type), batter_overall_rate)

    features = pd.DataFrame(
        [
            {
                "bowling_type": bowling_type,
                "bowling_arm": bowling_arm,
                "bowling_detail": bowling_detail,
                "phase": phase,
                "batter_overall_dismissal_rate": batter_overall_rate,
                "batter_vs_type_dismissal_rate": batter_vs_type_rate,
            }
        ]
    )

    proba = model_pipeline.predict_proba(features)[0][1]

    note = (
        "Model has modest predictive power (ROC-AUC ~0.55 on held-out data); "
        "treat this as a rough indicator, not a precise forecast."
    )
    if batter not in batter_overall.index:
        note = "Batter not seen in training data — using league-average tendencies. " + note

    return RiskResponse(
        batter=batter,
        bowling_type=bowling_type,
        bowling_arm=bowling_arm,
        bowling_detail=bowling_detail,
        phase=phase,
        dismissal_risk_pct=round(proba * 100, 2),
        note=note,
    )

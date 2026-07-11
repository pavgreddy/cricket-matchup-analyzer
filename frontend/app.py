"""
app.py

Streamlit frontend for the Cricket Matchup Analyzer.

Talks to the FastAPI backend (backend/main.py) to show:
  - A batter search/dropdown
  - Their matchup stats table (bowling_type x phase)
  - A couple of simple charts
  - A quick "predicted dismissal risk" calculator

Run from project root (with the backend already running separately):
    streamlit run frontend/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Set CRICKET_API_URL env var to override (e.g. when pointing at the
# deployed Render backend instead of localhost).
API_URL = os.environ.get("CRICKET_API_URL", "http://127.0.0.1:8000")

PHASE_ORDER = ["powerplay", "middle", "death"]

st.set_page_config(page_title="Cricket Matchup Analyzer", page_icon="🏏", layout="wide")


# ---------------------------------------------------------------------------
# API helpers (cached so we don't hammer the backend on every rerun)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_batters():
    resp = requests.get(f"{API_URL}/batters", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_matchup(batter: str):
    resp = requests.get(f"{API_URL}/matchup/{batter}", timeout=10)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_risk(batter, bowling_type, bowling_arm, bowling_detail, phase):
    resp = requests.get(
        f"{API_URL}/risk",
        params={
            "batter": batter,
            "bowling_type": bowling_type,
            "bowling_arm": bowling_arm,
            "bowling_detail": bowling_detail,
            "phase": phase,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏏 Cricket Matchup Analyzer")
st.caption("IPL ball-by-ball matchup stats: how a batter performs against different bowling types and phases.")

# ---------------------------------------------------------------------------
# Backend connectivity check
# ---------------------------------------------------------------------------
try:
    batters = fetch_batters()
except requests.exceptions.RequestException:
    st.error(
        f"Can't reach the backend at {API_URL}. "
        "Make sure it's running: `uvicorn backend.main:app --reload`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Batter selection
# ---------------------------------------------------------------------------
# Streamlit's built-in selectbox search only does plain substring matching
# on the option text -- typing "Virat Kohli" won't find "V Kohli". Use our
# own search box + token-based filter instead, so searching by full name
# still finds the Cricsheet "Initial Surname" entry via the surname.
search_query = st.text_input("Search for a batter", placeholder="e.g. Virat Kohli, Rohit Sharma...")

if search_query:
    tokens = [t.lower() for t in search_query.split() if t]
    filtered_batters = [b for b in batters if any(t in b.lower() for t in tokens)]
else:
    filtered_batters = batters

if search_query and not filtered_batters:
    st.warning(f"No batter found matching '{search_query}'. Try just their surname.")
    st.stop()

batter = st.selectbox(
    "Select batter" + (f" ({len(filtered_batters)} match{'es' if len(filtered_batters) != 1 else ''})" if search_query else ""),
    options=filtered_batters,
    index=None,
    placeholder="Choose from the list...",
)

if not batter:
    st.info("👆 Pick a batter to see their matchup breakdown.")
    st.stop()

matchup_df = fetch_matchup(batter)

if matchup_df.empty:
    st.warning(f"No matchup data found for {batter}.")
    st.stop()

matchup_df["phase"] = pd.Categorical(matchup_df["phase"], categories=PHASE_ORDER, ordered=True)
matchup_df = matchup_df.sort_values(["bowling_type", "phase"])

# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------
st.subheader(f"{batter} — Matchup Stats")

display_df = matchup_df.rename(
    columns={
        "bowling_type": "Bowling Type",
        "phase": "Phase",
        "balls_faced": "Balls Faced",
        "runs_scored": "Runs",
        "dismissals": "Dismissals",
        "strike_rate": "Strike Rate",
        "balls_per_dismissal": "Balls / Dismissal",
        "dismissal_pct": "Dismissal %",
        "boundary_pct": "Boundary %",
        "sufficient_sample": "Reliable Sample",
    }
)
st.dataframe(display_df, width="stretch", hide_index=True)

low_sample_count = (~matchup_df["sufficient_sample"]).sum()
if low_sample_count > 0:
    st.caption(
        f"⚠️ {low_sample_count} row(s) above are based on fewer than 30 balls faced — "
        "treat those numbers as indicative, not conclusive."
    )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Strike Rate by Phase & Bowling Type**")
    pivot_sr = matchup_df.pivot_table(
        index="phase", columns="bowling_type", values="strike_rate", observed=True
    ).reindex(PHASE_ORDER)
    st.bar_chart(pivot_sr)

with col2:
    st.markdown("**Dismissal % by Phase & Bowling Type**")
    pivot_dis = matchup_df.pivot_table(
        index="phase", columns="bowling_type", values="dismissal_pct", observed=True
    ).reindex(PHASE_ORDER)
    st.bar_chart(pivot_dis)

# ---------------------------------------------------------------------------
# Risk calculator
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Predicted Dismissal Risk")
st.caption(
    "Estimates dismissal probability for a specific matchup using a logistic regression model. "
    "Modest predictive power (ROC-AUC ~0.55) — treat as a rough indicator, not a forecast."
)

r1, r2, r3, r4 = st.columns(4)
with r1:
    risk_bowling_type = st.selectbox("Bowling Type", ["Pace", "Spin"])
with r2:
    risk_bowling_arm = st.selectbox("Bowling Arm", ["Right", "Left"])
with r3:
    risk_bowling_detail = st.selectbox(
        "Bowling Detail", ["Fast", "Medium", "Offbreak", "Legbreak", "Slow Left Arm Orthodox"]
    )
with r4:
    risk_phase = st.selectbox("Phase", PHASE_ORDER)

if st.button("Calculate Risk"):
    try:
        result = fetch_risk(batter, risk_bowling_type, risk_bowling_arm, risk_bowling_detail, risk_phase)
        st.metric("Predicted Dismissal Risk", f"{result['dismissal_risk_pct']}%")
        st.caption(result["note"])
    except requests.exceptions.HTTPError as e:
        st.error(f"Couldn't calculate risk: {e.response.json().get('detail', str(e))}")

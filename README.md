# Cricket Matchup Analyzer

Given a batter, shows how they perform against different bowling types (Pace/Spin)
and match phases (Powerplay/Middle/Death) using IPL ball-by-ball data — plus a
logistic regression model that estimates dismissal risk for a specific matchup.

**Live app:** https://pavan-cricket-matchup-analyzer.streamlit.app/
**Backend API:** https://cricket-matchup-analyzer.onrender.com

> Note: the backend is on Render's free tier, which sleeps after ~15 min of
> inactivity. The first request after a while may take 30-60s to wake it up.

## Tech Stack

- **Data**: 1,243 IPL matches from [Cricsheet](https://cricsheet.org/) (295,732 ball-by-ball deliveries), bowling style data from the `cricketdata` R package (CRAN)
- **Backend**: FastAPI + pandas + scikit-learn, Dockerized, deployed on Render
- **Frontend**: Streamlit, deployed on Streamlit Community Cloud
- **Model**: Logistic regression predicting per-ball dismissal probability

## Architecture

```
Cricsheet JSON files
        │
        ▼
parse_data.py ──► deliveries.csv
        │
        ▼
get_player_meta.py + normalize_bowling_styles.py
        │
        ▼
deliveries_with_styles.csv (295,732 rows, bowling style tagged)
        │
        ├──► feature_engineering.py ──► batter_matchup_stats.csv
        │                                       │
        └──► train_model.py ──► dismissal_model.pkl + batter_lookup.pkl
                                                │
                                                ▼
                                    backend/main.py (FastAPI)
                                     /batters  /matchup/{batter}  /risk
                                                │
                                                ▼
                                    frontend/app.py (Streamlit)
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check — batter count, model load status |
| `GET /batters?search=` | List batters, optional token-based name search |
| `GET /matchup/{batter}` | Matchup stats table for one batter |
| `GET /risk` | Predicted dismissal risk for a bowling_type/arm/detail/phase combo |

Interactive docs: `<backend-url>/docs`

## Model Limitations (documented honestly)

The dismissal-prediction model has modest predictive power: **ROC-AUC ≈ 0.55**
on a held-out season (barely better than random). This is expected — predicting
whether *one specific ball* results in a wicket is a genuinely hard, high-variance
problem without ball-tracking data (line/length/pace), which is proprietary and
out of scope for this project. The model is a supplementary signal; the core
value of this app is the **descriptive matchup stats table**, which reflects
real cricketing patterns (e.g. bowlers/tail-enders struggling against pace).

## Local Setup

```bash
git clone https://github.com/pavgreddy/cricket-matchup-analyzer.git
cd cricket-matchup-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Run the backend
uvicorn backend.main:app --reload

# In a separate terminal, run the frontend
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

## Deployment Notes / Issues Hit

A few real deployment issues came up building this — documenting them here
since they're common gotchas:

- **NaN isn't valid JSON.** `balls_per_dismissal` is `NaN` for batter/bowling-type/phase
  combos with zero dismissals. Starlette's JSON renderer uses `allow_nan=False`,
  so this crashed the `/matchup` endpoint until NaN values were explicitly
  converted to `None` on the raw Python dicts (not via `DataFrame.where()`,
  which silently coerces `None` back to `NaN` on `float64` columns).
- **Streamlit Cloud defaulted to Python 3.14**, which doesn't have stable
  `numpy`/`pyarrow` wheels yet, causing a segmentation fault on every run.
  Fixed by adding `frontend/runtime.txt` (`3.11`) and explicitly selecting
  Python 3.11 during a fresh app deploy (Python version can't be changed on
  an already-running app).
- **Unpinned `numpy`/`pyarrow` versions** in `requirements.txt` let Streamlit
  Cloud install the newest releases, which also segfaulted (likely a CPU
  instruction-set mismatch on the container). Fixed by pinning to a
  conservative, known-stable range.
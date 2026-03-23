# Bernese Oberland Thermal Bubble Map

Interactive web app for paragliding route intuition over Switzerland.

The app pulls forecast pressure-level data for many Bernese Oberland peaks and computes a **mid-layer thermal smoothness score** using the red/blue line separation logic from soundings:

- Mid-layer used: **850, 700, 600, 500 hPa**
- Core metric: **dewpoint depression** $T - T_d$
- High score: drier + more consistent mid-layer structure
- Visualization: **orange 300 m radius bubble** at **peak elevation + 300 m**

## What is included

- Real basemap of Switzerland (interactive map)
- 25+ Bernese Oberland peaks with coordinates and elevation
- Live forecast ingestion from Open-Meteo pressure-level variables
- Bubble coloring by score intensity (orange scale)
- Vertical dry-air cylinders from dry-layer top to bottom
- Forecast movie playback (automatic day-by-day map animation)
- Suggested high-score circuit line with visible PG waypoint markers and estimated chain distance
- Peak-level sounding profile panel with red temperature and blue dewpoint lines

## Quick start (2 minutes)

Use one of the two options below.

### Option A: Fully isolated local environment (.venv)

1. Open PowerShell in this folder.
1. Create and provision project-local virtual environment:

```bash
.\scripts\setup_venv.ps1
```

1. Run the app:

```bash
.\scripts\run_local.ps1
```

### Option B: Fully reproducible container (Docker)

1. Build and run:

```bash
docker compose up --build
```

1. Open `http://localhost:8501`

1. In the app, choose forecast hour and optional area filter.
1. Read the highest orange bubbles as strongest dry/stable mid-layer candidates.
1. Use the per-peak profile view to inspect where red and blue lines are farthest apart.

## Forecast horizon explained

- Forecast horizon means comparing multiple future days.
- In `Daily horizon` mode, choose one reference hour (for example 13:00 local).
- The app renders one map per day at that same hour so comparisons are fair.
- Display styles:
  - `Tabs`: one day per tab.
  - `Stacked maps`: all days visible one under another.
- Optional `Forecast movie`: automatically plays the day maps in sequence (with frame duration and loop controls).

Why this is useful:

- Thermal quality changes strongly with daytime heating. Fixing the hour avoids comparing morning conditions on one day versus afternoon conditions on another.

## What it does

- Scores each peak using mid-layer dewpoint depression.
- Ranks peaks by predicted "smooth thermal" potential.
- Draws comparable bubbles and cylinders with shared radius controls.
- Draws cylinders spanning dry-layer bottom to top (pressure-to-altitude approximation).
- Uses a gray-to-red gradient where better conditions are red and weaker conditions are gray.
- Allows coloring by different variables: score, mean gap, max gap, or consistency.
- Can autoplay daily horizon maps as a forecast movie to track weather evolution.
- Suggests a high-score peak chain to help route intuition.
- Lets you inspect the sounding profile for any selected peak.

## What can be improved next

- Add wind speed/direction and vertical shear penalty in score.
- Include cloud-base and overdevelopment risk index.
- Add sunrise/terrain shadow logic by slope aspect.
- Run multi-objective route optimization (distance, glide, retrieve risk).
- Export route as GPX/KML for XC planning tools.
- Add historical verification against flown tracks and observed weather.

## Project files

- `app.py` - Streamlit app
- `requirements.txt` - Python dependencies
- `requirements.lock.txt` - pinned dependency versions for reproducibility
- `data/bernese_oberland_peaks.csv` - Peak catalog
- `.streamlit/config.toml` - clean webhost/server configuration
- `scripts/setup_venv.ps1` - one-command local isolated setup
- `scripts/run_local.ps1` - one-command app start from `.venv`
- `Dockerfile` and `docker-compose.yml` - containerized reproducible runtime

## Run locally

1. Use isolated local environment:

```bash
.\scripts\setup_venv.ps1
```

1. Start app:

```bash
.\scripts\run_local.ps1
```

1. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Reproducibility model

- App dependencies are version-pinned in `requirements.lock.txt`.
- Local workflow is isolated in `.venv` under this project folder (no global packages required).
- Container workflow uses the same lock file for deterministic builds.
- Streamlit host behavior is configured in `.streamlit/config.toml` for consistent port and host binding.

## Clean webhost display

- Local clean URL: `http://localhost:8501`
- Docker clean URL: `http://localhost:8501`
- Both workflows use the same app port and host configuration for predictable behavior.

## Score definition

For each peak and selected forecast hour:

1. Fetch temperature and dewpoint at 850, 700, 600, 500 hPa.
2. Compute layer-wise depression $D = T - T_d$.
3. Build score from:

- mean depression (dryness, dominant)
- standard deviation across layers (stability/consistency)
- max depression bonus

The final score is clipped to $[0, 100]$.

## Notes on deployment

- **Local hosting** is fully supported via Streamlit.
- **Vercel** is optimized for frontend/serverless workloads and is usually not ideal for a full Streamlit runtime.
- Recommended hosted options for this app style:
  - Streamlit Community Cloud
  - Render
  - Railway

## Safety and interpretation

This tool is a **decision-support visualization**, not a flight-safety guarantee.

Always cross-check with official meteorological products, NOTAMs, and local pilot knowledge before flying.

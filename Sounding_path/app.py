from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

PRESSURE_LEVELS = [850, 700, 600, 500]
TEMPERATURE_COLS = [f"temperature_{p}hPa" for p in PRESSURE_LEVELS]
DEWPOINT_COLS = [f"dew_point_{p}hPa" for p in PRESSURE_LEVELS]
BASEMAP_STYLES = {
    "Clean Light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Topo Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Dark Matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}


@dataclass
class ScoreComponents:
    mean_depression: float
    max_depression: float
    std_depression: float
    score: float
    driest_layer_hpa: int
    dry_top_hpa: int
    dry_bottom_hpa: int


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(a))


def lerp_color(c1: List[int], c2: List[int], t: float) -> List[int]:
    t = float(np.clip(t, 0.0, 1.0))
    return [
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    ]


def metric_value_to_unit(value: float, metric_mode: str) -> float:
    if metric_mode == "Score":
        return float(np.clip(value / 100.0, 0.0, 1.0))
    if metric_mode == "Mean dry gap (T-Td)":
        return float(np.clip(value / 25.0, 0.0, 1.0))
    if metric_mode == "Max dry gap (T-Td)":
        return float(np.clip(value / 30.0, 0.0, 1.0))
    # For variability, lower is better, so invert.
    return float(1.0 - np.clip(value / 8.0, 0.0, 1.0))


def compute_visual_color(row: pd.Series, metric_mode: str) -> List[int]:
    metric_key = {
        "Score": "score",
        "Mean dry gap (T-Td)": "mean_dep_c",
        "Max dry gap (T-Td)": "max_dep_c",
        "Consistency (low std)": "std_dep_c",
    }[metric_mode]
    unit = metric_value_to_unit(float(row[metric_key]), metric_mode)
    # Low = gray, high = red.
    return lerp_color([145, 145, 145], [218, 38, 38], unit)


def pressure_to_altitude_m(pressure_hpa: float) -> float:
    # Standard atmosphere approximation, suitable for layer visualization.
    return 44330.0 * (1.0 - (float(pressure_hpa) / 1013.25) ** 0.1903)


def compute_stability_score(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> ScoreComponents:
    depression = temperature_c - dewpoint_c
    mean_dep = float(np.mean(depression))
    max_dep = float(np.max(depression))
    std_dep = float(np.std(depression))

    # Dryness dominates. Consistency across layers adds confidence for smooth climbs.
    dry_norm = float(np.clip((mean_dep - 2.0) / 22.0, 0.0, 1.0))
    consistency_norm = 1.0 - float(np.clip(std_dep / 8.0, 0.0, 1.0))
    max_dep_bonus = float(np.clip((max_dep - 5.0) / 20.0, 0.0, 1.0))

    score = 100.0 * (0.65 * dry_norm + 0.25 * consistency_norm + 0.10 * max_dep_bonus)
    driest_idx = int(np.argmax(depression))
    driest_layer_hpa = PRESSURE_LEVELS[driest_idx]

    # Define dry-air span: levels with depression above a practical threshold.
    dry_threshold = max(4.0, float(np.mean(depression)))
    dry_indices = np.where(depression >= dry_threshold)[0]
    if len(dry_indices) == 0:
        dry_indices = np.array([driest_idx])

    dry_pressures = [PRESSURE_LEVELS[int(i)] for i in dry_indices]
    dry_top_hpa = int(min(dry_pressures))
    dry_bottom_hpa = int(max(dry_pressures))

    return ScoreComponents(
        mean_depression=mean_dep,
        max_depression=max_dep,
        std_depression=std_dep,
        score=float(np.clip(score, 0.0, 100.0)),
        driest_layer_hpa=driest_layer_hpa,
        dry_top_hpa=dry_top_hpa,
        dry_bottom_hpa=dry_bottom_hpa,
    )


@st.cache_data(show_spinner=False)
def load_peaks(path: str) -> pd.DataFrame:
    peaks = pd.read_csv(path)
    required = {"name", "lat", "lon", "elev_m", "area"}
    missing = required - set(peaks.columns)
    if missing:
        raise ValueError(f"Missing required columns in peaks CSV: {sorted(missing)}")
    return peaks


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peak_forecast(lat: float, lon: float, forecast_days: int) -> pd.DataFrame:
    endpoint = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        *TEMPERATURE_COLS,
        *DEWPOINT_COLS,
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "forecast_days": forecast_days,
        "timezone": "Europe/Zurich",
    }
    response = requests.get(endpoint, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError("Open-Meteo response did not include hourly data.")

    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for col in hourly_vars:
        if col not in hourly:
            raise RuntimeError(f"Missing hourly variable '{col}' in weather response.")
        frame[col] = hourly[col]
    return frame


def build_peak_hour_table(peaks: pd.DataFrame, target_time: pd.Timestamp, forecast_days: int) -> pd.DataFrame:
    rows = []
    failed_peaks: List[str] = []
    for peak in peaks.itertuples(index=False):
        try:
            profile = fetch_peak_forecast(float(peak.lat), float(peak.lon), forecast_days)
        except Exception:
            failed_peaks.append(str(peak.name))
            continue

        row = profile.loc[profile["time"] == target_time]
        if row.empty:
            failed_peaks.append(str(peak.name))
            continue

        temperatures = row.iloc[0][TEMPERATURE_COLS].to_numpy(dtype=float)
        dewpoints = row.iloc[0][DEWPOINT_COLS].to_numpy(dtype=float)
        score_parts = compute_stability_score(temperatures, dewpoints)
        dry_top_alt = pressure_to_altitude_m(score_parts.dry_top_hpa)
        dry_bottom_alt = pressure_to_altitude_m(score_parts.dry_bottom_hpa)
        cylinder_height = max(100.0, dry_top_alt - dry_bottom_alt)

        rows.append(
            {
                "name": peak.name,
                "area": peak.area,
                "lat": float(peak.lat),
                "lon": float(peak.lon),
                "elev_m": float(peak.elev_m),
                "bubble_alt_m": float(peak.elev_m) + 300.0,
                "radius_m": 300.0,
                "score": score_parts.score,
                "mean_dep_c": score_parts.mean_depression,
                "max_dep_c": score_parts.max_depression,
                "std_dep_c": score_parts.std_depression,
                "driest_layer_hpa": score_parts.driest_layer_hpa,
                "dry_top_hpa": score_parts.dry_top_hpa,
                "dry_bottom_hpa": score_parts.dry_bottom_hpa,
                "dry_top_alt_m": dry_top_alt,
                "dry_bottom_alt_m": dry_bottom_alt,
                "cylinder_base_m": dry_bottom_alt,
                "cylinder_height_m": cylinder_height,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    result.attrs["failed_peaks"] = failed_peaks
    return result


def choose_timeslot(peaks: pd.DataFrame, forecast_days: int) -> Tuple[pd.Timestamp, List[pd.Timestamp]]:
    sample = fetch_peak_forecast(float(peaks.iloc[0]["lat"]), float(peaks.iloc[0]["lon"]), forecast_days)
    times = sorted(sample["time"].tolist())
    now = pd.Timestamp(datetime.now()).tz_localize(None)

    nearest = min(times, key=lambda t: abs(t - now))
    return nearest, times


def choose_daily_times(all_times: List[pd.Timestamp], target_hour: int) -> Dict[str, pd.Timestamp]:
    per_day: Dict[str, List[pd.Timestamp]] = {}
    for ts in all_times:
        day_key = ts.strftime("%Y-%m-%d")
        per_day.setdefault(day_key, []).append(ts)

    day_selection: Dict[str, pd.Timestamp] = {}
    for day_key, times in per_day.items():
        selected = min(times, key=lambda t: abs(int(t.hour) - int(target_hour)))
        day_selection[day_key] = selected
    return day_selection


def build_map_layers(
    peak_scores: pd.DataFrame,
    show_bubbles: bool,
    show_cylinders: bool,
    show_circuit: bool,
    color_metric_mode: str,
    visual_radius_m: int,
    min_score_for_circuit: float,
    max_circuit_points: int,
) -> Tuple[List[pdk.Layer], float]:
    layers: List[pdk.Layer] = []
    vis_df = peak_scores.copy()
    vis_df["color"] = vis_df.apply(lambda row: compute_visual_color(row, color_metric_mode), axis=1)
    vis_df["radius_m"] = float(visual_radius_m)

    if show_cylinders:
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=vis_df,
                get_position="[lon, lat, cylinder_base_m]",
                get_elevation="cylinder_height_m",
                elevation_scale=1,
                radius=int(visual_radius_m * 0.75),
                get_fill_color="color",
                get_line_color=[65, 65, 65],
                pickable=True,
                auto_highlight=True,
                opacity=0.16,
                extruded=True,
                stroked=True,
            )
        )

    if show_bubbles:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=vis_df,
                get_position="[lon, lat, bubble_alt_m]",
                get_radius="radius_m",
                radius_units="meters",
                radius_min_pixels=5,
                radius_max_pixels=45,
                get_fill_color="color",
                get_line_color=[70, 70, 70],
                line_width_min_pixels=2,
                pickable=True,
                stroked=True,
                opacity=0.2,
            )
        )

    route_distance = 0.0
    if show_circuit:
        route, route_distance = build_suggested_circuit(vis_df, min_score_for_circuit, max_circuit_points)
        if not route.empty:
            path_data = [
                {
                    "name": "Suggested route",
                    "path": route[["lon", "lat", "bubble_alt_m"]].values.tolist(),
                    "distance_km": route_distance,
                }
            ]
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=path_data,
                    get_path="path",
                    get_width=28,
                    width_units="meters",
                    get_color=[57, 255, 20, 180],
                    pickable=True,
                )
            )

            route_markers = route.copy().reset_index(drop=True)
            route_markers["route_idx"] = route_markers.index + 1
            route_markers["route_label"] = route_markers["route_idx"].apply(lambda x: f"PG-{x}")
            route_markers["route_color"] = route_markers["route_idx"].apply(
                lambda i: [20, 170, 80] if i == 1 else [255, 215, 0]
            )

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=route_markers,
                    get_position="[lon, lat, bubble_alt_m]",
                    get_radius=max(90, int(visual_radius_m * 0.25)),
                    radius_units="meters",
                    get_fill_color="route_color",
                    get_line_color=[30, 30, 30],
                    line_width_min_pixels=1,
                    pickable=True,
                    stroked=True,
                    opacity=0.45,
                )
            )

            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=route_markers,
                    get_position="[lon, lat, bubble_alt_m]",
                    get_text="route_label",
                    get_size=10,
                    size_units="meters",
                    size_scale=3,
                    get_color=[20, 20, 20, 150],
                    get_angle=0,
                    get_text_anchor="middle",
                    get_alignment_baseline="center",
                    pickable=True,
                )
            )

    return layers, route_distance


def render_map_block(
    peak_scores: pd.DataFrame,
    title: str,
    show_bubbles: bool,
    show_cylinders: bool,
    show_circuit: bool,
    color_metric_mode: str,
    visual_radius_m: int,
    basemap_style: str,
    min_score_for_circuit: float,
    max_circuit_points: int,
    key_suffix: str,
) -> float:
    layers, route_distance = build_map_layers(
        peak_scores=peak_scores,
        show_bubbles=show_bubbles,
        show_cylinders=show_cylinders,
        show_circuit=show_circuit,
        color_metric_mode=color_metric_mode,
        visual_radius_m=visual_radius_m,
        min_score_for_circuit=min_score_for_circuit,
        max_circuit_points=max_circuit_points,
    )

    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "Area: {area}<br/>"
            "Score: {score}<br/>"
            "Mean T-Td: {mean_dep_c} C<br/>"
            "Max T-Td: {max_dep_c} C<br/>"
            "Std T-Td: {std_dep_c} C<br/>"
            "Dry span: {dry_bottom_hpa} to {dry_top_hpa} hPa<br/>"
            "Driest layer: {driest_layer_hpa} hPa"
        ),
        "style": {"backgroundColor": "#111111", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=float(peak_scores["lat"].mean()),
        longitude=float(peak_scores["lon"].mean()),
        zoom=8.4,
        pitch=50,
        bearing=20,
    )

    st.subheader(title)
    st.pydeck_chart(
        pdk.Deck(
            map_style=basemap_style,
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip,
        ),
        width="stretch",
        key=f"deck_{key_suffix}",
    )
    return route_distance


def build_deck(
    peak_scores: pd.DataFrame,
    show_bubbles: bool,
    show_cylinders: bool,
    show_circuit: bool,
    color_metric_mode: str,
    visual_radius_m: int,
    basemap_style: str,
    min_score_for_circuit: float,
    max_circuit_points: int,
) -> Tuple[pdk.Deck, float]:
    layers, route_distance = build_map_layers(
        peak_scores=peak_scores,
        show_bubbles=show_bubbles,
        show_cylinders=show_cylinders,
        show_circuit=show_circuit,
        color_metric_mode=color_metric_mode,
        visual_radius_m=visual_radius_m,
        min_score_for_circuit=min_score_for_circuit,
        max_circuit_points=max_circuit_points,
    )

    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "Area: {area}<br/>"
            "Score: {score}<br/>"
            "Mean T-Td: {mean_dep_c} C<br/>"
            "Max T-Td: {max_dep_c} C<br/>"
            "Std T-Td: {std_dep_c} C<br/>"
            "Dry span: {dry_bottom_hpa} to {dry_top_hpa} hPa<br/>"
            "Driest layer: {driest_layer_hpa} hPa"
        ),
        "style": {"backgroundColor": "#111111", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=float(peak_scores["lat"].mean()),
        longitude=float(peak_scores["lon"].mean()),
        zoom=8.4,
        pitch=50,
        bearing=20,
    )

    deck = pdk.Deck(
        map_style=basemap_style,
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    )
    return deck, route_distance


def build_suggested_circuit(df: pd.DataFrame, min_score: float, max_points: int) -> Tuple[pd.DataFrame, float]:
    pool = df[df["score"] >= min_score].copy()
    if len(pool) < 2:
        pool = df.sort_values("score", ascending=False).head(max(4, min(max_points, len(df)))).copy()
        if len(pool) < 2:
            return pd.DataFrame(), 0.0

    pool = pool.sort_values("score", ascending=False).head(max_points).reset_index(drop=True)
    visited = [pool.loc[0, "name"]]
    current_name = visited[0]

    while len(visited) < len(pool):
        current = pool[pool["name"] == current_name].iloc[0]
        remaining = pool[~pool["name"].isin(visited)]
        if remaining.empty:
            break

        remaining = remaining.assign(
            dist_km=remaining.apply(
                lambda r: haversine_km(current["lat"], current["lon"], r["lat"], r["lon"]),
                axis=1,
            )
        )
        next_name = remaining.sort_values(["dist_km", "score"], ascending=[True, False]).iloc[0]["name"]
        visited.append(next_name)
        current_name = str(next_name)

    route = pool.set_index("name").loc[visited].reset_index()
    distance = 0.0
    for i in range(len(route) - 1):
        distance += haversine_km(
            float(route.loc[i, "lat"]),
            float(route.loc[i, "lon"]),
            float(route.loc[i + 1, "lat"]),
            float(route.loc[i + 1, "lon"]),
        )

    return route, distance


def make_profile_plot(peak_name: str, lat: float, lon: float, when: pd.Timestamp, forecast_days: int) -> go.Figure:
    prof = fetch_peak_forecast(lat, lon, forecast_days)
    row = prof.loc[prof["time"] == when]
    if row.empty:
        raise RuntimeError("No profile available for selected time.")

    t_vals = row.iloc[0][TEMPERATURE_COLS].to_numpy(dtype=float)
    td_vals = row.iloc[0][DEWPOINT_COLS].to_numpy(dtype=float)
    pressure = np.array(PRESSURE_LEVELS)
    depression = t_vals - td_vals
    max_idx = int(np.argmax(depression))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=pressure,
            mode="lines+markers",
            name="Temperature (red line)",
            line=dict(color="red", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=td_vals,
            y=pressure,
            mode="lines+markers",
            name="Dewpoint (blue line)",
            line=dict(color="royalblue", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[(t_vals[max_idx] + td_vals[max_idx]) / 2],
            y=[pressure[max_idx]],
            mode="markers",
            marker=dict(size=14, color="orange", line=dict(width=1, color="black")),
            name="Max T-Td gap",
            hovertemplate=(
                f"{peak_name}<br>Layer: {pressure[max_idx]} hPa"
                f"<br>Gap: {depression[max_idx]:.1f} C<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{peak_name}: Mid-layer sounding profile",
        xaxis_title="Temperature / Dewpoint (C)",
        yaxis_title="Pressure (hPa)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Bernese Oberland Thermal Bubble Map", layout="wide")
    st.title("Bernese Oberland Thermal Smoothness Map")
    st.caption(
        "Scores are derived from dewpoint depression (T - Td) in the 850 to 500 hPa mid-layer. "
        "Higher score = drier and more uniform layer, often smoother thermal structure."
    )

    with st.sidebar:
        st.header("Settings")
        forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=7, value=2)
        basemap_name = st.selectbox("Basemap", options=list(BASEMAP_STYLES.keys()), index=0)
        color_metric_mode = st.selectbox(
            "Color by",
            options=["Score", "Mean dry gap (T-Td)", "Max dry gap (T-Td)", "Consistency (low std)"],
            index=0,
        )
        visual_radius_m = st.slider("Bubble/cylinder radius (m)", min_value=140, max_value=500, value=260, step=20)
        view_mode = st.radio("Forecast view mode", options=["Single hour", "Daily horizon"], index=0)
        daily_target_hour = st.slider("Daily horizon reference hour", min_value=6, max_value=20, value=13)
        horizon_layout = st.radio("Daily horizon layout", options=["Tabs", "Stacked maps"], index=0)
        timelapse_enabled = st.toggle("Enable forecast movie (daily)", value=False)
        timelapse_seconds = st.slider("Movie frame duration (seconds)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        timelapse_loops = st.slider("Movie loops", min_value=1, max_value=5, value=1)
        min_score_visible = st.slider("Minimum score to display", min_value=0, max_value=95, value=0)
        show_bubbles = st.toggle("Show bubble markers", value=True)
        show_cylinders = st.toggle("Show dry-air cylinders", value=True)
        show_circuit = st.toggle("Show suggested circuit", value=True)
        min_score_for_circuit = st.slider("Circuit score threshold", min_value=30, max_value=95, value=45)
        max_circuit_points = st.slider("Max peaks in circuit", min_value=4, max_value=18, value=10)

    basemap_style = BASEMAP_STYLES[basemap_name]

    with st.expander("Quick start: what this app contains and how to use it", expanded=False):
        st.markdown(
            """
            - Contains: a Bernese Oberland peak catalog, live pressure-level forecasts, and a map with 300 m orange bubbles.
            - Does: computes a mid-layer dryness/stability score from $T - T_d$ at 850/700/600/500 hPa.
            - Use: pick a forecast hour (or daily horizon), inspect top scores, follow bubble/cylinder clusters, and inspect red/blue profile lines.
            - Forecast horizon: compares multiple days at the same local hour so day-to-day weather evolution is visible.
            - Improve further: add wind/shear penalties, cloud-base checks, and optimization for full XC task route sequencing.
            """
        )

    peaks = load_peaks("data/bernese_oberland_peaks.csv")
    areas = ["All"] + sorted(peaks["area"].unique().tolist())
    selected_area = st.selectbox("Area filter", options=areas, index=0)
    if selected_area != "All":
        peaks = peaks[peaks["area"] == selected_area].reset_index(drop=True)

    if peaks.empty:
        st.error("No peaks available after area filtering.")
        return

    default_time, all_times = choose_timeslot(peaks, forecast_days)
    if view_mode == "Single hour":
        selected_time = st.selectbox(
            "Forecast hour",
            options=all_times,
            index=all_times.index(default_time),
            format_func=lambda t: t.strftime("%Y-%m-%d %H:%M"),
        )

        with st.spinner("Computing scores for Bernese Oberland peaks..."):
            peak_scores = build_peak_hour_table(peaks, selected_time, forecast_days)

        failed_peaks = peak_scores.attrs.get("failed_peaks", [])
        if failed_peaks:
            st.warning(
                f"Forecast unavailable for {len(failed_peaks)} peak(s) at this time. "
                "They were skipped in scoring."
            )

        if peak_scores.empty:
            st.error("No forecast data returned for the selected time.")
            return

        peak_scores = peak_scores[peak_scores["score"] >= min_score_visible].reset_index(drop=True)
        if peak_scores.empty:
            st.warning("No peaks pass the minimum score filter. Lower the threshold.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Peaks shown", f"{len(peak_scores)}")
        col2.metric("Top score", f"{peak_scores['score'].max():.1f}")
        col3.metric("Mean score", f"{peak_scores['score'].mean():.1f}")

        st.subheader("Top peaks by smooth thermal potential")
        st.dataframe(
            peak_scores[["name", "area", "score", "mean_dep_c", "max_dep_c", "driest_layer_hpa", "dry_top_hpa", "dry_bottom_hpa", "elev_m"]]
            .rename(
                columns={
                    "name": "Peak",
                    "area": "Area",
                    "score": "Score",
                    "mean_dep_c": "Mean T-Td (C)",
                    "max_dep_c": "Max T-Td (C)",
                    "driest_layer_hpa": "Driest layer (hPa)",
                    "dry_top_hpa": "Dry top (hPa)",
                    "dry_bottom_hpa": "Dry bottom (hPa)",
                    "elev_m": "Peak elev (m)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        route_distance = render_map_block(
            peak_scores=peak_scores,
            title="Interactive map (clear route + comparable bubbles/cylinders)",
            show_bubbles=show_bubbles,
            show_cylinders=show_cylinders,
            show_circuit=show_circuit,
            color_metric_mode=color_metric_mode,
            visual_radius_m=visual_radius_m,
            basemap_style=basemap_style,
            min_score_for_circuit=min_score_for_circuit,
            max_circuit_points=max_circuit_points,
            key_suffix="single_hour",
        )

        if route_distance > 0:
            st.info(f"Suggested high-score chain distance: {route_distance:.1f} km")
        elif show_circuit:
            st.info("No suggested circuit found for the current score threshold. Try lowering it.")

        selected_peak_name = st.selectbox("Inspect sounding for one peak", options=peak_scores["name"].tolist(), index=0)
        selected_peak = peak_scores[peak_scores["name"] == selected_peak_name].iloc[0]
        profile_fig = make_profile_plot(
            peak_name=str(selected_peak["name"]),
            lat=float(selected_peak["lat"]),
            lon=float(selected_peak["lon"]),
            when=selected_time,
            forecast_days=forecast_days,
        )
        st.plotly_chart(profile_fig, width="stretch")
    else:
        st.info(
            "Forecast horizon compares each next day at the same local hour. "
            "This makes day-to-day changes directly comparable."
        )
        day_time_map = choose_daily_times(all_times, daily_target_hour)
        ordered_days = sorted(day_time_map.keys())
        day_results: List[Tuple[str, pd.Timestamp, pd.DataFrame]] = []

        with st.spinner("Computing daily horizon maps..."):
            for day_key in ordered_days:
                ts = day_time_map[day_key]
                scores = build_peak_hour_table(peaks, ts, forecast_days)
                scores = scores[scores["score"] >= min_score_visible].reset_index(drop=True)
                if not scores.empty:
                    day_results.append((day_key, ts, scores))

        if not day_results:
            st.warning("No data available for the selected horizon settings.")
            return

        if timelapse_enabled:
            st.markdown("### Forecast movie")
            st.caption(
                "Plays each day in sequence at the same local hour to visualize weather evolution."
            )
            if st.button("Play movie", type="primary"):
                title_slot = st.empty()
                map_slot = st.empty()
                progress_slot = st.empty()
                total_frames = len(day_results) * timelapse_loops
                frame_counter = 0

                for loop_idx in range(timelapse_loops):
                    for day_key, ts, scores in day_results:
                        frame_counter += 1
                        title_slot.markdown(
                            f"#### Frame {frame_counter}/{total_frames}: {day_key} ({ts.strftime('%H:%M')})"
                        )
                        deck, _ = build_deck(
                            peak_scores=scores,
                            show_bubbles=show_bubbles,
                            show_cylinders=show_cylinders,
                            show_circuit=show_circuit,
                            color_metric_mode=color_metric_mode,
                            visual_radius_m=visual_radius_m,
                            basemap_style=basemap_style,
                            min_score_for_circuit=min_score_for_circuit,
                            max_circuit_points=max_circuit_points,
                        )
                        map_slot.pydeck_chart(deck, width="stretch")
                        progress_slot.progress(frame_counter / total_frames)
                        time.sleep(timelapse_seconds)

                st.success("Forecast movie completed.")

        if horizon_layout == "Tabs":
            tabs = st.tabs([f"{d} ({t.strftime('%H:%M')})" for d, t, _ in day_results])
            for tab, (day_key, ts, scores) in zip(tabs, day_results):
                with tab:
                    st.write(f"Local forecast hour: {ts.strftime('%Y-%m-%d %H:%M')}")
                    st.dataframe(
                        scores[["name", "area", "score", "driest_layer_hpa", "dry_top_hpa", "dry_bottom_hpa"]]
                        .rename(
                            columns={
                                "name": "Peak",
                                "area": "Area",
                                "score": "Score",
                                "driest_layer_hpa": "Driest layer",
                                "dry_top_hpa": "Dry top",
                                "dry_bottom_hpa": "Dry bottom",
                            }
                        )
                        .head(12),
                        width="stretch",
                        hide_index=True,
                    )
                    render_map_block(
                        peak_scores=scores,
                        title=f"Daily horizon map - {day_key}",
                        show_bubbles=show_bubbles,
                        show_cylinders=show_cylinders,
                        show_circuit=show_circuit,
                        color_metric_mode=color_metric_mode,
                        visual_radius_m=visual_radius_m,
                        basemap_style=basemap_style,
                        min_score_for_circuit=min_score_for_circuit,
                        max_circuit_points=max_circuit_points,
                        key_suffix=f"tab_{day_key}",
                    )
        else:
            for day_key, ts, scores in day_results:
                st.markdown(f"### {day_key} ({ts.strftime('%H:%M')})")
                render_map_block(
                    peak_scores=scores,
                    title=f"Daily horizon map - {day_key}",
                    show_bubbles=show_bubbles,
                    show_cylinders=show_cylinders,
                    show_circuit=show_circuit,
                    color_metric_mode=color_metric_mode,
                    visual_radius_m=visual_radius_m,
                    basemap_style=basemap_style,
                    min_score_for_circuit=min_score_for_circuit,
                    max_circuit_points=max_circuit_points,
                    key_suffix=f"stack_{day_key}",
                )

    st.caption(
        "Important: This score is a meteorological heuristic, not a flight safety guarantee. "
        "Always validate with official forecasts, NOTAMs, and local pilot knowledge."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = APP_ROOT / "outputs" / "bald_explorer"


def _load_local_artifacts(artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(artifacts_dir / "latent_points.csv")
    traj = pd.read_csv(artifacts_dir / "trajectory_summary.csv")
    genes = None
    for name in [
        "top_genes_by_group_integrated_gradients.csv",
        "top_genes_by_group_shap.csv",
        "top_genes_by_group.csv",
    ]:
        candidate = artifacts_dir / name
        if candidate.exists():
            genes = pd.read_csv(candidate)
            break
    if genes is None:
        genes = pd.DataFrame()
    return points, traj, genes


def _load_from_api(base_url: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_resp = requests.get(f"{base_url}/artifacts/trajectories", timeout=30)
    traj_resp.raise_for_status()
    traj_data = traj_resp.json().get("data", {})

    genes_resp = requests.get(f"{base_url}/artifacts/top-genes", params={"top_k": 5000}, timeout=30)
    genes_resp.raise_for_status()
    genes_data = genes_resp.json().get("data", {})

    points = pd.DataFrame(traj_data.get("points", []))
    traj = pd.DataFrame(traj_data.get("trajectories", []))
    genes = pd.DataFrame(genes_data.get("rows", []))
    return points, traj, genes


def _load_publication_tables(artifacts_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name in [
        "publication_top_genes.csv",
        "publication_trajectory_table.csv",
        "publication_pathway_table.csv",
        "pathway_enrichment_ig.csv",
    ]:
        path = artifacts_dir / name
        if path.exists():
            out[name] = pd.read_csv(path)
    return out


def _df_download(df: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def _plot_download(fig, label: str, file_name: str) -> None:
    payload = fig.to_json().encode("utf-8")
    st.download_button(
        label=label,
        data=payload,
        file_name=file_name,
        mime="application/json",
        use_container_width=True,
    )


def _show_jobs_panel(api_url: str) -> None:
    st.subheader("Async Runs")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Refresh Jobs", use_container_width=True):
            pass

    try:
        jobs_resp = requests.get(f"{api_url}/jobs", timeout=20)
        jobs_resp.raise_for_status()
        jobs = jobs_resp.json().get("data", [])
    except Exception as exc:
        st.info(f"Jobs unavailable: {exc}")
        return

    if not jobs:
        st.caption("No jobs yet.")
        return

    jobs_df = pd.DataFrame(jobs)
    keep = [c for c in ["id", "kind", "status", "progress", "message", "created_at", "finished_at"] if c in jobs_df.columns]
    st.dataframe(jobs_df[keep], use_container_width=True, height=220)


def _safe_day_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "day_order" in df.columns:
        return df.sort_values("day_order")
    if "day_label" in df.columns:
        return df.sort_values("day_label")
    return df


def main() -> None:
    st.set_page_config(page_title="BALD-Explorer", layout="wide")
    st.title("BALD-Explorer Publication Prototype")
    st.caption("Trajectory storytelling, model-driven gene rankings, pathway evidence, and export presets")

    default_api = os.environ.get("BALD_API_URL", "http://127.0.0.1:8000")

    with st.sidebar:
        st.header("Data Source")
        mode = st.radio("Load from", ["Local artifacts", "FastAPI backend"], index=0)
        artifacts_dir = Path(
            st.text_input("Artifacts directory", value=str(DEFAULT_ARTIFACTS_DIR))
        )
        api_url = st.text_input("API URL", value=default_api)

        st.divider()
        st.header("Run Controls")
        st.caption("Use API async jobs for long-running artifact builds")
        counts_path = st.text_input("Counts path", value="")
        metadata_path = st.text_input("Metadata path", value="")
        checkpoint_path = st.text_input("Checkpoint path (optional)", value="")
        gmt_paths = st.text_input("GMT paths (comma-separated, optional)", value="")
        run_top_k = st.slider("Top K for run", min_value=20, max_value=500, value=100, step=10)
        run_device = st.selectbox("Run device", ["cpu", "cuda"], index=0)
        if st.button("Start Async Build Job", use_container_width=True):
            if not counts_path or not metadata_path:
                st.warning("Counts and metadata paths are required to start a run.")
            else:
                payload = {
                    "counts_path": counts_path,
                    "metadata_path": metadata_path,
                    "output_dir": str(artifacts_dir),
                    "top_k": run_top_k,
                    "checkpoint_path": checkpoint_path or None,
                    "gmt_paths": [x.strip() for x in gmt_paths.split(",") if x.strip()],
                    "device": run_device,
                }
                try:
                    resp = requests.post(f"{api_url.rstrip('/')}/jobs/build-artifacts", json=payload, timeout=30)
                    resp.raise_for_status()
                    job = resp.json().get("data", {})
                    st.success(f"Started job {job.get('id', '')}")
                except Exception as exc:
                    st.error(f"Could not start job: {exc}")

    try:
        if mode == "FastAPI backend":
            points, traj, genes = _load_from_api(api_url.rstrip("/"))
            publication_tables = {}
        else:
            points, traj, genes = _load_local_artifacts(artifacts_dir)
            publication_tables = _load_publication_tables(artifacts_dir)
    except Exception as exc:
        st.error(f"Could not load artifacts: {exc}")
        st.stop()

    if points.empty or traj.empty:
        st.warning("Trajectory artifacts are empty. Build artifacts first.")
        st.stop()

    if "BioProject" not in points.columns:
        st.error("Expected 'BioProject' column in latent_points.csv")
        st.stop()

    projects = sorted(points["BioProject"].astype(str).unique().tolist())
    selected_projects = st.multiselect("BioProject filter", projects, default=projects[: min(8, len(projects))])

    p_df = points[points["BioProject"].astype(str).isin(selected_projects)].copy()
    t_df = traj[traj["BioProject"].astype(str).isin(selected_projects)].copy()
    t_df = _safe_day_sort(t_df)

    tabs = st.tabs(["Story", "Trajectories", "Top Genes", "Pathways", "Exports", "Jobs"])

    with tabs[0]:
        st.subheader("Trajectory Storyline")
        if {"BioProject", "day_label", "pc1", "pc2"}.issubset(t_df.columns):
            story = (
                t_df.sort_values([c for c in ["BioProject", "day_order"] if c in t_df.columns])
                .groupby("BioProject", as_index=False)
                .agg(
                    n_timepoints=("day_label", "nunique"),
                    pc1_shift=("pc1", lambda s: float(s.max() - s.min())),
                    pc2_shift=("pc2", lambda s: float(s.max() - s.min())),
                )
                .sort_values("pc1_shift", ascending=False)
            )
            st.dataframe(story, use_container_width=True, height=280)
            st.caption("Use this table directly as a manuscript supplement for trajectory shift magnitude.")
        else:
            st.info("Not enough trajectory columns for storyline table.")

    with tabs[1]:
        st.subheader("Latent Trajectories")
        fig = None
        if {"pc1", "pc2"}.issubset(t_df.columns):
            fig = px.line(
                t_df,
                x="pc1",
                y="pc2",
                color="BioProject",
                line_group="BioProject",
                hover_data=[c for c in ["day_label", "day_order"] if c in t_df.columns],
                markers=True,
            )
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("pc1/pc2 columns missing in trajectory_summary.csv")

        st.subheader("Sample-Level Latent Map")
        fig2 = None
        if {"pc1", "pc2"}.issubset(p_df.columns):
            color_col = "day_label" if "day_label" in p_df.columns else "BioProject"
            fig2 = px.scatter(
                p_df,
                x="pc1",
                y="pc2",
                color=color_col,
                hover_data=[c for c in ["Run", "BioProject", "SampleTimepoint"] if c in p_df.columns],
            )
            fig2.update_layout(height=520)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("pc1/pc2 columns missing in latent_points.csv")

    with tabs[2]:
        st.subheader("Top Genes")
        if genes.empty:
            st.info("No top gene table found.")
        else:
            g_df = genes.copy()
            if "group" in g_df.columns:
                groups = sorted(g_df["group"].astype(str).unique().tolist())
                selected_group = st.selectbox("Group", groups, index=0)
                g_df = g_df[g_df["group"].astype(str) == selected_group]

            if "day_order" in g_df.columns:
                day_values = sorted(g_df["day_order"].dropna().astype(int).unique().tolist())
                if day_values:
                    selected_day = st.selectbox("Day", day_values, index=0)
                    g_df = g_df[g_df["day_order"].astype(int) == selected_day]

            top_k = st.slider("Top K", min_value=10, max_value=300, value=80, step=10)
            if "rank" in g_df.columns:
                g_df = g_df.sort_values("rank").head(top_k)
            else:
                g_df = g_df.head(top_k)

            st.dataframe(g_df, use_container_width=True, height=620)

    with tabs[3]:
        st.subheader("Pathway Evidence")
        path_df = publication_tables.get("pathway_enrichment_ig.csv", pd.DataFrame()) if mode == "Local artifacts" else pd.DataFrame()
        if path_df.empty:
            st.info("No pathway enrichment artifact found. Build with --gmt to generate pathway tables.")
        else:
            if "group" in path_df.columns:
                grp = st.selectbox("Pathway group", sorted(path_df["group"].astype(str).unique()))
                path_df = path_df[path_df["group"].astype(str) == str(grp)]
            st.dataframe(path_df.head(200), use_container_width=True, height=560)

    with tabs[4]:
        st.subheader("Publication Exports")
        if mode == "Local artifacts":
            _df_download(t_df, "Download trajectory_summary.csv", "trajectory_summary.csv")
            _df_download(genes if not genes.empty else pd.DataFrame(), "Download top_genes.csv", "top_genes.csv")
            if "publication_top_genes.csv" in publication_tables:
                _df_download(
                    publication_tables["publication_top_genes.csv"],
                    "Download publication_top_genes.csv",
                    "publication_top_genes.csv",
                )
            if "publication_trajectory_table.csv" in publication_tables:
                _df_download(
                    publication_tables["publication_trajectory_table.csv"],
                    "Download publication_trajectory_table.csv",
                    "publication_trajectory_table.csv",
                )
            if "publication_pathway_table.csv" in publication_tables:
                _df_download(
                    publication_tables["publication_pathway_table.csv"],
                    "Download publication_pathway_table.csv",
                    "publication_pathway_table.csv",
                )
            if "fig" in locals() and fig is not None:
                _plot_download(fig, "Download trajectory_figure.json", "trajectory_figure.json")
            if "fig2" in locals() and fig2 is not None:
                _plot_download(fig2, "Download sample_map_figure.json", "sample_map_figure.json")
        else:
            st.caption("Switch to Local artifacts mode to use direct file export controls.")

    with tabs[5]:
        _show_jobs_panel(api_url.rstrip("/"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import traceback
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.bald_data import build_trajectory_artifacts, load_logcpm_and_metadata
from src.bald_explainability import RealAttributionConfig, run_model_attributions

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = APP_ROOT / "outputs" / "bald_explorer"
DEFAULT_STATUS_PATH = DEFAULT_ARTIFACTS_DIR / "run_status.json"
JOBS_DIR = DEFAULT_ARTIFACTS_DIR / "jobs"

app = FastAPI(title="BALD-Explorer API", version="0.1.0")
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOB_LOCK = Lock()
JOB_STATE: Dict[str, Dict[str, Any]] = {}


class RunStatusUpdate(BaseModel):
    state: str = Field(..., description="idle|running|completed|failed")
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    artifacts: Dict[str, str] = Field(default_factory=dict)


class BuildArtifactsJobRequest(BaseModel):
    counts_path: str
    metadata_path: str
    output_dir: str = str(DEFAULT_ARTIFACTS_DIR)
    top_k: int = 100
    checkpoint_path: Optional[str] = None
    methods: List[str] = Field(default_factory=lambda: ["integrated_gradients", "shap"])
    feature_list_path: Optional[str] = None
    gmt_paths: List[str] = Field(default_factory=list)
    group_by_project_only: bool = False
    device: str = "cpu"
    model_hidden_dim: int = 256
    model_output_dim: int = 16
    shap_feature_cap: int = 300
    shap_max_samples: int = 96
    shap_nsamples: int = 128


class JobEnvelope(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    artifacts: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None


def _read_status(status_path: Path) -> Dict[str, object]:
    if not status_path.exists():
        return {
            "state": "idle",
            "progress": 0.0,
            "message": "No run has been started yet.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {},
        }
    with status_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_status(status_path: Path, payload: Dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_job_state(job: Dict[str, Any]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = JOBS_DIR / f"{job['id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)


def _upsert_job(job_id: str, **updates: Any) -> Dict[str, Any]:
    with JOB_LOCK:
        base = JOB_STATE.get(job_id, {})
        base.update(updates)
        JOB_STATE[job_id] = base
        _write_job_state(base)
        return base


def _create_job(kind: str, message: str = "Queued") -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    job = JobEnvelope(
        id=uuid.uuid4().hex,
        kind=kind,
        status="queued",
        created_at=now,
        progress=0.0,
        message=message,
    ).model_dump()
    with JOB_LOCK:
        JOB_STATE[job["id"]] = job
        _write_job_state(job)
    return job


def _build_job_worker(job_id: str, req: BuildArtifactsJobRequest) -> None:
    _upsert_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        progress=0.1,
        message="Loading counts and metadata",
    )

    output_dir = Path(req.output_dir)
    status_path = output_dir / "run_status.json"

    _write_status(
        status_path,
        {
            "state": "running",
            "progress": 0.1,
            "message": "Loading data",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {},
        },
    )

    try:
        expr, meta, _genes = load_logcpm_and_metadata(
            counts_path=Path(req.counts_path),
            metadata_path=Path(req.metadata_path),
        )
        _upsert_job(job_id, progress=0.35, message="Building trajectory artifacts")

        artifacts = build_trajectory_artifacts(
            expr_samples_by_genes=expr,
            meta=meta,
            output_dir=output_dir,
            top_k_genes=req.top_k,
        )

        if req.checkpoint_path:
            _upsert_job(job_id, progress=0.55, message="Running model attributions")
            traj_df = pd.read_csv(artifacts["trajectory_summary"])
            attr_artifacts = run_model_attributions(
                expr_samples_by_genes=expr,
                meta=meta,
                trajectory_summary=traj_df,
                config=RealAttributionConfig(
                    checkpoint_path=Path(req.checkpoint_path),
                    output_dir=output_dir,
                    methods=req.methods,
                    top_k=req.top_k,
                    group_by_day=not req.group_by_project_only,
                    device=req.device,
                    model_hidden_dim=req.model_hidden_dim,
                    model_output_dim=req.model_output_dim,
                    shap_feature_cap=req.shap_feature_cap,
                    shap_max_samples=req.shap_max_samples,
                    shap_nsamples=req.shap_nsamples,
                    feature_list_path=Path(req.feature_list_path) if req.feature_list_path else None,
                    gmt_paths=[Path(p) for p in req.gmt_paths] if req.gmt_paths else None,
                ),
            )
            artifacts.update(attr_artifacts)

        payload = {
            "state": "completed",
            "progress": 1.0,
            "message": "Artifacts built successfully.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {k: str(v) for k, v in artifacts.items()},
        }
        _write_status(status_path, payload)

        _upsert_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            progress=1.0,
            message="Build complete",
            artifacts=payload["artifacts"],
            error=None,
        )
    except Exception as exc:
        err = "\n".join(traceback.format_exception_only(type(exc), exc)).strip()
        _write_status(
            status_path,
            {
                "state": "failed",
                "progress": 1.0,
                "message": err,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": {},
            },
        )
        _upsert_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            progress=1.0,
            message="Build failed",
            error=err,
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "BALD-Explorer API"}


@app.get("/jobs")
def list_jobs() -> Dict[str, object]:
    with JOB_LOCK:
        jobs = sorted(JOB_STATE.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return {"success": True, "data": jobs}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, object]:
    with JOB_LOCK:
        job = JOB_STATE.get(job_id)
    if not job:
        path = JOBS_DIR / f"{job_id}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                job = json.load(f)
        else:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"success": True, "data": job}


@app.post("/jobs/build-artifacts")
async def start_build_artifacts_job(req: BuildArtifactsJobRequest) -> Dict[str, object]:
    job = _create_job(kind="build_artifacts", message="Queued build job")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(JOB_EXECUTOR, _build_job_worker, job["id"], req)
    return {"success": True, "data": job}


@app.get("/run-status")
def get_run_status(
    status_path: Optional[str] = Query(default=None),
):
    path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
    return {"success": True, "data": _read_status(path)}


@app.post("/run-status")
def update_run_status(
    update: RunStatusUpdate,
    status_path: Optional[str] = Query(default=None),
):
    path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
    payload = update.model_dump()
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(path, payload)
    return {"success": True, "data": payload}


@app.get("/artifacts")
def list_artifacts(artifacts_dir: Optional[str] = Query(default=None)):
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    if not base.exists():
        return {"success": True, "data": []}

    files = []
    for p in sorted(base.glob("*")):
        if p.is_file():
            files.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "modified_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )
    return {"success": True, "data": files}


@app.get("/artifacts/publication")
def publication_artifacts(artifacts_dir: Optional[str] = Query(default=None)):
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    candidates = [
        "publication_top_genes.csv",
        "publication_trajectory_table.csv",
        "publication_pathway_table.csv",
        "pathway_enrichment_ig.csv",
    ]
    out: Dict[str, List[Dict[str, object]]] = {"files": []}
    for name in candidates:
        path = base / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        out["files"].append({
            "name": name,
            "n_rows": int(df.shape[0]),
            "columns": df.columns.tolist(),
        })
    return {"success": True, "data": out}


@app.get("/artifacts/file/{file_name}")
def get_artifact_file(
    file_name: str,
    artifacts_dir: Optional[str] = Query(default=None),
):
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    path = base / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {file_name}")
    return FileResponse(str(path))


@app.get("/artifacts/top-genes")
def top_genes(
    group: Optional[str] = Query(default=None),
    day_order: Optional[int] = Query(default=None),
    top_k: int = Query(default=50, ge=1, le=5000),
    artifacts_dir: Optional[str] = Query(default=None),
):
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    candidate_names = [
        "top_genes_by_group_integrated_gradients.csv",
        "top_genes_by_group_shap.csv",
        "top_genes_by_group.csv",
    ]
    path = None
    for name in candidate_names:
        candidate = base / name
        if candidate.exists():
            path = candidate
            break
    if path is None:
        raise HTTPException(status_code=404, detail="No top-genes artifact found")

    df = pd.read_csv(path)
    if group is not None and "group" in df.columns:
        df = df[df["group"].astype(str) == str(group)]
    if day_order is not None and "day_order" in df.columns:
        df = df[df["day_order"] == day_order]

    if "rank" in df.columns:
        df = df.sort_values("rank").head(top_k)
    else:
        df = df.head(top_k)

    return {
        "success": True,
        "data": {
            "rows": df.to_dict(orient="records"),
            "count": int(df.shape[0]),
        },
    }


@app.get("/artifacts/trajectories")
def trajectories(artifacts_dir: Optional[str] = Query(default=None)):
    base = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    points_path = base / "latent_points.csv"
    traj_path = base / "trajectory_summary.csv"

    if not points_path.exists() or not traj_path.exists():
        raise HTTPException(status_code=404, detail="Required trajectory artifacts were not found")

    points_df = pd.read_csv(points_path)
    traj_df = pd.read_csv(traj_path)

    return {
        "success": True,
        "data": {
            "points": points_df.to_dict(orient="records"),
            "trajectories": traj_df.to_dict(orient="records"),
            "n_points": int(points_df.shape[0]),
            "n_trajectories": int(traj_df.shape[0]),
        },
    }

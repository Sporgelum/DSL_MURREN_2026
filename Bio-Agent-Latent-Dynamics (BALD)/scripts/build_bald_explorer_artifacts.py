from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bald_data import build_trajectory_artifacts, load_logcpm_and_metadata
from src.bald_explainability import RealAttributionConfig, run_model_attributions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BALD-Explorer starter artifacts from logCPM and metadata")
    p.add_argument(
        "--counts",
        type=Path,
        default=Path("../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv"),
        help="Path to logCPM matrix (genes x samples)",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv"),
        help="Path to metadata file",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bald_explorer"),
        help="Output artifact directory",
    )
    p.add_argument("--top-k", type=int, default=100, help="Top genes per group/day for export")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint path for model-driven attributions",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["integrated_gradients", "shap"],
        choices=["integrated_gradients", "gradient_x_input", "shap"],
        help="Attribution methods to run when checkpoint is provided",
    )
    p.add_argument(
        "--feature-list",
        type=Path,
        default=None,
        help="Optional text file with one model feature (gene) per line",
    )
    p.add_argument(
        "--gmt",
        nargs="*",
        default=None,
        help="Optional GMT files for pathway enrichment export",
    )
    p.add_argument("--model-hidden-dim", type=int, default=256)
    p.add_argument("--model-output-dim", type=int, default=16)
    p.add_argument("--shap-feature-cap", type=int, default=300)
    p.add_argument("--shap-max-samples", type=int, default=96)
    p.add_argument("--shap-nsamples", type=int, default=128)
    p.add_argument("--group-by-project-only", action="store_true", help="Group attributions by BioProject only")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    counts_path = args.counts.resolve()
    metadata_path = args.metadata.resolve()
    output_dir = args.output_dir.resolve()

    expr, meta, _genes = load_logcpm_and_metadata(
        counts_path=counts_path,
        metadata_path=metadata_path,
    )

    artifacts = build_trajectory_artifacts(
        expr_samples_by_genes=expr,
        meta=meta,
        output_dir=output_dir,
        top_k_genes=args.top_k,
    )

    if args.checkpoint is not None:
        methods: List[str] = list(args.methods)
        trajectory_summary = pd.read_csv(artifacts["trajectory_summary"])
        attr_artifacts = run_model_attributions(
            expr_samples_by_genes=expr,
            meta=meta,
            trajectory_summary=trajectory_summary,
            config=RealAttributionConfig(
                checkpoint_path=args.checkpoint.resolve(),
                output_dir=output_dir,
                methods=methods,
                top_k=args.top_k,
                group_by_day=not args.group_by_project_only,
                device=args.device,
                model_hidden_dim=args.model_hidden_dim,
                model_output_dim=args.model_output_dim,
                shap_feature_cap=args.shap_feature_cap,
                shap_max_samples=args.shap_max_samples,
                shap_nsamples=args.shap_nsamples,
                feature_list_path=args.feature_list.resolve() if args.feature_list else None,
                gmt_paths=[Path(p).resolve() for p in args.gmt] if args.gmt else None,
            ),
        )
        artifacts.update(attr_artifacts)

    status = {
        "state": "completed",
        "progress": 1.0,
        "message": "Artifacts built successfully.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    status_path = output_dir / "run_status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print("BALD-Explorer artifacts built:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")
    print(f"- run_status: {status_path}")


if __name__ == "__main__":
    main()

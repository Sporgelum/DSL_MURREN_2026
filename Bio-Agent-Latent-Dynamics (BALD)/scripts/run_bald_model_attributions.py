from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bald_data import load_logcpm_and_metadata
from src.bald_explainability import RealAttributionConfig, run_model_attributions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model-driven BALD attributions on real logCPM data")
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectory-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/bald_explorer"))
    p.add_argument("--methods", nargs="+", default=["integrated_gradients", "shap"], choices=["integrated_gradients", "gradient_x_input", "shap"])
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--feature-list", type=Path, default=None)
    p.add_argument("--gmt", nargs="*", default=None)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--model-hidden-dim", type=int, default=256)
    p.add_argument("--model-output-dim", type=int, default=16)
    p.add_argument("--shap-feature-cap", type=int, default=300)
    p.add_argument("--shap-max-samples", type=int, default=96)
    p.add_argument("--shap-nsamples", type=int, default=128)
    p.add_argument("--group-by-project-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    expr, meta, _genes = load_logcpm_and_metadata(
        counts_path=args.counts.resolve(),
        metadata_path=args.metadata.resolve(),
    )
    traj = pd.read_csv(args.trajectory_summary.resolve())

    artifacts = run_model_attributions(
        expr_samples_by_genes=expr,
        meta=meta,
        trajectory_summary=traj,
        config=RealAttributionConfig(
            checkpoint_path=args.checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            methods=list(args.methods),
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

    print("Model attribution artifacts exported:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

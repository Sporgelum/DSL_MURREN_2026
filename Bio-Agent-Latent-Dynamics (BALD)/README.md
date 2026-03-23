# BioAgent-Latent-Dynamics (BALD)

A multi-agent and interpretable framework for forecasting transcriptomic trajectories and identifying trajectory-driving genes.

## Why BALD

Most transcriptomics pipelines answer what changed on average.

BALD is designed to answer:

- where an individual sample is moving across time,
- why the model predicts that movement,
- which genes are most responsible for the predicted trajectory.

The target use case is vaccine or perturbation-response data with repeated time points (for example Day 0, Day 1, Day 7).

## Core Idea

BALD combines three layers:

1. Manifold layer: compress high-dimensional expression into a structured latent space.
2. Dynamics layer: predict sample trajectories through latent space over time.
3. Interaction layer: model context effects between samples via graph-based interactions.

Then a dedicated explainability layer maps predictions back to gene-level importance and stability.

## What We Learned from MiroFish and Reused

We do not copy MiroFish domain logic. We reuse system design patterns that are strong for complex simulation products:

1. Stage-based workflow

- MiroFish uses explicit stages from setup to report.
- BALD adopts this as data -> latent training -> dynamics -> explainability -> counterfactuals -> explorer.

1. State-first execution

- MiroFish persists simulation state and status transitions.
- BALD should persist run metadata, checkpoints, and explainability outputs as first-class run state.

1. Structured event logs

- MiroFish records actions per round in structured logs.
- BALD should log per-sample trajectory events, attribution outputs, and perturbation outcomes for auditability.

1. Real-time monitoring APIs

- MiroFish supports live status and detailed run endpoints.
- BALD-Explorer should expose live endpoints for model progress and post-hoc analysis playback.

1. Report generation mindset

- MiroFish ends with report-centric interaction.
- BALD should end with publication-centric exports: stable gene panels, pathway tables, and figure-ready assets.

## Current Repository Assets

- Improved proposal: project_description_improved_v1.md
- Implementation roadmap: implementation_ready_plan.md
- Explainability package:
- src/bald_explainability/attribution.py
- src/bald_explainability/exporter.py
- src/bald_explainability/stability.py
- src/bald_explainability/pipeline.py
- Data/artifact pipeline:
- src/bald_data/loaders.py
- src/bald_data/trajectory_artifacts.py
- scripts/build_bald_explorer_artifacts.py
- BALD-Explorer services:
- src/bald_explorer_api/main.py
- app/bald_explorer_streamlit.py
- Dependency list:
- requirements_bald_explorer.txt

## Explainability Skeleton (Now Included)

The initial code in src/bald_explainability supports:

- Integrated Gradients attribution
- Gradient x input attribution
- SHAP adapter (KernelExplainer-based)
- Group-level consensus rankings
- CSV export of ranked genes per group
- Stability metrics:
- top-k Jaccard overlap
- Spearman rank correlation

This gives a practical baseline for identifying genes that drive trajectory predictions.

## Quick Start

Prerequisites:

- Python 3.10+
- PyTorch
- NumPy
- FastAPI
- Streamlit

Install dependencies:

```bash
pip install -r requirements_bald_explorer.txt
```

Example run:

```bash
python examples/minimal_explainability_demo.py
```

Expected output:

- top gene indices for each sample group
- stability summary metrics

## Build Artifacts From Your logCPM + Metadata

Your data source is expected at:

- ../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv
- ../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv

Build BALD-Explorer artifacts:

```bash
python scripts/build_bald_explorer_artifacts.py \
  --counts "../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv" \
  --metadata "../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv" \
  --output-dir "outputs/bald_explorer" \
  --top-k 100
```

PowerShell command using your Python executable:

```powershell
& "C:/Users/emari/OneDrive - Universitaet Bern (1)/Documents/Environments/scimilarity_2024_local/Scripts/python.exe" scripts/build_bald_explorer_artifacts.py --counts "../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv" --metadata "../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv" --output-dir "outputs/bald_explorer" --top-k 100
```

Build artifacts plus model-driven attributions (Integrated Gradients + SHAP), pathway enrichment, and publication tables:

```powershell
& "C:/Users/emari/OneDrive - Universitaet Bern (1)/Documents/Environments/scimilarity_2024_local/Scripts/python.exe" scripts/build_bald_explorer_artifacts.py --counts "../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv" --metadata "../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv" --output-dir "outputs/bald_explorer" --checkpoint "path/to/your_bald_checkpoint.pth" --feature-list "path/to/model_feature_order.txt" --methods integrated_gradients shap --gmt "path/to/go_bp.gmt" "path/to/reactome.gmt" --device cpu --top-k 150
```

Run attribution-only export when trajectory artifacts already exist:

```powershell
& "C:/Users/emari/OneDrive - Universitaet Bern (1)/Documents/Environments/scimilarity_2024_local/Scripts/python.exe" scripts/run_bald_model_attributions.py --counts "../Project_plan/counts_and_metadata/logCPM_matrix_filtered_samples.csv" --metadata "../Project_plan/counts_and_metadata/metadata_with_sample_annotations.csv" --trajectory-summary "outputs/bald_explorer/trajectory_summary.csv" --checkpoint "path/to/your_bald_checkpoint.pth" --feature-list "path/to/model_feature_order.txt" --methods integrated_gradients shap --gmt "path/to/go_bp.gmt" "path/to/reactome.gmt" --output-dir "outputs/bald_explorer"
```

Generated files:

- outputs/bald_explorer/latent_points.csv
- outputs/bald_explorer/trajectory_summary.csv
- outputs/bald_explorer/top_genes_by_group.csv
- outputs/bald_explorer/top_genes_by_group_integrated_gradients.csv (when checkpoint is provided)
- outputs/bald_explorer/top_genes_by_group_shap.csv (when checkpoint is provided)
- outputs/bald_explorer/pathway_enrichment_ig.csv (when GMT files are provided)
- outputs/bald_explorer/publication_top_genes.csv
- outputs/bald_explorer/publication_trajectory_table.csv
- outputs/bald_explorer/publication_pathway_table.csv
- outputs/bald_explorer/run_status.json

## Start FastAPI Backend

```bash
uvicorn src.bald_explorer_api.main:app --reload --port 8000
```

Key endpoints:

- GET /health
- GET /run-status
- POST /run-status
- GET /jobs
- GET /jobs/{job_id}
- POST /jobs/build-artifacts
- GET /artifacts
- GET /artifacts/trajectories
- GET /artifacts/top-genes
- GET /artifacts/publication
- GET /artifacts/file/{file_name}

## Start Streamlit Prototype

```bash
streamlit run app/bald_explorer_streamlit.py
```

The UI supports:

- trajectory storyline table for publication supplements
- latent trajectory view (BioProject x day)
- sample-level latent map (colored by day)
- top-gene ranking filters (project/day/method)
- pathway evidence panel from enrichment artifacts
- export presets for CSV tables + figure JSON bundles
- local artifact mode and FastAPI mode
- async job monitor and API-triggered long-run execution

## BALD-Explorer Vision

BALD-Explorer is the publication-facing interface.

MVP modules:

1. Latent Map

- animate Day 0 -> Day 1 -> Day 7 paths with uncertainty ribbons.

1. Trajectory Inspector

- inspect one sample against cohort context.

1. Gene Importance

- show top-k genes per sample/group and method.

1. Counterfactual Lab

- perturb selected genes and compare predicted endpoint shift.

1. Export Center

- one-click SVG/PNG and methods metadata bundles for figures.

## Publication Strategy

BALD is built to support publication-ready claims with reproducibility:

- deterministic seed and config snapshots,
- robust attribution over folds/seeds,
- external signature overlap and pathway enrichment,
- explicit ablations (with and without interaction module).

## Immediate Next Steps

1. Plug your final production checkpoint and feature-order file into scripts/build_bald_explorer_artifacts.py.
2. Replace the PCA starter trajectory with your trained manifold embeddings.
3. Add pathway libraries specific to your biological question set.
4. Use async /jobs/build-artifacts execution for long cluster-like runs.

## Collaboration Note

Yes, this is achievable. The repository now has:

- a refined project narrative,
- a starter explainability codebase,
- and a concrete implementation-ready plan.

Next, we can move directly into model-training contracts and the first BALD-Explorer MVP endpoint set.

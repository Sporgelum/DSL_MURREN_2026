# BALD Implementation-Ready Plan (MiroFish-informed)

## Goal
Turn BALD into an end-to-end system with:
- trajectory prediction,
- robust gene-level explainability,
- an interactive BALD-Explorer for publication-grade outputs.

## What We Reuse from MiroFish (Design Patterns)
1. Stage-based workflow orchestration.
2. Persistent run state and simulation status APIs.
3. Structured action/event logs for replay and analytics.
4. Real-time monitoring endpoints for front-end dashboards.
5. Report-oriented final layer that turns outputs into narrative artifacts.

## BALD System Blueprint
1. Stage 1: Data and split manager
- Inputs: count matrix, metadata, gene annotations.
- Outputs: versioned train/val/test splits and preprocessing artifacts.

2. Stage 2: Latent model trainer
- Models: beta-VAE and conditional alternatives.
- Outputs: latent embeddings, decoder checkpoints, uncertainty estimates.

3. Stage 3: Trajectory engine
- Models: Neural ODE baseline and sequence baseline.
- Outputs: Day 1 and Day 7 predictions in latent and gene space.

4. Stage 4: Explainability engine
- Methods: Integrated Gradients and gradient x input first.
- Outputs: sample-level and cohort-level ranked genes plus stability metrics.

5. Stage 5: Counterfactual lab
- In-silico perturbation at gene level.
- Outputs: effect-size tables for endpoint and phenotype shifts.

6. Stage 6: BALD-Explorer
- Live trajectory map, ranked genes, pathway panel, and export panel.
- Outputs: publication-ready figures and reproducible report bundle.

## Proposed Repo Layout
- src/bald_data/
- src/bald_models/
- src/bald_dynamics/
- src/bald_explainability/
- src/bald_counterfactual/
- src/bald_explorer_api/
- app/
- configs/
- outputs/
- docs/

## Immediate Build Sequence (First 3 Weeks)
1. Week 1
- Freeze data contracts: AnnData schema, split format, model IO format.
- Add a single CLI command for dry-run validation.

2. Week 2
- Train minimal beta-VAE and store embeddings/checkpoint.
- Train one baseline trajectory model and benchmark endpoint error.

3. Week 3
- Plug in explainability module from src/bald_explainability.
- Export top genes by group and stability report (Jaccard, Spearman).

## Engineering Contracts
1. Prediction contract
- Input: Day 0 expression vector and optional metadata.
- Output: predicted Day 1 and Day 7 latent vectors and decoded expression.

2. Explainability contract
- Input: model checkpoint and sample matrix.
- Output: attribution matrix plus ranked gene tables by group.

3. Counterfactual contract
- Input: sample, selected genes, perturbation magnitudes.
- Output: shifted trajectory and effect-size summary.

## BALD-Explorer MVP
1. Tab A: Latent map
- Animate Day 0 to Day 7 with confidence ribbons.

2. Tab B: Trajectory inspector
- Per-sample path and nearest-neighbor context.

3. Tab C: Gene importance
- Top-k bars with method selector and stability badge.

4. Tab D: Counterfactual sandbox
- Pick a gene, set perturbation, compare before and after trajectory.

5. Tab E: Export center
- One-click SVG figures and methods metadata JSON.

## Metrics to Lock for Publication
1. Prediction
- MAE and MSE in latent and expression space.

2. Explainability robustness
- Mean pairwise top-k Jaccard.
- Mean pairwise Spearman of rankings.

3. Biological relevance
- Pathway enrichment significance and overlap with known signatures.

## Risks and Mitigation
1. Small sample size
- Nested CV, bootstrap, and strict hold-out reporting.

2. Attribution variance
- Multi-run consensus and minimum stability thresholds.

3. Tooling drift
- Versioned config snapshots and deterministic seeds.

## Next Actions in This Repo
1. Expand src/bald_explainability to include SHAP adapters.
2. Add a training/evaluation CLI for reproducible batch runs.
3. Start a FastAPI backend for BALD-Explorer data endpoints.
4. Add a lightweight front-end (Streamlit first, then React if needed).

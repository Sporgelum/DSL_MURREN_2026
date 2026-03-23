# BioAgent-Latent-Dynamics (BALD) - Improved Project Description v1

## Subtitle
A Multi-Agent, Interpretable Framework for Predicting and Explaining Transcriptomic Trajectories.

## 1) Project Vision
BALD aims to move from static transcriptomics to a dynamic and explainable simulation of immune response trajectories.  
Each biological sample is represented as an agent evolving in a learned latent manifold under vaccine perturbation, time, and interaction constraints.

The core scientific shift is:
- from group-level average changes -> to individual-level trajectory forecasts
- from black-box prediction -> to gene-level mechanistic interpretation

## 2) Scientific Questions and Hypotheses
### Q1: Can we forecast sample-specific trajectories across Day 0 -> Day 1 -> Day 7?
Hypothesis: A latent dynamical model (Neural ODE/LSTM/Transformer) can predict future states better than static baselines.

### Q2: Which genes most strongly drive trajectory predictions?
Hypothesis: A small, stable set of genes/pathways explains most trajectory variance and differs by responder phenotype.

### Q3: Can interactions between agents improve prediction and support counterfactual simulation?
Hypothesis: Interaction-aware models (graph dynamics or MARL-inspired message passing) improve both predictive performance and biological plausibility.

## 3) Proposed Architecture (Refined)
### Layer A: Manifold Learning (Representation)
- Input: RNA-seq matrix (samples x genes), metadata (timepoint, condition, responder status)
- Model: beta-VAE or conditional VAE with regularized latent space
- Output: Compact latent embedding z (32-64 dims), uncertainty estimates, and reconstructed expression

Design improvements:
- Add batch correction and covariate control (e.g., donor, sex, age, library effects)
- Enforce temporal smoothness regularization between consecutive timepoints
- Track latent uncertainty to flag unreliable trajectory regions

### Layer B: Temporal Dynamics (Per-Agent Evolution)
- Initial state: each sample agent starts at its Day 0 latent vector
- Dynamics model options:
  - Neural ODE for continuous trajectories
  - Sequence model (LSTM/Transformer) for discrete Day 0, Day 1, Day 7 transitions
- Output: Predicted latent state at future timepoints and decoded gene expression profile

Design improvements:
- Multi-task objective: latent prediction + expression reconstruction + phenotype classification
- Calibrated confidence intervals for trajectory endpoints

### Layer C: Agent Interaction (Population Context)
- Build a dynamic graph of samples in latent space (kNN or phenotype-aware graph)
- Let agents exchange signals via graph neural message passing
- Use controlled interaction strength to test "social" influence of perturbation

Design improvements:
- Compare no-interaction vs interaction models via ablations
- Add counterfactual interaction mode: "vaccinated version" of control profiles

## 4) Explainability and Learned-Feature Extraction Strategy
This is the critical new component for publication-quality insight.

### 4.1 Feature Attribution at Prediction Time
For each trajectory prediction, compute gene contributions using multiple complementary methods:
- Integrated Gradients (gene-level attribution from input to predicted future state)
- SHAP (model-agnostic or deep variants)
- Input-gradient and gradient x input saliency
- Attention weights (if using transformer-based dynamics)

Output:
- per-sample ranked genes
- per-group consensus ranking (high responders, low responders, controls)

### 4.2 Latent-to-Gene Sensitivity Mapping
Quantify how each latent dimension impacts genes and pathways:
- compute Jacobian-based sensitivities d(decoded_gene)/d(z_i)
- discover interpretable latent axes (e.g., inflammation, IFN response, antigen presentation)

Output:
- latent dimension annotation table
- pathway-labeled latent directions

### 4.3 Trajectory Velocity Decomposition
Decompose movement between timepoints into gene programs:
- map latent velocity vectors back to gene space
- identify genes that dominate Day 0 -> Day 1 and Day 1 -> Day 7 shifts

Output:
- early-response vs late-response gene modules
- per-time-interval importance scores

### 4.4 Counterfactual Gene Screening
Perform in-silico perturbation to test causal relevance:
- knock-down / knock-up candidate genes in input
- simulate resulting trajectory shift
- rank genes by effect size on endpoint movement and phenotype probability

Output:
- prioritized candidate biomarkers/intervention targets
- effect-size ranked candidate list with confidence intervals

### 4.5 Robustness and Stability of Important Genes
A feature is publication-grade only if stable:
- repeat attribution across seeds, folds, and bootstrap samples
- report stability metrics (rank correlation, Jaccard overlap, consensus score)
- retain only robust genes/pathways

Output:
- robust importance panel
- reproducibility report for supplementary material

## 5) Validation and Benchmarking Plan
### Predictive performance
- endpoint error in latent and gene space (MSE/MAE)
- trajectory similarity metrics (cosine, dynamic time warping if densified)
- phenotype prediction AUROC/AUPRC from predicted Day 7 state

### Biological validity
- pathway enrichment (GSEA/Reactome/GO) of top-ranked genes
- overlap with known vaccine response signatures
- external cohort transfer test (if available)

### Baselines
- static models (random forest, elastic net on Day 0)
- non-interaction latent dynamics
- simple linear latent transition model

## 6) Engineering Plan for a Beautiful Tool (BALD-Explorer)
Build an interactive research app for model inspection and storytelling.

### Core modules
- Latent Map View: UMAP/3D latent landscape with timepoint animation
- Agent Trajectory View: individual and cohort trajectories with uncertainty ribbons
- Gene Importance View: top genes per sample/group/time interval
- Counterfactual Lab: choose a gene, simulate perturbation, visualize trajectory shift
- Pathway Dashboard: enrichment plots linked to selected trajectory segments

### Suggested stack
- Backend: Python + FastAPI + PyTorch inference service
- Data layer: AnnData/HDF5 + parquet summaries
- Frontend: Plotly Dash or Streamlit (fast start), optional React + deck.gl (advanced)

### Publication-ready UX outputs
- exportable figure panels (SVG/PNG)
- reproducible report bundle (JSON + markdown narrative)
- automatic methods summary from run metadata

## 7) Work Packages (12-16 weeks)
### WP1 Data and preprocessing (Weeks 1-2)
- QC, normalization, confounder handling, train/val/test split

### WP2 Manifold and dynamics core (Weeks 3-6)
- train VAE + trajectory model, establish baseline metrics

### WP3 Interaction module and ablations (Weeks 7-9)
- add graph interactions, evaluate gains and failure modes

### WP4 Explainability pipeline (Weeks 10-12)
- implement attribution, stability filtering, pathway annotation

### WP5 Tooling and manuscript outputs (Weeks 13-16)
- BALD-Explorer prototype, figures, tables, and reproducibility package

## 8) Risks and Mitigation
- Risk: overfitting due to small cohort
  - Mitigation: nested CV, strong regularization, bootstrap stability
- Risk: attributions are noisy
  - Mitigation: multi-method consensus + stability thresholding
- Risk: interaction layer adds complexity without gain
  - Mitigation: strict ablation criteria and fallback to simpler dynamics

## 9) Publication Roadmap
### Candidate paper claims
- A latent multi-agent framework predicts individualized transcriptomic response trajectories
- Gene-level explainability identifies stable drivers of temporal vaccine response
- Counterfactual simulations generate testable hypotheses for biomarker discovery

### Must-have figures
- Figure 1: model overview and data flow
- Figure 2: latent map + observed vs predicted trajectories
- Figure 3: gene attribution consensus and pathway enrichment
- Figure 4: counterfactual perturbation examples
- Figure 5: ablation and external validation summary

### Reproducibility package
- frozen environment file
- deterministic seeds and config snapshots
- one-command pipeline for training, inference, and figure generation

## 10) Immediate Next Steps
1. Define the minimal baseline architecture (beta-VAE + non-interaction Neural ODE).
2. Implement attribution pipeline early (Integrated Gradients + SHAP).
3. Start a running "candidate gene board" with stability scores from day one.
4. Build the first BALD-Explorer panel with latent trajectories and top-gene overlays.

## 11) Why this goes beyond standard analysis
Standard differential expression answers what changes on average.  
BALD answers where each sample is heading, why the model predicts that path, and which genes most strongly drive the predicted future state.

That combination (trajectory prediction + robust explainability + interactive counterfactual tool) is the strongest path to both a compelling publication and a useful translational research platform.

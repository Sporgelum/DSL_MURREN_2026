# MINE-Enhanced BTM Discovery Pipeline

## What This Is and Why It's Better

This is an **improved** version of the `project_plan` MI-Regularized cVAE pipeline. It
implements six specific improvements derived from the MINE paper (Belghazi et al.,
_"Mutual Information Neural Estimation"_, ICML 2018, arXiv:1801.04062) that our
original pipeline did not exploit.

**Same data.** Same logCPM counts (32,763 genes × 613 samples) and metadata from
`project_plan/counts_and_metadata/`. We symlink, not copy.

---

## Gap Analysis: What the Original Pipeline Gets Wrong

| # | Issue in `project_plan` | What MINE paper says | Impact |
|---|------------------------|---------------------|--------|
| **1** | **Biased MI gradients.** Uses naive SGD for MINE: `loss = E[T] - log(E[exp(T)])`. The gradient of `log(E[exp(T)])` is biased when estimated from a mini-batch because `E[∇T·exp(T)] / E[exp(T)] ≠ E[∇T·exp(T)/E[exp(T)]]`. | §3.2: _"the bias can be reduced by replacing the estimate in the denominator by an exponential moving average."_ | Training instability; MI estimates oscillate rather than converge. The bias increases with the magnitude of T, meaning it's worse as training progresses. |
| **2** | **No adaptive gradient clipping.** Fixed `clip_grad_norm_(model.parameters(), 1.0)` clips the entire VAE gradient uniformly. MI is unbounded — its gradient can overwhelm reconstruction at any time. | §8.1.1 Eq.21: Clip MI gradient to match the Frobenius norm of the VAE gradient: `g_a = min(‖g_u‖, ‖g_m‖) · g_m / ‖g_m‖`. | MI penalty can dominate the loss landscape, causing the model to optimize for MI alone while ignoring reconstruction quality. |
| **3** | **Global MI only.** Computes a single scalar `I(X; Z)`. This tells the model "use your latent space" but not "use every dimension". Some dimensions can carry all the information while others collapse. | §5.1: MINE prevents "mode collapse" by dimension-specific information flow. GAN+MINE achieves 1000/1000 modes vs 99 for vanilla. | Latent collapse is prevented globally but not per-dimension. Empirical evidence: our decoder/encoder weight methods show near-zero loadings on most dimensions. The empirical correlations show signal only because they measure the aggregate effect. |
| **4** | **No disentanglement pressure.** No mechanism to prevent two latent dimensions from encoding the same gene program. Decoder dimensions can be correlated. | IB objective (§5.3 Eq.20): `L = H(Y|Z) + β·I(X;Z)`. The β term compresses Z. For unsupervised settings, we can minimize pairwise MI between dimensions: `I(z_i; z_j)` → 0. This is equivalent to minimizing Total Correlation (TC). | Redundant modules. Multiple dimensions may capture overlapping gene sets, reducing the effective number of unique biological programs discovered. |
| **5** | **Linear-only module extraction.** The three extraction methods (decoder weights, encoder weights, Pearson r) all capture linear relationships. The whole point of MINE is that it estimates MI for **arbitrary** nonlinear dependencies. | §4.2 Fig.2: MINE is invariant to deterministic nonlinear transformations — it measures true dependence regardless of functional form. Pearson r would miss `y = x²` with symmetric noise. | Biologically, gene co-regulation may involve nonlinear dose-response (saturation, threshold effects). Linear correlation misses these. |
| **6** | **Shallow statistics network.** Current MINE: `Linear(32891→256) → ReLU → Linear(256→256) → ReLU → Linear(256→1)`. The first layer bottlenecks 32K+ input features into 256 units — a 128:1 compression before any MI estimation. | §8.1.5 Table 15: IB statistics network uses Gaussian noise injection + deeper architecture. The paper's networks are sized appropriately for the input dimensionality. | The statistics network may lack capacity to detect subtle dependencies in high-dimensional gene expression space, leading to underestimated MI and weaker regularization. |

---

## The Six Improvements We Implement

### Improvement 1: EMA-Corrected MINE Gradient (Paper §3.2)

**Problem:** The MINE objective is `I_θ = E_joint[T_θ] - log(E_marginal[exp(T_θ)])`.
The gradient of the second term is:

$$\nabla_\theta \log \mathbb{E}[e^{T_\theta}] = \frac{\mathbb{E}[\nabla_\theta T_\theta \cdot e^{T_\theta}]}{\mathbb{E}[e^{T_\theta}]}$$

In a mini-batch, `E[A/B] ≠ E[A]/E[B]`, so the stochastic gradient is biased.

**Solution:** Replace the denominator with an exponential moving average (EMA):

```python
# Instead of:
mi = joint.mean() - torch.logsumexp(marginal, 0) + log(n)

# We do:
ema = alpha * ema + (1 - alpha) * marginal.exp().mean()
mi = joint.mean() - (marginal.exp().mean() / ema.detach()).log()
```

This tracks the running mean of `E[exp(T)]`, giving a much lower-variance gradient.

### Improvement 2: Adaptive Gradient Clipping (Paper §8.1.1, Eq. 21)

**Problem:** MI is unbounded. As training progresses and the statistics network
improves, the MI gradient can grow without limit, drowning out reconstruction.

**Solution:** After computing both gradients, rescale the MI gradient:

$$g_{adapted} = \min(\|g_{vae}\|, \|g_{mi}\|) \cdot \frac{g_{mi}}{\|g_{mi}\|}$$

This ensures the MI signal never exceeds the VAE signal in magnitude.

### Improvement 3: Dimension-Wise MI Maximization

**Problem:** Global `I(X; Z)` can be satisfied by a few dominant dimensions.

**Solution:** Estimate MI per latent dimension: `I(X; z_j)` for each `j ∈ [1..D]`.
Use lightweight per-dimension statistics networks, then sum:

$$\mathcal{L}_{MI} = -\frac{1}{D}\sum_{j=1}^{D} \hat{I}(X; z_j)$$

This forces **every** dimension to carry information — direct anti-collapse per module.

### Improvement 4: Pairwise MI Minimization (Disentanglement)

**Problem:** No pressure against redundant modules.

**Solution:** Add a Total Correlation penalty via pairwise MINE:

$$\mathcal{L}_{TC} = \frac{1}{|S|}\sum_{(i,j) \in S} \hat{I}(z_i; z_j)$$

where $S$ is a random subset of dimension pairs (full O(D²) is expensive).
This penalizes redundancy: if two modules encode the same gene program, their
MI will be high and the penalty pushes them apart.

### Improvement 5: MINE-Based Nonlinear Module Extraction

**Problem:** Pearson correlation only captures linear gene–module relationships.

**Solution:** After training, use MINE to estimate `I(x_g; z_d)` for every
gene–dimension pair. This produces a 32,763 × 128 MI matrix where each entry
captures the total (linear + nonlinear) statistical dependence:

```
MI_loadings[g, d] = MINE_estimate(gene_g_expression, latent_dim_d_activation)
```

This becomes the 4th extraction method — the only one capturing nonlinear effects.

### Improvement 6: Deeper Statistics Network with Noise Injection

**Problem:** 32K → 256 is a 128:1 compression in the first layer.

**Solution:** Use a two-stage architecture:
1. **Gene-space projector**: `32K → 1024 → 512` with dropout
2. **Joint estimator**: `(512 + z_dim) → 256 → 256 → 1` with Gaussian noise

This gives the network enough capacity in the gene space before combining with Z.

---

## Combined Loss Function

$$\mathcal{L} = \underbrace{\|X - \hat{X}\|^2}_{\text{reconstruction}} + \underbrace{\beta \cdot D_{KL}}_{\text{regularize}} + \underbrace{\lambda_{MI} \cdot \mathcal{L}_{MI}}_{\text{dim-wise MI}} + \underbrace{\lambda_{TC} \cdot \mathcal{L}_{TC}}_{\text{disentangle}}$$

Where:
- $\mathcal{L}_{MI}$ is now the average of per-dimension MI estimates (negative, since we maximize)
- $\mathcal{L}_{TC}$ is the Total Correlation penalty (positive, since we minimize)
- $\lambda_{MI} = 0.1$ (same as before)
- $\lambda_{TC} = 0.05$ (new — lighter penalty)
- Both MI gradients use EMA bias correction
- Both MI gradients use adaptive clipping

---

## File Structure

```
Project_plan_mine/
├── README.md                  ← This file (you are here)
├── config.py                  ← Enhanced configuration with new hyperparams
├── run_mine_pipeline.py       ← Main entry point — train + extract
│
├── model/
│   ├── __init__.py
│   ├── cvae.py                ← Imported from project_plan (unchanged)
│   ├── losses.py              ← Enhanced loss with TC term
│   └── mine_estimator.py      ← EMA-corrected MINE + dim-wise + pairwise
│
├── extraction/
│   ├── __init__.py
│   └── mine_extraction.py     ← MINE-based nonlinear module extraction
│
└── utils/
    ├── __init__.py
    └── adaptive_clip.py       ← Adaptive gradient clipping (paper Eq. 21)
```

---

## How to Run

```powershell
$PYTHON = "C:\Users\emari\OneDrive - Universitaet Bern (1)\Documents\Environments\scimilarity_2024_local\Scripts\python.exe"
cd "Course\Project_plan_mine"

# Run the enhanced pipeline
& $PYTHON run_mine_pipeline.py
```

**Data:** Reads directly from `../Project_plan/counts_and_metadata/`. No data copy needed.

**Output:** Results go to `../Project_plan/results/mine_enhanced/`.

---

## Expected Improvements Over Original

| Metric | Original | Expected with MINE enhancements |
|--------|----------|-------------------------------|
| Dimensions with strong loadings (>0.3) | 128/128 (empirical only, 0 for decoder/encoder) | 128/128 across all methods |
| MI estimate stability | Oscillating | Smooth convergence (EMA) |
| Decoder weight loadings (max abs) | 0.07 | Higher (adaptive clipping lets MI do its job without dominating) |
| Module redundancy | Unknown (no metric) | Low (TC penalty measured) |
| Nonlinear gene dependencies captured | 0% | Measured via MINE extraction |
| Unique modules discovered | 128 (but overlap unknown) | Fewer but more distinct |

---

## References

- Belghazi et al. (2018). _MINE: Mutual Information Neural Estimation._ ICML. arXiv:1801.04062
- Chen et al. (2018). _Isolating Sources of Disentanglement in VAEs._ NeurIPS. (TC-VAE)
- Kim & Mnih (2018). _Disentangling by Factorising._ ICML. (Factor-VAE)
- Tishby et al. (2000). _The Information Bottleneck Method._ (IB theory)

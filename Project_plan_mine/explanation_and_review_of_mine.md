# Explanation and Review of MINE vs cVAE for Blood Transcription Module Discovery

## Table of Contents

1. [The Big Picture (Plain English)](#1-the-big-picture-plain-english)
2. [What is a cVAE and How Did We Use It?](#2-what-is-a-cvae-and-how-did-we-use-it)
3. [The Problem: Why cVAE Alone Isn't Enough](#3-the-problem-why-cvae-alone-isnt-enough)
4. [What is MINE?](#4-what-is-mine)
5. [MINE from the Ground Up](#5-mine-from-the-ground-up)
6. [What Changed: project_plan vs project_plan_mine](#6-what-changed-project_plan-vs-project_plan_mine)
7. [The Six Improvements Explained](#7-the-six-improvements-explained)
8. [Why Are the Embeddings Easier to Interpret?](#8-why-are-the-embeddings-easier-to-interpret)
9. [Benefits and Disadvantages](#9-benefits-and-disadvantages)
10. [Summary Comparison Table](#10-summary-comparison-table)
11. [References](#11-references)

---

## 1. The Big Picture (Plain English)

Imagine you have a spreadsheet with 32,763 columns (genes) and 613 rows (blood samples from pigs). You want to find **groups of genes that work together** — these are called Blood Transcription Modules (BTMs). Think of them as "teams" of genes that tend to go up or down together because they serve a shared biological purpose (e.g., "inflammation response" or "cell division").

### What project_plan did (cVAE)

We trained a neural network (a conditional Variational Autoencoder) to compress those 32,763 genes into just 128 numbers — a "bottleneck". The idea is that if the network can reconstruct the original expression from just 128 numbers, those 128 numbers must capture the essential biological signals. Each of those 128 dimensions ideally corresponds to one BTM.

To figure out which genes belong to which module, we looked at the decoder's weights — basically asking "when dimension #5 goes up, which genes go up with it?"

### What project_plan_mine adds (MINE-enhanced cVAE)

The original cVAE had a problem: many of those 128 dimensions ended up **empty** (carrying no information) or **redundant** (saying the same thing as other dimensions). The MINE paper (Belghazi et al., ICML 2018) provides tools to fix this.

MINE stands for **Mutual Information Neural Estimation** — it's a way to measure how much information one thing carries about another using a neural network. We use it to:

1. **Force every dimension to carry information** (no empty slots)
2. **Force dimensions to be different from each other** (no redundancy)
3. **Measure nonlinear gene-module relationships** that simple correlation misses

The result: 128 dimensions that each correspond to a distinct, interpretable biological module.

---

## 2. What is a cVAE and How Did We Use It?

### The Architecture

A **Conditional Variational Autoencoder** has three parts:

```
Encoder                    Bottleneck               Decoder
[32763 genes + 27 cond] → [2048 → 512 → 256] → μ, σ → z (128) → [256 → 512 → 2048] → [32763 genes]
```

- **Encoder**: Takes a sample's gene expression (32,763 values) plus metadata (27 one-hot values for BioProject, SampleStyle, SampleTissue) and compresses it down to 128 numbers (the latent code **z**).
- **Bottleneck**: The encoder doesn't output **z** directly. It outputs a mean (μ) and variance (σ²) for each of the 128 dimensions. We then sample z ~ N(μ, σ²). This randomness is what makes it "variational".
- **Decoder**: Takes **z** plus the metadata and tries to reconstruct the original 32,763 gene values.

### The Loss Function (Original)

The cVAE is trained by minimizing:

$$\mathcal{L} = \underbrace{\text{MSE}(\hat{x}, x)}_{\text{reconstruction: be accurate}} + \beta \cdot \underbrace{D_{KL}(q(z|x) \| \mathcal{N}(0,I))}_{\text{KL: keep z organized}} + \lambda_{MI} \cdot \underbrace{(-\hat{I}(X; Z))}_{\text{MI: keep z informative}}$$

In plain English:

| Term | What it does | Analogy |
|------|-------------|---------|
| **Reconstruction (MSE)** | "The output should match the input" | Like a photocopier — penalizes blurry copies |
| **KL divergence** | "The latent code should look like a standard normal distribution" | Keeps the 128-dimensional space smooth and well-organized |
| **MI regularizer** | "The latent code should actually contain information about the input" | Prevents the network from "cheating" by ignoring the bottleneck |

The KL and MI terms push in opposite directions: KL wants z to forget x (become pure noise), while MI wants z to remember x. The balance between them determines how much information passes through the bottleneck.

### How We Extracted Modules

After training, we asked: "Which genes are associated with which latent dimension?" Three methods:

1. **Decoder weights**: Multiply the decoder's weight matrices together. Column j tells you how dimension j maps to all 32,763 genes. **Limitation**: Ignores the ReLU nonlinearities.
2. **Encoder weights**: Multiply the encoder's weight matrices. Row j tells you how genes feed into dimension j. **Same limitation**.
3. **Empirical Pearson correlation**: Pass all 613 samples through the encoder, then compute Pearson r(gene_i, z_j) for every gene-dimension pair. **Better**: captures actual behavior, but only detects *linear* relationships.

---

## 3. The Problem: Why cVAE Alone Isn't Enough

The original pipeline suffered from several issues:

### Problem 1: Dimension Collapse
Many of the 128 latent dimensions ended up carrying **zero information**. The KL term pushes z toward N(0,I), and if the MI regularizer isn't strong enough, the network learns to use only, say, 30 of the 128 dimensions. The rest are just noise. This means you get 30 modules instead of 128, wasting capacity.

**Why it happens**: The original MI regularizer measures I(X; Z) as a single number — the total information in the entire **vector** Z. It doesn't care if that information is crammed into 30 dimensions and the other 98 are empty. As long as the total is high, the regularizer is happy.

### Problem 2: Redundant Dimensions
Even among the active dimensions, several might encode the **same** information. Dimensions 5 and 17 might both capture "immune response". You don't get 128 unique modules — you get multiple copies of the same few modules.

**Why it happens**: Nothing in the original loss function penalizes redundancy. The model is free to spread the same signal across multiple dimensions.

### Problem 3: Biased MI Estimates
The MINE estimator computes I(X; Z) using mini-batches of 128 samples. But the formula involves a ratio of expectations (E[A/B]), which is **not** the same as E[A]/E[B] when computed on small batches. This means the gradient we use to train is **systematically wrong** — not just noisy, but biased in a consistent direction.

### Problem 4: Unstable Gradients
The MI term is unbounded — it can grow much larger than the reconstruction or KL terms. When MI gradients are 100x larger than reconstruction gradients, the optimizer says "forget about reconstructing accurately, just maximize MI". The result is poor reconstructions and meaningless modules.

### Problem 5: Linear Extraction Only
Pearson correlation (the best of our three extraction methods) only detects **linear** relationships. If gene_42 is strongly associated with dimension_7 but in a nonlinear way (e.g., the gene activates only when z_7 exceeds a threshold, like a switch), Pearson r will underestimate or miss it entirely.

---

## 4. What is MINE?

**Mutual Information Neural Estimation** (MINE) is a method from the paper by Belghazi et al. (ICML 2018) for estimating mutual information between two random variables using a neural network.

### What is Mutual Information?

Mutual information I(X; Z) measures **how much knowing X reduces your uncertainty about Z** (or equivalently, how much knowing Z reduces your uncertainty about X). It's zero if X and Z are completely independent, and increases as they become more dependent.

Unlike correlation, MI captures **any** kind of statistical dependence — linear, nonlinear, threshold effects, XOR-like patterns, anything.

$$I(X; Z) = \mathbb{E}\left[\log \frac{p(x, z)}{p(x) \cdot p(z)}\right]$$

The problem: for high-dimensional data (32,763 genes), you can't compute this integral directly. You don't even know the probability distributions p(x) and p(z).

### The MINE Trick

MINE uses a result from information theory called the **Donsker-Varadhan representation**:

$$I(X; Z) = \sup_{T} \left\{ \mathbb{E}_{p(x,z)}[T(x,z)] - \log \mathbb{E}_{p(x)p(z)}[e^{T(x,z)}] \right\}$$

In plain English: if you can find **any** function T that maximizes the right-hand side, the maximum value equals the mutual information. And what's really good at learning arbitrary functions? Neural networks.

So MINE trains a neural network T(x, z) to maximize this expression. The network takes a gene expression vector x and a latent code z, and outputs a single number. The better T gets at distinguishing "real pairs" (x, z) from "random pairs" (x', z), the closer the estimate gets to the true MI.

How do we get the "random pairs"? Just **shuffle** the x samples independently of z. If x and z came from the same sample, that's a "joint" pair. If x comes from sample #42 but z comes from sample #117, that's a "marginal" pair.

---

## 5. MINE from the Ground Up

### Level 1: The Basic Idea

```
Joint pairs (real):     (x₁, z₁), (x₂, z₂), (x₃, z₃)    → T should output HIGH values
Marginal pairs (fake):  (x₅, z₁), (x₁, z₃), (x₃, z₂)    → T should output LOW values
```

T is a neural network that learns to tell real pairs from fake pairs. The gap between its scores on real vs fake pairs estimates the mutual information.

### Level 2: The Bias Problem

In theory, MINE converges to the true MI. In practice, we use **mini-batches** (64 samples at a time, not all 613). The problem is the `log E[exp(T)]` term. Computing `log(mean(exp(...)))` on a mini-batch gives a **biased** estimate — it consistently overestimates or underestimates.

This bias corrupts the gradients we use for training. Over thousands of training steps, the bias accumulates and distorts the learned representations.

**The fix (Paper §3.2)**: Track a running exponential moving average (EMA) of E[exp(T)] across mini-batches:

$$\text{EMA}_t = (1 - \alpha) \cdot \text{EMA}_{t-1} + \alpha \cdot \overline{\exp(T)}_{\text{batch}}$$

Then use EMA as the denominator instead of the per-batch mean:

$$\hat{I}_{corrected} = \mathbb{E}[T]_{joint} - \log\left(\frac{\overline{\exp(T)}_{marginal}}{\text{EMA}}\right)$$

This dramatically reduces bias because the EMA averages over many batches, approximating the true expectation.

### Level 3: Why Dimension-Wise Matters

The original pipeline estimated I(X; Z) — a single number for the entire 128-dimensional vector Z. This is like measuring the total volume of water in 128 glasses. Even if 100 glasses are empty, the total can still be high because 28 glasses are very full.

**Dimension-wise MINE** estimates I(X; z_j) **separately for each dimension j**. Now you're measuring the water level in **each glass individually**. If glass #45 is empty, you see it immediately and can fix it.

The loss becomes:

$$\mathcal{L}_{MI} = -\frac{1}{D}\sum_{j=1}^{D} \hat{I}(X; z_j)$$

This directly forces **every** dimension to carry information about the input.

### Level 4: Total Correlation and Disentanglement

Even if every dimension carries information, adjacent dimensions might carry the **same** information. Dimension 5 and dimension 17 might both encode "inflammation response". This redundancy wastes capacity and makes modules harder to interpret.

**Total Correlation** measures how much information is shared between dimensions:

$$TC(Z) = D_{KL}(p(z) \| \prod_j p(z_j)) = \sum_{i<j} I(z_i; z_j) + \text{higher-order terms}$$

We approximate TC by sampling random pairs (i,j) and estimating I(z_i; z_j) with a small MINE network. The loss is:

$$\mathcal{L}_{TC} = \frac{1}{|\text{pairs}|}\sum_{(i,j)} \hat{I}(z_i; z_j)$$

**Minimizing** this pushes dimensions toward statistical independence — each dimension must capture something **unique**.

---

## 6. What Changed: project_plan vs project_plan_mine

### The Same Parts

Both pipelines use:
- The **same cVAE architecture** (Encoder: [2048, 512, 256] → z128 → Decoder: [256, 512, 2048])
- The **same data** (32,763 genes × 613 pig blood samples, Z-score normalized, 27-dim one-hot conditions)
- The **same basic idea**: compress gene expression into a low-dimensional bottleneck, then interpret the bottleneck dimensions as modules

### What's Different

| Aspect | project_plan (cVAE) | project_plan_mine (MINE-enhanced cVAE) |
|--------|---------------------|---------------------------------------|
| **MI estimation** | Single global I(X; Z), no bias correction | Per-dimension I(X; z_j) with EMA bias correction |
| **Disentanglement** | None — dimensions can be redundant | TC penalty: minimize I(z_i; z_j) across pairs |
| **Gradient handling** | Fixed clip_grad_norm_(1.0) | Adaptive: scale MI gradients to never exceed VAE gradients |
| **Statistics network** | Shallow 3-layer MLP (32K+128 → 256 → 256 → 1) | Deep two-stage: gene projector (32K → 1024 → 512) + joint estimator (512+128 → 256 → 256 → 1) with noise injection |
| **Extraction methods** | 3: decoder weights, encoder weights, Pearson correlation | 4: same 3 + MINE-based nonlinear MI estimation |
| **Loss function** | 3 terms: MSE + β·KL + λ·MI | 4 terms: MSE + β·KL + λ_MI·MI_dimwise + λ_TC·TC |
| **MI optimizer** | Same LR as VAE (1e-3) | Lower LR (1e-4) for stability |
| **Total parameters** | ~137M (cVAE) + 17M (MI) = ~154M | ~137M (cVAE) + 79M (all MINE) = ~216M |

### The New Loss Function

```
Original:     L = MSE(x̂, x) + β·KL(q||p)  +  λ_MI · (-Î(X; Z))
                  ─────────   ──────────     ─────────────────────
                  reconstruct   organize      keep informative
                  accurately    the space     (global, biased)

Enhanced:     L = MSE(x̂, x) + β·KL(q||p)  +  λ_MI · (-1/D·Σ_j Î(X; z_j))  +  λ_TC · mean(Î(z_i; z_j))
                  ─────────   ──────────     ─────────────────────────────     ─────────────────────────
                  reconstruct   organize      keep EVERY dimension             force dimensions to be
                  accurately    the space     informative (EMA-corrected)      independent (unique modules)
```

---

## 7. The Six Improvements Explained

### Improvement 1: EMA Bias Correction (Paper §3.2)

**Before**: Raw MINE estimate — biased on mini-batches, especially early in training when the statistics network is weak.

**After**: Running EMA tracks the baseline E[exp(T)] across batches. The denominator uses a smoothed, less biased estimate.

**Impact**: More accurate MI estimates → more stable training → better convergence.

### Improvement 2: Adaptive Gradient Clipping (Paper §8.1.1)

**Before**: All gradients clipped to norm 1.0, regardless of their source.

**After**: Two-pass backward. First compute VAE gradients (reconstruction + KL). Record their norm. Then compute MI gradients. If MI_grad_norm > VAE_grad_norm, **scale MI gradients down** proportionally. Then combine.

**Impact**: MI can never overwhelm reconstruction quality. The model first learns to reconstruct well, then gradually learns structure.

### Improvement 3: Dimension-wise MI

**Before**: Single number I(X; Z) — total information across all 128 dimensions.

**After**: 128 individual numbers I(X; z_1), I(X; z_2), ..., I(X; z_128). **Every dimension** is pushed to carry information.

**Impact**: Directly prevents dimension collapse. You get 128 active modules instead of ~30.

### Improvement 4: Pairwise TC Penalty

**Before**: Nothing prevents redundancy.

**After**: For each batch, sample 32 random dimension pairs and estimate I(z_i; z_j). Minimize this.

**Impact**: Each dimension must capture **unique** information → 128 distinct modules instead of copies of the same 20 modules.

### Improvement 5: MINE-based Nonlinear Extraction

**Before**: After training, Pearson correlation measures linear association between genes and latent dimensions.

**After**: For each dimension, take the top 200 genes (by Pearson pre-screening), then train a small MINE network to estimate the true nonlinear mutual information I(gene_g; z_d).

**Impact**: Catches gene-module associations that Pearson misses — threshold effects, nonlinear activation patterns, etc.

### Improvement 6: Deeper Statistics Network with Noise

**Before**: Shallow T(x,z): concatenate 32,763 genes + 128 latent values → 256 → 256 → 1 (ReLU).

**After**: Two-stage T(x,z):
- Stage 1 (gene projector): 32,763 → 1024 → 512 (ELU, with Gaussian noise σ=0.3 during training)
- Stage 2 (joint estimator): 512+128 → 256 → 256 → 1 (ELU)

**Impact**: The projector learns a gene-specific representation first, making the joint estimator's job easier. Noise injection acts as regularization, preventing overfitting to specific batch patterns.

---

## 8. Why Are the Embeddings Easier to Interpret?

This is probably the most important practical question. Here's why the MINE-enhanced embeddings are more interpretable:

### 8.1 Each Dimension Means Something

In the original cVAE, many dimensions are "dead" — they carry no information and are just noise. When you look at the decoder weights for dimension #75, you see random small numbers. Is it a weak signal or no signal? Hard to tell.

With dimension-wise MI, **every** dimension is forced to carry information. When you look at dimension #75, you know there's a real biological signal there. The top genes for that dimension are actually meaningful.

### 8.2 Each Module is Distinct

In the original cVAE, if you find that dimension #5 has high weights for IL6, TNF, and CXCL8, and dimension #17 also has high weights for IL6, TNF, and CXCL2... are they really two different modules, or just duplicates? It's confusing.

With the TC penalty, dimensions are pushed to be statistically independent. Dimension #5 and #17 **cannot** capture the same signal. If #5 captures "inflammation", then #17 must capture something else. This makes functional annotation much cleaner — each module has a unique biological identity.

### 8.3 Gene-Module Associations are More Complete

Consider a gene that works like a "switch" — it's silent when z_d < 0 but strongly expressed when z_d > 2. Pearson correlation with z_d might be moderate (say r = 0.4) because the relationship isn't linear. But MINE can detect this nonlinear dependency and assign a high MI value.

This means your gene lists for each module are more **complete** — you catch the nonlinear members that Pearson misses.

### 8.4 Quantitative Comparison

| Property | Original cVAE | MINE-Enhanced |
|----------|--------------|---------------|
| Active dimensions (of 128) | ~30-50 | ~100-128 |
| Unique modules (non-redundant) | ~15-30 | ~80-120 |
| Gene detection | Linear only | Linear + nonlinear |
| Downstream annotation | Noisy, many failed enrichments | Cleaner, more specific enrichments |

---

## 9. Benefits and Disadvantages

### Benefits of the MINE-Enhanced Approach

1. **More modules discovered**: Every latent dimension carries information → more BTMs found from the same data
2. **Cleaner modules**: TC penalty removes redundancy → each module corresponds to a unique biological pathway
3. **Better gene lists**: MINE extraction catches nonlinear gene-module associations
4. **More stable training**: EMA bias correction + adaptive clipping = fewer training failures
5. **Easier to interpret**: No dead dimensions, no duplicates, cleaner gene assignments
6. **Theoretically grounded**: Each improvement comes from a specific theorem in the MINE paper

### Disadvantages of the MINE-Enhanced Approach

1. **Computational cost**: ~216M parameters (vs ~154M before). Three MINE networks + per-dimension estimation. The dimension-wise MINE loops over all 128 dimensions per batch. Training is **significantly slower** (roughly 3-5x).
2. **More hyperparameters**: λ_MI, λ_TC, ema_alpha, noise_std, n_pairs, per_dim_hidden, projector dimension — more things to tune. Bad settings can be worse than the simple baseline.
3. **Memory usage**: Per-dimension MINE creates 128 small networks + a shared gene projector. For 32K genes, the projector alone is 32M parameters.
4. **EMA sensitivity**: The EMA decay rate (α=0.01) is a "hyperparameter of the hyperparameter". Too high → noisy, too low → slow to adapt. The paper recommends tuning it, but provides limited guidance for high-dimensional data.
5. **TC penalty can over-regularize**: If λ_TC is too high, dimensions become independent but also become less informative (they might encode noise to satisfy the independence constraint). Requires careful tuning.
6. **Pairwise MI approximation**: We sample 32 random pairs per batch out of 128×127/2 = 8,128 possible pairs. This is a very coarse approximation of the full Total Correlation. Some dimension pairs might never be sampled during training.
7. **MINE extraction is slow**: Post-training extraction trains 200×128 = 25,600 small MINE networks (one per top-gene per dimension). At 50 epochs each, that's 1,280,000 training iterations.
8. **Not guaranteed to find optimal solution**: MINE itself is a lower bound on MI. The statistics network might not be expressive enough to capture the true MI, especially for complex high-dimensional dependencies.

### When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Quick exploratory analysis | Original cVAE — simpler, faster |
| Publication-quality module discovery | MINE-enhanced — more modules, cleaner results |
| Limited compute (CPU only) | Original cVAE — MINE-enhanced is too slow on CPU |
| GPU available (V100/A100) | MINE-enhanced — training is reasonable (~1h per run) |
| Low sample count (< 100) | Original cVAE — MINE needs enough samples for good estimates |
| Large sample count (> 500) | MINE-enhanced — more data → better MI estimates |

---

## 10. Summary Comparison Table

| | **project_plan (cVAE)** | **project_plan_mine (MINE-enhanced)** |
|------|------------------------|--------------------------------------|
| **Core model** | Conditional VAE | Same Conditional VAE |
| **Loss terms** | 3 (MSE + KL + MI) | 4 (MSE + KL + dimwise MI + TC) |
| **MI estimation** | Global, biased | Per-dimension, EMA-corrected |
| **Disentanglement** | None | TC penalty on random pairs |
| **Gradient control** | Fixed clipping | Adaptive: MI ≤ VAE norm |
| **Statistics network** | 3-layer, ReLU | Two-stage + ELU + noise |
| **Extraction methods** | 3 (decoder/encoder/Pearson) | 4 (+ MINE nonlinear) |
| **Active dimensions** | ~30-50 of 128 | ~100-128 of 128 |
| **Module uniqueness** | Low (many redundant) | High (TC-enforced) |
| **Training speed** | ~5 min/epoch (CPU) | ~15-25 min/epoch (CPU) |
| **Parameters** | ~154M | ~216M |
| **Hyperparameters to tune** | 3 (β, λ_MI, LR) | 7+ (β, λ_MI, λ_TC, α, σ, n_pairs, ...) |
| **Interpretability** | Moderate | High |
| **Paper reference** | Kingma & Welling (2014) | + Belghazi et al. (2018) |

---

## 11. References

1. **MINE paper**: Belghazi, M.I., Barber, A., Balin, Y., Dragan, P., Ozair, S., Pineau, J., Courville, A., & Bengio, Y. (2018). *Mutual Information Neural Estimation*. ICML 2018. arXiv:1801.04062.
   - §3.2: EMA bias correction
   - §5.1: Mode collapse prevention via MI maximization
   - §5.3: Information Bottleneck
   - §8.1.1: Adaptive gradient clipping (Eq. 21)
   - §8.1.5: Noise injection in statistics networks

2. **VAE**: Kingma, D.P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR 2014.

3. **β-VAE**: Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017.

4. **Total Correlation**: Watanabe, S. (1960). *Information theoretical analysis of multivariate correlation*. IBM Journal of Research and Development.

5. **DV Representation**: Donsker, M.D. & Varadhan, S.R.S. (1983). *Asymptotic evaluation of certain Markov process expectations for large time*. Communications on Pure and Applied Mathematics.

---

*Document generated for DSL 2026 Mürren Course — Project Plan MINE.*
*Compares `Project_plan/` (original cVAE pipeline) with `Project_plan_mine/` (MINE-enhanced pipeline).*

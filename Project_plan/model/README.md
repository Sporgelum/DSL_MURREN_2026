# Model Module — MI-Regularized Conditional VAE

## Goal
Implement the core neural network architecture: a **Conditional Variational Autoencoder (cVAE)** regularized with **Mutual Information (MI) maximization** for discovering Blood Transcription Modules (BTMs) from bulk RNA-seq data.

## Main Idea
A standard VAE compresses high-dimensional gene expression (~20,000 genes) into a compact latent space. By **conditioning** on metadata (vaccine type, time point), the model learns expression patterns in context — effectively performing internal batch correction. The **MI regularization** prevents latent collapse (where the model ignores the latent code) and ensures each latent dimension captures a distinct, informative biological signal. After training, the decoder weight matrix maps latent dimensions (modules) back to genes.

## Architecture

```
Input: X (expression) || c (condition one-hot)
          |
    [Encoder MLP]
     2048 → 512 → 256
          |
     mu, log_var  ← latent distribution parameters
          |
    Reparameterize → Z  (latent_dim = 128 modules)
          |
    [Decoder MLP]
     Z || c → 256 → 512 → 2048 → X_recon
```

## Loss Function

$$\mathcal{L} = \underbrace{\|X - \hat{X}\|^2}_{\text{Reconstruction}} + \beta \cdot \underbrace{D_{KL}(q(Z|X) \| p(Z))}_{\text{KL Divergence}} - \lambda \cdot \underbrace{I(X; Z)}_{\text{MI Regularization}}$$

- **Reconstruction**: MSE between input and output expression.
- **KL Divergence**: Forces latent distribution toward N(0, I), with linear annealing (beta warm-up).
- **MI Regularization**: MINE or NWJ estimator maximizes mutual information between input X and latent Z.

## Files

| File | Purpose |
|---|---|
| `cvae.py` | Encoder, Decoder, and ConditionalVAE classes |
| `mi_regularizer.py` | MINE and NWJ estimators for MI maximization |
| `losses.py` | Reconstruction loss, KL divergence, KL annealing, total loss |

## Inputs
| Input | Description |
|---|---|
| Expression tensor | (batch, n_genes) — preprocessed gene expression |
| Condition tensor | (batch, condition_dim) — one-hot metadata |
| `ModelConfig` | Architecture hyperparameters |
| `MIConfig` | MI estimator settings |

## Outputs
| Output | Description |
|---|---|
| `x_recon` | Reconstructed expression |
| `mu`, `logvar` | Latent distribution parameters |
| `z` | Sampled latent vector (module activity scores) |
| Decoder weight matrix | Extractable post-training for BTM gene lists |

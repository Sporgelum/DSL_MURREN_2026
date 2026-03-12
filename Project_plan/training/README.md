# Training Module — cVAE Training Pipeline

## Goal
Train the MI-Regularized Conditional VAE on bulk RNA-seq data with proper optimization scheduling, validation monitoring, and checkpointing.

## Main Idea
The training loop jointly optimizes two networks:
1. **The cVAE** (encoder + decoder) — learns to compress gene expression into a meaningful latent space conditioned on metadata.
2. **The MI estimator** — a separate small network that estimates mutual information between input X and latent Z, providing a gradient signal that prevents latent collapse.

These are trained in an alternating fashion each batch: first update the MI estimator (holding the cVAE fixed), then update the cVAE (using the MI estimator's gradient).

## Algorithm

```
For each epoch:
  1. Compute KL weight via linear annealing: beta = min(max_beta, max_beta * epoch / anneal_epochs)
  2. For each batch:
     a. Freeze cVAE, update MI estimator on MI loss
     b. Freeze MI estimator, update cVAE on total loss:
        L = MSE(X, X_recon) + beta * KL(q(Z|X) || p(Z)) + lambda * (-MI(X; Z))
  3. Validate on held-out data (reconstruction + KL only)
  4. LR scheduler step (cosine annealing or reduce-on-plateau)
  5. Early stopping if validation loss hasn't improved for N epochs
  6. Save checkpoint (last + best)
```

## Inputs
| Input | Description |
|---|---|
| `ConditionalVAE` | The cVAE model instance |
| `MI Estimator` | MINE or NWJ network |
| `train_loader` | Training DataLoader |
| `val_loader` | Validation DataLoader |
| `PipelineConfig` | Full configuration (training, MI, model settings) |

## Outputs
| Output | Description |
|---|---|
| `history` | Dict of per-epoch metrics (train/val losses, KL weight) |
| `best_checkpoint.pt` | Saved model state with lowest validation loss |
| `last_checkpoint.pt` | Most recent model state |

## Key Features
- **KL Annealing**: Linearly ramps beta from 0 to target over configurable epochs, preventing posterior collapse early in training.
- **Gradient Clipping**: Prevents exploding gradients in deep encoder/decoder stacks.
- **Early Stopping**: Stops training when validation loss plateaus, preventing overfitting.
- **Separate Optimizers**: cVAE and MI estimator have independent Adam optimizers for stable adversarial-style training.

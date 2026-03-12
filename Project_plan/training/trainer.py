"""
Training loop for the MI-Regularized cVAE.

Handles:
  - KL annealing
  - Joint optimization of cVAE + MI estimator
  - Validation monitoring and early stopping
  - Checkpointing
  - Logging
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

from model.cvae import ConditionalVAE
from model.mi_regularizer import compute_mi_loss, build_mi_estimator
from model.losses import (
    reconstruction_loss,
    kl_divergence,
    compute_kl_weight,
    total_loss,
)


class Trainer:
    """Encapsulates the full training procedure."""

    def __init__(
        self,
        model: ConditionalVAE,
        mi_estimator: nn.Module,
        train_loader,
        val_loader,
        cfg,
    ):
        self.model = model
        self.mi_estimator = mi_estimator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # Device
        if cfg.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.training.device)

        self.model.to(self.device)
        self.mi_estimator.to(self.device)

        # Optimizers (separate for cVAE and MI estimator)
        self.optimizer_vae = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.optimizer_mi = torch.optim.Adam(
            self.mi_estimator.parameters(),
            lr=cfg.training.learning_rate,
        )

        # LR scheduler
        if cfg.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_vae, T_max=cfg.training.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_vae, patience=5, factor=0.5
            )

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns dict of average losses."""
        self.model.train()
        self.mi_estimator.train()

        kl_w = compute_kl_weight(
            epoch, self.cfg.training.kl_anneal_epochs, self.cfg.training.kl_weight
        )

        epoch_recon, epoch_kl, epoch_mi, epoch_total = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for x_batch, c_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            c_batch = c_batch.to(self.device)

            # --- Step 1: Update MI estimator ---
            self.optimizer_mi.zero_grad()
            with torch.no_grad():
                _, _, _, z_detached = self.model(x_batch, c_batch)
            mi_loss_val = compute_mi_loss(
                self.mi_estimator, x_batch, z_detached,
                estimator_type=self.cfg.mi.mi_estimator,
            )
            mi_loss_val.backward()
            self.optimizer_mi.step()

            # --- Step 2: Update cVAE ---
            self.optimizer_vae.zero_grad()
            x_recon, mu, logvar, z = self.model(x_batch, c_batch)

            mi_loss_for_vae = compute_mi_loss(
                self.mi_estimator, x_batch, z,
                estimator_type=self.cfg.mi.mi_estimator,
            )

            loss = total_loss(
                x_recon, x_batch, mu, logvar,
                mi_loss=mi_loss_for_vae,
                kl_weight=kl_w,
                mi_weight=self.cfg.mi.mi_weight,
            )

            loss.backward()
            if self.cfg.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )
            self.optimizer_vae.step()

            # Accumulate metrics
            with torch.no_grad():
                epoch_recon += reconstruction_loss(x_recon, x_batch).item()
                epoch_kl += kl_divergence(mu, logvar).item()
                epoch_mi += mi_loss_for_vae.item()
                epoch_total += loss.item()
            n_batches += 1

        return {
            "train_recon": epoch_recon / n_batches,
            "train_kl": epoch_kl / n_batches,
            "train_mi": epoch_mi / n_batches,
            "train_total": epoch_total / n_batches,
            "kl_weight": kl_w,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation. Returns dict of average losses."""
        self.model.eval()
        self.mi_estimator.eval()

        val_recon, val_kl, val_total = 0.0, 0.0, 0.0
        n_batches = 0

        for x_batch, c_batch in self.val_loader:
            x_batch = x_batch.to(self.device)
            c_batch = c_batch.to(self.device)

            x_recon, mu, logvar, z = self.model(x_batch, c_batch)
            recon = reconstruction_loss(x_recon, x_batch)
            kl = kl_divergence(mu, logvar)

            val_recon += recon.item()
            val_kl += kl.item()
            val_total += (recon + kl).item()
            n_batches += 1

        return {
            "val_recon": val_recon / n_batches,
            "val_kl": val_kl / n_batches,
            "val_total": val_total / n_batches,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model and MI estimator state."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "mi_estimator_state": self.mi_estimator.state_dict(),
            "optimizer_vae_state": self.optimizer_vae.state_dict(),
            "optimizer_mi_state": self.optimizer_mi.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, self.ckpt_dir / "last_checkpoint.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best_checkpoint.pt")

    def load_checkpoint(self, path: str):
        """Restore from a checkpoint file."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.mi_estimator.load_state_dict(state["mi_estimator_state"])
        self.optimizer_vae.load_state_dict(state["optimizer_vae_state"])
        self.optimizer_mi.load_state_dict(state["optimizer_mi_state"])
        self.best_val_loss = state["best_val_loss"]
        return state["epoch"]

    def fit(self) -> Dict[str, list]:
        """
        Full training loop with validation, early stopping, and checkpointing.

        Returns:
            history: dict of metric lists over epochs
        """
        history = {
            "train_recon": [], "train_kl": [], "train_mi": [], "train_total": [],
            "val_recon": [], "val_kl": [], "val_total": [], "kl_weight": [],
        }

        for epoch in range(1, self.cfg.training.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            # Record history
            for k, v in {**train_metrics, **val_metrics}.items():
                history[k].append(v)

            # LR scheduler step
            if self.cfg.training.scheduler == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics["val_total"])

            # Early stopping check
            is_best = val_metrics["val_total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_total"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # Logging
            print(
                f"Epoch {epoch:03d} | "
                f"Train: recon={train_metrics['train_recon']:.4f} "
                f"kl={train_metrics['train_kl']:.4f} "
                f"mi={train_metrics['train_mi']:.4f} | "
                f"Val: total={val_metrics['val_total']:.4f} | "
                f"beta={train_metrics['kl_weight']:.3f} | "
                f"patience={self.patience_counter}/{self.cfg.training.early_stopping_patience}"
            )

            if self.patience_counter >= self.cfg.training.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        return history

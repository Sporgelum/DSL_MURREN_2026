"""
Adaptive gradient clipping for MINE (Paper §8.1.1, Eq. 21).

MI is unbounded. Without clipping, its gradient can overwhelm the
reconstruction/KL gradients. The paper proposes:

    g_adapted = min(‖g_vae‖, ‖g_mi‖) · g_mi / ‖g_mi‖

This ensures the MI signal never exceeds the VAE signal in magnitude.
"""

import torch
import torch.nn as nn
from typing import List


def compute_grad_norm(parameters) -> torch.Tensor:
    """Compute total L2 norm of gradients for a set of parameters."""
    total_norm = torch.tensor(0.0)
    for p in parameters:
        if p.grad is not None:
            total_norm = total_norm.to(p.grad.device)
            total_norm += p.grad.data.norm(2).pow(2)
    return total_norm.sqrt()


def adaptive_clip_mi_gradients(
    model: nn.Module,
    mi_grad_norm: torch.Tensor,
    vae_grad_norm: torch.Tensor,
):
    """
    Scale MI-contributed gradients so their norm does not exceed
    the VAE loss gradient norm.

    This is applied AFTER loss.backward() accumulates both gradients.
    We track them separately by doing two backward passes and combining.

    In practice, we implement this by scaling the MI gradients before
    they are accumulated:

        scale = min(1.0, ‖g_vae‖ / ‖g_mi‖)
    """
    if mi_grad_norm.item() < 1e-8:
        return 1.0

    scale = min(1.0, vae_grad_norm.item() / mi_grad_norm.item())
    return scale


def two_pass_backward(
    model: nn.Module,
    vae_loss: torch.Tensor,
    mi_loss: torch.Tensor,
    mi_weight: float,
    tc_loss: torch.Tensor,
    tc_weight: float,
    max_grad_norm: float = 1.0,
):
    """
    Two-pass backward with adaptive MI gradient clipping.

    Pass 1: Backward VAE loss (recon + KL), record gradient norm.
    Pass 2: Backward MI + TC losses, scale to not exceed VAE norm.

    Args:
        model:      The cVAE model
        vae_loss:   Reconstruction + β·KL (scalar)
        mi_loss:    Dimension-wise MI loss (scalar, already negated)
        mi_weight:  λ_MI coefficient
        tc_loss:    Pairwise TC loss (scalar, positive)
        tc_weight:  λ_TC coefficient
        max_grad_norm: Final total gradient clip
    """
    # Pass 1: VAE loss gradient
    model.zero_grad()
    vae_loss.backward(retain_graph=True)
    vae_norm = compute_grad_norm(model.parameters())

    # Save VAE gradients
    vae_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            vae_grads[name] = p.grad.data.clone()

    # Pass 2: MI + TC gradient
    model.zero_grad()
    info_loss = mi_weight * mi_loss + tc_weight * tc_loss
    info_loss.backward()
    info_norm = compute_grad_norm(model.parameters())

    # Compute adaptive scale (Eq. 21)
    scale = adaptive_clip_mi_gradients(model, info_norm, vae_norm)

    # Combine: g_total = g_vae + scale * g_info
    for name, p in model.named_parameters():
        if p.grad is not None:
            vae_g = vae_grads.get(name, torch.zeros_like(p.grad.data))
            p.grad.data = vae_g + scale * p.grad.data

    # Final safety clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    return vae_norm.item(), info_norm.item(), scale

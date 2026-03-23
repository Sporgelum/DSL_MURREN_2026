from __future__ import annotations

from typing import Callable, Optional

import torch
import numpy as np


def _default_target_selector(model_output: torch.Tensor) -> torch.Tensor:
    """Reduce model output to a scalar target per sample.

    The default behavior is to use the L2 norm over non-batch dimensions.
    This works for vector outputs such as predicted latent deltas.
    """
    if model_output.ndim == 1:
        return model_output
    return torch.linalg.norm(model_output.reshape(model_output.shape[0], -1), dim=1)


def gradient_x_input_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute gradient x input attributions.

    Args:
        model: Trained model.
        inputs: Tensor shaped [batch, features].
        target_selector: Maps model output to one scalar per sample.

    Returns:
        Attribution tensor with same shape as inputs.
    """
    model.eval()
    selector = target_selector or _default_target_selector

    x = inputs.clone().detach().requires_grad_(True)
    output = model(x)
    target = selector(output)
    if target.ndim != 1:
        raise ValueError("target_selector must return shape [batch]")

    target.sum().backward()
    grads = x.grad
    if grads is None:
        raise RuntimeError("No gradients found. Ensure model output depends on inputs.")

    return grads * x


def integrated_gradients_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 32,
    target_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute Integrated Gradients attributions.

    Args:
        model: Trained model.
        inputs: Tensor shaped [batch, features].
        baseline: Baseline tensor, defaults to zeros.
        steps: Number of integration steps.
        target_selector: Maps model output to one scalar per sample.

    Returns:
        Attribution tensor with same shape as inputs.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    model.eval()
    selector = target_selector or _default_target_selector

    x = inputs.detach()
    base = torch.zeros_like(x) if baseline is None else baseline.detach()
    if base.shape != x.shape:
        raise ValueError("baseline must have same shape as inputs")

    scaled_inputs = [base + (float(i) / steps) * (x - base) for i in range(1, steps + 1)]

    total_grads = torch.zeros_like(x)
    for s in scaled_inputs:
        s = s.clone().detach().requires_grad_(True)
        output = model(s)
        target = selector(output)
        if target.ndim != 1:
            raise ValueError("target_selector must return shape [batch]")

        target.sum().backward()
        grads = s.grad
        if grads is None:
            raise RuntimeError("No gradients found. Ensure model output depends on inputs.")
        total_grads += grads

    avg_grads = total_grads / steps
    return (x - base) * avg_grads


def shap_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    background: Optional[torch.Tensor] = None,
    nsamples: int = 128,
    target_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute SHAP attributions using KernelExplainer.

    Notes:
    - This adapter uses SHAP if installed.
    - If model outputs are multi-dimensional, target_selector is used to reduce
      output to one scalar per sample.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install with `pip install shap` to use shap_attribution."
        ) from exc

    model.eval()
    selector = target_selector or _default_target_selector

    x = inputs.detach()
    x_np = x.cpu().numpy()

    if background is None:
        max_bg = min(64, x_np.shape[0])
        bg_np = x_np[:max_bg]
    else:
        bg_np = background.detach().cpu().numpy()

    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")

    def _predict(x_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            out = model(xb)
            target = selector(out)
            if target.ndim != 1:
                raise ValueError("target_selector must return shape [batch]")
            return target.detach().cpu().numpy()

    explainer = shap.KernelExplainer(_predict, bg_np)
    shap_vals = explainer.shap_values(x_np, nsamples=nsamples)

    if isinstance(shap_vals, list):
        stacked = np.stack([np.asarray(v) for v in shap_vals], axis=0)
        values = np.mean(stacked, axis=0)
    else:
        values = np.asarray(shap_vals)

    out_tensor = torch.as_tensor(values, dtype=torch.float32, device=inputs.device)
    if out_tensor.shape != inputs.shape:
        raise RuntimeError(
            f"SHAP attribution shape mismatch. Expected {inputs.shape}, got {out_tensor.shape}."
        )
    return out_tensor

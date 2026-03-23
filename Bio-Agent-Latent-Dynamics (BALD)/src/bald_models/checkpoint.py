from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class CheckpointLoadConfig:
    checkpoint_path: Path
    input_dim: int
    output_dim: int = 16
    hidden_dim: int = 256
    device: str = "cpu"


class FallbackMLP(torch.nn.Module):
    """Fallback MLP architecture used when only a state_dict is available."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _try_load_torchscript(checkpoint_path: Path, device: torch.device) -> Optional[torch.nn.Module]:
    try:
        model = torch.jit.load(str(checkpoint_path), map_location=device)
        model.eval()
        return model
    except Exception:
        return None


def _extract_state_dict(payload: object) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload  # raw state_dict
    return None


def _infer_linear_shapes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[int, int]]:
    pattern = re.compile(r".*?(\d+)\.weight$")
    indexed: List[Tuple[int, Tuple[int, int]]] = []
    for key, value in state_dict.items():
        if value.ndim != 2:
            continue
        m = pattern.match(key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        indexed.append((layer_idx, (int(value.shape[1]), int(value.shape[0]))))

    if indexed:
        return [shape for _, shape in sorted(indexed, key=lambda x: x[0])]

    generic = [(int(v.shape[1]), int(v.shape[0])) for v in state_dict.values() if v.ndim == 2]
    return generic


def _build_model_for_state_dict(
    state_dict: Dict[str, torch.Tensor],
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
) -> torch.nn.Module:
    shapes = _infer_linear_shapes_from_state_dict(state_dict)

    if not shapes:
        return FallbackMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    first_in = shapes[0][0]
    last_out = shapes[-1][1]

    model = FallbackMLP(
        input_dim=first_in if first_in > 0 else input_dim,
        hidden_dim=max(hidden_dim, shapes[0][1]),
        output_dim=last_out if last_out > 0 else output_dim,
    )
    return model


def load_checkpoint_model(config: CheckpointLoadConfig) -> torch.nn.Module:
    device = torch.device(config.device)

    scripted = _try_load_torchscript(config.checkpoint_path, device=device)
    if scripted is not None:
        return scripted.to(device)

    payload = torch.load(str(config.checkpoint_path), map_location=device)
    if isinstance(payload, torch.nn.Module):
        payload.eval()
        return payload.to(device)

    state_dict = _extract_state_dict(payload)
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Expected TorchScript, nn.Module, or dict/state_dict payload."
        )

    model = _build_model_for_state_dict(
        state_dict=state_dict,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing and len(missing) > 2:
        raise ValueError(
            f"Checkpoint partially incompatible with fallback model. Missing keys: {missing[:8]}"
        )
    if unexpected and len(unexpected) > 25:
        raise ValueError(
            f"Checkpoint appears incompatible with fallback model. Unexpected keys sample: {unexpected[:8]}"
        )

    model.eval()
    return model

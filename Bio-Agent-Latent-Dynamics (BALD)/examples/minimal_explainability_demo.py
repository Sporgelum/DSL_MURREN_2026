from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bald_explainability.pipeline import ExplainabilityEngine


class TinyTrajectoryModel(torch.nn.Module):
    """Toy model that maps Day0 expression to latent delta prediction."""

    def __init__(self, n_genes: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_genes, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    n_samples = 40
    n_genes = 500

    x = torch.randn(n_samples, n_genes)
    groups = ["high_responder"] * 20 + ["control"] * 20

    model = TinyTrajectoryModel(n_genes=n_genes)

    engine = ExplainabilityEngine(
        model=model,
        method="integrated_gradients",
        top_k=20,
    )

    result = engine.explain(x, groups)

    print("Group top genes (indices):")
    for group_name, indices in result.top_gene_indices_per_group.items():
        print(f"- {group_name}: {indices[:10].tolist()}")

    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    output_csv = Path("outputs/bald_explorer/demo_ranked_genes.csv")
    result.to_ranked_gene_csv(
        gene_names=gene_names,
        output_csv=output_csv,
        top_k=20,
        method="integrated_gradients",
    )

    print(f"Ranked-gene CSV exported: {output_csv}")
    print("Stability summary:", result.stability)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import torch


def _enable_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    passes: int = 30,
) -> dict[str, object]:
    _enable_dropout(model)
    samples = []
    chunk_size = min(10, passes)
    with torch.inference_mode():
        for start in range(0, passes, chunk_size):
            current_size = min(chunk_size, passes - start)
            repeated = inputs.repeat(current_size, 1, 1)
            samples.append(torch.softmax(model(repeated), dim=1).cpu().numpy())
    model.eval()
    mc_probabilities = np.concatenate(samples, axis=0)
    mean_probabilities = mc_probabilities.mean(axis=0)
    eps = 1e-12
    entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + eps))
    expected_entropy = np.mean(
        -np.sum(mc_probabilities * np.log(mc_probabilities + eps), axis=1)
    )
    return {
        "mean_probabilities": mean_probabilities,
        "entropy": float(entropy),
        "mutual_information": float(entropy - expected_entropy),
        "passes": passes,
    }

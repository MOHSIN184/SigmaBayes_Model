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
    with torch.inference_mode():
        for _ in range(passes):
            samples.append(torch.softmax(model(inputs), dim=1).cpu().numpy()[0])
    model.eval()
    mc_probabilities = np.stack(samples)
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

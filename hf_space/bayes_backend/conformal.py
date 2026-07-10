from __future__ import annotations

from collections.abc import Mapping, Sequence


def prediction_set(
    probabilities: Sequence[float],
    labels: Sequence[str],
    qhat: float,
) -> list[str]:
    threshold = 1.0 - float(qhat)
    return [
        label
        for label, probability in zip(labels, probabilities)
        if float(probability) >= threshold
    ]


def conformal_sets(
    probabilities: Sequence[float],
    labels: Sequence[str],
    thresholds: Mapping[str, float] | None,
) -> tuple[list[str], list[str]]:
    if not thresholds:
        return [], []
    return (
        prediction_set(probabilities, labels, thresholds["90"]),
        prediction_set(probabilities, labels, thresholds["95"]),
    )

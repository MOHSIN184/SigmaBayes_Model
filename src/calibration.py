from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score


def _get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_logits_tensor(logits, device):
    if isinstance(logits, torch.Tensor):
        return logits.detach().clone().float().to(device)
    return torch.tensor(logits, dtype=torch.float32, device=device)


def _to_labels_tensor(labels, device):
    if isinstance(labels, torch.Tensor):
        return labels.detach().clone().long().to(device)
    return torch.tensor(labels, dtype=torch.long, device=device)


def _softmax_numpy(logits):
    if isinstance(logits, torch.Tensor):
        logits_tensor = logits.detach().clone().float()
        return torch.softmax(logits_tensor, dim=1).cpu().numpy()
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits_tensor, dim=1).numpy()


class TemperatureScaler:
    def __init__(self, init_temperature=1.0):
        initial_value = max(float(init_temperature), 1e-3)
        self.temperature = torch.nn.Parameter(torch.tensor(initial_value))

    def fit(self, logits, labels, max_iter=100, lr=0.01, device=None):
        device = _get_device(device)
        logits_tensor = _to_logits_tensor(logits, device)
        labels_tensor = _to_labels_tensor(labels, device)
        self.temperature = torch.nn.Parameter(self.temperature.detach().clone().to(device))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                self.temperature.clamp_(min=1e-3)
            temperature = self.temperature
            loss = criterion(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)

        return self

    def transform(self, logits):
        input_was_numpy = not isinstance(logits, torch.Tensor)
        device = self.temperature.device
        logits_tensor = _to_logits_tensor(logits, device)
        scaled_logits = logits_tensor / torch.clamp(self.temperature, min=1e-3)

        if input_was_numpy:
            return scaled_logits.detach().cpu().numpy()
        return scaled_logits

    def predict_proba(self, logits):
        scaled_logits = self.transform(logits)
        if isinstance(scaled_logits, torch.Tensor):
            return torch.softmax(scaled_logits, dim=1).detach().cpu().numpy()
        return _softmax_numpy(scaled_logits)

    def get_temperature(self):
        return float(torch.clamp(self.temperature.detach().cpu(), min=1e-3).item())


def _calibration_bins(y_true, y_proba, n_bins=15):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = predictions == y_true
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        count = int(np.sum(in_bin))
        if count == 0:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "proportion": 0.0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        accuracy = float(np.mean(correct[in_bin]))
        confidence = float(np.mean(confidences[in_bin]))
        gap = abs(accuracy - confidence)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "proportion": count / len(y_true),
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": gap,
            }
        )

    return bins


def expected_calibration_error(y_true, y_proba, n_bins=15):
    bins = _calibration_bins(y_true, y_proba, n_bins=n_bins)
    return float(sum(item["proportion"] * item["gap"] for item in bins))


def maximum_calibration_error(y_true, y_proba, n_bins=15):
    bins = _calibration_bins(y_true, y_proba, n_bins=n_bins)
    return float(max((item["gap"] for item in bins), default=0.0))


def brier_score_multiclass(y_true, y_proba, num_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if num_classes is None:
        num_classes = y_proba.shape[1]

    one_hot = np.zeros((len(y_true), num_classes), dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true, y_proba, eps=1e-12):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    true_probabilities = y_proba[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(true_probabilities, eps, 1.0))))


def evaluate_calibration(y_true, y_proba, num_classes=None, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.argmax(y_proba, axis=1)

    return {
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_proba, n_bins=n_bins),
        "brier_score": brier_score_multiclass(
            y_true, y_proba, num_classes=num_classes
        ),
        "nll": negative_log_likelihood(y_true, y_proba),
        "mean_confidence": float(np.mean(np.max(y_proba, axis=1))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def plot_reliability_diagram(
    y_true,
    y_proba,
    save_path=None,
    n_bins=15,
    title="Reliability Diagram",
):
    bins = _calibration_bins(y_true, y_proba, n_bins=n_bins)
    bin_centers = np.array([(item["lower"] + item["upper"]) / 2 for item in bins])
    bin_accuracies = np.array([item["accuracy"] for item in bins])
    bin_confidences = np.array([item["confidence"] for item in bins])
    bin_counts = np.array([item["count"] for item in bins])

    fig, (ax_reliability, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax_reliability.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    non_empty = bin_counts > 0
    ax_reliability.plot(
        bin_confidences[non_empty],
        bin_accuracies[non_empty],
        marker="o",
        label="Model",
    )
    ax_reliability.set_ylabel("Accuracy")
    ax_reliability.set_title(title)
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.legend()

    width = 1.0 / n_bins
    ax_hist.bar(bin_centers, bin_counts, width=width * 0.9, color="steelblue")
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0, 1)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def calibrate_and_evaluate(
    val_logits,
    val_labels,
    test_logits,
    test_labels,
    n_bins=15,
):
    test_proba_before = _softmax_numpy(test_logits)
    before_metrics = evaluate_calibration(
        test_labels,
        test_proba_before,
        num_classes=test_proba_before.shape[1],
        n_bins=n_bins,
    )

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    test_proba_after = scaler.predict_proba(test_logits)
    after_metrics = evaluate_calibration(
        test_labels,
        test_proba_after,
        num_classes=test_proba_after.shape[1],
        n_bins=n_bins,
    )

    return {
        "temperature": scaler.get_temperature(),
        "before": before_metrics,
        "after": after_metrics,
        "test_proba_before": test_proba_before,
        "test_proba_after": test_proba_after,
    }


def save_calibration_metrics(before_metrics, after_metrics, temperature, save_path):
    rows = []
    for stage, metrics in [("before", before_metrics), ("after", after_metrics)]:
        row = {"stage": stage, "temperature": temperature}
        row.update(metrics)
        rows.append(row)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


def save_calibrated_probabilities(y_true, proba_before, proba_after, save_path):
    y_true = np.asarray(y_true, dtype=int)
    proba_before = np.asarray(proba_before, dtype=float)
    proba_after = np.asarray(proba_after, dtype=float)

    data = {
        "true_label": y_true,
        "before_pred": np.argmax(proba_before, axis=1),
        "before_confidence": np.max(proba_before, axis=1),
        "after_pred": np.argmax(proba_after, axis=1),
        "after_confidence": np.max(proba_after, axis=1),
    }

    for class_index in range(proba_before.shape[1]):
        data[f"before_proba_class_{class_index}"] = proba_before[:, class_index]
    for class_index in range(proba_after.shape[1]):
        data[f"after_proba_class_{class_index}"] = proba_after[:, class_index]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df

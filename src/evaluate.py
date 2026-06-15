from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict_logits(model, data_loader, device=None):
    device = _get_device(device)
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0))
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,))
    return logits, y_true


def predict_proba(model, data_loader, device=None):
    logits, y_true = predict_logits(model, data_loader, device=device)
    if logits.size == 0:
        return logits, y_true
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    return probabilities, y_true


def predict_classes_from_proba(y_proba):
    return np.argmax(y_proba, axis=1)


def _safe_metric(metric_fn, *args, **kwargs):
    try:
        value = metric_fn(*args, **kwargs)
        if np.isscalar(value) and not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def evaluate_classification(y_true, y_pred, y_proba=None, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.arange(len(class_names)) if class_names is not None else np.unique(y_true)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_f1": {},
    }

    per_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    for index, label in enumerate(labels):
        name = class_names[index] if class_names is not None else str(label)
        metrics["per_class_precision"][name] = per_precision[index]
        metrics["per_class_recall"][name] = per_recall[index]
        metrics["per_class_f1"][name] = per_f1[index]

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        num_classes = y_proba.shape[1]

        if num_classes == 2:
            positive_scores = y_proba[:, 1]
            metrics["auroc"] = _safe_metric(roc_auc_score, y_true, positive_scores)
            metrics["auprc"] = _safe_metric(
                average_precision_score, y_true, positive_scores
            )
        else:
            y_bin = label_binarize(y_true, classes=np.arange(num_classes))
            metrics["auroc_macro_ovr"] = _safe_metric(
                roc_auc_score,
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
            metrics["auprc_macro_ovr"] = _safe_metric(
                average_precision_score,
                y_bin,
                y_proba,
                average="macro",
            )

    return metrics


def classification_report_dataframe(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def _save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return path


def plot_roc_pr_curves(y_true, y_proba, class_names, save_dir=None, prefix="model"):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    num_classes = y_proba.shape[1]
    generated_paths = {}
    save_dir = Path(save_dir) if save_dir is not None else None

    if num_classes == 2:
        scores = y_proba[:, 1]

        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auroc = roc_auc_score(y_true, scores)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            fig.tight_layout()
            if save_dir is not None:
                generated_paths["roc_curve"] = _save_figure(
                    fig, save_dir / f"{prefix}_roc_curve.png"
                )
        except Exception:
            pass

        try:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            auprc = average_precision_score(y_true, scores)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend()
            fig.tight_layout()
            if save_dir is not None:
                generated_paths["pr_curve"] = _save_figure(
                    fig, save_dir / f"{prefix}_pr_curve.png"
                )
        except Exception:
            pass

        return generated_paths

    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for class_index, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_proba[:, class_index])
            ax.plot(fpr, tpr, label=class_name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend()
        fig.tight_layout()
        if save_dir is not None:
            generated_paths["roc_curve"] = _save_figure(
                fig, save_dir / f"{prefix}_roc_curve.png"
            )
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for class_index, class_name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(
                y_bin[:, class_index], y_proba[:, class_index]
            )
            ax.plot(recall, precision, label=class_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("One-vs-Rest Precision-Recall Curves")
        ax.legend()
        fig.tight_layout()
        if save_dir is not None:
            generated_paths["pr_curve"] = _save_figure(
                fig, save_dir / f"{prefix}_pr_curve.png"
            )
    except Exception:
        pass

    return generated_paths


def plot_training_history(history_df, save_path=None):
    metric_columns = [
        column
        for column in ["val_accuracy", "val_f1_macro"]
        if column in history_df.columns
    ]
    num_plots = 1 + int(bool(metric_columns))
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    axes = np.atleast_1d(axes)

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    if metric_columns:
        for column in metric_columns:
            axes[1].plot(history_df["epoch"], history_df[column], label=column)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def save_metrics_table(metrics, save_path):
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            value = str(value)
        rows.append({"metric": key, "value": value})

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


def save_classification_report(y_true, y_pred, class_names, save_path):
    report_df = classification_report_dataframe(y_true, y_pred, class_names)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(save_path)
    return report_df


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predictive_entropy(probs, eps=1e-12):
    probs = np.asarray(probs)
    return -np.sum(probs * np.log(probs + eps), axis=1)


def probability_variance(mc_probs):
    mc_probs = np.asarray(mc_probs)
    return np.var(mc_probs, axis=0).mean(axis=1)


def mutual_information(mc_probs, eps=1e-12):
    mc_probs = np.asarray(mc_probs)
    mean_probs = mc_probs.mean(axis=0)
    entropy_mean = predictive_entropy(mean_probs, eps=eps)
    expected_entropy = np.mean(
        -np.sum(mc_probs * np.log(mc_probs + eps), axis=2),
        axis=0,
    )
    return entropy_mean - expected_entropy


def mc_dropout_predict(model, data_loader, T=30, device=None):
    device = _get_device(device)
    model = model.to(device)
    model.eval()
    enable_dropout(model)

    all_passes = []
    y_true = None

    with torch.no_grad():
        for _ in range(T):
            pass_probs = []
            pass_labels = []
            for x, y in data_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pass_probs.append(probs.detach().cpu().numpy())
                pass_labels.append(y.detach().cpu().numpy())

            all_passes.append(np.concatenate(pass_probs, axis=0))
            if y_true is None:
                y_true = np.concatenate(pass_labels, axis=0)

    mc_proba = np.stack(all_passes, axis=0)
    mean_proba = mc_proba.mean(axis=0)
    y_pred = np.argmax(mean_proba, axis=1)
    confidence = np.max(mean_proba, axis=1)

    return {
        "mean_proba": mean_proba,
        "mc_proba": mc_proba,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
        "entropy": predictive_entropy(mean_proba),
        "variance": probability_variance(mc_proba),
        "mutual_information": mutual_information(mc_proba),
    }


def create_uncertainty_dataframe(mc_result, class_names):
    y_true = np.asarray(mc_result["y_true"])
    y_pred = np.asarray(mc_result["y_pred"])

    return pd.DataFrame(
        {
            "true_label": y_true,
            "true_label_name": [class_names[label] for label in y_true],
            "predicted_label": y_pred,
            "predicted_label_name": [class_names[label] for label in y_pred],
            "confidence": mc_result["confidence"],
            "entropy": mc_result["entropy"],
            "variance": mc_result["variance"],
            "mutual_information": mc_result["mutual_information"],
            "correct": y_true == y_pred,
        }
    )


def save_uncertainty_results(mc_result, class_names, save_path):
    uncertainty_df = create_uncertainty_dataframe(mc_result, class_names)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_df.to_csv(save_path, index=False)
    return uncertainty_df


def plot_uncertainty_histogram(uncertainty_df, save_path=None, column="entropy"):
    fig, ax = plt.subplots(figsize=(7, 5))

    correct_values = uncertainty_df.loc[uncertainty_df["correct"], column]
    incorrect_values = uncertainty_df.loc[~uncertainty_df["correct"], column]

    if len(correct_values) > 0:
        ax.hist(correct_values, bins=30, alpha=0.7, label="Correct")
    if len(incorrect_values) > 0:
        ax.hist(incorrect_values, bins=30, alpha=0.7, label="Incorrect")

    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"{column.replace('_', ' ').title()} Distribution")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig

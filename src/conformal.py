from pathlib import Path

import numpy as np
import pandas as pd


def compute_nonconformity_scores(y_cal, proba_cal):
    y_cal = np.asarray(y_cal, dtype=int)
    proba_cal = np.asarray(proba_cal, dtype=float)
    true_class_probabilities = proba_cal[np.arange(len(y_cal)), y_cal]
    return 1.0 - true_class_probabilities


def conformal_quantile(scores, alpha=0.1):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores must contain at least one value.")

    n = scores.size
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


def conformal_prediction_sets(proba_test, qhat, class_names=None):
    proba_test = np.asarray(proba_test, dtype=float)
    threshold = 1.0 - qhat
    prediction_sets = []

    for probabilities in proba_test:
        included_indexes = np.where(probabilities >= threshold)[0].tolist()
        if class_names is None:
            prediction_sets.append(included_indexes)
        else:
            prediction_sets.append([class_names[index] for index in included_indexes])

    return prediction_sets


def _set_contains_label(prediction_set, label, class_names=None):
    if class_names is not None:
        label_name = class_names[int(label)]
        return label_name in prediction_set or int(label) in prediction_set
    return int(label) in prediction_set


def _label_key(label, class_names=None):
    if class_names is None:
        return str(int(label))
    return class_names[int(label)]


def evaluate_conformal_sets(y_true, prediction_sets, class_names=None):
    y_true = np.asarray(y_true, dtype=int)
    set_sizes = np.array([len(prediction_set) for prediction_set in prediction_sets])
    covered = np.array(
        [
            _set_contains_label(prediction_set, label, class_names=class_names)
            for label, prediction_set in zip(y_true, prediction_sets)
        ],
        dtype=bool,
    )

    class_wise_coverage = {}
    class_wise_average_set_size = {}
    labels = np.arange(len(class_names)) if class_names is not None else np.unique(y_true)

    for label in labels:
        label_mask = y_true == label
        key = _label_key(label, class_names=class_names)
        if np.any(label_mask):
            class_wise_coverage[key] = float(np.mean(covered[label_mask]))
            class_wise_average_set_size[key] = float(np.mean(set_sizes[label_mask]))
        else:
            class_wise_coverage[key] = None
            class_wise_average_set_size[key] = None

    return {
        "coverage": float(np.mean(covered)) if len(covered) else 0.0,
        "average_set_size": float(np.mean(set_sizes)) if len(set_sizes) else 0.0,
        "median_set_size": float(np.median(set_sizes)) if len(set_sizes) else 0.0,
        "singleton_rate": float(np.mean(set_sizes == 1)) if len(set_sizes) else 0.0,
        "empty_set_rate": float(np.mean(set_sizes == 0)) if len(set_sizes) else 0.0,
        "max_set_size": int(np.max(set_sizes)) if len(set_sizes) else 0,
        "min_set_size": int(np.min(set_sizes)) if len(set_sizes) else 0,
        "class_wise_coverage": class_wise_coverage,
        "class_wise_average_set_size": class_wise_average_set_size,
    }


def run_conformal_prediction(
    y_cal,
    proba_cal,
    y_test,
    proba_test,
    class_names=None,
    alpha=0.1,
):
    scores = compute_nonconformity_scores(y_cal, proba_cal)
    qhat = conformal_quantile(scores, alpha=alpha)
    prediction_sets = conformal_prediction_sets(
        proba_test, qhat, class_names=class_names
    )
    metrics = evaluate_conformal_sets(
        y_test, prediction_sets, class_names=class_names
    )

    return {
        "alpha": alpha,
        "confidence_level": 1 - alpha,
        "qhat": qhat,
        "prediction_sets": prediction_sets,
        "metrics": metrics,
    }


def _prediction_set_to_string(prediction_set):
    return "|".join(str(item) for item in prediction_set)


def prediction_sets_to_dataframe(
    y_true,
    y_pred,
    y_proba,
    prediction_sets,
    class_names=None,
):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    if class_names is None:
        class_names = [str(index) for index in range(y_proba.shape[1])]

    data = {
        "true_label": y_true,
        "true_label_name": [class_names[label] for label in y_true],
        "predicted_label": y_pred,
        "predicted_label_name": [class_names[label] for label in y_pred],
        "confidence": np.max(y_proba, axis=1),
        "prediction_set": [
            _prediction_set_to_string(prediction_set)
            for prediction_set in prediction_sets
        ],
        "set_size": [len(prediction_set) for prediction_set in prediction_sets],
        "covered": [
            _set_contains_label(prediction_set, label, class_names=class_names)
            for label, prediction_set in zip(y_true, prediction_sets)
        ],
    }

    for class_index, class_name in enumerate(class_names):
        safe_class_name = str(class_name).replace(" ", "_").replace("-", "_")
        data[f"proba_{safe_class_name}"] = y_proba[:, class_index]

    return pd.DataFrame(data)


def save_conformal_results(result, y_true, y_proba, save_path, class_names=None):
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.argmax(y_proba, axis=1)
    df = prediction_sets_to_dataframe(
        y_true,
        y_pred,
        y_proba,
        result["prediction_sets"],
        class_names=class_names,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def save_conformal_metrics(results_by_alpha, save_path):
    rows = []
    for result in results_by_alpha:
        metrics = result["metrics"]
        rows.append(
            {
                "alpha": result["alpha"],
                "confidence_level": result["confidence_level"],
                "qhat": result["qhat"],
                "coverage": metrics["coverage"],
                "average_set_size": metrics["average_set_size"],
                "median_set_size": metrics["median_set_size"],
                "singleton_rate": metrics["singleton_rate"],
                "empty_set_rate": metrics["empty_set_rate"],
                "min_set_size": metrics["min_set_size"],
                "max_set_size": metrics["max_set_size"],
                "class_wise_coverage": str(metrics["class_wise_coverage"]),
                "class_wise_average_set_size": str(
                    metrics["class_wise_average_set_size"]
                ),
            }
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df

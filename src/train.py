import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels, num_classes):
    labels_array = np.asarray(labels, dtype=int)
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_array,
    )
    return torch.tensor(weights, dtype=torch.float32)


def create_data_loader(dataset, batch_size=64, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples else 0.0


def evaluate_on_loader(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    average_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
    f1_macro = (
        f1_score(all_labels, all_predictions, average="macro", zero_division=0)
        if all_labels
        else 0.0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    num_classes,
    epochs=50,
    lr=1e-3,
    patience=8,
    device=None,
    use_class_weights=True,
    class_labels=None,
    model_save_path=None,
):
    if device is None:
        device = get_device()
    device = torch.device(device)
    model = model.to(device)

    if use_class_weights and class_labels is not None:
        class_weights = compute_class_weights(class_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 2),
    )

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_on_loader(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "learning_rate": learning_rate,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            if model_save_path is not None:
                save_model(model, model_save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    return model, pd.DataFrame(history)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    if device is None:
        device = get_device()
    device = torch.device(device)
    state_dict = torch.load(Path(path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

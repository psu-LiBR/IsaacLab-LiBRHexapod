"""Train a basic terrain neural network using the PyTorch Quickstart structure.

Reference: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
Uses the existing features.csv, including its original train/validation/test splits.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from terrain_network import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_epoch(dataloader, model, loss_fn, optimizer):
    """Learn from each batch, following the tutorial's training loop."""
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        prediction = model(x)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


def evaluate(dataloader, model, loss_fn):
    """Make predictions without changing the network's weights."""
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for x, y in dataloader:
            scores = model(x)
            total_loss += loss_fn(scores, y).item() * len(y)
            predictions.extend(scores.argmax(1).tolist())
    return total_loss / len(dataloader.dataset), predictions


def train(features: Path, output: Path, epochs: int = 100) -> None:
    """Read features, train on CPU, select an epoch on validation data, and test."""
    if epochs < 1:
        raise ValueError("Epochs must be at least one.")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    with features.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The feature CSV is empty.")

    feature_names = []
    for name in rows[0]:
        if name.startswith(("feature_acc_", "feature_gyro_")):
            feature_names.append(name)
    if len(feature_names) != 24:
        raise ValueError("Expected 24 IMU features from prepare_recordings.py.")
    labels = sorted({row["label"] for row in rows})
    x = []
    y = []
    splits = []
    for row in rows:
        values = []
        for name in feature_names:
            values.append(float(row[name]))
        x.append(values)
        y.append(labels.index(row["label"]))
        splits.append(row["split"])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    splits = np.array(splits)
    if not np.isfinite(x).all():
        raise ValueError("Sensor features must all be finite.")
    if not set(splits).issubset({"train", "validation", "test"}):
        raise ValueError("Unknown split in feature CSV.")

    gait_groups = {}
    sources = {}
    for row in rows:
        for field, assignments in [("gait_sha256", gait_groups), ("source", sources)]:
            if assignments.setdefault(row[field], row["split"]) != row["split"]:
                raise ValueError(f"{field} appears in more than one split.")
    for split in ("train", "validation", "test"):
        if set(y[splits == split]) != set(range(len(labels))):
            raise ValueError(f"Every terrain must appear in {split}.")

    feature_mean = x[splits == "train"].mean(axis=0)
    std = x[splits == "train"].std(axis=0)
    std[std == 0] = 1.0
    x = (x - feature_mean) / std
    loaders = {}
    for split in ("train", "validation", "test"):
        dataset = TensorDataset(torch.tensor(x[splits == split]), torch.tensor(y[splits == split]))
        loaders[split] = DataLoader(dataset, batch_size=64, shuffle=(split == "train"))

    model = NeuralNetwork(len(feature_names), len(labels))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    best_f1 = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        training_loss = train_epoch(loaders["train"], model, loss_fn, optimizer)
        validation_loss, validation_predictions = evaluate(loaders["validation"], model, loss_fn)
        validation_f1 = float(f1_score(y[splits == "validation"], validation_predictions, average="macro"))
        history.append(
            dict(
                epoch=epoch + 1,
                training_loss=training_loss,
                validation_loss=validation_loss,
                validation_macro_f1=validation_f1,
            )
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={training_loss:.4f}, validation macro F1={validation_f1:.3f}", flush=True)

    model.load_state_dict(best_weights)
    test_loss, predictions = evaluate(loaders["test"], model, loss_fn)
    test_y = y[splits == "test"]
    report = dict(
        features_path=str(features.resolve()),
        features_sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
        labels=labels,
        feature_names=feature_names,
        epochs=epochs,
        best_epoch=best_epoch,
        validation_macro_f1=best_f1,
        test_loss=test_loss,
        test_accuracy=float(accuracy_score(test_y, predictions)),
        test_balanced_accuracy=float(balanced_accuracy_score(test_y, predictions)),
        test_macro_f1=float(f1_score(test_y, predictions, average="macro")),
        confusion_matrix=confusion_matrix(test_y, predictions).tolist(),
        per_class=classification_report(test_y, predictions, target_names=labels, output_dict=True, zero_division=0),
        split_counts={split: int((splits == split).sum()) for split in ("train", "validation", "test")},
        torch_version=str(torch.__version__),
        history=history,
        limitations=[
            "Collection days overlap splits; new-day performance is unverified.",
            "The same historical test set was previously evaluated with the random forest.",
            "Unknown terrains are assigned a known label. This model is offline only.",
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            feature_names=feature_names,
            labels=labels,
            mean=torch.tensor(feature_mean),
            std=torch.tensor(std),
            schema_version=1,
        ),
        output / "terrain_model.pt",
    )
    (output / "pytorch_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "pytorch_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "cycle", "label", "prediction"])
        test_rows = [row for row in rows if row["split"] == "test"]
        for row, prediction in zip(test_rows, predictions):
            writer.writerow([row["source"], row["cycle"], row["label"], labels[prediction]])
    print(
        f"Selected epoch {best_epoch}; test accuracy={report['test_accuracy']:.3f}, "
        f"macro F1={report['test_macro_f1']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="Existing features.csv with train/validation/test assignments.")
    parser.add_argument("output", type=Path, help="Output directory for the PyTorch model and report.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of passes through the training data.")
    args = parser.parse_args()
    train(args.features, args.output, args.epochs)

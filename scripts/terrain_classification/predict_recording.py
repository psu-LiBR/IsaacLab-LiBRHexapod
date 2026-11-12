"""Classify complete gait cycles in a recording using a locally trained model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from prepare_recordings import extract


def predict(model_path: Path, recording: Path) -> list[dict]:
    """Return a terrain label and probability score for each complete gait cycle."""
    if model_path.suffix == ".pt":
        return predict_pytorch(model_path, recording)
    artifact = joblib.load(model_path)
    rows = extract(recording, "inference", "test")
    x = np.array([[row[name] for name in artifact["feature_names"]] for row in rows])
    predictions = artifact["model"].predict(x)
    probabilities = artifact["model"].predict_proba(x)
    return [
        dict(cycle=row["cycle"], terrain=str(label), score=float(max(scores)))
        for row, label, scores in zip(rows, predictions, probabilities)
    ]


def predict_pytorch(model_path: Path, recording: Path) -> list[dict]:
    """Run the tutorial-based network on complete recording cycles."""
    import torch
    from terrain_network import NeuralNetwork

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    rows = extract(recording, "inference", "test")
    values = []
    for row in rows:
        cycle_features = []
        for name in checkpoint["feature_names"]:
            cycle_features.append(row[name])
        values.append(cycle_features)
    x = torch.tensor(values, dtype=torch.float32)
    x = (x - checkpoint["mean"]) / checkpoint["std"]

    model = NeuralNetwork(len(checkpoint["feature_names"]), len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        scores = model(x)
        probabilities = torch.softmax(scores, dim=1)
        predictions = probabilities.argmax(1)
    results = []
    for row, prediction, probability in zip(rows, predictions, probabilities):
        label = checkpoint["labels"][prediction.item()]
        results.append(dict(cycle=row["cycle"], terrain=label, score=probability.max().item()))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a trained terrain model (.joblib or .pt).")
    parser.add_argument("recording", type=Path, help="Path to a recording folder containing metadata and sensor CSVs.")
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.recording), indent=2))

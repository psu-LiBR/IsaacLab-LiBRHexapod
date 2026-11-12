"""Train and evaluate a terrain classifier on labeled recording folders."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from prepare_recordings import extract
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TERRAINS = {"10mmBead", "20mm", "20mmBead", "30mm", "30mmBead", "40mmBead", "antislip", "flatwood", "rubberwood"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def grouped_split(y, groups):
    """Keep every use of a gait hash together, including across terrain labels."""
    classes = set(y)
    for seed in range(1000):
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(y, y, groups))
        train_local, val_local = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000).split(
                y[trainval], y[trainval], groups[trainval]
            )
        )
        train, val = trainval[train_local], trainval[val_local]
        if all(set(y[index]) == classes for index in (train, val, test)):
            assert not (
                set(groups[train]) & set(groups[val])
                or set(groups[train]) & set(groups[test])
                or set(groups[val]) & set(groups[test])
            )
            return train, val, test, seed
    raise ValueError("Cannot cover every terrain with disjoint gait groups; collect more independent gaits.")


def metrics(y, prediction, labels):
    return dict(
        accuracy=float(accuracy_score(y, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y, prediction, labels=labels).tolist(),
        per_class=classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0),
    )


def train(root, output, merge_bead_labels=False):
    output.mkdir(parents=True, exist_ok=True)
    rows, accepted, rejected = [], [], []
    for path in sorted(root.rglob("meta.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8-sig"))
            label = meta.get("surface", "").strip()
            if label not in TERRAINS:
                raise ValueError(f"Excluded non-terrain or unknown label: {label!r}")
            session = f"{meta['host']}:{meta['start_time_iso'][:10]}"
            features = extract(path.parent, session, "train")
            if merge_bead_labels:
                label = {"20mm": "20mmBead", "30mm": "30mmBead"}.get(label, label)
                for row in features:
                    row["label"] = label
            hashes = {
                name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
                for name in ("meta.json", "imu.csv", "joints.csv")
            }
            accepted.append(
                dict(
                    source=str(path.parent),
                    label=label,
                    session_id=session,
                    gait_sha256=meta["gait"]["sha256"],
                    cycles=len(features),
                    input_sha256=hashes,
                )
            )
            rows.extend(features)
        except (ValueError, KeyError, OSError, TypeError, OverflowError) as error:
            rejected.append(dict(source=str(path.parent), reason=str(error)))
    gait_counts = {
        label: len({row["gait_sha256"] for row in rows if row["label"] == label})
        for label in sorted({row["label"] for row in rows})
    }
    excluded_labels = {label: count for label, count in gait_counts.items() if count < 3}
    withheld = [run for run in accepted if run["label"] in excluded_labels]
    accepted = [run for run in accepted if run["label"] not in excluded_labels]
    rows = [row for row in rows if row["label"] not in excluded_labels]
    write_json(
        output / "data_audit.json",
        dict(
            accepted=accepted,
            rejected=rejected,
            withheld_insufficient_gait_groups=withheld,
            excluded_label_gait_counts=excluded_labels,
            folders_without_metadata=[
                str(p.parent) for p in sorted(root.rglob("imu.csv")) if not (p.parent / "meta.json").exists()
            ],
        ),
    )
    if not rows:
        raise ValueError("No usable recordings; see data_audit.json")
    names = [key for key in rows[0] if key.startswith("feature_")]
    x = np.array([[row[key] for key in names] for row in rows], dtype=float)
    y = np.array([row["label"] for row in rows])
    groups = np.array([row["gait_sha256"] for row in rows])
    sources = np.array([row["source"] for row in rows])
    labels = sorted(set(y))
    train_idx, val_idx, test_idx, seed = grouped_split(y, groups)
    splits = np.full(len(rows), "", dtype="<U10")
    for split, index in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        splits[index] = split
        for i in index:
            rows[i]["split"] = split
    with (output / "features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "session_id", "gait_sha256", "split", "label"])
        writer.writeheader()
        for run in accepted:
            index = np.flatnonzero(sources == run["source"])[0]
            writer.writerow(
                dict(
                    path=run["source"],
                    session_id=run["session_id"],
                    gait_sha256=run["gait_sha256"],
                    split=str(splits[index]),
                    label=run["label"],
                )
            )
    candidates = {}
    fitted = {}
    imu = np.array([i for i, name in enumerate(names) if name.startswith(("feature_acc_", "feature_gyro_"))])
    for subset, columns in [("full", np.arange(len(names))), ("imu", imu)]:
        estimators = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
            "random_forest": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
            ),
        }
        for name, estimator in estimators.items():
            key = subset + "_" + name
            estimator.fit(x[train_idx][:, columns], y[train_idx])
            score = metrics(y[val_idx], estimator.predict(x[val_idx][:, columns]), labels)
            candidates[key] = score
            fitted[key] = (estimator, columns)
            print(f"{key}: validation macro F1={score['macro_f1']:.3f}", flush=True)
    best = max(candidates, key=lambda key: candidates[key]["macro_f1"])
    estimator, columns = fitted[best]
    # Keep the validation-selected, training-only model; test is evaluated once.
    prediction = estimator.predict(x[test_idx][:, columns])
    baseline = DummyClassifier(strategy="most_frequent").fit(x[train_idx], y[train_idx])
    test_metrics = metrics(y[test_idx], prediction, labels)
    per_run = []
    for source in sorted(set(sources[test_idx])):
        mask = sources[test_idx] == source
        per_run.append(
            dict(
                source=source,
                label=str(y[test_idx][mask][0]),
                cycles=int(mask.sum()),
                accuracy=float(accuracy_score(y[test_idx][mask], prediction[mask])),
            )
        )
    report = dict(
        dataset_root=str(root.resolve()),
        labels=labels,
        accepted_runs=len(accepted),
        rejected_runs=len(rejected),
        withheld_runs=len(withheld),
        excluded_label_gait_counts=excluded_labels,
        cycles=len(rows),
        features=len(names),
        merge_bead_labels=merge_bead_labels,
        split_seed=seed,
        split_policy="Disjoint gait SHA256 groups across all terrains; entire runs kept together.",
        limitations=[
            "Collection days may overlap splits; this does not establish performance on a new day or robot.",
            "Terrain labels are taken from metadata and have not been independently verified.",
            "Predictions require the same sensors, units, motor order and cycle feature extraction.",
            "This is an offline classifier, not yet connected to live robot control.",
        ],
        split_counts={
            split: dict(
                cycles=int((splits == split).sum()),
                runs=len(set(sources[splits == split])),
                gait_groups=len(set(groups[splits == split])),
                per_class=dict(Counter(y[splits == split])),
            )
            for split in ("train", "validation", "test")
        },
        validation_candidates=candidates,
        selected_model=best,
        test=test_metrics,
        test_majority_baseline=metrics(y[test_idx], baseline.predict(x[test_idx]), labels),
        test_per_run=per_run,
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
    )
    write_json(output / "report.json", report)
    artifact = dict(
        model=estimator,
        feature_names=[names[i] for i in columns],
        labels=labels,
        schema_version=1,
        model_name=best,
        versions=report["versions"],
    )
    joblib.dump(artifact, output / "terrain_model.joblib")
    restored = joblib.load(output / "terrain_model.joblib")
    assert np.array_equal(prediction, restored["model"].predict(x[test_idx][:, columns]))
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "cycle", "label", "prediction"])
        writer.writerows(
            (rows[i]["source"], rows[i]["cycle"], rows[i]["label"], str(p)) for i, p in zip(test_idx, prediction)
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("accepted_runs", "rejected_runs", "cycles", "selected_model", "split_counts")
            },
            indent=2,
        )
    )
    print(
        f"Test accuracy={test_metrics['accuracy']:.3f}, balanced accuracy={test_metrics['balanced_accuracy']:.3f}, macro F1={test_metrics['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--merge-bead-labels", action="store_true", help="Only use if 20mm/30mm aliases are confirmed.")
    args = parser.parse_args()
    train(args.dataset, args.output, args.merge_bead_labels)

#!/usr/bin/env python3
"""Learn a component scorer and combine it with leakage-safe row OOF gating.

The frozen segmentation trunk is not retrained.  Candidate pixels are the
existing probability-threshold mask, so the scorer and row gate can only
remove predicted pixels; they can never invent bridges or modify gold masks.
The component scorer is fitted on writer-disjoint fitting rows and calibrated
on a deterministic writer holdout before the fixed segmenter validation is
read.  Row gating uses stored writer-held-out predictions; missing OOF rows are
conservatively kept.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOLD300_PATH = ROOT / "scripts/380_revision_gold300_factorial_seed42.py"
COMPARE_PATH = ROOT / "scripts/372_revision_gold105_seed42_compare.py"
GOLD300_SHA256 = "c680ebfe44598043e831f9df621d38091869d0f5fcb4a02ad0650672f58c8686"
COMPARE_SHA256 = "4d84393bd2b5bdf6437df39ee86ca2b0c644899ddf094fe3157f4517e83cb95d"
SCHEMA = "hme-revision-component-scorer-oof-gate-dev-v1"
SEED = 42
PROBABILITY_THRESHOLD = 0.32
HEURISTIC_THRESHOLD = 0.45
ROW_GATE_THRESHOLD = 0.005197631195187569
FEATURE_NAMES = (
    "log_area", "width", "height", "log_aspect", "fill",
    "mean_probability", "max_probability", "p75_probability",
    "p90_probability", "probability_std", "mean_ink", "bbox_ink",
    "centroid_x", "centroid_y", "border_touch", "area_over_major",
    "heuristic_score",
)


class Experiment381Error(RuntimeError):
    """Raised when the locked component-scoring protocol changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str | None = None):
    actual = digest_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise Experiment381Error(
            f"dependency digest mismatch for {path.name}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Experiment381Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment381Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment381Error(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_bucket(group_uuid: str, buckets: int = 5) -> int:
    if not group_uuid or buckets < 2:
        raise Experiment381Error("writer bucket requires an identity and >=2 buckets")
    digest = hashlib.sha256(("381-writer:" + group_uuid).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    import cv2

    count, labels = cv2.connectedComponents(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    return labels.astype(np.int32, copy=False), int(count - 1)


def _major_axis(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) <= 1:
        return 1.0
    points = np.column_stack((xs.astype(float), ys.astype(float)))
    centred = points - points.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(1, len(xs) - 1)
    values, vectors = np.linalg.eigh(covariance)
    projection = centred @ vectors[:, int(np.argmax(values))]
    return max(1.0, float(np.ptp(projection) + 1.0))


def component_feature_vector(
    component: np.ndarray,
    probabilities: np.ndarray,
    image: np.ndarray,
    *,
    probability_threshold: float = PROBABILITY_THRESHOLD,
    heuristic_score: float = 0.0,
) -> np.ndarray:
    component = np.asarray(component, dtype=bool)
    probability = np.asarray(probabilities, dtype=np.float32)
    gray = np.asarray(image, dtype=np.float32)
    if component.ndim != 2 or probability.shape != component.shape \
            or gray.shape != component.shape or not bool(component.any()):
        raise Experiment381Error("invalid component feature inputs")
    height, width = component.shape
    ys, xs = np.nonzero(component)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_width, box_height = x1 - x0, y1 - y0
    area = int(len(xs))
    values = probability[component]
    ink = np.clip(1.0 - gray / 255.0, 0.0, 1.0)
    major = _major_axis(xs, ys)
    vector = np.asarray([
        math.log1p(area) / math.log1p(height * width),
        box_width / width,
        box_height / height,
        math.log(max(box_width, box_height) / max(1, min(box_width, box_height))),
        area / max(1, box_width * box_height),
        float(values.mean()),
        float(values.max()),
        float(np.quantile(values, 0.75)),
        float(np.quantile(values, 0.90)),
        float(values.std()),
        float(ink[component].mean()),
        float(ink[y0:y1, x0:x1].mean()),
        float(xs.mean() / max(1, width - 1)),
        float(ys.mean() / max(1, height - 1)),
        float(x0 == 0 or y0 == 0 or x1 == width or y1 == height),
        float(area / major / max(height, 1)),
        float(heuristic_score),
    ], dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise Experiment381Error("non-finite component features")
    return vector


def component_training_records(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    compare: Any,
) -> list[dict[str, Any]]:
    if len(rows) != len(probabilities):
        raise Experiment381Error("component training rows/probabilities differ")
    records: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities):
        support = np.asarray(probability >= PROBABILITY_THRESHOLD, dtype=bool)
        labels, count = _label_components(support)
        base_features = compare.component_features(support, probability)[1]
        truth = np.asarray(row["mask"], dtype=bool)
        truth_labels, truth_count = _label_components(truth)
        truth_areas = np.bincount(truth_labels.ravel(), minlength=truth_count + 1)
        for label in range(1, count + 1):
            component = labels == label
            feature = base_features[label - 1]
            heuristic = compare.heuristic_component_score(
                feature, PROBABILITY_THRESHOLD)
            intersections = np.bincount(
                truth_labels[component], minlength=truth_count + 1)
            pred_area = int(component.sum())
            matched_iou = 0.0
            for truth_label in range(1, truth_count + 1):
                intersection = int(intersections[truth_label])
                if intersection:
                    union = pred_area + int(truth_areas[truth_label]) - intersection
                    matched_iou = max(matched_iou, intersection / max(1, union))
            records.append({
                "features": component_feature_vector(
                    component, probability, row["image"],
                    heuristic_score=heuristic,
                ),
                "group_uuid": str(row["group_uuid"]),
                "label": float(matched_iou >= 0.10),
                "line_id": str(row["line_id"]),
                "matched_iou": matched_iou,
            })
    return records


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if len(labels) == 0 or positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def threshold_metrics(labels: np.ndarray, scores: np.ndarray,
                      threshold: float) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores >= threshold, dtype=bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "false_negative": fn, "false_positive": fp, "f1": f1,
        "precision": precision, "recall": recall, "true_positive": tp,
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray, *,
                     minimum_recall: float = 0.0) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(labels) == 0 or len(labels) != len(scores) or len(np.unique(labels)) != 2:
        raise Experiment381Error("threshold selection requires aligned two-class data")
    candidates = np.unique(np.concatenate((
        np.linspace(0.05, 0.95, 181, dtype=np.float32), scores)))
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = threshold_metrics(labels, scores, float(threshold))
        if float(metrics["recall"]) + 1e-12 < minimum_recall:
            continue
        rank = (float(metrics["f1"]), float(metrics["precision"]),
                float(metrics["recall"]), float(threshold))
        if best is None or rank > best[0]:
            best = rank, float(threshold), metrics
    if best is None:
        raise Experiment381Error("no scorer threshold satisfies recall guard")
    return best[1], {**best[2], "threshold": best[1]}


def train_mlp(records: Sequence[Mapping[str, Any]], torch: Any):
    training = [record for record in records
                if stable_bucket(str(record["group_uuid"])) != 0]
    development = [record for record in records
                   if stable_bucket(str(record["group_uuid"])) == 0]
    if min(len(training), len(development)) == 0:
        raise Experiment381Error("writer split produced an empty scorer partition")
    x_train = np.stack([record["features"] for record in training])
    y_train = np.asarray([record["label"] for record in training], dtype=np.float32)
    x_dev = np.stack([record["features"] for record in development])
    y_dev = np.asarray([record["label"] for record in development], dtype=np.float32)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_dev)) != 2:
        raise Experiment381Error("writer split lacks a scorer class")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), 32), torch.nn.GELU(),
        torch.nn.Linear(32, 16), torch.nn.GELU(), torch.nn.Linear(16, 1),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    train_x = torch.from_numpy(((x_train - mean) / scale).astype(np.float32))
    train_y = torch.from_numpy(y_train[:, None])
    positive = float(y_train.sum())
    pos_weight = torch.tensor([(len(y_train) - positive) / max(positive, 1.0)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = math.inf
    best_state = None
    stale = 0
    dev_x = torch.from_numpy(((x_dev - mean) / scale).astype(np.float32))
    dev_y = torch.from_numpy(y_dev[:, None])
    for epoch in range(600):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = float(loss_fn(model(dev_x), dev_y))
        if dev_loss < best_loss - 1e-6:
            best_loss = dev_loss
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 80:
            break
    if best_state is None:
        raise Experiment381Error("component scorer did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        dev_scores = torch.sigmoid(model(dev_x)).numpy().ravel()
    heuristic_index = FEATURE_NAMES.index("heuristic_score")
    heuristic_dev = threshold_metrics(
        y_dev, x_dev[:, heuristic_index], HEURISTIC_THRESHOLD)
    threshold, dev_metrics = choose_threshold(
        y_dev, dev_scores, minimum_recall=float(heuristic_dev["recall"]))
    audit = {
        "development": {
            **dev_metrics,
            "average_precision": average_precision(y_dev, dev_scores),
            "components": len(development),
            "negative": int((y_dev == 0).sum()),
            "positive": int((y_dev == 1).sum()),
            "writers": len({str(record["group_uuid"]) for record in development}),
            "recall_guard": float(heuristic_dev["recall"]),
            "reference_heuristic": heuristic_dev,
        },
        "epochs": epoch + 1,
        "feature_names": list(FEATURE_NAMES),
        "training": {
            "components": len(training),
            "negative": int((y_train == 0).sum()),
            "positive": int((y_train == 1).sum()),
            "writers": len({str(record["group_uuid"]) for record in training}),
        },
        "writer_overlap": len(
            {str(record["group_uuid"]) for record in training}
            & {str(record["group_uuid"]) for record in development}),
    }
    return model, mean.astype(np.float32), scale.astype(np.float32), threshold, audit


def score_components(model: Any, mean: np.ndarray, scale: np.ndarray,
                     features: np.ndarray, torch: Any) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    tensor = torch.from_numpy(((features - mean) / scale).astype(np.float32))
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(tensor)).numpy().ravel().astype(np.float32)


def learned_filter(
    probabilities: np.ndarray,
    images: np.ndarray,
    model: Any,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    compare: Any,
    torch: Any,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    output = np.zeros_like(probabilities, dtype=bool)
    decisions: list[list[dict[str, Any]]] = []
    for index, (probability, image) in enumerate(zip(probabilities, images)):
        support = probability >= PROBABILITY_THRESHOLD
        labels, count = _label_components(support)
        base_features = compare.component_features(support, probability)[1]
        vectors = []
        for label in range(1, count + 1):
            component = labels == label
            heuristic = compare.heuristic_component_score(
                base_features[label - 1], PROBABILITY_THRESHOLD)
            vectors.append(component_feature_vector(
                component, probability, image, heuristic_score=heuristic))
        vectors_array = np.stack(vectors) if vectors else np.empty(
            (0, len(FEATURE_NAMES)), dtype=np.float32)
        scores = score_components(model, mean, scale, vectors_array, torch)
        row_decisions = []
        for label, score in enumerate(scores, start=1):
            keep = bool(score >= threshold)
            if keep:
                output[index][labels == label] = True
            row_decisions.append({
                "component": label, "kept": keep, "score": float(score)})
        decisions.append(row_decisions)
    return output, decisions


def row_gate_keep(line_ids: Sequence[str], oof_path: Path,
                  threshold: float = ROW_GATE_THRESHOLD):
    wanted = set(line_ids)
    scores: dict[str, float] = {}
    old_gold: dict[str, bool] = {}
    with oof_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            row = json.loads(line)
            line_id = str(row["line_id"])
            if line_id in wanted:
                scores[line_id] = float(row["probability"])
                old_gold[line_id] = bool(row["gold_has_edit"])
    keep = np.asarray([
        scores.get(line_id, 1.0) >= threshold for line_id in line_ids
    ], dtype=bool)
    return keep, scores, old_gold


def mask_metrics(compare: Any, predicted: np.ndarray, targets: np.ndarray,
                 positive_count: int) -> dict[str, Any]:
    result = compare.score_masks(
        predicted[:positive_count], predicted[positive_count:], targets)
    result["fragmentation"] = compare.fragmentation_metrics(
        predicted[:positive_count], targets)
    return result


def compact_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "boundary_f1": result["boundary_f1"],
        "clean_any_component_rate": result["clean_any_component_rate"],
        "clean_fp_components_per_line": result["clean_fp_components_per_line"],
        "component": result["component"],
        "fragmentation": result["fragmentation"],
        "pixel": result["pixel"],
    }


def _materialise_training(source_output: Path):
    gold300 = load_module(GOLD300_PATH, "hme380_for381", GOLD300_SHA256)
    wavelet = gold300.load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for381", ablation.BASE_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for381", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for381", ablation.FULL1000_SHA256)
    manifest = read_json(source_output / "run_manifest.json")
    command = manifest.get("command")
    if not isinstance(command, list) or len(command) < 3:
        raise Experiment381Error("source command is absent")
    args = gold300.build_parser(corrective, base, runner, full).parse_args(command[2:])
    if args.source_family != "current_best" or args.correction_set != "all300" \
            or args.seeds != "42":
        raise Experiment381Error("source is not current-best all300 seed42")
    full.configure_base(base)
    gold300.patch_prepare(base, args)
    prepared = base.prepare(args)
    recipe = base.RECIPES[args.recipe]
    images_dir = Path(args.crop_images_dir)
    fit_positive, clipped_positive = base.materialise(
        prepared["fit_positive"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    fit_clean, clipped_clean = base.materialise(
        prepared["fit_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    validation_positive, clipped_val_positive = base.materialise(
        prepared["validation_positive"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    validation_clean, clipped_val_clean = base.materialise(
        prepared["validation_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    return {
        "ablation": ablation, "args": args, "base": base,
        "clipped": clipped_positive + clipped_clean + clipped_val_positive
        + clipped_val_clean,
        "fit_clean": fit_clean, "fit_positive": fit_positive,
        "prepared": prepared,
        "validation_clean": validation_clean,
        "validation_positive": validation_positive,
    }


def _predict_training(rows: Sequence[dict[str, Any]], model: Any,
                      ablation: Any, batch_size: int, device: Any):
    probabilities = []
    targets = []
    for start in range(0, len(rows), 64):
        probability, target = ablation.predict(
            model, list(rows[start:start + 64]), batch_size, device)
        probabilities.append(probability.astype(np.float32))
        targets.append(target.astype(np.float32))
    return np.concatenate(probabilities), np.concatenate(targets)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != SEED or not math.isclose(
            args.probability_threshold, PROBABILITY_THRESHOLD):
        raise Experiment381Error("formal seed/segmentation threshold changed")
    source = args.source_output.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment381Error(f"create-only output exists: {output}")
    if not (source / "complete.json").is_file() \
            or not (source / "segmenter_seed42.pt").is_file():
        raise Experiment381Error("strongest source output is incomplete")
    source_summary = read_json(source / "summary.json")
    if source_summary.get("variant") != "current_best_all300":
        raise Experiment381Error("source output is not the strongest pixel model")
    compare = load_module(COMPARE_PATH, "hme372_for381", COMPARE_SHA256)
    materialised = _materialise_training(source)
    if materialised["clipped"] != int(
            read_json(source / "run_manifest.json")["clipped_coordinate_values"]):
        raise Experiment381Error("materialisation clipping audit changed")

    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise Experiment381Error("formal scorer extraction requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(
        source / "segmenter_seed42.pt", map_location="cpu", weights_only=True)
    model = materialised["ablation"].make_unetpp_resnet34(
        materialised["args"].resnet34_weights).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    fit_rows = materialised["fit_positive"] + materialised["fit_clean"]
    train_probability, _ = _predict_training(
        fit_rows, model, materialised["ablation"], args.batch_size, device)
    del model
    torch.cuda.empty_cache()
    records = component_training_records(
        fit_rows, train_probability, compare)
    scorer, mean, scale, scorer_threshold, scorer_audit = train_mlp(records, torch)

    archive = np.load(source / "validation_predictions_mean.npz")
    positive_probability = archive["positive_probabilities"].astype(np.float32)
    clean_probability = archive["clean_probabilities"].astype(np.float32)
    targets = archive["positive_targets"].astype(bool)
    probabilities = np.concatenate((positive_probability, clean_probability))
    positive_rows = sorted(
        materialised["validation_positive"], key=lambda row: row["line_id"])
    clean_rows = sorted(
        materialised["validation_clean"], key=lambda row: row["line_id"])
    validation_rows = positive_rows + clean_rows
    positive_key = json.loads(
        (source / "validation_key.json").read_text(encoding="utf-8"))
    if [row["line_id"] for row in positive_rows] \
            != [row["line_id"] for row in positive_key]:
        raise Experiment381Error("positive validation order differs from source arrays")
    images = np.stack([row["image"] for row in validation_rows])
    support, heuristic = compare.filter_components(
        probabilities, PROBABILITY_THRESHOLD)
    learned, decisions = learned_filter(
        probabilities, images, scorer, mean, scale, scorer_threshold,
        compare, torch)
    line_ids = [str(row["line_id"]) for row in validation_rows]
    row_keep, row_scores, old_gold = row_gate_keep(
        line_ids, args.row_oof_predictions.expanduser().resolve())
    variants = {
        "raw_probability_mask": support,
        "heuristic_component_baseline": heuristic,
        "heuristic_plus_oof_row_gate": heuristic & row_keep[:, None, None],
        "learned_component_scorer": learned,
        "learned_scorer_plus_oof_row_gate": learned & row_keep[:, None, None],
    }
    metrics = {
        name: compact_metrics(mask_metrics(
            compare, mask, targets, len(positive_rows)))
        for name, mask in variants.items()
    }
    validation_writers = {str(row["group_uuid"]) for row in validation_rows}
    fitting_writers = {str(row["group_uuid"]) for row in fit_rows}
    if validation_writers & fitting_writers:
        raise Experiment381Error("segmenter/scorer fitting writers overlap validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    torch.save({
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_scale": scale,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "schema_version": SCHEMA,
        "seed": SEED,
        "state_dict": scorer.state_dict(),
        "threshold": scorer_threshold,
    }, staging / "component_scorer.pt")
    write_json(staging / "metrics.json", metrics)
    row_audit = {
        "coverage": len(row_scores),
        "missing_conservatively_kept": len(line_ids) - len(row_scores),
        "old_gold_agreement": sum(
            old_gold.get(row["line_id"], bool(index < len(positive_rows)))
            == bool(index < len(positive_rows))
            for index, row in enumerate(validation_rows)
            if row["line_id"] in old_gold),
        "positive_kept": int(row_keep[:len(positive_rows)].sum()),
        "positive_rows": len(positive_rows),
        "row_gate_threshold": ROW_GATE_THRESHOLD,
        "scored_clean_kept": int(row_keep[len(positive_rows):].sum()),
        "scored_clean_rows": len(clean_rows),
    }
    summary = {
        "best_by_component_f1": max(
            metrics, key=lambda name: metrics[name]["component"]["f1"]),
        "data": {
            "fit_clean": len(materialised["fit_clean"]),
            "fit_positive": len(materialised["fit_positive"]),
            "validation_clean": len(clean_rows),
            "validation_positive": len(positive_rows),
            "writer_overlap_fit_validation": 0,
        },
        "metrics": metrics,
        "protocol": {
            "candidate_pixels_added": False,
            "component_match_iou": 0.10,
            "frozen_test_touched": False,
            "gold_masks_modified": False,
            "heuristic_component_threshold": HEURISTIC_THRESHOLD,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "row_gate": "writer-held-out OOF; missing rows conservatively kept",
            "scorer_threshold_selected_on": "deterministic fitting-writer holdout only",
            "seed": SEED,
        },
        "row_gate": row_audit,
        "schema_version": SCHEMA,
        "scorer": scorer_audit,
        "status": "complete",
    }
    write_json(staging / "summary.json", summary)
    with (staging / "validation_component_decisions.jsonl").open(
            "x", encoding="utf-8") as handle:
        for row, components, keep in zip(validation_rows, decisions, row_keep):
            handle.write(json.dumps({
                "components": components,
                "line_id": row["line_id"],
                "oof_row_gate_keep": bool(keep),
                "oof_row_probability": row_scores.get(row["line_id"]),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "command": [sys.executable, *sys.argv],
        "component_scorer_training_rows": len(fit_rows),
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frozen_test_touched": False,
        "gold_masks_modified": False,
        "network_used": False,
        "row_oof_predictions_sha256": digest_file(
            args.row_oof_predictions.expanduser().resolve()),
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "source_checkpoint_sha256": digest_file(source / "segmenter_seed42.pt"),
        "source_manifest_sha256": digest_file(source / "run_manifest.json"),
    }
    write_json(staging / "run_manifest.json", manifest)
    write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    files = sorted(path for path in staging.iterdir()
                   if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text("".join(
        f"{digest_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8")
    staging.rename(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--row-oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--probability-threshold", type=float,
                        default=PROBABILITY_THRESHOLD)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    started = time.monotonic()
    try:
        summary = run(build_parser().parse_args())
    except Experiment381Error as exc:
        raise SystemExit(f"experiment 381 refused: {exc}") from exc
    print(json.dumps({
        "best": summary["best_by_component_f1"],
        "elapsed_seconds": time.monotonic() - started,
        "metrics": summary["metrics"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

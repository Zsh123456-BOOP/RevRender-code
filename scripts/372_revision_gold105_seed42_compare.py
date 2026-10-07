#!/usr/bin/env python3
"""Compare the four seed-42 experiment-371 arms on one locked dev split.

This is a read-only, validation-only comparison.  It preserves the ordinary
pixel and connected-component metrics, applies the previously frozen 0.45
component score, and reports a separate fragmentation diagnostic.  No pixels
are bridged and no model or scorer is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = "hme-revision-gold105-ablation-dev-v1"
BASELINE_SCHEMA = "hme-revision-full1000-three-seed-ensemble-v1"
SCHEMA = "hme-revision-gold105-seed42-compare-v1"
VARIANTS = (
    "data_only", "unetpp_resnet34", "boundary_head", "line_connectivity",
)
REQUIRED_FILES = (
    "complete.json", "run_manifest.json", "summary.json",
    "validation_key.json", "validation_predictions_mean.npz",
)
COMPONENT_SCORE_THRESHOLD = 0.45
FIXED_PROBABILITY_THRESHOLD = 0.30
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
SCORE_WEIGHTS = {
    "mean_probability_margin": 0.35,
    "max_probability_margin": 0.20,
    "log_area": 0.20,
    "major_axis_length": 0.15,
    "aspect_ratio": 0.10,
}
AREA_REFERENCE_PX = 256.0
LENGTH_REFERENCE_PX = 64.0
ASPECT_REFERENCE = 8.0


class Compare372Error(RuntimeError):
    """Raised when an integrity, scope, or protocol invariant is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Compare372Error(f"invalid JSON: {path}") from exc


def verify_sums(root: Path) -> None:
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in entries or len(digest) != 64:
            raise Compare372Error(f"malformed SHA256SUMS: {root}")
        entries[name] = digest
    if not set(REQUIRED_FILES) <= set(entries):
        raise Compare372Error(f"incomplete SHA256SUMS: {root}")
    for name in REQUIRED_FILES:
        if digest_file(root / name) != entries[name]:
            raise Compare372Error(f"digest mismatch: {root / name}")


def load_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    verify_sums(root)
    complete = read_json(root / "complete.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    key = read_json(root / "validation_key.json")
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"}:
        raise Compare372Error(f"invalid completion marker: {root}")
    if summary.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("schema_version") != SOURCE_SCHEMA:
        raise Compare372Error(f"source schema mismatch: {root}")
    if manifest.get("m6_frozen_test_touched") is not False \
            or manifest.get("reference_evaluation_materialised") is not False \
            or manifest.get("evaluation_fold_trained_on") is not False:
        raise Compare372Error(f"source safety contract changed: {root}")
    if summary.get("data", {}).get("writer_overlap_fit_validation") != 0:
        raise Compare372Error(f"writer overlap in source: {root}")
    protocol = summary.get("protocol", {})
    if protocol.get("seeds") != [42]:
        raise Compare372Error(f"source is not seed 42: {root}")
    variant = summary.get("variant")
    if variant not in VARIANTS:
        raise Compare372Error(f"unexpected variant: {variant}")
    if not isinstance(key, list) or len(key) != 127:
        raise Compare372Error(f"validation key changed: {root}")
    with np.load(root / "validation_predictions_mean.npz") as archive:
        arrays = {
            "positive": archive["positive_probabilities"].astype(np.float32),
            "clean": archive["clean_probabilities"].astype(np.float32),
            "targets": archive["positive_targets"].astype(bool),
        }
    if arrays["positive"].shape != arrays["targets"].shape \
            or len(arrays["positive"]) != 127 \
            or len(arrays["clean"]) != 381:
        raise Compare372Error(f"validation arrays changed: {root}")
    return {
        "arrays": arrays,
        "key": key,
        "manifest": manifest,
        "root": root,
        "summary": summary,
        "variant": variant,
    }


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    count, labels = cv2.connectedComponents(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8)
    return labels.astype(np.int32, copy=False), int(count - 1)


def component_features(mask: np.ndarray, probabilities: np.ndarray) \
        -> tuple[np.ndarray, list[dict[str, float | int]]]:
    labels, count = label_components(mask)
    features = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        area = int(len(xs))
        points = np.column_stack((xs.astype(float), ys.astype(float)))
        if area <= 1:
            major, minor = 1.0, 1.0
        else:
            centred = points - points.mean(axis=0, keepdims=True)
            covariance = centred.T @ centred / max(1, area - 1)
            values, vectors = np.linalg.eigh(covariance)
            projections = centred @ vectors[:, np.argsort(values)[::-1]]
            major = float(np.ptp(projections[:, 0]) + 1.0)
            minor = float(np.ptp(projections[:, 1]) + 1.0)
        major, minor = max(1.0, major), max(1.0, minor)
        values = probabilities[labels == label]
        features.append({
            "area": area,
            "aspect_ratio": major / minor,
            "label": label,
            "major_axis_length": major,
            "max_probability": float(values.max()),
            "mean_probability": float(values.mean()),
        })
    return labels, features


def heuristic_component_score(feature: dict[str, Any],
                              probability_threshold: float) -> float:
    scale = max(1e-8, 1.0 - probability_threshold)
    mean_margin = float(np.clip(
        (feature["mean_probability"] - probability_threshold) / scale, 0, 1))
    max_margin = float(np.clip(
        (feature["max_probability"] - probability_threshold) / scale, 0, 1))
    area = min(1.0, math.log1p(feature["area"])
               / math.log1p(AREA_REFERENCE_PX))
    length = min(1.0, feature["major_axis_length"] / LENGTH_REFERENCE_PX)
    aspect = min(1.0, max(0.0, feature["aspect_ratio"] - 1.0)
                 / (ASPECT_REFERENCE - 1.0))
    return (
        SCORE_WEIGHTS["mean_probability_margin"] * mean_margin
        + SCORE_WEIGHTS["max_probability_margin"] * max_margin
        + SCORE_WEIGHTS["log_area"] * area
        + SCORE_WEIGHTS["major_axis_length"] * length
        + SCORE_WEIGHTS["aspect_ratio"] * aspect
    )


def score_and_filter_components(mask: np.ndarray, probabilities: np.ndarray,
                                probability_threshold: float) -> np.ndarray:
    labels, features = component_features(mask, probabilities)
    output = np.zeros_like(mask, dtype=bool)
    for feature in features:
        if heuristic_component_score(feature, probability_threshold) \
                >= COMPONENT_SCORE_THRESHOLD:
            output[labels == feature["label"]] = True
    return output


def pixel_metrics(predicted: np.ndarray,
                  targets: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=bool)
    truth = np.asarray(targets, dtype=bool)
    intersection = (prediction & truth).reshape(len(prediction), -1).sum(axis=1)
    union = (prediction | truth).reshape(len(prediction), -1).sum(axis=1)
    predicted_sum = prediction.reshape(len(prediction), -1).sum(axis=1)
    truth_sum = truth.reshape(len(prediction), -1).sum(axis=1)
    micro_intersection = int(intersection.sum())
    micro_union = int(union.sum())
    micro_prediction = int(predicted_sum.sum())
    micro_truth = int(truth_sum.sum())
    per_iou = np.divide(intersection, union, out=np.ones_like(
        intersection, dtype=float), where=union > 0)
    per_dice = np.divide(
        2 * intersection, predicted_sum + truth_sum,
        out=np.ones_like(intersection, dtype=float),
        where=(predicted_sum + truth_sum) > 0)
    return {
        "dice": 2 * micro_intersection / (micro_prediction + micro_truth)
            if micro_prediction + micro_truth else 1.0,
        "iou": micro_intersection / micro_union if micro_union else 1.0,
        "macro_dice": float(per_dice.mean()) if len(per_dice) else 1.0,
        "macro_iou": float(per_iou.mean()) if len(per_iou) else 1.0,
        "precision": micro_intersection / micro_prediction
            if micro_prediction else 1.0,
        "recall": micro_intersection / micro_truth if micro_truth else 1.0,
    }


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy <= radius * radius).astype(np.uint8)


def boundary_f1(predicted: np.ndarray, targets: np.ndarray,
                tolerance: int = 5) -> float:
    matched_prediction = matched_truth = prediction_total = truth_total = 0
    for prediction, truth in zip(predicted, targets):
        prediction_u8 = np.ascontiguousarray(prediction, dtype=np.uint8)
        truth_u8 = np.ascontiguousarray(truth, dtype=np.uint8)
        prediction_boundary = prediction & ~cv2.erode(
            prediction_u8, CONNECTIVITY, borderType=cv2.BORDER_CONSTANT).astype(bool)
        truth_boundary = truth & ~cv2.erode(
            truth_u8, CONNECTIVITY, borderType=cv2.BORDER_CONSTANT).astype(bool)
        prediction_total += int(prediction_boundary.sum())
        truth_total += int(truth_boundary.sum())
        kernel = disk(tolerance)
        dilated_truth = cv2.dilate(
            truth_boundary.astype(np.uint8), kernel).astype(bool)
        dilated_prediction = cv2.dilate(
            prediction_boundary.astype(np.uint8), kernel).astype(bool)
        matched_prediction += int((prediction_boundary & dilated_truth).sum())
        matched_truth += int((truth_boundary & dilated_prediction).sum())
    precision = matched_prediction / prediction_total if prediction_total else 1.0
    recall = matched_truth / truth_total if truth_total else 1.0
    return 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0


def match_components(predicted: np.ndarray, truth: np.ndarray,
                     match_iou: float = 0.10) -> dict[str, Any]:
    predicted_labels, predicted_count = label_components(predicted)
    truth_labels, truth_count = label_components(truth)
    predicted_areas = np.bincount(
        predicted_labels.ravel(), minlength=predicted_count + 1)[1:]
    truth_areas = np.bincount(
        truth_labels.ravel(), minlength=truth_count + 1)[1:]
    adjacency = [[] for _ in range(predicted_count)]
    ious: dict[tuple[int, int], float] = {}
    if predicted_count and truth_count:
        joint = predicted_labels.astype(np.int64) * (truth_count + 1) \
            + truth_labels.astype(np.int64)
        overlaps = np.bincount(joint.ravel(), minlength=(predicted_count + 1)
                               * (truth_count + 1)).reshape(
                                   predicted_count + 1, truth_count + 1)
        pred_indices, truth_indices = np.nonzero(overlaps[1:, 1:])
        for pred_index, truth_index in zip(
                pred_indices.tolist(), truth_indices.tolist()):
            intersection = int(overlaps[pred_index + 1, truth_index + 1])
            union = int(predicted_areas[pred_index]
                        + truth_areas[truth_index] - intersection)
            iou = intersection / union
            if iou >= match_iou:
                adjacency[pred_index].append(truth_index)
                ious[(pred_index, truth_index)] = iou
    for pred_index in range(predicted_count):
        adjacency[pred_index].sort(key=lambda index: (-ious[(pred_index, index)], index))
    matched_truth: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for truth_index in adjacency[pred_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = matched_truth.get(truth_index)
            if previous is None or augment(previous, visited):
                matched_truth[truth_index] = pred_index
                return True
        return False

    for pred_index in range(predicted_count):
        augment(pred_index, set())
    matches = [(pred_index, truth_index)
               for truth_index, pred_index in matched_truth.items()]
    return {
        "matches": matches,
        "matched_ious": [ious[pair] for pair in matches],
        "predicted_count": predicted_count,
        "truth_areas": truth_areas.astype(int).tolist(),
        "truth_count": truth_count,
    }


def component_metrics(predicted: np.ndarray, targets: np.ndarray, *,
                      match_iou: float = 0.10,
                      small_gold_max_area: int = 128) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    predicted_total = truth_total = small_total = small_matched = 0
    matched_ious: list[float] = []
    for prediction, truth in zip(predicted, targets):
        matched = match_components(prediction, truth, match_iou)
        matched_truth = {truth_index for _, truth_index in matched["matches"]}
        true_positive += len(matched["matches"])
        false_positive += matched["predicted_count"] - len(matched["matches"])
        false_negative += matched["truth_count"] - len(matched["matches"])
        predicted_total += matched["predicted_count"]
        truth_total += matched["truth_count"]
        matched_ious.extend(matched["matched_ious"])
        for truth_index, area in enumerate(matched["truth_areas"]):
            if area <= small_gold_max_area:
                small_total += 1
                small_matched += int(truth_index in matched_truth)
    precision = true_positive / predicted_total if predicted_total else 1.0
    recall = true_positive / truth_total if truth_total else 1.0
    return {
        "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "fp_components_per_line": false_positive / len(predicted)
            if len(predicted) else 0.0,
        "match_iou": match_iou,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "precision": precision,
        "predicted_components": predicted_total,
        "recall": recall,
        "small_gold_absolute_recall": small_matched / small_total
            if small_total else None,
        "small_gold_components": small_total,
        "small_gold_matched": small_matched,
        "small_gold_max_area": small_gold_max_area,
        "true_components": truth_total,
        "true_positive": true_positive,
    }


def score_masks(positive_masks: np.ndarray, clean_masks: np.ndarray,
                targets: np.ndarray) -> dict[str, Any]:
    predictions = np.concatenate((positive_masks, clean_masks))
    truths = np.concatenate((targets, np.zeros_like(clean_masks, dtype=bool)))
    clean_counts = [label_components(mask)[1] for mask in clean_masks]
    return {
        "boundary_f1": boundary_f1(positive_masks, targets),
        "clean_any_component_rate": sum(value > 0 for value in clean_counts)
            / len(clean_counts),
        "clean_fp_components_per_line": float(np.mean(clean_counts)),
        "component": component_metrics(predictions, truths),
        "pixel": pixel_metrics(positive_masks, targets),
    }


def filter_components(probabilities: np.ndarray,
                      probability_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    support = probabilities >= probability_threshold
    filtered = []
    for mask, probability in zip(support, probabilities):
        filtered.append(score_and_filter_components(
            mask, probability, probability_threshold))
    return support, np.stack(filtered)


def fragmentation_metrics(predicted: np.ndarray,
                          targets: np.ndarray) -> dict[str, Any]:
    overlap_counts: list[int] = []
    hit_counts: list[int] = []
    for prediction, truth in zip(predicted, targets):
        pred_labels, pred_count = label_components(prediction)
        truth_labels, truth_count = label_components(truth)
        if not truth_count:
            continue
        joint = truth_labels.astype(np.int64) * (pred_count + 1) \
            + pred_labels.astype(np.int64)
        overlaps = np.bincount(
            joint.ravel(), minlength=(truth_count + 1) * (pred_count + 1),
        ).reshape(truth_count + 1, pred_count + 1)
        for truth_index in range(1, truth_count + 1):
            count = int(np.count_nonzero(overlaps[truth_index, 1:]))
            overlap_counts.append(count)
            if count:
                hit_counts.append(count)
    split = sum(value >= 2 for value in overlap_counts)
    return {
        "hit_true_components": len(hit_counts),
        "mean_overlapping_prediction_components_per_hit":
            float(np.mean(hit_counts)) if hit_counts else 0.0,
        "maximum_overlapping_prediction_components": max(hit_counts, default=0),
        "split_true_components": split,
        "split_true_component_rate": split / len(overlap_counts)
            if overlap_counts else 0.0,
        "true_components": len(overlap_counts),
    }


def score_at(bundle: dict[str, Any], threshold: float) -> dict[str, Any]:
    arrays = bundle["arrays"]
    probabilities = np.concatenate((arrays["positive"], arrays["clean"]))
    support, filtered = filter_components(probabilities, threshold)
    positive_count = len(arrays["positive"])
    standard = score_masks(
        filtered[:positive_count], filtered[positive_count:], arrays["targets"])
    standard["fragmentation"] = fragmentation_metrics(
        filtered[:positive_count], arrays["targets"])
    type_slices = {}
    for mark_type in ("line", "blot", "overwrite"):
        indices = [index for index, row in enumerate(bundle["key"])
                   if mark_type in row.get("types", [])]
        if not indices:
            continue
        selected = np.asarray(indices, dtype=np.int64)
        subset_predictions = filtered[selected]
        subset_targets = arrays["targets"][selected]
        type_slices[mark_type] = {
            "positive_rows": len(indices),
            "component": component_metrics(subset_predictions, subset_targets),
            "fragmentation": fragmentation_metrics(
                subset_predictions, subset_targets),
            "pixel": pixel_metrics(subset_predictions, subset_targets),
        }
    standard["type_slices_nonexclusive"] = type_slices
    return standard


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundles = [load_source(path) for path in args.input_dirs]
    variants = [bundle["variant"] for bundle in bundles]
    if sorted(variants) != sorted(VARIANTS) or len(set(variants)) != 4:
        raise Compare372Error("exactly one source per predeclared variant is required")
    key_digests = {digest_file(bundle["root"] / "validation_key.json")
                   for bundle in bundles}
    split_digests = {bundle["summary"]["protocol"]["split_manifest_sha256"]
                     for bundle in bundles}
    targets = bundles[0]["arrays"]["targets"]
    if len(key_digests) != 1 or len(split_digests) != 1 \
            or any(not np.array_equal(bundle["arrays"]["targets"], targets)
                   for bundle in bundles[1:]):
        raise Compare372Error("sources do not share one identical validation split")

    rows = {}
    for bundle in bundles:
        summary = bundle["summary"]
        threshold = float(summary["ensemble"]["operating_points"]
                          ["localization"]["threshold"])
        rows[bundle["variant"]] = {
            "fit_clean": summary["data"]["total_fit_clean"],
            "fit_positive": summary["data"]["fit_positive"],
            "fixed_probability_0.30_component_score_0.45": score_at(
                bundle, FIXED_PROBABILITY_THRESHOLD),
            "raw_selected_checkpoint": summary["ensemble"],
            "selected_probability_threshold": threshold,
            "selected_probability_component_score_0.45": score_at(
                bundle, threshold),
            "source": str(bundle["root"]),
            "source_manifest_sha256": digest_file(
                bundle["root"] / "run_manifest.json"),
        }

    baseline_root = args.baseline_dir.resolve()
    baseline = read_json(baseline_root / "results.json")
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        raise Compare372Error("baseline ensemble schema mismatch")
    result = {
        "arms": rows,
        "baseline_three_seed_reference_not_seed_matched": {
            "note": "context only; three-seed ensemble is not a seed-42 arm",
            "results": baseline,
            "source": str(baseline_root),
        },
        "protocol": {
            "component_score_threshold": COMPONENT_SCORE_THRESHOLD,
            "fixed_probability_threshold": FIXED_PROBABILITY_THRESHOLD,
            "fragmentation_adds_or_deletes_pixels": False,
            "m6_frozen_or_reference_read": False,
            "model_or_scorer_fitted": False,
            "seed": 42,
            "split_manifest_sha256": next(iter(split_digests)),
            "validation_key_sha256": next(iter(key_digests)),
        },
        "schema_version": SCHEMA,
    }
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise Compare372Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        (staging / "results.json").write_text(json.dumps(
            result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        manifest = {
            "create_only": True,
            "held_reference_m6_frozen_read": False,
            "network_used": False,
            "platform": platform.platform(),
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "training_performed": False,
            "validation_only": True,
        }
        (staging / "run_manifest.json").write_text(json.dumps(
            manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "complete.json").write_text(json.dumps({
            "schema_version": SCHEMA, "status": "complete"},
            sort_keys=True) + "\n", encoding="utf-8")
        sums = [f"{digest_file(path)}  {path.name}"
                for path in sorted(staging.iterdir())]
        (staging / "SHA256SUMS").write_text(
            "\n".join(sums) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dirs", nargs=4, type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    try:
        result = run(build_parser().parse_args())
    except Compare372Error as exc:
        raise SystemExit(f"experiment 372 refused: {exc}") from exc
    compact = {}
    for variant, arm in result["arms"].items():
        metrics = arm["selected_probability_component_score_0.45"]
        compact[variant] = {
            "component_f1": metrics["component"]["f1"],
            "fp_components_per_line": metrics["component"]["fp_components_per_line"],
            "pixel_iou": metrics["pixel"]["iou"],
            "split_true_component_rate":
                metrics["fragmentation"]["split_true_component_rate"],
        }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

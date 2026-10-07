#!/usr/bin/env python3
"""Train coverage-preserving component scorers for experiment 389 outputs.

The scorer only removes connected components from the frozen pixel mask.  It
never adds pixels or changes gold masks.  Three scorer thresholds are selected
on a fitting-writer holdout (F1, >=90% recall, and >=95% recall).  The existing
writer-OOF row gate is evaluated as a frozen optional post-filter, not trained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COVERAGE_PATH = ROOT / "scripts/389_revision_coverage_finetune_seed42.py"
COVERAGE_SHA256 = "fa507c04c750d20619505265b962a232baa16c9517c64e7026a57402806d38ef"
COMPONENT_PATH = ROOT / "scripts/381_revision_component_scorer_oof_gate.py"
COMPONENT_SHA256 = "956b21c7f553d481ce6289fa1f22938e1302c0abe1c9c61a0aa0a466b28a76ea"
SCHEMA = "hme-revision-coverage-component-scorer-dev-v1"
LABEL_POLICIES = {
    "iou10": 0.10,
    "iou05": 0.05,
    "iou01": 0.01,
    "any_overlap": 0.0,
}


class Experiment390Error(RuntimeError):
    """Raised when a scorer provenance or safety invariant changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str):
    actual = digest_file(path)
    if actual != expected_sha256:
        raise Experiment390Error(
            f"dependency digest mismatch for {path.name}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Experiment390Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment390Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment390Error(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_sha256s(root: Path) -> None:
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in entries or len(digest) != 64:
            raise Experiment390Error("source SHA256SUMS is malformed")
        entries[name] = digest
    for name, digest in entries.items():
        if digest_file(root / name) != digest:
            raise Experiment390Error(f"source checksum mismatch: {name}")


def materialise_training(coverage, source_output: Path):
    active = coverage.load_active()
    gold300 = active.load_gold300()
    wavelet = gold300.load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for390", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for390", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for390", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for390", ablation.FULL1000_SHA256)
    manifest = read_json(source_output / "run_manifest.json")
    command = manifest.get("command")
    if not isinstance(command, list) or len(command) < 3:
        raise Experiment390Error("source command is absent")
    args = coverage.build_parser(
        active, gold300, corrective, base, runner, full).parse_args(command[2:])
    coverage.validate_args(args)
    full.configure_base(base)
    coverage.patch_prepare_locked_validation(active, gold300, base, args)
    prepared = base.prepare(args)
    recipe = base.RECIPES[args.recipe]
    images_dir = Path(args.crop_images_dir)

    def materialise(rows):
        return base.materialise(
            rows, prepared["segmenter"], prepared["base"],
            prepared["catalogue"], images_dir, recipe["crop"],
            prepared["fold_lookup"])

    fit_positive, clipped_positive = materialise(prepared["fit_positive"])
    fit_clean, clipped_clean = materialise(prepared["fit_clean"])
    validation_positive, clipped_val_positive = materialise(
        prepared["validation_positive"])
    validation_clean, clipped_val_clean = materialise(
        prepared["validation_clean"])
    return {
        "ablation": ablation,
        "args": args,
        "clipped": clipped_positive + clipped_clean + clipped_val_positive
                   + clipped_val_clean,
        "fit_clean": fit_clean,
        "fit_positive": fit_positive,
        "validation_clean": validation_clean,
        "validation_positive": validation_positive,
    }


def predict_training(rows: Sequence[dict[str, Any]], model: Any,
                     ablation: Any, batch_size: int, device: Any):
    probabilities = []
    for start in range(0, len(rows), 64):
        probability, _ = ablation.predict(
            model, list(rows[start:start + 64]), batch_size, device)
        probabilities.append(probability.astype(np.float32))
    return np.concatenate(probabilities)


def scorer_thresholds(component, records: Sequence[Mapping[str, Any]],
                      model: Any, mean: np.ndarray, scale: np.ndarray,
                      f1_threshold: float, torch: Any) -> dict[str, Any]:
    development = [record for record in records
                   if component.stable_bucket(str(record["group_uuid"])) == 0]
    features = np.stack([record["features"] for record in development])
    labels = np.asarray([record["label"] for record in development],
                        dtype=np.float32)
    scores = component.score_components(model, mean, scale, features, torch)
    output: dict[str, Any] = {
        "f1": {
            **component.threshold_metrics(labels, scores, f1_threshold),
            "threshold": float(f1_threshold),
        }
    }
    for name, minimum_recall in (("recall90", 0.90), ("recall95", 0.95)):
        threshold, metrics = component.choose_threshold(
            labels, scores, minimum_recall=minimum_recall)
        output[name] = {**metrics, "minimum_recall": minimum_recall,
                        "threshold": float(threshold)}
    return output


def apply_label_policy(records: Sequence[Mapping[str, Any]],
                       policy: str) -> list[dict[str, Any]]:
    if policy not in LABEL_POLICIES:
        raise Experiment390Error("unknown component label policy")
    threshold = LABEL_POLICIES[policy]
    output = []
    for record in records:
        matched_iou = float(record["matched_iou"])
        positive = matched_iou > 0.0 if policy == "any_overlap" \
            else matched_iou >= threshold
        output.append({**record, "label": float(positive)})
    if len({record["label"] for record in output}) != 2:
        raise Experiment390Error("component label policy lacks two classes")
    return output


def fixed_mask_coverage(predicted: np.ndarray, targets: np.ndarray,
                        positive_count: int,
                        expected_gold_components: int | None = 237) \
        -> dict[str, Any]:
    import cv2

    predicted = np.asarray(predicted, dtype=bool)
    targets = np.asarray(targets, dtype=bool)
    positive = predicted[:positive_count]
    clean = predicted[positive_count:]
    if positive.shape != targets.shape:
        raise Experiment390Error("coverage mask/target shapes differ")
    coverages: list[float] = []
    for mask, target in zip(positive, targets):
        count, labels = cv2.connectedComponents(
            np.ascontiguousarray(target, dtype=np.uint8), connectivity=8)
        for label in range(1, count):
            component = labels == label
            coverages.append(float(np.mean(mask[component])))
    if expected_gold_components is not None \
            and len(coverages) != expected_gold_components:
        raise Experiment390Error("gold component count changed")
    values = np.asarray(coverages, dtype=float)
    true_positive = int((positive & targets).sum())
    false_positive = int((positive & ~targets).sum()) + int(clean.sum())
    truth = int(targets.sum())
    predicted_pixels = true_positive + false_positive
    return {
        "clean_false_alarm": float(np.mean(clean.reshape(len(clean), -1).any(1))),
        "covered_ge_10_count": int(np.sum(values >= 0.10)),
        "covered_ge_10_rate": float(np.mean(values >= 0.10)),
        "covered_ge_50_count": int(np.sum(values >= 0.50)),
        "covered_ge_50_rate": float(np.mean(values >= 0.50)),
        "gold_components": len(coverages),
        "mean_gold_component_coverage": float(np.mean(values)),
        "median_gold_component_coverage": float(np.median(values)),
        "pixel_iou": true_positive / max(1, truth + false_positive),
        "pixel_precision": true_positive / max(1, predicted_pixels),
        "pixel_recall": true_positive / max(1, truth),
    }


def mask_from_decisions(probabilities: np.ndarray,
                        decisions: Sequence[Sequence[Mapping[str, Any]]],
                        threshold: float) -> np.ndarray:
    import cv2

    output = np.zeros_like(probabilities, dtype=bool)
    for index, (probability, row_decisions) in enumerate(
            zip(probabilities, decisions)):
        # The caller passes already thresholded support as float/bool arrays.
        support = np.asarray(probability, dtype=bool)
        labels_count, labels = cv2.connectedComponents(
            np.ascontiguousarray(support, dtype=np.uint8), connectivity=8)
        if labels_count - 1 != len(row_decisions):
            raise Experiment390Error("component decision count changed")
        for decision in row_decisions:
            if float(decision["score"]) >= threshold:
                output[index][labels == int(decision["component"])] = True
    return output


def coverage_guard_threshold(component, records: Sequence[Mapping[str, Any]],
                             fit_rows: Sequence[dict[str, Any]],
                             train_probability: np.ndarray, model: Any,
                             mean: np.ndarray, scale: np.ndarray,
                             compare: Any, torch: Any,
                             probability_threshold: float,
                             positive_count: int) \
        -> tuple[float, dict[str, Any]]:
    if not 0 < positive_count < len(fit_rows):
        raise Experiment390Error("coverage-guard positive count is invalid")
    positive_pairs = [
        (row, probability) for row, probability in
        zip(fit_rows[:positive_count], train_probability[:positive_count])
        if component.stable_bucket(str(row["group_uuid"])) == 0
    ]
    clean_pairs = [
        (row, probability) for row, probability in
        zip(fit_rows[positive_count:], train_probability[positive_count:])
        if component.stable_bucket(str(row["group_uuid"])) == 0
    ]
    pairs = positive_pairs + clean_pairs
    rows = [pair[0] for pair in pairs]
    probabilities = np.stack([pair[1] for pair in pairs])
    targets = np.stack([
        np.asarray(row["mask"], dtype=bool) for row, _ in positive_pairs])
    images = np.stack([row["image"] for row in rows])
    support = probabilities >= probability_threshold
    _all_kept, decisions = component.learned_filter(
        probabilities, images, model, mean, scale, -1.0, compare, torch)
    scores = np.asarray([
        float(decision["score"]) for row in decisions for decision in row],
        dtype=np.float32)
    labels = np.asarray([
        record["label"] for record in records
        if component.stable_bucket(str(record["group_uuid"])) == 0
    ], dtype=np.float32)
    if len(scores) != len(labels):
        raise Experiment390Error("coverage-guard scorer records differ")
    raw = fixed_mask_coverage(
        support, targets, len(positive_pairs), expected_gold_components=None)
    candidates = np.unique(np.concatenate((
        np.linspace(0.0, 1.0, 201, dtype=np.float32), scores)))
    best = None
    for threshold in candidates:
        mask = mask_from_decisions(
            support.astype(np.float32), decisions, float(threshold))
        coverage = fixed_mask_coverage(
            mask, targets, len(positive_pairs), expected_gold_components=None)
        if coverage["covered_ge_50_rate"] + 1e-12 \
                < raw["covered_ge_50_rate"] - 0.005 \
                or coverage["covered_ge_10_rate"] + 1e-12 \
                < raw["covered_ge_10_rate"] - 0.005:
            continue
        classification = component.threshold_metrics(
            labels, scores, float(threshold))
        rank = (
            float(classification["f1"]),
            float(classification["precision"]),
            float(classification["recall"]),
            float(threshold),
        )
        if best is None or rank > best[0]:
            best = rank, float(threshold), coverage, classification
    if best is None:
        raise Experiment390Error("no component threshold preserves fit coverage")
    return best[1], {
        "coverage": best[2],
        "coverage_guard_absolute": 0.005,
        "development_components": len(labels),
        "development_rows": len(rows),
        "raw_coverage": raw,
        **best[3],
        "threshold": best[1],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    coverage = load_module(COVERAGE_PATH, "hme389_for390", COVERAGE_SHA256)
    component = load_module(COMPONENT_PATH, "hme381_for390", COMPONENT_SHA256)
    if args.seed != 42:
        raise Experiment390Error("formal scorer seed changed")
    source = args.source_output.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment390Error(f"create-only output exists: {output}")
    if not (source / "complete.json").is_file() \
            or not (source / "segmenter_seed42.pt").is_file():
        raise Experiment390Error("coverage source output is incomplete")
    verify_sha256s(source)
    source_summary = read_json(source / "summary.json")
    source_manifest = read_json(source / "run_manifest.json")
    if source_summary.get("schema_version") != coverage.SCHEMA \
            or source_summary.get("variant") not in coverage.VARIANTS \
            or source_manifest.get("m6_frozen_test_touched") is not False \
            or source_manifest.get("reference_evaluation_materialised") is not False:
        raise Experiment390Error("coverage source safety contract changed")
    probability_threshold = float(source_summary["coverage"]["threshold"])
    if not 0.0 < probability_threshold < 1.0:
        raise Experiment390Error("coverage probability threshold is invalid")
    component.PROBABILITY_THRESHOLD = probability_threshold
    compare = component.load_module(
        component.COMPARE_PATH, "hme372_for390", component.COMPARE_SHA256)
    materialised = materialise_training(coverage, source)
    if materialised["clipped"] != int(
            source_manifest["clipped_coordinate_values"]):
        raise Experiment390Error("materialisation clipping audit changed")

    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise Experiment390Error("formal scorer extraction requires CUDA")
    device = torch.device("cuda")
    try:
        checkpoint = torch.load(
            source / "segmenter_seed42.pt", map_location="cpu",
            weights_only=True)
    except TypeError:
        checkpoint = torch.load(
            source / "segmenter_seed42.pt", map_location="cpu")
    model = materialised["ablation"].make_unetpp_resnet34(
        materialised["args"].resnet34_weights).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    fit_rows = materialised["fit_positive"] + materialised["fit_clean"]
    train_probability = predict_training(
        fit_rows, model, materialised["ablation"], args.batch_size, device)
    del model
    torch.cuda.empty_cache()
    records = apply_label_policy(component.component_training_records(
        fit_rows, train_probability, compare), args.label_policy)
    scorer, mean, scale, f1_threshold, scorer_audit = \
        component.train_mlp(records, torch)
    thresholds = scorer_thresholds(
        component, records, scorer, mean, scale, f1_threshold, torch)
    guarded_threshold, guarded_audit = coverage_guard_threshold(
        component, records, fit_rows, train_probability, scorer, mean, scale,
        compare, torch, probability_threshold,
        len(materialised["fit_positive"]))
    thresholds["coverage_guard"] = guarded_audit

    archive = np.load(source / "validation_predictions_seed42.npz")
    positive_probability = archive["positive_probabilities"].astype(np.float32)
    clean_probability = archive["clean_probabilities"].astype(np.float32)
    targets = archive["targets"].astype(bool)
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
        raise Experiment390Error("positive validation order differs from arrays")
    images = np.stack([row["image"] for row in validation_rows])
    support, heuristic = compare.filter_components(
        probabilities, probability_threshold)
    masks = {
        "raw_probability_mask": support,
        "heuristic_component_baseline": heuristic,
    }
    decisions: dict[str, list[list[dict[str, Any]]]] = {}
    for name, threshold_audit in thresholds.items():
        mask, threshold_decisions = component.learned_filter(
            probabilities, images, scorer, mean, scale,
            float(threshold_audit["threshold"]), compare, torch)
        masks[f"learned_{name}"] = mask
        decisions[name] = threshold_decisions
    line_ids = [str(row["line_id"]) for row in validation_rows]
    row_keep, row_scores, old_gold = component.row_gate_keep(
        line_ids, args.row_oof_predictions.expanduser().resolve())
    for name in tuple(masks):
        if name.startswith("learned_"):
            masks[f"{name}_plus_frozen_row_gate"] = \
                masks[name] & row_keep[:, None, None]
    metrics = {}
    for name, mask in masks.items():
        compact = component.compact_metrics(component.mask_metrics(
            compare, mask, targets, len(positive_rows)))
        compact["coverage"] = fixed_mask_coverage(
            mask, targets, len(positive_rows))
        metrics[name] = compact
    raw_iou = metrics["raw_probability_mask"]["coverage"]["pixel_iou"]
    if not math.isclose(
            raw_iou, float(source_summary["coverage"]["pixel_iou"]),
            rel_tol=0.0, abs_tol=2e-4):
        raise Experiment390Error("raw coverage differs from source")
    validation_writers = {str(row["group_uuid"]) for row in validation_rows}
    fitting_writers = {str(row["group_uuid"]) for row in fit_rows}
    if validation_writers & fitting_writers:
        raise Experiment390Error("scorer fitting writers overlap validation")
    no_gate_names = [name for name in metrics
                     if "frozen_row_gate" not in name]
    best_component = max(
        no_gate_names, key=lambda name: metrics[name]["component"]["f1"])
    raw_ge50 = metrics["raw_probability_mask"]["coverage"][
        "covered_ge_50_rate"]
    preserving = [name for name in no_gate_names
                  if metrics[name]["coverage"]["covered_ge_50_rate"]
                  >= raw_ge50 - 0.02]
    best_preserving = max(
        preserving,
        key=lambda name: (
            metrics[name]["component"]["f1"],
            metrics[name]["component"]["precision"],
            -metrics[name]["clean_fp_components_per_line"],
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        torch.save({
            "feature_names": component.FEATURE_NAMES,
            "feature_mean": mean,
            "feature_scale": scale,
            "probability_threshold": probability_threshold,
            "label_policy": args.label_policy,
            "schema_version": SCHEMA,
            "seed": 42,
            "state_dict": scorer.state_dict(),
            "thresholds": {name: value["threshold"]
                           for name, value in thresholds.items()},
        }, staging / "component_scorer.pt")
        write_json(staging / "metrics.json", metrics)
        summary = {
            "best_by_component_f1_without_row_gate": best_component,
            "best_coverage_preserving_without_row_gate": best_preserving,
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
                "label_policy": args.label_policy,
                "label_positive_definition": "matched_iou > 0"
                    if args.label_policy == "any_overlap"
                    else f"matched_iou >= {LABEL_POLICIES[args.label_policy]}",
                "probability_threshold": probability_threshold,
                "row_gate_trained": False,
                "row_gate_usage": "optional frozen OOF comparison only",
                "scorer_threshold_selected_on":
                    "deterministic fitting-writer holdout only",
                "seed": 42,
                "source_variant": source_summary["variant"],
            },
            "row_gate": {
                "coverage": len(row_scores),
                "missing_conservatively_kept": len(line_ids) - len(row_scores),
                "old_gold_agreement": sum(
                    old_gold.get(row["line_id"], bool(index < len(positive_rows)))
                    == bool(index < len(positive_rows))
                    for index, row in enumerate(validation_rows)
                    if row["line_id"] in old_gold),
                "positive_kept": int(row_keep[:len(positive_rows)].sum()),
                "positive_rows": len(positive_rows),
                "scored_clean_kept": int(row_keep[len(positive_rows):].sum()),
                "scored_clean_rows": len(clean_rows),
            },
            "schema_version": SCHEMA,
            "scorer": scorer_audit,
            "status": "complete",
            "thresholds": thresholds,
        }
        write_json(staging / "summary.json", summary)
        with (staging / "validation_component_decisions.jsonl").open(
                "x", encoding="utf-8") as handle:
            for index, row in enumerate(validation_rows):
                handle.write(json.dumps({
                    "line_id": row["line_id"],
                    "oof_row_gate_keep": bool(row_keep[index]),
                    "oof_row_probability": row_scores.get(row["line_id"]),
                    "threshold_decisions": {
                        name: value[index] for name, value in decisions.items()},
                }, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "command": [sys.executable, *sys.argv],
            "component_scorer_training_rows": len(fit_rows),
            "create_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "frozen_test_touched": False,
            "gold_masks_modified": False,
            "label_policy": args.label_policy,
            "network_used": False,
            "row_gate_trained": False,
            "row_oof_predictions_sha256": digest_file(
                args.row_oof_predictions.expanduser().resolve()),
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "source_checkpoint_sha256": digest_file(
                source / "segmenter_seed42.pt"),
            "source_manifest_sha256": digest_file(source / "run_manifest.json"),
            "source_variant": source_summary["variant"],
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
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--row-oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--label-policy", choices=tuple(LABEL_POLICIES),
                        default="iou10")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    started = time.monotonic()
    try:
        summary = run(build_parser().parse_args())
    except Experiment390Error as exc:
        raise SystemExit(f"experiment 390 refused: {exc}") from exc
    print(json.dumps({
        "best_component": summary["best_by_component_f1_without_row_gate"],
        "best_coverage_preserving":
            summary["best_coverage_preserving_without_row_gate"],
        "elapsed_seconds": time.monotonic() - started,
        "label_policy": summary["protocol"]["label_policy"],
        "metrics": summary["metrics"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

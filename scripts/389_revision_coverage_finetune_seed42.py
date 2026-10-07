#!/usr/bin/env python3
"""Coverage-first fine-tuning of the 1000-line revision segmenter.

Four create-only development arms share the exact 1000-line seed-42 source,
the later corrected 300 and active-review 200 packages, and one locked
writer-disjoint validation set.  The arms cross the existing loss versus a
recall-weighted Tversky auxiliary loss with 1:1 versus 1:2 positive/clean
sampling.  Checkpoints are selected by gold-component coverage subject to a
pixel-precision guardrail; component filtering and the row gate are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATH = ROOT / "scripts/383_revision_active200_hard_negative_seed42.py"
ACTIVE_SHA256 = "5901dbd4526085b7867b2ea451d52c04c21625bae48e190d2de3308a270922eb"
SCHEMA = "hme-revision-coverage-finetune-seed42-dev-v1"
SOURCE_SCHEMA = "hme-revision-unetpp-original-seed42-dev-v1"
SOURCE_VARIANT = "unetpp_resnet34_scratch_original1000"
SOURCE_CHECKPOINT_SHA256 = (
    "ef41cbd2de95c5352a53a690dfb9cff17f65de32ee2d781f88a9e99747c58d2b"
)
VALIDATION_KEY_SHA256 = (
    "4d9dcf5db1d37cb0725a80cf3d5b4848e0351051c9d4121904cd8a6b5ac15cd0"
)
LOCKED_VALIDATION_NEGATIVE_RATIO = 3
LOCKED_VALIDATION_POSITIVE = 127
LOCKED_VALIDATION_CLEAN = 381
LOCKED_VALIDATION_WRITERS = 54
PRECISION_FLOOR = 0.60
TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70
TVERSKY_WEIGHT = 0.50
THRESHOLDS = tuple(float(value) for value in np.arange(0.02, 1.00, 0.02))
VARIANTS = {
    "standard_neg1": {"loss": "standard", "negative_ratio": 1},
    "standard_neg2": {"loss": "standard", "negative_ratio": 2},
    "tversky_neg1": {"loss": "recall_tversky", "negative_ratio": 1},
    "tversky_neg2": {"loss": "recall_tversky", "negative_ratio": 2},
}


class Experiment389Error(RuntimeError):
    """Raised when a provenance, safety, or protocol invariant changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment389Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment389Error(f"expected JSON object: {path}")
    return value


def load_active():
    if digest_file(ACTIVE_PATH) != ACTIVE_SHA256:
        raise Experiment389Error("experiment-383 dependency digest mismatch")
    specification = importlib.util.spec_from_file_location(
        "hme383_for389", ACTIVE_PATH)
    if specification is None or specification.loader is None:
        raise Experiment389Error("cannot import experiment 383")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sha256_entries(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in entries or len(digest) != 64:
            raise Experiment389Error("source SHA256SUMS is malformed")
        entries[name] = digest
    return entries


def verify_source(path: Path) -> dict[str, Any]:
    import torch

    path = Path(path).resolve()
    root = path.parent
    for required in (
            "complete.json", "summary.json", "run_manifest.json",
            "SHA256SUMS", "validation_key.json"):
        if not (root / required).is_file():
            raise Experiment389Error(f"source is missing {required}")
    complete = read_json(root / "complete.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"} \
            or summary.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("schema_version") != SOURCE_SCHEMA \
            or summary.get("variant") != SOURCE_VARIANT \
            or manifest.get("variant") != SOURCE_VARIANT \
            or summary.get("data", {}).get("data_arm") != "new1000" \
            or summary.get("data", {}).get("fit_positive") != 718 \
            or summary.get("data", {}).get("writer_overlap_fit_validation") != 0 \
            or manifest.get("m6_frozen_test_touched") is not False \
            or manifest.get("reference_evaluation_materialised") is not False:
        raise Experiment389Error("1000-line source safety contract changed")
    entries = _sha256_entries(root)
    checkpoint_sha = digest_file(path)
    if checkpoint_sha != SOURCE_CHECKPOINT_SHA256 \
            or entries.get(path.name) != checkpoint_sha \
            or digest_file(root / "validation_key.json") != VALIDATION_KEY_SHA256:
        raise Experiment389Error("1000-line source digest changed")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    architecture = checkpoint.get("architecture", {})
    if checkpoint.get("seed") != 42 \
            or checkpoint.get("data_arm") != "new1000" \
            or checkpoint.get("training_stage") != "scratch_original1000" \
            or architecture.get("variant") != "unetpp_resnet34" \
            or architecture.get("in_channels") != 1:
        raise Experiment389Error("1000-line source checkpoint metadata changed")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": digest_file(root / "run_manifest.json"),
        "root": root,
    }


def validate_args(args: argparse.Namespace) -> None:
    expected = VARIANTS.get(args.variant)
    if expected is None \
            or args.source_family != "original1000" \
            or args.correction_set != "all300" \
            or args.extra_fingerprint != "b279d1e8e824041a2492498486726559" \
            or args.active_fingerprint != "1342533e3090e15fc95e838b8a841e03" \
            or args.stage != "finetune_replay" \
            or args.mode != "dev" or args.data_arm != "new1000" \
            or args.recipe != "optimized" or args.seeds != "42" \
            or args.input_mode != "image" or args.merge_radius_factor != 0 \
            or not args.gold_confirmed:
        raise Experiment389Error("coverage fine-tuning base protocol changed")
    if args.optimizer_steps != 10000 or args.validation_interval != 250 \
            or not math.isclose(args.selection_warmup_fraction, 0.2) \
            or not math.isclose(args.learning_rate, 3e-5) \
            or args.batch_size != 12 \
            or int(args.negative_ratio) != expected["negative_ratio"] \
            or not math.isclose(args.isolated_fp_weight, 0.0):
        raise Experiment389Error("coverage fine-tuning optimisation changed")


def tag_rows(fit_positive: list[dict], fit_clean: list[dict],
             args: argparse.Namespace) -> dict[str, int]:
    active_ids = set(getattr(args, "active_fit_line_ids", ()))
    gold_ids = set(getattr(args, "gold300_fit_line_ids", ()))
    for row in fit_positive + fit_clean:
        line_id = str(row["line_id"])
        row["active200"] = line_id in active_ids
        row["gold300"] = line_id in gold_ids
        if row["active200"] and row["gold300"]:
            raise Experiment389Error("materialised row has two correction tags")
    counts = {
        "active_clean": sum(bool(row["active200"]) for row in fit_clean),
        "active_positive": sum(bool(row["active200"]) for row in fit_positive),
        "gold300_clean": sum(bool(row["gold300"]) for row in fit_clean),
        "gold300_positive": sum(bool(row["gold300"]) for row in fit_positive),
        "original_clean": sum(
            not row["active200"] and not row["gold300"] for row in fit_clean),
        "original_positive": sum(
            not row["active200"] and not row["gold300"]
            for row in fit_positive),
    }
    if counts != getattr(args, "expected_partition", None):
        raise Experiment389Error(f"materialised partition changed: {counts}")
    if len(fit_positive) != 1086 \
            or len(fit_clean) != int(args.negative_ratio) * len(fit_positive):
        raise Experiment389Error("requested positive/clean ratio changed")
    return counts


def training_batch(base, rng: random.Random, args: argparse.Namespace,
                   cycles: dict[str, Any]) -> list[dict]:
    ratio = int(args.negative_ratio)
    positive_count = args.batch_size // (1 + ratio)
    clean_count = args.batch_size - positive_count
    if positive_count * (1 + ratio) != args.batch_size \
            or positive_count % 2:
        raise Experiment389Error("batch cannot express exact 50% correction share")
    corrected_count = positive_count // 2
    rows = cycles["corrected_positive"].take(corrected_count) \
        + cycles["original_positive"].take(corrected_count) \
        + cycles["clean"].take_distinct(clean_count)
    identities = [str(row["line_id"]) for row in rows]
    if len(rows) != args.batch_size or len(identities) != len(set(identities)):
        raise Experiment389Error("coverage batch size or identity changed")
    rng.shuffle(rows)
    return rows


def trim_locked_training_clean(prepared: dict[str, Any],
                               args: argparse.Namespace) -> dict[str, Any]:
    """Change only the fitting clean prefix after building the locked split."""
    audit = prepared["audit"]
    if audit.get("validation_positive") != LOCKED_VALIDATION_POSITIVE \
            or audit.get("validation_clean") != LOCKED_VALIDATION_CLEAN \
            or audit.get("validation_writer_count") != LOCKED_VALIDATION_WRITERS:
        raise Experiment389Error("locked validation population changed")
    total_positive = len(prepared["fit_positive"])
    gold_clean_count = int(audit["fit_gold_clean"])
    wanted_total_clean = int(args.negative_ratio) * total_positive
    wanted_catalogue_clean = wanted_total_clean - gold_clean_count
    fit_clean = prepared["fit_clean"]
    if total_positive != 1086 or gold_clean_count != 27 \
            or wanted_catalogue_clean < 0 \
            or len(fit_clean) < wanted_total_clean:
        raise Experiment389Error("locked fitting population changed")
    selected_catalogue = fit_clean[
        gold_clean_count:gold_clean_count + wanted_catalogue_clean]
    selected = fit_clean[:gold_clean_count] + selected_catalogue
    selected_ids = [str(row["line_id"]) for row in selected_catalogue]
    if len(selected_ids) != len(set(selected_ids)):
        raise Experiment389Error("trimmed clean prefix contains duplicates")
    hard_selection = dict(prepared["hard_selection"])
    hard_selection.update({
        "prefix_count": len(selected_catalogue),
        "prefix_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")).hexdigest(),
    })
    prepared["fit_clean"] = selected
    prepared["hard_selection"] = hard_selection
    audit.update({
        "hard_selection": hard_selection,
        "locked_validation_negative_ratio": LOCKED_VALIDATION_NEGATIVE_RATIO,
        "sampled_catalogue_clean": len(selected_catalogue),
        "total_fit_clean": len(selected),
        "training_negative_ratio": int(args.negative_ratio),
    })
    expected = dict(getattr(args, "expected_partition", {}))
    corrected_clean = int(expected.get("active_clean", -1)) \
        + int(expected.get("gold300_clean", -1))
    if corrected_clean != 13:
        raise Experiment389Error("corrected clean partition changed")
    expected["original_clean"] = wanted_total_clean - corrected_clean
    args.expected_partition = expected
    return prepared


def patch_prepare_locked_validation(active, gold300, base,
                                    args: argparse.Namespace) -> None:
    """Prepare with the canonical split, then vary training negatives only."""
    active.patch_prepare(gold300, base, args)
    corrected_prepare = base.prepare

    def prepare(run_args: argparse.Namespace):
        requested_ratio = int(run_args.negative_ratio)
        run_args.negative_ratio = LOCKED_VALIDATION_NEGATIVE_RATIO
        try:
            prepared = corrected_prepare(run_args)
        finally:
            run_args.negative_ratio = requested_ratio
        return trim_locked_training_clean(prepared, run_args)

    base.prepare = prepare


def recall_tversky_loss(logits, target):
    import torch

    probability = torch.sigmoid(logits)
    axes = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=axes)
    false_positive = (probability * (1.0 - target)).sum(dim=axes)
    false_negative = ((1.0 - probability) * target).sum(dim=axes)
    score = (true_positive + 1.0) / (
        true_positive + TVERSKY_ALPHA * false_positive
        + TVERSKY_BETA * false_negative + 1.0)
    return (1.0 - score).mean()


def _component_probability_values(probabilities: np.ndarray,
                                  targets: np.ndarray) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise Experiment389Error("OpenCV is required for coverage metrics") from exc
    values: list[np.ndarray] = []
    for probability, target in zip(probabilities, targets):
        count, labels = cv2.connectedComponents(
            np.ascontiguousarray(target, dtype=np.uint8), connectivity=8)
        for label in range(1, count):
            component = probability[labels == label]
            if not len(component):
                raise Experiment389Error("empty gold component")
            values.append(component)
    if len(values) != 237:
        raise Experiment389Error(
            f"gold component count changed: {len(values)}")
    return values


def coverage_operating_point(positive_probability: np.ndarray,
                             targets: np.ndarray,
                             clean_probability: np.ndarray,
                             precision_floor: float = PRECISION_FLOOR) \
        -> dict[str, Any]:
    positive_probability = np.asarray(positive_probability, dtype=np.float32)
    clean_probability = np.asarray(clean_probability, dtype=np.float32)
    targets = np.asarray(targets, dtype=bool)
    if positive_probability.shape != targets.shape \
            or positive_probability.ndim != 3 \
            or clean_probability.shape[1:] != targets.shape[1:]:
        raise Experiment389Error("coverage arrays differ")
    component_values = _component_probability_values(
        positive_probability, targets)
    truth_values = np.sort(positive_probability[targets])
    false_values = np.sort(np.concatenate((
        positive_probability[~targets], clean_probability.reshape(-1))))
    clean_max = clean_probability.reshape(len(clean_probability), -1).max(axis=1)
    truth_pixels = int(len(truth_values))
    candidates = []
    for threshold in THRESHOLDS:
        true_positive = truth_pixels - int(np.searchsorted(
            truth_values, threshold, side="left"))
        false_positive = len(false_values) - int(np.searchsorted(
            false_values, threshold, side="left"))
        predicted = true_positive + false_positive
        precision = true_positive / predicted if predicted else 1.0
        recall = true_positive / truth_pixels if truth_pixels else 1.0
        iou = true_positive / (truth_pixels + false_positive) \
            if truth_pixels + false_positive else 1.0
        coverages = np.asarray([
            float(np.mean(values >= threshold)) for values in component_values
        ], dtype=float)
        candidates.append({
            "clean_false_alarm": float(np.mean(clean_max >= threshold)),
            "covered_ge_10_count": int(np.sum(coverages >= 0.10)),
            "covered_ge_10_rate": float(np.mean(coverages >= 0.10)),
            "covered_ge_50_count": int(np.sum(coverages >= 0.50)),
            "covered_ge_50_rate": float(np.mean(coverages >= 0.50)),
            "eligible": bool(precision >= precision_floor),
            "gold_components": int(len(coverages)),
            "mean_gold_component_coverage": float(np.mean(coverages)),
            "median_gold_component_coverage": float(np.median(coverages)),
            "pixel_iou": float(iou),
            "pixel_precision": float(precision),
            "pixel_recall": float(recall),
            "predicted_to_gold_area": float(predicted / truth_pixels),
            "precision_floor": float(precision_floor),
            "threshold": float(threshold),
        })
    eligible = [point for point in candidates if point["eligible"]]
    if not eligible:
        raise Experiment389Error("no coverage threshold meets precision floor")
    return max(eligible, key=coverage_key)


def coverage_key(point: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(point["covered_ge_50_rate"]),
        float(point["median_gold_component_coverage"]),
        float(point["covered_ge_10_rate"]),
        float(point["pixel_recall"]),
        float(point["pixel_iou"]),
        float(point["pixel_precision"]),
        -float(point["clean_false_alarm"]),
        float(point["threshold"]),
    )


def predict(ablation, model, rows: Sequence[dict[str, Any]], batch_size: int,
            device):
    return ablation.predict(model, rows, batch_size, device)


def train_once(active, corrective, ablation, base, extension,
               fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path,
               source: dict[str, Any]):
    import torch

    if seed != 42 or not torch.cuda.is_available():
        raise Experiment389Error("formal coverage fine-tuning requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    counts = tag_rows(fit_positive, fit_clean, args)
    model = ablation.make_unetpp_resnet34(args.resnet34_weights).to(device)
    model.load_state_dict(source["checkpoint"]["state_dict"], strict=True)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(seed)
    partitions = {
        "corrected_positive": [
            row for row in fit_positive if row["active200"] or row["gold300"]],
        "original_positive": [
            row for row in fit_positive
            if not row["active200"] and not row["gold300"]],
        "clean": list(fit_clean),
    }
    if len(partitions["corrected_positive"]) != 368 \
            or len(partitions["original_positive"]) != 718:
        raise Experiment389Error("coverage positive partition changed")
    cycles = {name: base.ShuffledCycle(rows, rng)
              for name, rows in partitions.items()}
    recipe = base.RECIPES[args.recipe]
    validation_target = np.stack([
        np.asarray(row["mask"], dtype=np.float32) / 255.0
        for row in validation_positive])
    source_positive, source_target = predict(
        ablation, model, validation_positive, args.batch_size, device)
    source_clean, _ = predict(
        ablation, model, validation_clean, args.batch_size, device)
    source_coverage = coverage_operating_point(
        source_positive, source_target > 0.5, source_clean)
    trace_path = staging / f"coverage_trace_seed{seed}.jsonl"
    best_key = best_state = best_points = best_coverage = None
    best_step = -1
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}
    loss_family = VARIANTS[args.variant]["loss"]

    with trace_path.open("x", encoding="utf-8") as trace:
        trace.write(json.dumps({
            "coverage": source_coverage, "event": "source", "step": 0,
            "variant": args.variant,
        }, sort_keys=True) + "\n")
        for step in range(1, args.optimizer_steps + 1):
            model.train()
            rows = training_batch(base, rng, args, cycles)
            images, masks, skeletons = [], [], []
            for row in rows:
                image, mask, skeleton, _ = extension.paired_affine_with_aux(
                    row["image"], row["mask"], row["line_skeleton"], None, rng)
                images.append(image)
                masks.append(mask)
                skeletons.append(skeleton)
            gray = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            target = torch.from_numpy(
                np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
            logits = ablation.main_logits(model(gray))
            standard, skeleton_loss, neighbourhood = \
                corrective.standard_loss_terms(
                    extension, logits, target, skeleton, args)
            tversky = recall_tversky_loss(logits, target)
            tversky_weight = TVERSKY_WEIGHT \
                if loss_family == "recall_tversky" else 0.0
            loss = standard + recipe["skeleton_weight"] * skeleton_loss \
                + tversky_weight * tversky
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            last_losses = {
                "loss": float(loss.detach().cpu()),
                "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                "skeleton": float(skeleton_loss.detach().cpu()),
                "standard": float(standard.detach().cpu()),
                "tversky": float(tversky.detach().cpu()),
                "tversky_weight": float(tversky_weight),
            }
            if step % args.validation_interval != 0 \
                    and step != args.optimizer_steps:
                continue
            positive_probability, positive_target = predict(
                ablation, model, validation_positive, args.batch_size, device)
            clean_probability, _ = predict(
                ablation, model, validation_clean, args.batch_size, device)
            coverage = coverage_operating_point(
                positive_probability, positive_target > 0.5, clean_probability)
            points = base.choose_operating_points(
                positive_probability, positive_target, clean_probability,
                args.max_clean_fp,
                base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
            key = coverage_key(coverage)
            if step >= start_select and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_coverage, best_points = coverage, points
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            record = {
                "best_step": best_step,
                "coverage": coverage,
                "event": "validation",
                "seed": seed,
                "standard_iou": points["localization"]["iou"],
                "step": step,
                "variant": args.variant,
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({
                "best_step": best_step,
                "covered_ge_50_rate": round(coverage["covered_ge_50_rate"], 6),
                "median_coverage": round(
                    coverage["median_gold_component_coverage"], 6),
                "pixel_precision": round(coverage["pixel_precision"], 6),
                "step": step,
                "variant": args.variant,
            }, sort_keys=True), flush=True)

    if best_state is None or best_points is None or best_coverage is None:
        raise Experiment389Error("coverage fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    positive_probability, positive_target = predict(
        ablation, model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        ablation, model, validation_clean, args.batch_size, device)
    final_coverage = coverage_operating_point(
        positive_probability, positive_target > 0.5, clean_probability)
    if coverage_key(final_coverage) != coverage_key(best_coverage):
        raise Experiment389Error("reloaded coverage checkpoint differs")
    points = base.choose_operating_points(
        positive_probability, positive_target, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": {
            "base_variant": "unetpp_resnet34",
            "input_channels": 1,
            "variant": "coverage_finetune",
        },
        "best_step": best_step,
        "coverage_operating_point": final_coverage,
        "data_arm": "new1000_plus_all300_plus_active200",
        "negative_ratio": int(args.negative_ratio),
        "operating_points": points,
        "seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "state_dict": model.state_dict(),
        "variant": args.variant,
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=positive_probability.astype(np.float16),
        targets=(positive_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "coverage": final_coverage,
        "last_losses": last_losses,
        "loss_family": loss_family,
        "partition": counts,
        "sampling": {
            "batch_size": args.batch_size,
            "corrected_positive_per_batch":
                args.batch_size // (1 + int(args.negative_ratio)) // 2,
            "corrected_positive_pool": len(partitions["corrected_positive"]),
            "negative_ratio": int(args.negative_ratio),
            "original_positive_per_batch":
                args.batch_size // (1 + int(args.negative_ratio)) // 2,
            "original_positive_pool": len(partitions["original_positive"]),
        },
        "seed": seed,
        "source_coverage": source_coverage,
        "validation": points,
        "validation_target_shape": list(validation_target.shape),
        "variant": args.variant,
    }, {"clean": clean_probability, "positive": positive_probability}


def finalise(base, staging: Path, args: argparse.Namespace,
             source: dict[str, Any]) -> dict[str, Any]:
    summary_path = staging / "summary.json"
    manifest_path = staging / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    data = summary.get("data", {})
    if data.get("writer_overlap_fit_validation") != 0 \
            or data.get("fit_positive") != 1086 \
            or data.get("validation_positive") != LOCKED_VALIDATION_POSITIVE \
            or data.get("validation_clean") != LOCKED_VALIDATION_CLEAN \
            or data.get("validation_writer_count") != LOCKED_VALIDATION_WRITERS \
            or data.get("active200", {}).get("training_rows") != 151 \
            or data.get("extra300", {}).get("training_rows") != 230 \
            or data.get("total_fit_clean") \
            != int(args.negative_ratio) * data.get("fit_positive", -1) \
            or data.get("m6_frozen_test_touched") is not False:
        raise Experiment389Error("coverage final data contract changed")
    if digest_file(staging / "validation_key.json") != VALIDATION_KEY_SHA256:
        raise Experiment389Error("coverage validation split changed")
    if len(summary.get("per_seed", [])) != 1:
        raise Experiment389Error("coverage run is not single-seed")
    coverage = summary["per_seed"][0].get("coverage")
    if not isinstance(coverage, dict) or coverage.get("gold_components") != 237:
        raise Experiment389Error("coverage summary is absent")
    summary["coverage"] = coverage
    summary["schema_version"] = SCHEMA
    summary["variant"] = args.variant
    summary["protocol"].update({
        "candidate_pixels_used": False,
        "checkpoint_selection": (
            "gold-component >=50%, median, >=10%, pixel recall; "
            "pixel precision >=0.60"),
        "corrected_positive_share": 0.50,
        "gold_masks_modified": False,
        "loss_family": VARIANTS[args.variant]["loss"],
        "row_gate_used": False,
        "seed_policy": "single fixed seed 42",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_training_rows": 1000,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "tversky_weight": TVERSKY_WEIGHT
            if VARIANTS[args.variant]["loss"] == "recall_tversky" else 0.0,
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
    })
    manifest.update({
        "active_audit_sha256": digest_file(Path(args.active_audit)),
        "active_key_sha256": digest_file(Path(args.active_key)),
        "active_returned_sha256": digest_file(Path(args.active_returned)),
        "data_arm": "new1000_plus_all300_plus_active200",
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "seed": 42,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "variant": args.variant,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(active, gold300, corrective, base, runner, full):
    parser = gold300.build_parser(corrective, base, runner, full)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--active-key", required=True)
    parser.add_argument("--active-returned", required=True)
    parser.add_argument("--active-audit", required=True)
    parser.add_argument("--active-fingerprint", required=True)
    return parser


def run(args, active, gold300, corrective, ablation, base, extension,
        runner, full):
    validate_args(args)
    source = verify_source(Path(args.source_checkpoint))
    full.configure_base(base)
    patch_prepare_locked_validation(active, gold300, base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(active, corrective, ablation, b, e, fp, fc, vp, vc, a,
                   seed, staging, source)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment389Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inner = output.parent / f".{output.name}.inner-{uuid.uuid4().hex}"
    original_output = args.output_dir
    args.output_dir = inner
    try:
        runner.run(args, base, extension)
        summary = finalise(base, inner, args, source)
        inner.rename(output)
    except BaseException:
        if inner.is_dir():
            shutil.rmtree(inner)
        raise
    finally:
        args.output_dir = original_output
    return summary


def main() -> int:
    active = load_active()
    gold300 = active.load_gold300()
    wavelet = gold300.load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for389", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for389", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for389", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for389", ablation.FULL1000_SHA256)
    args = build_parser(
        active, gold300, corrective, base, runner, full).parse_args()
    try:
        summary = run(
            args, active, gold300, corrective, ablation, base, extension,
            runner, full)
    except (Experiment389Error, active.Experiment383Error,
            gold300.Experiment380Error, wavelet.Experiment376Error,
            corrective.Experiment374Error, ablation.Ablation371Error,
            base.LocalizerError, runner.MechanismError,
            full.Full1000Error) as exc:
        raise SystemExit(f"experiment 389 refused: {exc}") from exc
    if args.validate_only:
        payload = {"data": summary, "status": "validated",
                   "variant": args.variant}
    else:
        payload = {
            "coverage": summary["coverage"],
            "data": summary["data"],
            "variant": args.variant,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

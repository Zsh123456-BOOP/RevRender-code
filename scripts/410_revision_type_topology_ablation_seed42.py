#!/usr/bin/env python3
"""Ablate cleaned, type-aware topology training from the current pixel base.

All four seed-42 arms start from the same current best pixel checkpoint and the
same writer-disjoint development split.  The repeated bottom-right scan-corner
OCR lines are removed through experiment 409's immutable exclusion manifest.
Treatments compare stronger line-skeleton recall, morphology experts, and a
line-only SCNP proxy.  Blot pixels are never subjected to the line-topology
penalty.  Frozen test/reference material is not materialised.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE402_PATH = ROOT / "scripts/402_revision_active300_coverage_finetune_seed42.py"
ACTIVE402_SHA256 = "d06065056506cd80e3c13789b133f0d5f69ddbfb76d817d00a32a8f0ec3a6dd1"
SCHEMA = "hme-revision-type-topology-ablation-seed42-dev-v1"
INITIAL_CHECKPOINT_SHA256 = (
    "80bddc85a8c7ac0b2cfed4b4ef4059fb1f3bee07673dd2908e2c2a75ed288bfe"
)
EXCLUSION_MANIFEST_SHA256 = (
    "9d59d00b384935bea1142665988848b7286a18905e074ab45ca416ac8b7ae29c"
)
EXPECTED_EXCLUDED_IDS = 28
EXPECTED_CLEAN_VALIDATION_POSITIVE = 126
EXPECTED_CLEAN_VALIDATION_CLEAN = 379
EXPECTED_CLEAN_GOLD_COMPONENTS = 236

VARIANTS = {
    "clean_control": {
        "dual_expert": False,
        "expert_aux_weight": 0.0,
        "gate_weight": 0.0,
        "line_scnp_weight": 0.0,
        "skeleton_weight": 0.15,
    },
    "line_skeleton035": {
        "dual_expert": False,
        "expert_aux_weight": 0.0,
        "gate_weight": 0.0,
        "line_scnp_weight": 0.0,
        "skeleton_weight": 0.35,
    },
    "type_dual_expert": {
        "dual_expert": True,
        "expert_aux_weight": 0.10,
        "gate_weight": 0.05,
        "line_scnp_weight": 0.0,
        "skeleton_weight": 0.15,
    },
    "type_dual_topology": {
        "dual_expert": True,
        "expert_aux_weight": 0.10,
        "gate_weight": 0.05,
        "line_scnp_weight": 0.10,
        "skeleton_weight": 0.35,
    },
}


class Experiment410Error(RuntimeError):
    """Raised when a provenance, data, or create-only contract changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str):
    actual = digest_file(path)
    if actual != expected_sha256:
        raise Experiment410Error(
            f"dependency digest mismatch for {path.name}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Experiment410Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment410Error(f"invalid JSON: {path}") from exc
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def load_exclusions(path: Path) -> set[str]:
    if digest_file(path) != EXCLUSION_MANIFEST_SHA256:
        raise Experiment410Error("scan-corner exclusion manifest changed")
    document = read_json(path)
    if not isinstance(document, dict):
        raise Experiment410Error(f"expected JSON object: {path}")
    values = document.get("excluded_line_ids")
    if document.get("m6_frozen_test_touched") is not False \
            or document.get("source_data_modified") is not False \
            or not isinstance(values, list):
        raise Experiment410Error("scan-corner exclusion contract changed")
    output = set(map(str, values))
    if len(output) != EXPECTED_EXCLUDED_IDS:
        raise Experiment410Error("scan-corner exclusion count changed")
    return output


def filter_scan_corners(data: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name in ("fit_positive", "fit_clean", "validation_positive",
                 "validation_clean"):
        rows = list(data[name])
        kept = [row for row in rows if str(row["line_id"]) not in excluded]
        audit[name] = {
            "after": len(kept),
            "before": len(rows),
            "excluded": len(rows) - len(kept),
        }
        data[name] = kept
    if len(data["validation_positive"]) != EXPECTED_CLEAN_VALIDATION_POSITIVE \
            or len(data["validation_clean"]) != EXPECTED_CLEAN_VALIDATION_CLEAN:
        raise Experiment410Error("cleaned validation row counts changed")
    return audit


def load_initial_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = Path(path) / "segmenter_seed42.pt"
    if digest_file(checkpoint) != INITIAL_CHECKPOINT_SHA256:
        raise Experiment410Error("current-best pixel checkpoint changed")
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if value.get("seed") != 42 \
            or value.get("variant") != "source_late_encoder_lr1" \
            or not isinstance(value.get("state_dict"), dict):
        raise Experiment410Error("current-best pixel checkpoint metadata changed")
    return {"checkpoint": value, "path": checkpoint,
            "sha256": INITIAL_CHECKPOINT_SHA256}


def make_model(ablation: Any, weights_path: Path,
               initial_state: Mapping[str, Any], dual_expert: bool):
    import torch
    from torch import nn
    from torch.nn import functional

    core = ablation.make_unetpp_resnet34(weights_path)
    core.load_state_dict(initial_state, strict=True)
    if not dual_expert:
        return core

    class TypeAwareRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.features = nn.Sequential(
                nn.Conv2d(2, 32, 5, padding=2, bias=False),
                nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                nn.Conv2d(32, 32, 1, bias=False),
                nn.GroupNorm(8, 32), nn.GELU(),
            )
            self.line_delta = nn.Conv2d(32, 1, 3, padding=1)
            self.blot_delta = nn.Conv2d(32, 1, 5, padding=2)
            self.gate = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.line_delta.weight)
            nn.init.zeros_(self.line_delta.bias)
            nn.init.zeros_(self.blot_delta.weight)
            nn.init.zeros_(self.blot_delta.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

        def forward_with_aux(self, gray):
            base_logits = ablation.main_logits(self.core(gray))
            probability = torch.sigmoid(base_logits)
            features = self.features(torch.cat((gray, probability), dim=1))
            line_logits = base_logits + self.line_delta(features)
            blot_logits = base_logits + self.blot_delta(features)
            gate_logits = self.gate(features)
            gate = torch.sigmoid(gate_logits)
            final = gate * line_logits + (1.0 - gate) * blot_logits
            return final, line_logits, blot_logits, gate_logits

        def forward(self, gray):
            return self.forward_with_aux(gray)[0]

    return TypeAwareRefiner()


def configure_trainable(model: Any, dual_expert: bool) -> dict[str, int]:
    core = model.core if dual_expert else model
    for parameter in core.parameters():
        parameter.requires_grad = False
    for name, parameter in core.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad = True
    if not hasattr(core.encoder, "layer4"):
        raise Experiment410Error("ResNet encoder has no layer4")
    for parameter in core.encoder.layer4.parameters():
        parameter.requires_grad = True
    if dual_expert:
        for name, parameter in model.named_parameters():
            if not name.startswith("core."):
                parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def keep_frozen_encoder_eval(model: Any, dual_expert: bool) -> None:
    core = model.core if dual_expert else model
    core.encoder.eval()
    core.encoder.layer4.train()


def balanced_dice_bce(logits: Any, target: Any):
    import torch
    from torch.nn import functional

    dimensions = tuple(range(1, target.ndim))
    positive_pixels = target.sum(dimensions)
    total_pixels = float(np.prod(target.shape[1:]))
    positive_weight = ((total_pixels - positive_pixels) /
                       positive_pixels.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    weights = 1.0 + target * (positive_weight.view(-1, 1, 1, 1) - 1.0)
    bce = (bce * weights).mean()
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + positive_pixels + 1.0
    valid = positive_pixels > 0
    dice = (1.0 - numerator[valid] / denominator[valid]).mean() \
        if bool(valid.any()) else torch.zeros((), device=logits.device)
    return bce + dice


def type_targets(target: Any, skeleton: Any):
    from torch.nn import functional

    line_region = functional.max_pool2d(
        skeleton, kernel_size=5, stride=1, padding=2) * target
    line_region = line_region.clamp(0.0, 1.0)
    blot_region = (target - line_region).clamp(0.0, 1.0)
    return line_region, blot_region


def count_gold_components(targets: np.ndarray) -> int:
    import cv2

    total = 0
    for target in np.asarray(targets) > 0.5:
        count, _labels = cv2.connectedComponents(
            np.ascontiguousarray(target, dtype=np.uint8), connectivity=8)
        total += count - 1
    return total


def component_probability_values(probabilities: np.ndarray,
                                 targets: np.ndarray) -> list[np.ndarray]:
    import cv2

    values: list[np.ndarray] = []
    for probability, target in zip(probabilities, targets):
        count, labels = cv2.connectedComponents(
            np.ascontiguousarray(target, dtype=np.uint8), connectivity=8)
        for label in range(1, count):
            values.append(np.asarray(probability[labels == label]))
    if len(values) != EXPECTED_CLEAN_GOLD_COMPONENTS:
        raise Experiment410Error(
            f"cleaned gold component count changed: {len(values)}")
    return values


def cleaned_coverage_operating_point(coverage: Any,
                                     positive_probability: np.ndarray,
                                     targets: np.ndarray,
                                     clean_probability: np.ndarray,
                                     precision_floor: float = 0.60) \
        -> dict[str, Any]:
    """Experiment-389 coverage metric with the audited 236-component guard."""
    positive_probability = np.asarray(positive_probability, dtype=np.float32)
    clean_probability = np.asarray(clean_probability, dtype=np.float32)
    targets = np.asarray(targets, dtype=bool)
    if positive_probability.shape != targets.shape \
            or positive_probability.ndim != 3 \
            or clean_probability.shape[1:] != targets.shape[1:]:
        raise Experiment410Error("cleaned coverage arrays differ")
    component_values = component_probability_values(
        positive_probability, targets)
    truth_values = np.sort(positive_probability[targets])
    false_values = np.sort(np.concatenate((
        positive_probability[~targets], clean_probability.reshape(-1))))
    clean_max = clean_probability.reshape(len(clean_probability), -1).max(axis=1)
    truth_pixels = int(len(truth_values))
    candidates = []
    for threshold in coverage.THRESHOLDS:
        true_positive = truth_pixels - int(np.searchsorted(
            truth_values, threshold, side="left"))
        false_positive = len(false_values) - int(np.searchsorted(
            false_values, threshold, side="left"))
        predicted = true_positive + false_positive
        precision = true_positive / predicted if predicted else 1.0
        recall = true_positive / truth_pixels if truth_pixels else 1.0
        iou = true_positive / (truth_pixels + false_positive) \
            if truth_pixels + false_positive else 1.0
        component_coverages = np.asarray([
            float(np.mean(values >= threshold)) for values in component_values
        ], dtype=float)
        candidates.append({
            "clean_false_alarm": float(np.mean(clean_max >= threshold)),
            "covered_ge_10_count": int(np.sum(component_coverages >= 0.10)),
            "covered_ge_10_rate": float(np.mean(component_coverages >= 0.10)),
            "covered_ge_50_count": int(np.sum(component_coverages >= 0.50)),
            "covered_ge_50_rate": float(np.mean(component_coverages >= 0.50)),
            "eligible": bool(precision >= precision_floor),
            "gold_components": int(len(component_coverages)),
            "mean_gold_component_coverage": float(np.mean(component_coverages)),
            "median_gold_component_coverage": float(np.median(component_coverages)),
            "pixel_iou": float(iou),
            "pixel_precision": float(precision),
            "pixel_recall": float(recall),
            "predicted_to_gold_area": float(predicted / truth_pixels),
            "precision_floor": float(precision_floor),
            "threshold": float(threshold),
        })
    eligible = [point for point in candidates if point["eligible"]]
    if not eligible:
        raise Experiment410Error("no cleaned coverage threshold meets precision floor")
    return max(eligible, key=coverage.coverage_key)


def predict(ablation: Any, model: Any, rows: list[dict[str, Any]],
            batch_size: int, device: Any):
    return ablation.predict(model, rows, batch_size, device)


def train(data: Mapping[str, Any], initial: Mapping[str, Any],
          args: argparse.Namespace, config: Mapping[str, Any], staging: Path):
    import torch
    from torch.nn import functional

    if args.seed != 42 or not torch.cuda.is_available():
        raise Experiment410Error("formal topology training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dual = bool(config["dual_expert"])
    model = make_model(
        data["ablation"], data["run_args"].resnet34_weights,
        initial["checkpoint"]["state_dict"], dual).to(device)
    parameter_count = configure_trainable(model, dual)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=data["run_args"].weight_decay)
    rng = random.Random(args.seed)
    partitions = {
        "active_positive": [row for row in data["fit_positive"]
                            if row["active300"]],
        "replay_positive": [row for row in data["fit_positive"]
                            if not row["active300"]],
        "active_clean": [row for row in data["fit_clean"] if row["active300"]],
        "clean": [row for row in data["fit_clean"] if not row["active300"]],
    }
    if not partitions["active_positive"] or not partitions["replay_positive"] \
            or len(partitions["clean"]) < 8:
        raise Experiment410Error("cleaned training pools are incomplete")
    cycles = {name: data["base"].ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    batch_config = {
        "active_positive_per_batch": 1,
        "negative_ratio": 1,
    }
    initial_positive, initial_target = predict(
        data["ablation"], model, data["validation_positive"], 12, device)
    initial_clean, _ = predict(
        data["ablation"], model, data["validation_clean"], 12, device)
    if count_gold_components(initial_target) != EXPECTED_CLEAN_GOLD_COMPONENTS:
        raise Experiment410Error("cleaned gold component count changed")
    initial_coverage = cleaned_coverage_operating_point(
        data["coverage"], initial_positive, initial_target > 0.5,
        initial_clean)
    best_key = best_state = best_coverage = best_points = None
    best_step = -1
    last_losses: dict[str, float] = {}
    trace_path = staging / "training_trace.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        trace.write(json.dumps({"coverage": initial_coverage,
                                "event": "initial", "step": 0},
                               sort_keys=True) + "\n")
        for step in range(1, args.optimizer_steps + 1):
            model.train()
            keep_frozen_encoder_eval(model, dual)
            rows = data["active402"].training_batch(
                data["base"], rng, batch_config, cycles, step)
            images, masks, skeletons = [], [], []
            for row in rows:
                image, mask, skeleton, _hard = \
                    data["extension"].paired_affine_with_aux(
                        row["image"], row["mask"], row["line_skeleton"],
                        None, rng)
                images.append(image)
                masks.append(mask)
                skeletons.append(skeleton)
            gray = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            target = torch.from_numpy(
                np.stack(masks).astype(np.float32)[:, None] / 255.0
            ).to(device)
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0
            ).to(device)
            if dual:
                logits, line_logits, blot_logits, gate_logits = \
                    model.forward_with_aux(gray)
            else:
                logits = data["ablation"].main_logits(model(gray))
                line_logits = blot_logits = gate_logits = None
            standard, skeleton_loss, neighbourhood = \
                data["corrective"].standard_loss_terms(
                    data["extension"], logits, target, skeleton,
                    data["run_args"])
            line_target, blot_target = type_targets(target, skeleton)
            expert_loss = torch.zeros((), device=device)
            gate_loss = torch.zeros((), device=device)
            line_scnp = torch.zeros((), device=device)
            if dual:
                expert_loss = (balanced_dice_bce(line_logits, line_target)
                               + balanced_dice_bce(blot_logits, blot_target)) / 2.0
                selected = target >= 0.5
                if bool(selected.any()):
                    gate_loss = functional.binary_cross_entropy_with_logits(
                        gate_logits[selected], line_target[selected])
                if float(config["line_scnp_weight"]) > 0:
                    line_scnp = data["extension"].isolated_false_positive_proxy_loss(
                        line_logits, line_target, data["run_args"].neighbourhood_kernel)
            loss = standard \
                + float(config["skeleton_weight"]) * skeleton_loss \
                + float(config["expert_aux_weight"]) * expert_loss \
                + float(config["gate_weight"]) * gate_loss \
                + float(config["line_scnp_weight"]) * line_scnp
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad], 5.0)
            optimiser.step()
            last_losses = {
                "expert": float(expert_loss.detach().cpu()),
                "gate": float(gate_loss.detach().cpu()),
                "line_scnp": float(line_scnp.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                "skeleton": float(skeleton_loss.detach().cpu()),
                "standard": float(standard.detach().cpu()),
            }
            if step % args.validation_interval and step != args.optimizer_steps:
                continue
            positive_probability, positive_target = predict(
                data["ablation"], model, data["validation_positive"], 12, device)
            clean_probability, _ = predict(
                data["ablation"], model, data["validation_clean"], 12, device)
            coverage = cleaned_coverage_operating_point(
                data["coverage"], positive_probability,
                positive_target > 0.5, clean_probability)
            points = data["base"].choose_operating_points(
                positive_probability, positive_target, clean_probability,
                data["run_args"].max_clean_fp,
                data["base"].scaled_min_fire_pixels(
                    data["run_args"].recipe, data["run_args"].min_fire_pixels))
            key = data["coverage"].coverage_key(coverage)
            if step >= args.selection_warmup_steps \
                    and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_coverage, best_points = coverage, points
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            record = {
                "best_step": best_step,
                "covered_ge_10": coverage["covered_ge_10_count"],
                "covered_ge_50": coverage["covered_ge_50_count"],
                "event": "validation",
                "median": coverage["median_gold_component_coverage"],
                "pixel_iou": coverage["pixel_iou"],
                "step": step,
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({**record, "variant": args.variant},
                             sort_keys=True), flush=True)
    if best_state is None or best_coverage is None or best_points is None:
        raise Experiment410Error("topology arm produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_positive, final_target = predict(
        data["ablation"], model, data["validation_positive"], 12, device)
    final_clean, _ = predict(
        data["ablation"], model, data["validation_clean"], 12, device)
    final_coverage = cleaned_coverage_operating_point(
        data["coverage"], final_positive, final_target > 0.5, final_clean)
    torch.save({
        "architecture": {
            "base_variant": "unetpp_resnet34",
            "dual_expert": dual,
            "input_channels": 1,
            "variant": args.variant,
        },
        "best_step": best_step,
        "coverage_operating_point": final_coverage,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "source_checkpoint_sha256": initial["sha256"],
        "state_dict": model.state_dict(),
        "variant": args.variant,
    }, staging / "segmenter_seed42.pt")
    np.savez_compressed(
        staging / "validation_predictions_seed42.npz",
        clean_probabilities=final_clean.astype(np.float16),
        positive_probabilities=final_positive.astype(np.float16),
        targets=(final_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "coverage": final_coverage,
        "initial_coverage": initial_coverage,
        "last_losses": last_losses,
        "operating_points": best_points,
        "parameter_count": parameter_count,
        "partitions": {name: len(rows) for name, rows in partitions.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS or args.seed != 42:
        raise Experiment410Error("formal topology arms use known variants/seed42")
    if args.optimizer_steps != 3000 or args.validation_interval != 250 \
            or args.selection_warmup_steps != 500 \
            or not np.isclose(args.learning_rate, 1e-5):
        raise Experiment410Error("formal optimisation protocol changed")
    active402 = load_module(
        ACTIVE402_PATH, "hme402_for410", ACTIVE402_SHA256)
    scorer390 = load_module(
        active402.SCORER390_PATH, "hme390_for410", active402.SCORER390_SHA256)
    coverage = scorer390.load_module(
        scorer390.COVERAGE_PATH, "hme389_for410", scorer390.COVERAGE_SHA256)
    data_source = active402.verify_source(args.data_source_output)
    config_for_data = active402.VARIANTS["standard_neg1_natural_lr1"]
    data = active402.prepare_data(
        scorer390, coverage, data_source, args, config_for_data)
    exclusions = load_exclusions(args.exclusion_manifest)
    cleaning_audit = filter_scan_corners(data, exclusions)
    initial = load_initial_checkpoint(args.initial_output)
    data.update({"active402": active402, "coverage": coverage})
    if data["audit"]["writer_overlap_fit_validation"] != 0:
        raise Experiment410Error("writer isolation changed")
    if args.validate_only:
        return {"cleaning": cleaning_audit, "variant": args.variant}

    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment410Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result = train(data, initial, args, VARIANTS[args.variant], staging)
        original_key = read_json(data_source["root"] / "validation_key.json")
        if not isinstance(original_key, list):
            raise Experiment410Error("validation key must be a JSON list")
        clean_key = [row for row in original_key
                     if str(row["line_id"]) not in exclusions]
        write_json(staging / "validation_key_clean.json", clean_key)
        summary = {
            "cleaning": cleaning_audit,
            "coverage": result["coverage"],
            "gold_components": EXPECTED_CLEAN_GOLD_COMPONENTS,
            "protocol": {
                **VARIANTS[args.variant],
                "fixed_cleaned_validation": True,
                "frozen_test_touched": False,
                "learning_rate": args.learning_rate,
                "m6_frozen_test_touched": False,
                "seed_policy": "fixed seed 42",
                "source_checkpoint_sha256": initial["sha256"],
                "writer_overlap_fit_validation": 0,
            },
            "result": result,
            "schema_version": SCHEMA,
            "status": "complete",
            "variant": args.variant,
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "run_manifest.json", {
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_source_checkpoint_sha256": data_source["checkpoint_sha256"],
            "exclusion_manifest_sha256": EXCLUSION_MANIFEST_SHA256,
            "frozen_test_touched": False,
            "initial_checkpoint_sha256": initial["sha256"],
            "reference_evaluation_materialised": False,
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "seed": args.seed,
            "variant": args.variant,
        })
        write_json(staging / "complete.json", {
            "schema_version": SCHEMA, "status": "complete"})
        data["base"].write_recursive_sha256s(staging)
        staging.rename(output)
        return summary
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source-output", type=Path, required=True)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--active-key", type=Path, required=True)
    parser.add_argument("--active-returned", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer-steps", type=int, default=3000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--selection-warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    try:
        summary = run(build_parser().parse_args())
    except (Experiment410Error, RuntimeError) as exc:
        raise SystemExit(f"experiment 410 refused: {exc}") from exc
    print(json.dumps({
        "cleaning": summary.get("cleaning"),
        "coverage": summary.get("coverage"),
        "variant": summary["variant"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

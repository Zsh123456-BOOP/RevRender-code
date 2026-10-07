#!/usr/bin/env python3
"""Size-balanced, separation and anti-fragment losses on experiment 480.

The architecture is locked to experiment 480's best seed-42 screen:
full-resolution mask pixels, no duplicate-query suppression, and the unchanged
MiT-B0/query/quality/fusion structure.  Seven independent arms change only
matched-instance supervision:

* ``roi_bce`` -- BCE inside an 8-pixel dilation of gold-or-prediction;
* ``pos_weight`` -- per-instance area-balanced positive BCE;
* ``tversky`` -- Tversky(alpha=.3, beta=.7) in place of Dice;
* ``separation_025/050/100`` -- unique query ownership on gold pixels;
* ``anti_fragment`` -- soft-opening residual on each matched query.

The main metric is instance F1 at IoU>=.50.  Every validation record retains
the factory and dev-selected workpoints and adds fixed gold-area-bin recall and
score-agnostic query-mask oracle recall.  Only seed 42 is admitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOINT_PATH = ROOT / "scripts/445_revision_joint_head_probe.py"
BASELINE_PATH = ROOT / "scripts/422_revision_paper_baselines.py"
TYPEPILOT_PATH = ROOT / "scripts/418_revision_typepilot300_factorial_seed42.py"
FUSION_PATH = ROOT / "scripts/478_revision_instance_preserving_fusion_seed42.py"
FULLRES_PATH = ROOT / "scripts/480_revision_fullres_instance_seed42.py"
SCHEMA = "hme-revision-size-separation-seed42-v1"

SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236
EXPECTED_AREA_BIN_COUNTS = (75, 73, 88)
BASELINE_OUTPUT = \
    "work/revision_fullres_instance_seed42_v1/fullres_dup_off"

ARMS = (
    "roi_bce",
    "pos_weight",
    "tversky",
    "separation_025",
    "separation_050",
    "separation_100",
    "anti_fragment",
)
SEPARATION_WEIGHTS = {
    "separation_025": 0.25,
    "separation_050": 0.50,
    "separation_100": 1.00,
}
AREA_BINS = (
    ("lt_300", 0, 300),
    ("300_to_900", 300, 901),
    ("gt_900", 901, None),
)


class SizeSeparation481Error(RuntimeError):
    """Raised when experiment 481's locked protocol is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SizeSeparation481Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def dice_region_loss(probability: Any, gold: Any) -> Any:
    dimensions = tuple(range(1, probability.ndim))
    numerator = 2.0 * (probability * gold).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + gold.sum(dimensions) + 1.0
    return (1.0 - numerator / denominator).mean()


def roi_support(probability: Any, gold: Any, dilation: int) -> Any:
    """Detached union of gold/prediction, dilated by exactly ``dilation``."""
    from torch.nn import functional

    binary = torch_maximum(
        (probability.detach() >= 0.5).to(gold.dtype),
        (gold >= 0.5).to(gold.dtype))
    kernel = 2 * dilation + 1
    return functional.max_pool2d(
        binary[:, None], kernel, stride=1, padding=dilation)[:, 0]


def torch_maximum(left: Any, right: Any) -> Any:
    import torch

    return torch.maximum(left, right)


def roi_bce_loss(selected: Any, gold: Any, dilation: int) -> Any:
    from torch.nn import functional

    probability = selected.sigmoid()
    support = roi_support(probability, gold, dilation)
    pixel = functional.binary_cross_entropy_with_logits(
        selected, gold, reduction="none")
    numerator = (pixel * support).flatten(1).sum(1)
    denominator = support.flatten(1).sum(1).clamp_min(1.0)
    return (numerator / denominator).mean()


def positive_weighted_bce_loss(selected: Any, gold: Any) -> Any:
    """Balance foreground and background independently for every instance."""
    from torch.nn import functional

    dimensions = tuple(range(1, gold.ndim))
    positive = gold.sum(dimensions).clamp_min(1.0)
    total = float(gold.shape[-2] * gold.shape[-1])
    weight = ((total - positive) / positive).clamp_min(1.0)
    positive_term = functional.softplus(-selected) * gold \
        * weight[:, None, None]
    negative_term = functional.softplus(selected) * (1.0 - gold)
    return (positive_term + negative_term).mean(dimensions).mean()


def tversky_loss(probability: Any, gold: Any,
                 alpha: float = 0.3, beta: float = 0.7) -> Any:
    dimensions = tuple(range(1, gold.ndim))
    true_positive = (probability * gold).sum(dimensions)
    false_positive = (probability * (1.0 - gold)).sum(dimensions)
    false_negative = ((1.0 - probability) * gold).sum(dimensions)
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive
        + beta * false_negative + 1.0)
    return (1.0 - score).mean()


def separation_loss(selected: Any, gold: Any) -> Any:
    """Make each gold pixel belong to its Hungarian-matched query uniquely."""
    import torch
    from torch.nn import functional

    if selected.shape[0] == 0:
        return selected.sum() * 0.0
    background = torch.zeros_like(selected[:1])
    class_logits = torch.cat((selected, background), dim=0)[None]
    labels = torch.full(
        gold.shape[-2:], -100, dtype=torch.long, device=gold.device)
    for index, target in enumerate(gold):
        labels[target >= 0.5] = index
    if not bool((labels >= 0).any()):
        return selected.sum() * 0.0
    return functional.cross_entropy(
        class_logits, labels[None], ignore_index=-100)


def anti_fragment_loss(selected: Any) -> Any:
    """Penalise query probability removed by a differentiable 3x3 opening."""
    from torch.nn import functional

    probability = selected.sigmoid()[:, None]
    eroded = -functional.max_pool2d(-probability, 3, stride=1, padding=1)
    opened = functional.max_pool2d(eroded, 3, stride=1, padding=1)
    residual = (probability - opened).abs().flatten(1).sum(1)
    support = probability.flatten(1).sum(1).clamp_min(1.0)
    return (residual / support).mean()


def matched_loss(selected: Any, gold: Any, arm: str,
                 roi_dilation: int) -> Any:
    from torch.nn import functional

    probability = selected.sigmoid()
    if arm == "roi_bce":
        bce = roi_bce_loss(selected, gold, roi_dilation)
        return bce + dice_region_loss(probability, gold)
    if arm == "pos_weight":
        bce = positive_weighted_bce_loss(selected, gold)
        return bce + dice_region_loss(probability, gold)
    bce = functional.binary_cross_entropy_with_logits(selected, gold)
    if arm == "tversky":
        return bce + tversky_loss(probability, gold)
    return bce + dice_region_loss(probability, gold)


def instance_losses(probe: Any, mask_logits: Any, objectness: Any,
                    batch_targets: Sequence[Sequence[Any]], arm: str,
                    roi_dilation: int) -> tuple[Any, Any, Any, Any]:
    import torch
    from torch.nn import functional

    mask_total = torch.zeros((), device=mask_logits.device)
    object_total = torch.zeros((), device=mask_logits.device)
    separation_total = torch.zeros((), device=mask_logits.device)
    fragment_total = torch.zeros((), device=mask_logits.device)
    for index, targets in enumerate(batch_targets):
        rows, columns = probe.hungarian_match(
            mask_logits[index], objectness[index], targets)
        label = torch.zeros(mask_logits.shape[1], device=mask_logits.device)
        if rows:
            label[torch.as_tensor(rows, device=label.device)] = 1.0
            selected = mask_logits[index][rows]
            gold = torch.stack([targets[column] for column in columns]).to(
                selected.dtype)
            mask_total = mask_total + matched_loss(
                selected, gold, arm, roi_dilation)
            if arm in SEPARATION_WEIGHTS:
                separation_total = separation_total + separation_loss(
                    selected, gold)
            if arm == "anti_fragment":
                fragment_total = fragment_total + anti_fragment_loss(selected)
        object_total = object_total + functional.binary_cross_entropy_with_logits(
            objectness[index], label)
    count = mask_logits.shape[0]
    return (mask_total / count, object_total / count,
            separation_total / count, fragment_total / count)


def greedy_matched_truths(predicted: Sequence[np.ndarray],
                          truths: Sequence[np.ndarray],
                          match_iou: float) -> set[int]:
    pairs = []
    for predicted_index, predicted_mask in enumerate(predicted):
        predicted_area = int(predicted_mask.sum())
        if predicted_area == 0:
            continue
        for truth_index, truth_mask in enumerate(truths):
            intersection = int(np.logical_and(
                predicted_mask, truth_mask).sum())
            if not intersection:
                continue
            union = predicted_area + int(truth_mask.sum()) - intersection
            iou = intersection / union
            if iou >= match_iou:
                pairs.append((iou, predicted_index, truth_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    for _iou, predicted_index, truth_index in pairs:
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
    return used_truth


def area_bin_name(area: int) -> str:
    if area < 300:
        return "lt_300"
    if area <= 900:
        return "300_to_900"
    return "gt_900"


def area_binned_recall(predicted: Sequence[Sequence[np.ndarray]],
                       truths: Sequence[Sequence[np.ndarray]],
                       match_iou: float = 0.50) -> dict[str, Any]:
    counts = {name: {"gold": 0, "true_positive": 0}
              for name, _lower, _upper in AREA_BINS}
    for row_predicted, row_truths in zip(predicted, truths):
        matched = greedy_matched_truths(row_predicted, row_truths, match_iou)
        for index, truth in enumerate(row_truths):
            bucket = area_bin_name(int(truth.sum()))
            counts[bucket]["gold"] += 1
            counts[bucket]["true_positive"] += int(index in matched)
    for name, values in counts.items():
        values["recall"] = values["true_positive"] / values["gold"] \
            if values["gold"] else 0.0
        values["definition"] = {
            "lt_300": "area < 300 px",
            "300_to_900": "300 <= area <= 900 px",
            "gt_900": "area > 900 px",
        }[name]
    return counts


def oracle_recall(masks: np.ndarray,
                  truths: Sequence[Sequence[np.ndarray]],
                  mask_threshold: float,
                  match_iou: float = 0.50) -> dict[str, Any]:
    """Per-gold best of all queries, deliberately ignoring query scores."""
    recovered = total = 0
    by_bin = {name: {"gold": 0, "recovered": 0}
              for name, _lower, _upper in AREA_BINS}
    for row_masks, row_truths in zip(masks, truths):
        binary_queries = row_masks >= mask_threshold
        query_areas = binary_queries.reshape(binary_queries.shape[0], -1).sum(1)
        for truth in row_truths:
            truth_area = int(truth.sum())
            bucket = area_bin_name(truth_area)
            total += 1
            by_bin[bucket]["gold"] += 1
            best = 0.0
            for query, query_area in zip(binary_queries, query_areas):
                intersection = int(np.logical_and(query, truth).sum())
                union = int(query_area) + truth_area - intersection
                best = max(best, intersection / union if union else 0.0)
            hit = best >= match_iou
            recovered += int(hit)
            by_bin[bucket]["recovered"] += int(hit)
    for values in by_bin.values():
        values["recall"] = values["recovered"] / values["gold"] \
            if values["gold"] else 0.0
    return {
        "gold": total,
        "recovered": recovered,
        "recall": recovered / total if total else 0.0,
        "by_gold_area": by_bin,
        "definition": "best IoU over all 16 queries for each gold; scores ignored",
        "mask_threshold": mask_threshold,
        "match_iou": match_iou,
    }


def evaluate_with_diagnostics(probe: Any, extensions: Any, union: np.ndarray,
                              masks: np.ndarray, scores: np.ndarray,
                              rows: np.ndarray, positive_count: int,
                              gold_masks: Sequence[np.ndarray],
                              args: argparse.Namespace, joint: Any,
                              fullres: Any) -> dict[str, Any]:
    result = fullres.evaluate(
        probe, extensions, union, masks, scores, rows, positive_count,
        gold_masks, args, joint)
    truth_union = np.concatenate((
        np.stack(gold_masks),
        np.zeros((union.shape[0] - positive_count, *union.shape[1:]),
                 dtype=np.float32)), axis=0)
    truths = [probe.gold_instances(row) for row in truth_union]
    selected_point = result["instance_operating_point"]["selected"]
    selected = fullres.build_predicted_instances(
        masks, scores, selected_point["mask_threshold"],
        selected_point["score_threshold"])
    binned = area_binned_recall(selected, truths)
    bin_counts = tuple(binned[name]["gold"] for name, _lower, _upper in AREA_BINS)
    if bin_counts != EXPECTED_AREA_BIN_COUNTS:
        raise SizeSeparation481Error(
            f"gold area-bin counts changed: {bin_counts}")
    result["instance_diagnostics"] = {
        "area_binned_recall_050": binned,
        "oracle_recall_050": oracle_recall(
            masks, truths, selected_point["mask_threshold"]),
        "oracle_recall_050_factory_mask_threshold": oracle_recall(
            masks, truths, args.mask_threshold),
    }
    return result


def make_loss_adapter(holder: Mapping[str, Any], args: argparse.Namespace,
                      fusion: Any):
    from torch.nn import functional

    def loss(probe: Any, mask_logits: Any, objectness: Any,
             batch_targets: Sequence[Sequence[Any]]) -> tuple[Any, Any]:
        model = holder["model"]
        mask_part, objectness_part, separate, fragment = instance_losses(
            probe, mask_logits, model.last_raw_objectness, batch_targets,
            args.optimization_arm, args.roi_dilation)
        dense_logits = functional.interpolate(
            model.last_dense_logits, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        fused_logits = functional.interpolate(
            model.last_fused_logits, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        query_union = functional.interpolate(
            model.last_query_union, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        target = fusion.union_targets(batch_targets, dense_logits)
        dense_part = fusion.binary_mask_loss(dense_logits, target)
        fused_part = fusion.binary_mask_loss(fused_logits, target)
        consistency = fusion.asymmetric_better_branch_consistency(
            dense_logits.sigmoid(), query_union, target)
        quality = fusion.quality_target_loss(
            probe, mask_logits, model.last_raw_objectness,
            model.last_quality_logits, batch_targets)
        separation_weight = SEPARATION_WEIGHTS.get(args.optimization_arm, 0.0)
        fragment_weight = args.anti_fragment_weight \
            if args.optimization_arm == "anti_fragment" else 0.0
        holder["last_auxiliary_losses"] = {
            "anti_fragment": float(fragment.detach()),
            "consistency": float(consistency.detach()),
            "dense": float(dense_part.detach()),
            "fused": float(fused_part.detach()),
            "quality": float(quality.detach()),
            "separation": float(separate.detach()),
        }
        return (
            mask_part
            + args.dense_loss_weight * dense_part
            + args.fused_loss_weight * fused_part
            + args.consistency_weight * consistency
            + args.quality_weight * quality
            + separation_weight * separate
            + fragment_weight * fragment,
            objectness_part,
        )

    return loss


def parser(joint: Any, baseline: Any, typepilot: Any,
           argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--optimization-arm", choices=ARMS, required=True)
    base.add_argument("--roi-dilation", type=int, default=8)
    base.add_argument("--anti-fragment-weight", type=float, default=0.10)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    base.add_argument("--quality-weight", type=float, default=0.5)
    arguments = list(sys.argv[1:] if argv is None else argv)
    forbidden = ("--arm", "--pixel-decoder", "--duplicate-mode", "--mask-loss")
    if any(any(value == flag or value.startswith(flag + "=")
               for flag in forbidden) for value in arguments):
        raise SizeSeparation481Error(
            "experiment 481 locks the experiment-480 architecture")
    args = base.parse_args([*arguments, "--arm", "instance"])
    args.pixel_decoder = "fullres"
    args.duplicate_mode = "off"
    args.mask_loss = "dice"
    args.cldice_weight = 0.20
    args.duplicate_weight = 0.0
    return args


def finalise(engine_output: Path, final_output: Path,
             args: argparse.Namespace, holder: Mapping[str, Any],
             dependencies: Mapping[str, Any]) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise SizeSeparation481Error(
            f"create-only output exists: {final_output}")
    for name in ("SHA256SUMS", "complete.json"):
        path = engine_output / name
        if path.exists():
            path.unlink()
    summary_path = engine_output / "summary.json"
    manifest_path = engine_output / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = summary["per_seed"][0]["validation"]
    if int(validation["gold_instances"]) != EXPECTED_GOLD_INSTANCES:
        raise SizeSeparation481Error("development gold-instance count changed")
    if "instance_diagnostics" not in validation:
        raise SizeSeparation481Error("required instance diagnostics are absent")
    architecture = dict(holder["configuration"])
    architecture.update({
        "instance_optimization_arm": args.optimization_arm,
        "roi_dilation": args.roi_dilation if args.optimization_arm == "roi_bce"
            else None,
        "separation_weight": SEPARATION_WEIGHTS.get(args.optimization_arm, 0.0),
        "anti_fragment_weight": args.anti_fragment_weight
            if args.optimization_arm == "anti_fragment" else 0.0,
    })
    protocol = {
        "baseline_output": BASELINE_OUTPUT,
        "factory_instance_workpoint": {
            "mask_threshold": args.mask_threshold,
            "score_threshold": args.score_threshold,
        },
        "frozen_test_touched": False,
        "gold_area_bins_pixels": [300, 900],
        "instance_workpoint_grid": {
            "mask_thresholds": list(dependencies["fullres"].MASK_THRESHOLDS),
            "score_thresholds": list(dependencies["fullres"].SCORE_THRESHOLDS),
        },
        "m6_or_reference_materialised": False,
        "optimization_arm": args.optimization_arm,
        "oracle_definition": "per-gold best IoU among 16 query masks; scores ignored",
        "primary_metric": "instance_f1_match_iou_0.50",
        "selected_instance_workpoint": validation[
            "instance_operating_point"]["selected"],
        "writer_overlap_fit_validation": 0,
    }
    summary.update({
        "architecture": architecture,
        "method": "size_balanced_separation_instance_loss",
        "schema_version": SCHEMA,
    })
    summary["protocol"].update(protocol)
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "architecture": architecture,
        "dependencies": {name: module.__source_sha256__
                         for name, module in dependencies.items()},
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
    })
    manifest["protocol"] = summary["protocol"]
    manifest_path.write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    import torch

    torch.save({
        "architecture": architecture,
        "schema_version": "hme-revision-size-separation-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"size_separation_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "optimization_arm": args.optimization_arm,
        "seed": args.seed,
        "status": "complete",
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for481_size")
    baseline = load_module(BASELINE_PATH, "hme422_for481_size")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for481_size")
    fusion = load_module(FUSION_PATH, "hme478_for481_size")
    fullres = load_module(FULLRES_PATH, "hme480_for481_size")
    args = parser(joint, baseline, typepilot, argv)
    if args.seed != SEED:
        raise SystemExit("481 loss probe refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"481 loss probe refused: output already exists: {final_output}")

    holder: dict[str, Any] = {}
    joint.build_model = lambda _baseline, _ablation, run_args: \
        fullres.build_model(baseline, fusion, run_args, holder)
    joint.instance_loss = make_loss_adapter(holder, args, fusion)
    joint.predict = fullres.predict
    joint.evaluate = lambda probe, extensions, union, masks, scores, rows, \
            positive_count, gold_masks, run_args: evaluate_with_diagnostics(
                probe, extensions, union, masks, scores, rows,
                positive_count, gold_masks, run_args, joint, fullres)
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("481 loss probe refused: sample counts changed")
        print(json.dumps({
            "counts": result["counts"],
            "frozen_test_touched": False,
            "optimization_arm": args.optimization_arm,
            "seed": args.seed,
            "validated": True,
        }, sort_keys=True))
        return 0

    final_output.parent.mkdir(parents=True, exist_ok=True)
    engine_output = final_output.parent / (
        f".{final_output.name}.engine-{uuid.uuid4().hex}")
    args.output_dir = engine_output
    try:
        joint.run(args, baseline, typepilot)
        finalise(engine_output, final_output, args, holder, {
            "baseline": baseline,
            "fullres": fullres,
            "fusion": fusion,
            "joint": joint,
            "typepilot": typepilot,
        })
    except BaseException:
        if engine_output.is_dir():
            shutil.rmtree(engine_output)
        raise
    validation = json.loads((final_output / "summary.json").read_text(
        encoding="utf-8"))["per_seed"][0]["validation"]
    diagnostics = validation["instance_diagnostics"]
    print(json.dumps({
        "area_binned_recall_050": diagnostics["area_binned_recall_050"],
        "instance_f1_050_factory":
            validation["instance_factory"]["0.50"]["f1"],
        "instance_f1_050_selected":
            validation["instance_selected"]["0.50"]["f1"],
        "optimization_arm": args.optimization_arm,
        "oracle_recall_050": diagnostics["oracle_recall_050"]["recall"],
        "output_dir": str(final_output),
        "pixel_iou": validation["pixel_iou"],
        "seed": args.seed,
        "selected_workpoint":
            validation["instance_operating_point"]["selected"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

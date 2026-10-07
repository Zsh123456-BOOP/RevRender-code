#!/usr/bin/env python3
"""Diagnose and strengthen the two-type revision pixel architecture.

The rare human ``overwrite`` type is merged into the dense-region target.  The
formal development-only arms are:

``oracle_route``
    No-training diagnostic of the current learned router versus a gold-type
    oracle router.  Oracle numbers are diagnostic only and never selectable.
``lomix_deep_supervision``
    Reuse all UNet++ nested full-resolution decoder nodes, supervise every
    output, and learn a small global mixture.
``fullres_dual_decoder``
    Add a full-resolution stroke stem and genuinely separate line/dense
    decoders.  A background/line/dense router is supervised on every pixel.

All arms preserve the writer-disjoint cleaned development split and never
materialise the frozen M6/reference evaluation set.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TYPE410_PATH = ROOT / "scripts/410_revision_type_topology_ablation_seed42.py"
TYPE410_SHA256 = "e2d018f3d54204511a5abeb956f7a081e81309495fe33426ec4a2eb2ffd9fb32"
SCHEMA = "hme-revision-two-type-architecture-seed42-dev-v1"
SEED = 42
VARIANTS = (
    "oracle_route", "lomix_deep_supervision", "fullres_dual_decoder")


class Experiment416Error(RuntimeError):
    """Raised when a provenance, split, architecture, or output guard changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str):
    actual = digest_file(path)
    if actual != expected_sha256:
        raise Experiment416Error(
            f"dependency digest mismatch for {path.name}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Experiment416Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def merge_human_types(type_masks: np.ndarray) -> np.ndarray:
    """Return [line, dense] with overwrite merged into the dense target."""
    value = np.asarray(type_masks)
    if value.ndim < 3 or value.shape[-3] != 3:
        raise Experiment416Error("expected line/blot/overwrite type channels")
    line = value[..., 0, :, :]
    dense = np.maximum(value[..., 1, :, :], value[..., 2, :, :])
    return np.stack((line, dense), axis=-3)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment416Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment416Error(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment416Error(f"invalid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise Experiment416Error(f"JSONL contains a non-object: {path}")
    return rows


def package_mark_rows(key_path: Path, returned_path: Path,
                      source_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_path = Path(key_path).expanduser().resolve()
    returned_path = Path(returned_path).expanduser().resolve()
    keys = read_jsonl(key_path)
    returned = read_json(returned_path)
    records = returned.get("records")
    if not isinstance(records, dict):
        raise Experiment416Error(f"annotation records are absent: {returned_path}")
    rows = []
    for key in keys:
        record_id = str(key["id"])
        record = records.get(record_id)
        if not isinstance(record, dict):
            continue
        rows.append({
            "flag": record.get("flag"),
            "group_uuid": str(key["group_uuid"]),
            "image_id": str(key["image_id"]),
            "line_id": str(key["line_id"]),
            "marks": [dict(mark) for mark in (record.get("marks") or [])],
            "source": source_name,
        })
    return rows, {
        "key": str(key_path), "key_sha256": digest_file(key_path),
        "records_loaded": len(rows), "returned": str(returned_path),
        "returned_sha256": digest_file(returned_path),
    }


def union_type_masks(type_masks: np.ndarray) -> np.ndarray:
    value = np.asarray(type_masks)
    if value.ndim != 3 or value.shape[0] != 3:
        raise Experiment416Error("type masks must be 3 x H x W")
    return np.any(value > 0, axis=0)


def reconcile_type_masks(type_masks: np.ndarray,
                         gold: np.ndarray) -> tuple[np.ndarray, int]:
    stack = np.asarray(type_masks, dtype=np.uint8).copy()
    truth = np.asarray(gold, dtype=bool)
    before = int(np.count_nonzero(union_type_masks(stack) != truth))
    extra = union_type_masks(stack) & ~truth
    stack[:, extra] = 0
    missing = truth & ~union_type_masks(stack)
    if bool(missing.any()):
        coordinates = [np.argwhere(channel > 0) for channel in stack]
        for y, x in np.argwhere(missing):
            distances = [
                float(np.min((points[:, 0] - y) ** 2
                             + (points[:, 1] - x) ** 2))
                if len(points) else math.inf
                for points in coordinates]
            stack[int(np.argmin(distances)), y, x] = 255
    if np.any(union_type_masks(stack) != truth):
        raise Experiment416Error("type-mask reconciliation failed")
    return stack, before


def attach_human_type_masks(data: dict[str, Any],
                            args: argparse.Namespace) -> dict[str, Any]:
    """Re-render the same audited human geometry used by the prior 415 run."""
    base = data["base"]
    run_args = data["run_args"]
    package_specs = (
        (Path(run_args.new_key), Path(run_args.new_returned), "new1000"),
        (Path(run_args.old_key), Path(run_args.old_returned), "old300"),
        (Path(run_args.extra_key), Path(run_args.extra_returned), "gold300"),
        (Path(run_args.active_key), Path(run_args.active_returned), "active200"),
        (ROOT / "outputs/revision_region_eval_annotation_v1/key.jsonl",
         ROOT / "work/revision_region_eval_returned_v2/revision_region_annotation.json",
         "held_mask_validation"),
        (Path(args.active_key), Path(args.active_returned), "active300"),
    )
    raw_rows: list[dict[str, Any]] = []
    package_audit = []
    for key_path, returned_path, source_name in package_specs:
        rows, audit = package_mark_rows(key_path, returned_path, source_name)
        raw_rows.extend(rows)
        package_audit.append(audit)
    raw_lookup: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in raw_rows:
        line_id = str(row["line_id"])
        if line_id in raw_lookup:
            duplicate_ids.add(line_id)
        raw_lookup[line_id] = row
    target_rows = data["fit_positive"] + data["validation_positive"]
    target_ids = {str(row["line_id"]) for row in target_rows}
    if len(target_ids) != len(target_rows) or not target_ids <= set(raw_lookup):
        raise Experiment416Error("materialised positives lack raw human marks")

    recipe = base.RECIPES[run_args.recipe]
    images_dir = Path(run_args.crop_images_dir)
    segmenter = base.load_module(
        base.SEGMENTER_SCRIPT, "hme344_for416_types", base.SEGMENTER_SHA256)
    _catalogue_rows, catalogue = base.load_catalogue(segmenter)
    base_protocol = segmenter.load_region().load_matched().load_gap().load_base()
    masks_by_type: dict[str, dict[str, np.ndarray]] = {}
    clipped_by_type: dict[str, int] = {}
    for mark_type in ("line", "blot", "overwrite"):
        typed_rows = []
        for line_id in sorted(target_ids):
            raw = raw_lookup[line_id]
            typed_rows.append({
                **raw,
                "marks": [dict(mark) for mark in raw["marks"]
                          if str(mark["type"]) == mark_type],
            })
        rendered, clipped = base.materialise(
            typed_rows, segmenter, base_protocol, catalogue,
            images_dir, recipe["crop"])
        masks_by_type[mark_type] = {
            str(row["line_id"]): np.asarray(row["mask"], dtype=np.uint8)
            for row in rendered}
        clipped_by_type[mark_type] = int(clipped)

    pixel_counts = collections.Counter()
    row_counts = collections.Counter()
    mismatch_pixels = 0
    for row in target_rows:
        line_id = str(row["line_id"])
        stack = np.stack([masks_by_type[name][line_id]
                          for name in ("line", "blot", "overwrite")])
        stack, difference = reconcile_type_masks(
            stack, np.asarray(row["mask"] > 0, dtype=bool))
        row["human_type_masks"] = stack
        mismatch_pixels += difference
        for index, name in enumerate(("line", "blot", "overwrite")):
            pixels = int(np.count_nonzero(stack[index]))
            pixel_counts[name] += pixels
            row_counts[name] += int(pixels > 0)
    zero = np.zeros((3, *target_rows[0]["mask"].shape), dtype=np.uint8)
    for row in data["fit_clean"] + data["validation_clean"]:
        row["human_type_masks"] = zero.copy()
    mark_counts = collections.Counter(
        str(mark["type"]) for line_id in target_ids
        for mark in raw_lookup[line_id]["marks"])
    if set(mark_counts) - {"line", "blot", "overwrite"}:
        raise Experiment416Error("human mark types changed")
    return {
        "clipped_coordinate_values": clipped_by_type,
        "duplicate_package_line_ids": len(duplicate_ids),
        "mark_counts": dict(sorted(mark_counts.items())),
        "packages": package_audit,
        "pixel_counts": dict(sorted(pixel_counts.items())),
        "row_counts": dict(sorted(row_counts.items())),
        "target_rows": len(target_rows),
        "union_mismatch_pixels_before_reconciliation": mismatch_pixels,
        "union_mismatch_pixels_after_reconciliation": 0,
    }


def augment_human_types(extension: Any, row: Mapping[str, Any],
                        rng: random.Random):
    initial_state = rng.getstate()
    typed = np.asarray(row["human_type_masks"], dtype=np.float32) / 255.0
    image, mask, skeleton, first = extension.paired_affine_with_aux(
        row["image"], row["mask"], row["line_skeleton"], typed[0], rng)
    final_state = rng.getstate()
    channels = [first]
    for channel in typed[1:]:
        rng.setstate(initial_state)
        _image, _mask, _skeleton, transformed = \
            extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], channel, rng)
        channels.append(transformed)
    rng.setstate(final_state)
    return image, mask, skeleton, np.stack(channels)


def two_type_router_targets(type_target: Any):
    """Create background=0, line=1, dense=2 labels for every pixel."""
    import torch

    if type_target.ndim != 4 or type_target.shape[1] != 2:
        raise Experiment416Error("two-type tensor must be Bx2xHxW")
    labels = torch.zeros(
        type_target.shape[0], *type_target.shape[2:], dtype=torch.long,
        device=type_target.device)
    labels[type_target[:, 0] > 0.5] = 1
    labels[type_target[:, 1] > 0.5] = 2
    return labels


def dense_area_boundary_loss(logits: Any, target: Any):
    """Penalise dense-region area drift and boundary disagreement."""
    import torch

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, target.ndim))
    area = torch.abs(probability.mean(dimensions) - target.mean(dimensions)).mean()

    def gradients(value):
        horizontal = torch.abs(value[..., :, 1:] - value[..., :, :-1])
        vertical = torch.abs(value[..., 1:, :] - value[..., :-1, :])
        return horizontal, vertical

    pred_h, pred_v = gradients(probability)
    true_h, true_v = gradients(target)
    boundary = torch.nn.functional.l1_loss(pred_h, true_h) \
        + torch.nn.functional.l1_loss(pred_v, true_v)
    return area + boundary


class _Tap:
    def __init__(self, core: Any):
        self.values: dict[str, Any] = {}
        self.handles = []
        for name in ("x0_1", "x0_2", "x0_3", "x0_4"):
            module = getattr(core, name, None)
            if module is None:
                raise Experiment416Error(f"UNet++ node absent: {name}")
            self.handles.append(module.register_forward_hook(
                self._capture(name)))

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self.values[name] = output
        return hook


def make_lomix_model(ablation: Any, weights_path: Path,
                     initial_state: Mapping[str, Any]):
    import torch
    from torch import nn
    from torch.nn import functional

    core = ablation.make_unetpp_resnet34(weights_path)
    core.load_state_dict(initial_state, strict=True)

    class LoMixUNetPP(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.tap = _Tap(core)
            self.aux_heads = nn.ModuleList([nn.Conv2d(64, 1, 1)
                                            for _ in range(3)])
            for head in self.aux_heads:
                head.load_state_dict(core.head.state_dict())
            self.mix_logits = nn.Parameter(torch.tensor([-2.0, -1.0, 0.0, 3.0]))

        def forward_training(self, gray):
            deepest = self.core(gray)
            auxiliaries = [functional.interpolate(
                head(self.tap.values[f"x0_{index}"]),
                size=gray.shape[-2:], mode="bilinear", align_corners=False)
                for index, head in zip((1, 2, 3), self.aux_heads)]
            logits = auxiliaries + [deepest]
            weights = torch.softmax(self.mix_logits, dim=0)
            final = sum(weight * value for weight, value in zip(weights, logits))
            return final, {"aux_logits": logits, "mix_weights": weights}

        def forward(self, gray):
            return self.forward_training(gray)[0]

    return LoMixUNetPP()


def make_fullres_dual_model(ablation: Any, weights_path: Path,
                            initial_state: Mapping[str, Any]):
    import torch
    from torch import nn
    from torch.nn import functional

    core = ablation.make_unetpp_resnet34(weights_path)
    core.load_state_dict(initial_state, strict=True)

    class ExpertDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.half_path = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.GroupNorm(16, 128), nn.GELU(),
                nn.Conv2d(128, 96, 3, padding=1, bias=False),
                nn.GroupNorm(12, 96), nn.GELU())
            self.full = nn.Sequential(
                nn.Conv2d(128, 96, 3, padding=1, bias=False),
                nn.GroupNorm(12, 96), nn.GELU(),
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64), nn.GELU())
            self.delta = nn.Conv2d(64, 1, 3, padding=1)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

        def forward(self, multiscale, stroke, size):
            half = self.half_path(multiscale)
            full = functional.interpolate(
                half, size=size, mode="bilinear", align_corners=False)
            feature = self.full(torch.cat((full, stroke), dim=1))
            return self.delta(feature), feature

    class FullResolutionDualDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.tap = _Tap(core)
            self.stroke_stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32), nn.GELU())
            self.line_decoder = ExpertDecoder()
            self.dense_decoder = ExpertDecoder()
            self.router = nn.Sequential(
                nn.Conv2d(160, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 3, 1))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

        def forward_training(self, gray):
            base = self.core(gray)
            reference = self.tap.values["x0_4"]
            multiscale = torch.cat([
                functional.interpolate(
                    self.tap.values[f"x0_{index}"],
                    size=reference.shape[-2:], mode="bilinear",
                    align_corners=False)
                for index in (1, 2, 3, 4)], dim=1)
            stroke = self.stroke_stem(gray)
            line_delta, line_feature = self.line_decoder(
                multiscale, stroke, gray.shape[-2:])
            dense_delta, dense_feature = self.dense_decoder(
                multiscale, stroke, gray.shape[-2:])
            line_logits = base + line_delta
            dense_logits = base + dense_delta
            router_logits = self.router(torch.cat(
                (stroke, line_feature, dense_feature), dim=1))
            weights = torch.softmax(router_logits, dim=1)
            line_gate = 0.25 + 0.75 * weights[:, 1:2]
            dense_gate = 0.25 + 0.75 * weights[:, 2:3]
            line_probability = torch.sigmoid(line_logits) * line_gate
            dense_probability = torch.sigmoid(dense_logits) * dense_gate
            probability = 1.0 - (1.0 - line_probability) \
                * (1.0 - dense_probability)
            final = torch.logit(probability.clamp(1e-5, 1.0 - 1e-5))
            return final, {
                "dense_logits": dense_logits,
                "line_logits": line_logits,
                "router_logits": router_logits,
            }

        def forward(self, gray):
            return self.forward_training(gray)[0]

    return FullResolutionDualDecoder()


def configure_trainable(model: Any) -> dict[str, int]:
    for parameter in model.core.parameters():
        parameter.requires_grad = False
    for name, parameter in model.core.named_parameters():
        if not name.startswith("encoder."):
            parameter.requires_grad = True
    for parameter in model.core.encoder.layer4.parameters():
        parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if not name.startswith("core."):
            parameter.requires_grad = True
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad),
    }


def keep_frozen_encoder_eval(model: Any) -> None:
    model.core.encoder.eval()
    model.core.encoder.layer4.train()


def predict_model(model: Any, rows: Sequence[Mapping[str, Any]],
                  batch_size: int, device: Any):
    import torch

    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            gray = torch.from_numpy(np.stack([
                row["image"] for row in batch
            ]).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
            probabilities.append(torch.sigmoid(model(gray)).cpu().numpy()[:, 0])
    return np.concatenate(probabilities), np.stack([
        row["mask"] for row in rows]).astype(np.float32) / 255.0


def per_type_component_coverage(probabilities: np.ndarray,
                                targets: np.ndarray,
                                threshold: float) -> dict[str, Any]:
    import cv2

    predicted = np.asarray(probabilities) >= float(threshold)
    output = {}
    for index, name in enumerate(("line", "dense")):
        values = []
        for mask, target in zip(predicted, targets[:, index] > 0):
            count, labels = cv2.connectedComponents(
                np.ascontiguousarray(target, dtype=np.uint8), connectivity=8)
            for component in range(1, count):
                values.append(float(mask[labels == component].mean()))
        array = np.asarray(values, dtype=np.float32)
        output[name] = {
            "components": int(len(array)),
            "covered_ge_10": int(np.sum(array >= 0.10)),
            "covered_ge_50": int(np.sum(array >= 0.50)),
            "median_coverage": float(np.median(array)) if len(array) else None,
        }
    return output


def prepare_data(args: argparse.Namespace):
    type410 = load_module(TYPE410_PATH, "hme410_for416", TYPE410_SHA256)
    active402 = type410.load_module(
        type410.ACTIVE402_PATH, "hme402_for416", type410.ACTIVE402_SHA256)
    scorer390 = active402.load_module(
        active402.SCORER390_PATH, "hme390_for416", active402.SCORER390_SHA256)
    coverage = scorer390.load_module(
        scorer390.COVERAGE_PATH, "hme389_for416", scorer390.COVERAGE_SHA256)
    source = active402.verify_source(args.data_source_output)
    data = active402.prepare_data(
        scorer390, coverage, source, args,
        active402.VARIANTS["standard_neg1_natural_lr1"])
    exclusions = type410.load_exclusions(args.exclusion_manifest)
    cleaning = type410.filter_scan_corners(data, exclusions)
    data.update({"active402": active402, "coverage": coverage})
    type_audit = attach_human_type_masks(data, args)
    initial = type410.load_initial_checkpoint(args.initial_output)
    if data["audit"]["writer_overlap_fit_validation"] != 0:
        raise Experiment416Error("writer isolation changed")
    return type410, data, source, exclusions, cleaning, type_audit, initial


def train_arm(type410: Any, data: Mapping[str, Any],
              initial: Mapping[str, Any], args: argparse.Namespace,
              staging: Path) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if args.variant == "lomix_deep_supervision":
        model = make_lomix_model(
            data["ablation"], data["run_args"].resnet34_weights,
            initial["checkpoint"]["state_dict"])
    else:
        model = make_fullres_dual_model(
            data["ablation"], data["run_args"].resnet34_weights,
            initial["checkpoint"]["state_dict"])
    model = model.to(device)
    parameter_count = configure_trainable(model)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=data["run_args"].weight_decay)
    rng = random.Random(SEED)
    partitions = {
        "active_positive": [row for row in data["fit_positive"] if row["active300"]],
        "replay_positive": [row for row in data["fit_positive"] if not row["active300"]],
        "active_clean": [row for row in data["fit_clean"] if row["active300"]],
        "clean": [row for row in data["fit_clean"] if not row["active300"]],
    }
    cycles = {name: data["base"].ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    batch_config = {"active_positive_per_batch": 1, "negative_ratio": 1}
    initial_positive, initial_target = predict_model(
        model, data["validation_positive"], 12, device)
    initial_clean, _ = predict_model(model, data["validation_clean"], 12, device)
    initial_coverage = type410.cleaned_coverage_operating_point(
        data["coverage"], initial_positive, initial_target > 0.5, initial_clean)
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
            keep_frozen_encoder_eval(model)
            rows = data["active402"].training_batch(
                data["base"], rng, batch_config, cycles, step)
            images, masks, skeletons, types = [], [], [], []
            for row in rows:
                image, mask, skeleton, typed = augment_human_types(
                    data["extension"], row, rng)
                images.append(image)
                masks.append(mask)
                skeletons.append(skeleton)
                types.append(merge_human_types(typed))
            gray = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            target = torch.from_numpy(
                np.stack(masks).astype(np.float32)[:, None] / 255.0
            ).to(device)
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0
            ).to(device)
            type_target = torch.from_numpy(
                np.stack(types).astype(np.float32)).to(device)
            logits, auxiliary = model.forward_training(gray)
            standard, skeleton_loss, neighbourhood = \
                data["corrective"].standard_loss_terms(
                    data["extension"], logits, target, skeleton,
                    data["run_args"])
            if args.variant == "lomix_deep_supervision":
                auxiliary_loss = sum(type410.balanced_dice_bce(value, target)
                                     for value in auxiliary["aux_logits"]) \
                    / len(auxiliary["aux_logits"])
                loss = standard + 0.35 * skeleton_loss + 0.15 * auxiliary_loss
                last_losses = {
                    "auxiliary": float(auxiliary_loss.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                    "skeleton": float(skeleton_loss.detach().cpu()),
                    "standard": float(standard.detach().cpu()),
                }
            else:
                line_target = type_target[:, 0:1]
                dense_target = type_target[:, 1:2]
                expert_loss = (
                    type410.balanced_dice_bce(
                        auxiliary["line_logits"], line_target)
                    + type410.balanced_dice_bce(
                        auxiliary["dense_logits"], dense_target)) / 2.0
                class_weights = torch.tensor(
                    [0.05, 1.0, 1.0], dtype=logits.dtype, device=device)
                router_loss = functional.cross_entropy(
                    auxiliary["router_logits"],
                    two_type_router_targets(type_target), weight=class_weights)
                line_scnp = data["extension"].isolated_false_positive_proxy_loss(
                    auxiliary["line_logits"], line_target,
                    data["run_args"].neighbourhood_kernel)
                dense_shape = dense_area_boundary_loss(
                    auxiliary["dense_logits"], dense_target)
                loss = standard + 0.35 * skeleton_loss + 0.15 * expert_loss \
                    + 0.05 * router_loss + 0.10 * line_scnp \
                    + 0.05 * dense_shape
                last_losses = {
                    "dense_shape": float(dense_shape.detach().cpu()),
                    "expert": float(expert_loss.detach().cpu()),
                    "line_scnp": float(line_scnp.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                    "router": float(router_loss.detach().cpu()),
                    "skeleton": float(skeleton_loss.detach().cpu()),
                    "standard": float(standard.detach().cpu()),
                }
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad], 5.0)
            optimiser.step()
            if step % args.validation_interval and step != args.optimizer_steps:
                continue
            positive_probability, positive_target = predict_model(
                model, data["validation_positive"], 12, device)
            clean_probability, _ = predict_model(
                model, data["validation_clean"], 12, device)
            coverage_point = type410.cleaned_coverage_operating_point(
                data["coverage"], positive_probability,
                positive_target > 0.5, clean_probability)
            points = data["base"].choose_operating_points(
                positive_probability, positive_target, clean_probability,
                data["run_args"].max_clean_fp,
                data["base"].scaled_min_fire_pixels(
                    data["run_args"].recipe,
                    data["run_args"].min_fire_pixels))
            key = data["coverage"].coverage_key(coverage_point)
            if step >= args.selection_warmup_steps \
                    and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_coverage, best_points = coverage_point, points
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            record = {
                "best_step": best_step,
                "covered_ge_10": coverage_point["covered_ge_10_count"],
                "covered_ge_50": coverage_point["covered_ge_50_count"],
                "event": "validation",
                "median": coverage_point["median_gold_component_coverage"],
                "pixel_iou": coverage_point["pixel_iou"],
                "step": step,
                "variant": args.variant,
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
    if best_state is None or best_coverage is None or best_points is None:
        raise Experiment416Error("architecture arm produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_positive, final_target = predict_model(
        model, data["validation_positive"], 12, device)
    final_clean, _ = predict_model(model, data["validation_clean"], 12, device)
    final_coverage = type410.cleaned_coverage_operating_point(
        data["coverage"], final_positive, final_target > 0.5, final_clean)
    two_types = merge_human_types(np.stack([
        row["human_type_masks"] for row in data["validation_positive"]]))
    per_type = per_type_component_coverage(
        final_positive, two_types, final_coverage["threshold"])
    import torch
    torch.save({
        "architecture": args.variant,
        "best_step": best_step,
        "coverage_operating_point": final_coverage,
        "human_types": ["line", "dense=blot+overwrite"],
        "parameter_count": parameter_count,
        "seed": SEED,
        "source_checkpoint_sha256": initial["sha256"],
        "state_dict": model.state_dict(),
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
        "per_type_coverage": per_type,
    }


def _predict_current_dual(model: Any, rows: Sequence[Mapping[str, Any]],
                          batch_size: int, device: Any):
    import torch

    learned, line, dense, gates = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            gray = torch.from_numpy(np.stack([
                row["image"] for row in batch
            ]).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
            final, line_logits, dense_logits, gate_logits = \
                model.forward_with_aux(gray)
            learned.append(torch.sigmoid(final).cpu().numpy()[:, 0])
            line.append(torch.sigmoid(line_logits).cpu().numpy()[:, 0])
            dense.append(torch.sigmoid(dense_logits).cpu().numpy()[:, 0])
            gates.append(torch.sigmoid(gate_logits).cpu().numpy()[:, 0])
    return tuple(np.concatenate(values) for values in (learned, line, dense, gates))


def oracle_diagnostic(type410: Any, data: Mapping[str, Any],
                      initial: Mapping[str, Any], args: argparse.Namespace,
                      staging: Path) -> dict[str, Any]:
    import torch

    source = args.source_pixel_output.expanduser().resolve()
    if not (source / "segmenter_seed42.pt").is_file():
        raise Experiment416Error("current dual-expert checkpoint is absent")
    checkpoint = torch.load(
        source / "segmenter_seed42.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "type_dual_topology":
        raise Experiment416Error("oracle source is not type_dual_topology")
    device = torch.device("cuda")
    model = type410.make_model(
        data["ablation"], data["run_args"].resnet34_weights,
        initial["checkpoint"]["state_dict"], True).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    positive = _predict_current_dual(
        model, data["validation_positive"], 12, device)
    clean = _predict_current_dual(model, data["validation_clean"], 12, device)
    target = np.stack([row["mask"] for row in data["validation_positive"]]) > 0
    typed = merge_human_types(np.stack([
        row["human_type_masks"] for row in data["validation_positive"]])) > 0
    line_target, dense_target = typed[:, 0], typed[:, 1]
    oracle_positive = positive[0].copy()
    oracle_positive[line_target] = positive[1][line_target]
    oracle_positive[dense_target] = positive[2][dense_target]
    arms = {
        "learned_router": (positive[0], clean[0]),
        "line_expert_only": (positive[1], clean[1]),
        "dense_expert_only": (positive[2], clean[2]),
        "gold_type_oracle_diagnostic_only": (oracle_positive, clean[0]),
    }
    metrics = {name: type410.cleaned_coverage_operating_point(
        data["coverage"], pos, target, neg)
        for name, (pos, neg) in arms.items()}
    selected = line_target | dense_target
    gate_accuracy = float(np.mean(
        (positive[3][selected] >= 0.5) == line_target[selected]))
    result = {
        "gate_accuracy_on_gold_pixels": gate_accuracy,
        "metrics": metrics,
        "oracle_used_for_training_or_selection": False,
        "source_checkpoint_sha256": digest_file(source / "segmenter_seed42.pt"),
    }
    write_json(staging / "oracle_metrics.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS or args.seed != SEED \
            or args.optimizer_steps != 3000 \
            or args.validation_interval != 250 \
            or args.selection_warmup_steps != 500 \
            or not np.isclose(args.learning_rate, 1e-5):
        raise Experiment416Error("formal fixed-seed protocol changed")
    import torch

    if not torch.cuda.is_available():
        raise Experiment416Error("formal architecture validation requires CUDA")
    type410, data, source, exclusions, cleaning, type_audit, initial = \
        prepare_data(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment416Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        if args.variant == "oracle_route":
            result = oracle_diagnostic(type410, data, initial, args, staging)
            coverage = result["metrics"]["learned_router"]
        else:
            result = train_arm(type410, data, initial, args, staging)
            coverage = result["coverage"]
        original_key = type410.read_json(source["root"] / "validation_key.json")
        clean_key = [row for row in original_key
                     if str(row["line_id"]) not in exclusions]
        write_json(staging / "validation_key_clean.json", clean_key)
        summary = {
            "cleaning": cleaning,
            "coverage": coverage,
            "gold_components": type410.EXPECTED_CLEAN_GOLD_COMPONENTS,
            "protocol": {
                "fixed_cleaned_validation": True,
                "frozen_test_touched": False,
                "human_types": ["line", "dense=blot+overwrite"],
                "learning_rate": args.learning_rate,
                "m6_frozen_test_touched": False,
                "oracle_diagnostic_only": args.variant == "oracle_route",
                "reference_evaluation_materialised": False,
                "seed": SEED,
                "writer_overlap_fit_validation": 0,
            },
            "result": result,
            "schema_version": SCHEMA,
            "status": "complete",
            "type_targets": type_audit,
            "variant": args.variant,
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "run_manifest.json", {
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_source_checkpoint_sha256": source["checkpoint_sha256"],
            "frozen_test_touched": False,
            "initial_checkpoint_sha256": initial["sha256"],
            "reference_evaluation_materialised": False,
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "seed": SEED,
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
    parser.add_argument("--source-pixel-output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--active-key", type=Path, required=True)
    parser.add_argument("--active-returned", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer-steps", type=int, default=3000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--selection-warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    try:
        summary = run(build_parser().parse_args())
    except (Experiment416Error, RuntimeError) as exc:
        raise SystemExit(f"experiment 416 refused: {exc}") from exc
    print(json.dumps({
        "coverage": summary["coverage"], "variant": summary["variant"]
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run seed-42 revision-localiser ablations with the safe subset of 105 new gold rows.

The locked full-1000 writer-disjoint validation population is reused unchanged.
Of the 105 reviewed rows, protected images and locked-validation writers are
excluded before labels are materialised for fitting.  The remaining 80 rows are
added to the fitting pool.  Four single-factor variants are supported:

``data_only``
    Existing shallow U-Net and locked losses, with the extra safe gold only.
``unetpp_resnet34``
    ImageNet-pretrained ResNet-34 encoder with a nested U-Net++ decoder.
``boundary_head``
    Existing U-Net plus a supervised boundary auxiliary head.
``line_connectivity``
    Existing U-Net plus a line-skeleton local-min continuity loss.

The script is development-only, create-only, and fixed to random seed 42.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/354_revision_localizer_optimized.py"
EXTENSION_PATH = ROOT / "scripts/357_revision_localizer_extensions.py"
RUNNER_PATH = ROOT / "scripts/358_revision_localizer_mechanism.py"
FULL1000_PATH = ROOT / "scripts/365_revision_localizer_full1000.py"
BASE_SHA256 = "73e3582b59186f97899571f832e01f983034fed4bba2269b2feb5d57fa4e2780"
EXTENSION_SHA256 = "10e003b28c4f9c6b1de45a22c27a1051351c95150ecbedd1d25ea8110b5d91ce"
RUNNER_SHA256 = "958b27a5230216c6a20d08f095de3c73b39adf5148dec9bcb562492b24e507d0"
FULL1000_SHA256 = "55712b1c2ada5c137f75a08891ec1660a5b768061b4ee93eeab492dcb6c4d998"

SCHEMA = "hme-revision-gold105-ablation-dev-v1"
VARIANTS = ("data_only", "unetpp_resnet34", "boundary_head", "line_connectivity")
EXPECTED_EXTRA_KEY_SHA256 = (
    "411052032fa831c8f40ce4c6a44dad4b3fcad9d847a2954c22b287da63ac545b"
)
EXPECTED_EXTRA_RETURNED_SHA256 = (
    "31b8c9f39cf04104bea33fa7c7a751c66599b30dd52d54f96a84993b09c38c8d"
)
EXPECTED_EXTRA_FINGERPRINT = "b279d1e8e824041a2492498486726559"
EXPECTED_RESNET34_SHA256 = (
    "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
)
EXPECTED_EXTRA_RECORDS = 105
EXPECTED_EXTRA_QA = {
    "clean_after_sanitation": 3,
    "dropped_zero_length_lines": 0,
    "flags": {"no_revision": 3},
    "mark_types": {"blot": 56, "line": 284, "overwrite": 1},
    "pinned_test_records_not_semantically_read": 0,
    "positive_after_sanitation": 86,
    "protected_image_records_not_semantically_read": 16,
    "raw_marks": 341,
    "safe_records": 89,
    "source_records": 105,
}
EXPECTED_TRAINING_ROWS = 80
EXPECTED_TRAINING_POSITIVE = 77
EXPECTED_TRAINING_CLEAN = 3
EXPECTED_VALIDATION_WRITER_EXCLUDED = 9


class Ablation371Error(RuntimeError):
    """Raised when an ablation or data-safety invariant is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected: str):
    if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
        raise Ablation371Error(f"dependency mismatch: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Ablation371Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Ablation371Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Ablation371Error(f"JSON root must be an object: {path}")
    return value


def boundary_target(target):
    """Return a one-pixel inner/outer boundary band for a binary target."""
    import torch
    from torch.nn import functional as functional

    foreground = target >= 0.5
    dilated = functional.max_pool2d(
        foreground.float(), kernel_size=3, stride=1, padding=1) >= 0.5
    eroded = -functional.max_pool2d(
        -foreground.float(), kernel_size=3, stride=1, padding=1) >= 0.5
    return (dilated & ~eroded).to(dtype=target.dtype)


def positive_dice_bce(logits, target):
    import torch
    from torch.nn import functional as functional

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    present = target.sum(dimensions) > 0
    dice = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if bool(present.any()):
        numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
        denominator = probability.sum(dimensions) + target.sum(dimensions) + 1.0
        dice = (1.0 - numerator[present] / denominator[present]).mean()
    return functional.binary_cross_entropy_with_logits(logits, target) + dice


def line_local_min_loss(probability, skeleton, kernel_size: int = 3):
    """Emphasise the weakest predicted point in each local line-skeleton patch."""
    import torch
    from torch.nn import functional as functional

    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("connectivity kernel must be a positive odd integer")
    support = skeleton >= 0.5
    if not bool(support.any()):
        return torch.zeros((), dtype=probability.dtype, device=probability.device)
    values = torch.where(support, probability, torch.ones_like(probability))
    local_min = -functional.max_pool2d(
        -values, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (1.0 - local_min[support]).mean()


def make_boundary_model(extension: Any, in_channels: int, base_channels: int):
    import torch

    core = extension.make_unet(in_channels, base_channels)

    class BoundaryUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.boundary_head = torch.nn.Conv2d(base_channels, 1, 1)
            self._feature = None
            self._hook = self.core.c1.register_forward_hook(self._capture)

        def _capture(self, _module, _inputs, output):
            self._feature = output

        def forward(self, value):
            main = self.core(value)
            if self._feature is None:
                raise RuntimeError("boundary feature hook did not fire")
            return main, self.boundary_head(self._feature)

    return BoundaryUNet()


def make_unetpp_resnet34(weights_path: Path):
    """Return a compact nested decoder over an offline ImageNet ResNet-34."""
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as functional

    weights_path = weights_path.expanduser().resolve()
    if weights_path.is_symlink() or not weights_path.is_file():
        raise Ablation371Error("ResNet-34 weights must be a regular offline file")
    if digest_file(weights_path) != EXPECTED_RESNET34_SHA256:
        raise Ablation371Error("ResNet-34 official weight digest mismatch")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    encoder = torchvision.models.resnet34(weights=None)
    encoder.load_state_dict(state, strict=True)

    def block(channels: int, output: int):
        return nn.Sequential(
            nn.Conv2d(channels, output, 3, padding=1, bias=False),
            nn.BatchNorm2d(output), nn.ReLU(inplace=True),
            nn.Conv2d(output, output, 3, padding=1, bias=False),
            nn.BatchNorm2d(output), nn.ReLU(inplace=True),
        )

    class UNetPlusPlus(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.input_channels = 1
            self.output_channels = 1
            self.head_count = 1
            self.register_buffer("mean", torch.tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.x3_1 = block(256 + 512, 256)
            self.x2_1 = block(128 + 256, 128)
            self.x1_1 = block(64 + 128, 64)
            self.x0_1 = block(64 + 64, 64)
            self.x2_2 = block(128 * 2 + 256, 128)
            self.x1_2 = block(64 * 2 + 128, 64)
            self.x0_2 = block(64 * 3, 64)
            self.x1_3 = block(64 * 3 + 128, 64)
            self.x0_3 = block(64 * 4, 64)
            self.x0_4 = block(64 * 5, 64)
            self.head = nn.Conv2d(64, 1, 1)

        @staticmethod
        def up(value, target):
            return functional.interpolate(
                value, size=target.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, gray):
            rgb = ((gray + 1.0) * 0.5).repeat(1, 3, 1, 1)
            rgb = (rgb - self.mean) / self.std
            x0_0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(rgb)))
            x1_0 = self.encoder.layer1(self.encoder.maxpool(x0_0))
            x2_0 = self.encoder.layer2(x1_0)
            x3_0 = self.encoder.layer3(x2_0)
            x4_0 = self.encoder.layer4(x3_0)
            x3_1 = self.x3_1(torch.cat((x3_0, self.up(x4_0, x3_0)), 1))
            x2_1 = self.x2_1(torch.cat((x2_0, self.up(x3_0, x2_0)), 1))
            x1_1 = self.x1_1(torch.cat((x1_0, self.up(x2_0, x1_0)), 1))
            x0_1 = self.x0_1(torch.cat((x0_0, self.up(x1_0, x0_0)), 1))
            x2_2 = self.x2_2(torch.cat((x2_0, x2_1, self.up(x3_1, x2_0)), 1))
            x1_2 = self.x1_2(torch.cat((x1_0, x1_1, self.up(x2_1, x1_0)), 1))
            x0_2 = self.x0_2(torch.cat((x0_0, x0_1, self.up(x1_1, x0_0)), 1))
            x1_3 = self.x1_3(torch.cat((x1_0, x1_1, x1_2,
                                       self.up(x2_2, x1_0)), 1))
            x0_3 = self.x0_3(torch.cat((x0_0, x0_1, x0_2,
                                       self.up(x1_2, x0_0)), 1))
            x0_4 = self.x0_4(torch.cat((x0_0, x0_1, x0_2, x0_3,
                                       self.up(x1_3, x0_0)), 1))
            return functional.interpolate(
                self.head(x0_4), size=gray.shape[-2:], mode="bilinear",
                align_corners=False)

    return UNetPlusPlus()


def main_logits(output):
    return output[0] if isinstance(output, tuple) else output


def predict(model, rows: Sequence[dict[str, Any]], batch_size: int, device):
    import torch

    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            image = np.stack([
                row["image"] for row in rows[start:start + batch_size]
            ]).astype(np.float32)
            value = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            probabilities.append(torch.sigmoid(main_logits(model(value))).cpu().numpy()[:, 0])
    model.train()
    return np.concatenate(probabilities), np.stack([row["mask"] for row in rows]) / 255.0


def build_model(args: argparse.Namespace, extension: Any, device):
    if args.variant == "unetpp_resnet34":
        if args.resnet34_weights is None:
            raise Ablation371Error("UNet++ requires --resnet34-weights")
        model = make_unetpp_resnet34(args.resnet34_weights)
    elif args.variant == "boundary_head":
        model = make_boundary_model(extension, 1, args.base_channels)
    else:
        model = extension.make_unet(1, args.base_channels)
    return model.to(device)


def train_once(base, extension, fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path):
    import torch
    from torch import nn

    if seed != 42:
        raise Ablation371Error("all experiment-371 arms are fixed to seed 42")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Ablation371Error("formal training requires CUDA")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build_model(args, extension, device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(seed)
    positive_cycle = base.ShuffledCycle(fit_positive, rng)
    clean_cycle = base.ShuffledCycle(fit_clean, rng)
    recipe = base.RECIPES[args.recipe]
    best_key, best_step, best_state, best_points = None, -1, None, None
    start_select = int(round(args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        chunk, _, _ = base.take_balanced_batch(
            positive_cycle, clean_cycle, args.batch_size, args.negative_ratio)
        rng.shuffle(chunk)
        images, masks, skeletons = [], [], []
        for row in chunk:
            image, mask, skeleton, _aux = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], None, rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        x = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
        y = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
        output = model(x)
        logits = main_logits(output)
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, probability.ndim))
        numerator = 2.0 * (probability * y).sum(dimensions) + 1.0
        denominator = probability.sum(dimensions) + y.sum(dimensions) + 1.0
        positive = y.sum(dimensions) > 0
        dice_loss = (1.0 - numerator[positive] / denominator[positive]).mean()
        standard_loss = bce(logits, y) + dice_loss
        skeleton_sum = skeleton.sum(dimensions)
        valid_skeleton = skeleton_sum > 0
        skeleton_loss = torch.zeros((), device=device)
        if recipe["skeleton_weight"] and bool(valid_skeleton.any()):
            covered = (probability * skeleton).sum(dimensions)
            skeleton_loss = (1.0 - (covered[valid_skeleton] + 1.0)
                             / (skeleton_sum[valid_skeleton] + 1.0)).mean()
        neighbourhood_loss = extension.isolated_false_positive_proxy_loss(
            logits, y, args.neighbourhood_kernel)
        boundary_loss = torch.zeros((), device=device)
        if args.variant == "boundary_head":
            boundary_loss = positive_dice_bce(output[1], boundary_target(y))
        continuity_loss = torch.zeros((), device=device)
        if args.variant == "line_connectivity":
            continuity_loss = line_local_min_loss(
                probability, skeleton, args.connectivity_kernel)
        loss = standard_loss + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood_loss \
            + args.boundary_weight * boundary_loss \
            + args.connectivity_weight * continuity_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "boundary": float(boundary_loss.detach().cpu()),
            "continuity": float(continuity_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood_loss.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard_loss.detach().cpu()),
        }
        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        val_probability, val_target = predict(
            model, validation_positive, args.batch_size, device)
        clean_probability, _ = predict(
            model, validation_clean, args.batch_size, device)
        points = base.choose_operating_points(
            val_probability, val_target, clean_probability, args.max_clean_fp,
            base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
        primary = points["localization"]
        key = (primary["iou"], primary["macro_iou"], primary["recall"],
               -primary["threshold"])
        if step >= start_select and (best_key is None or key > best_key):
            best_key, best_step, best_points = key, step, points
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(json.dumps({
            "best_step": best_step,
            "event": "validation",
            "seed": seed,
            "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
            "variant": args.variant,
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Ablation371Error("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    val_probability, val_target = predict(
        model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        model, validation_clean, args.batch_size, device)
    points = base.choose_operating_points(
        val_probability, val_target, clean_probability, args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": {
            "base_channels": args.base_channels,
            "boundary_head": args.variant == "boundary_head",
            "crop": list(recipe["crop"]),
            "in_channels": 1,
            "pretrained_resnet34_sha256": None if args.resnet34_weights is None
                else digest_file(args.resnet34_weights),
            "variant": args.variant,
        },
        "best_step": best_step,
        "data_arm": "new1000_plus_safe80",
        "operating_points": points,
        "seed": seed,
        "state_dict": model.state_dict(),
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=val_probability.astype(np.float16),
        targets=(val_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "last_losses": last_losses,
        "seed": seed,
        "validation": points,
        "variant": args.variant,
    }, {"clean": clean_probability, "positive": val_probability}


def load_extra_rows(base, prepared: dict[str, Any], args: argparse.Namespace):
    eval_package = base.load_module(
        base.EVAL_PACKAGE_SCRIPT, "hme_evalpkg325_for371", base.EVAL_PACKAGE_SHA256)
    catalogue_rows, _lookup = base.load_catalogue(prepared["segmenter"])
    protected_images = {
        str(entry["image_id"]) for entry in catalogue_rows
        if set(entry.get("data_roles") or [])
        & {"fixed_development_only", "fixed_external_test_only"}
    }
    rows, qa = base.read_package(
        Path(args.extra_key), Path(args.extra_returned), name="extra105",
        expected_key_sha=EXPECTED_EXTRA_KEY_SHA256,
        expected_returned_sha=EXPECTED_EXTRA_RETURNED_SHA256,
        expected_fingerprint=EXPECTED_EXTRA_FINGERPRINT,
        expected_records=EXPECTED_EXTRA_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=eval_package.collect_test_identifiers())
    if qa != EXPECTED_EXTRA_QA:
        raise Ablation371Error(f"extra105 QA changed: {qa}")
    validation_writers = {
        str(row["group_uuid"]) for row in
        prepared["validation_positive"] + prepared["validation_clean"]
    }
    fitting = [row for row in rows if str(row["group_uuid"]) not in validation_writers]
    excluded = [row for row in rows if str(row["group_uuid"]) in validation_writers]
    if len(fitting) != EXPECTED_TRAINING_ROWS \
            or sum(bool(row["marks"]) for row in fitting) != EXPECTED_TRAINING_POSITIVE \
            or sum(not row["marks"] for row in fitting) != EXPECTED_TRAINING_CLEAN \
            or len(excluded) != EXPECTED_VALIDATION_WRITER_EXCLUDED:
        raise Ablation371Error("extra105 fitting/validation isolation changed")
    return fitting, excluded, qa


def patch_prepare(base, args: argparse.Namespace) -> None:
    original_prepare = base.prepare

    def prepare(run_args: argparse.Namespace):
        prepared = original_prepare(run_args)
        if run_args.data_arm != "new1000":
            raise Ablation371Error("experiment 371 requires the new1000 arm")
        fitting, excluded, qa = load_extra_rows(base, prepared, run_args)
        extra_ids = {row["line_id"] for row in fitting}
        existing_ids = {
            row["line_id"] for row in prepared["fit_positive"]
            + prepared["fit_clean"] + prepared["validation_positive"]
            + prepared["validation_clean"]
        }
        if extra_ids & existing_ids:
            raise Ablation371Error("extra105 overlaps an existing split")
        extra_positive = [row for row in fitting if row["marks"]]
        extra_clean = [row for row in fitting if not row["marks"]]
        existing_gold_clean_count = int(prepared["audit"]["fit_gold_clean"])
        existing_gold_clean = prepared["fit_clean"][:existing_gold_clean_count]
        candidates = [row for row in prepared["fit_clean_candidates"]
                      if row["line_id"] not in extra_ids]
        total_positive = len(prepared["fit_positive"]) + len(extra_positive)
        total_gold_clean = len(existing_gold_clean) + len(extra_clean)
        wanted_catalogue_clean = base.reference_catalogue_clean_count(
            total_positive, total_gold_clean, run_args.negative_ratio)
        sampled, hard_selection = base.select_hard_prefix(
            Path(run_args.hard_negative_manifest), candidates,
            wanted_catalogue_clean,
            validation_seed=run_args.validation_seed,
            validation_fraction=run_args.validation_fraction)
        prepared["fit_positive"] = prepared["fit_positive"] + extra_positive
        prepared["fit_clean"] = existing_gold_clean + extra_clean + sampled
        prepared["hard_selection"] = hard_selection
        audit = prepared["audit"]
        audit["data_arm"] = "new1000_plus_safe80"
        audit["fit_gold_clean"] = total_gold_clean
        audit["fit_positive"] = total_positive
        audit["hard_selection"] = hard_selection
        audit["sampled_catalogue_clean"] = len(sampled)
        audit["total_fit_clean"] = len(prepared["fit_clean"])
        audit["extra105"] = {
            "all_105_used_for_training": False,
            "protected_image_rows_excluded_without_semantic_label_access": 16,
            "safe_records_after_protected_closure": len(fitting) + len(excluded),
            "training_clean": len(extra_clean),
            "training_positive": len(extra_positive),
            "training_rows": len(fitting),
            "validation_writer_rows_excluded": len(excluded),
            "qa": qa,
        }
        fit_writers = {str(row["group_uuid"]) for row in
                       prepared["fit_positive"] + prepared["fit_clean"]}
        validation_writers = {str(row["group_uuid"]) for row in
                              prepared["validation_positive"]
                              + prepared["validation_clean"]}
        if fit_writers & validation_writers:
            raise Ablation371Error("writer isolation failed after adding extra105")
        return prepared

    base.prepare = prepare


def finalise(base, staging: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = staging / "summary.json"
    manifest_path = staging / "run_manifest.json"
    complete_path = staging / "complete.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    summary["schema_version"] = SCHEMA
    summary["variant"] = args.variant
    summary["protocol"].update({
        "boundary_weight": args.boundary_weight,
        "connectivity_kernel": args.connectivity_kernel,
        "connectivity_weight": args.connectivity_weight,
        "seed_policy": "single fixed seed 42 until model recipe stabilises",
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
        "variant": args.variant,
    })
    manifest.update({
        "data_arm": "new1000_plus_safe80",
        "extra_key_sha256": digest_file(Path(args.extra_key)),
        "extra_returned_sha256": digest_file(Path(args.extra_returned)),
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "seed": 42,
        "variant": args.variant,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(complete_path, {"schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(base, runner, full) -> argparse.ArgumentParser:
    parser = full.build_parser(base, runner)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--extra-key", required=True, type=Path)
    parser.add_argument("--extra-returned", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--resnet34-weights", type=Path)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--connectivity-weight", type=float, default=0.10)
    parser.add_argument("--connectivity-kernel", type=int, default=3)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode != "dev" or args.data_arm != "new1000" \
            or args.recipe != "optimized" or args.seeds != "42" \
            or args.input_mode != "image" or args.merge_radius_factor != 0 \
            or not math.isclose(args.isolated_fp_weight, 0.5):
        raise Ablation371Error("experiment 371 protocol differs from the locked seed-42 plan")
    if args.boundary_weight < 0 or args.connectivity_weight < 0 \
            or args.connectivity_kernel < 1 or args.connectivity_kernel % 2 != 1:
        raise Ablation371Error("invalid auxiliary loss configuration")
    if args.variant != "boundary_head" and args.boundary_weight != 0.15:
        # The stored default remains fixed but is multiplied by an exact zero loss.
        raise Ablation371Error("boundary weight changed outside its arm")
    if args.variant != "line_connectivity" and args.connectivity_weight != 0.10:
        raise Ablation371Error("connectivity weight changed outside its arm")
    if args.variant == "unetpp_resnet34" and args.resnet34_weights is None:
        raise Ablation371Error("UNet++ arm lacks offline pretrained weights")
    if args.variant != "unetpp_resnet34" and args.resnet34_weights is not None:
        raise Ablation371Error("only UNet++ may consume ResNet-34 weights")


def run(args: argparse.Namespace, base, extension, runner, full):
    validate_args(args)
    full.configure_base(base)
    patch_prepare(base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(b, e, fp, fc, vp, vc, a, seed, staging)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Ablation371Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    inner = output.parent / f".{output.name}.inner-{uuid.uuid4().hex}"
    original_output = args.output_dir
    args.output_dir = inner
    try:
        runner.run(args, base, extension)
        summary = finalise(base, inner, args)
        inner.rename(output)
    except Exception:
        if inner.is_dir():
            shutil.rmtree(inner)
        raise
    finally:
        args.output_dir = original_output
    return summary


def main() -> int:
    base = load_module(BASE_PATH, "hme354_for371", BASE_SHA256)
    extension = load_module(EXTENSION_PATH, "hme357_for371", EXTENSION_SHA256)
    runner = load_module(RUNNER_PATH, "hme358_for371", RUNNER_SHA256)
    full = load_module(FULL1000_PATH, "hme365_for371", FULL1000_SHA256)
    args = build_parser(base, runner, full).parse_args()
    try:
        summary = run(args, base, extension, runner, full)
    except (Ablation371Error, base.LocalizerError, runner.MechanismError,
            full.Full1000Error) as exc:
        raise SystemExit(f"experiment 371 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"],
        "ensemble": summary["ensemble"],
        "status": summary["mode"],
        "variant": args.variant,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

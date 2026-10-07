#!/usr/bin/env python3
"""Evaluate LWTformer-inspired wavelet adapters on revision segmentation.

All four arms start from the same completed experiment-374 corrective
fine-tuning checkpoint.  The control receives the same additional replay
training as the wavelet arms.  The three treatment arms inject a half-scale
wavelet residual into the first ResNet feature map while preserving the
pretrained UNet++ trunk and the frozen writer-disjoint validation protocol.
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
CORRECTIVE_PATH = ROOT / "scripts/374_revision_corrective_learning_seed42.py"
CORRECTIVE_SHA256 = "d791c869b52332202c82dee4c826392b70690674ce5115dec09a5dae5476eafd"
SOURCE_SCHEMA = "hme-revision-corrective-learning-seed42-dev-v1"
SCHEMA = "hme-revision-wavelet-ablation-seed42-dev-v1"
VARIANTS = (
    "control_continue",
    "fixed_haar",
    "learnable_wavedown",
    "directional_type",
)


class Experiment376Error(RuntimeError):
    """Raised when the locked wavelet-ablation protocol changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment376Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment376Error(f"expected JSON object: {path}")
    return value


def load_corrective():
    if digest_file(CORRECTIVE_PATH) != CORRECTIVE_SHA256:
        raise Experiment376Error("experiment-374 dependency digest mismatch")
    specification = importlib.util.spec_from_file_location(
        "hme374_for376", CORRECTIVE_PATH)
    if specification is None or specification.loader is None:
        raise Experiment376Error("cannot import experiment 374")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def verify_source(source_checkpoint: Path) -> dict[str, Any]:
    import torch

    path = source_checkpoint.expanduser().resolve()
    root = path.parent
    if path.is_symlink() or not path.is_file() \
            or path.name != "segmenter_seed42.pt":
        raise Experiment376Error("source checkpoint must be a regular seed42 checkpoint")
    complete = read_json(root / "complete.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"} \
            or summary.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("schema_version") != SOURCE_SCHEMA:
        raise Experiment376Error("source experiment schema or completion changed")
    if summary.get("variant") != "finetune_replay" \
            or manifest.get("stage") != "finetune_replay" \
            or manifest.get("m6_frozen_test_touched") is not False \
            or manifest.get("reference_evaluation_materialised") is not False \
            or summary.get("data", {}).get("writer_overlap_fit_validation") != 0:
        raise Experiment376Error("source experiment safety contract changed")
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in sums:
            raise Experiment376Error("source SHA256SUMS is malformed")
        sums[name] = digest
    if sums.get(path.name) != digest_file(path):
        raise Experiment376Error("source checkpoint digest mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    architecture = payload.get("architecture", {})
    if payload.get("seed") != 42 \
            or payload.get("data_arm") != "new1000_plus_safe80" \
            or payload.get("stage") != "finetune_replay" \
            or architecture.get("variant") != "unetpp_resnet34" \
            or architecture.get("input_channels") != 1 \
            or architecture.get("residual_logits") is not False:
        raise Experiment376Error("source checkpoint metadata changed")
    return {
        "checkpoint": payload,
        "checkpoint_sha256": digest_file(path),
        "manifest_sha256": digest_file(root / "run_manifest.json"),
        "root": root,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.wavelet_variant not in VARIANTS \
            or args.stage != "finetune_replay" \
            or args.mode != "dev" or args.data_arm != "new1000" \
            or args.recipe != "optimized" or args.seeds != "42" \
            or args.input_mode != "image" or args.merge_radius_factor != 0 \
            or not math.isclose(args.isolated_fp_weight, 0.5):
        raise Experiment376Error("wavelet-ablation base protocol changed")
    if args.optimizer_steps != 3000 or args.validation_interval != 100 \
            or not math.isclose(args.selection_warmup_fraction, 0.2) \
            or not math.isclose(args.learning_rate, 3e-5) \
            or not math.isclose(args.adapter_learning_rate, 3e-4):
        raise Experiment376Error("wavelet optimisation protocol changed")
    if args.corrected_positive_per_batch != 1 \
            or args.corrected_clean_interval != 20 \
            or args.line_target_dilation != 13 \
            or not math.isclose(args.type_aux_weight, 0.1):
        raise Experiment376Error("wavelet supervision protocol changed")


def make_wavelet_model(ablation, weights_path: Path, source_state: dict[str, Any],
                       variant: str):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    core = ablation.make_unetpp_resnet34(weights_path)
    core.load_state_dict(source_state, strict=True)
    if variant == "control_continue":
        return core

    class HaarDWT(nn.Module):
        def __init__(self, learnable: bool):
            super().__init__()
            scale = 1.0 / math.sqrt(2.0)
            low = torch.tensor([scale, scale], dtype=torch.float32)
            high = torch.tensor([-scale, scale], dtype=torch.float32)
            if learnable:
                self.dec_lo = nn.Parameter(low)
                self.dec_hi = nn.Parameter(high)
            else:
                self.register_buffer("dec_lo", low)
                self.register_buffer("dec_hi", high)

        def forward(self, value):
            pad_bottom = int(value.shape[-2] % 2)
            pad_right = int(value.shape[-1] % 2)
            if pad_bottom or pad_right:
                value = functional.pad(
                    value, (0, pad_right, 0, pad_bottom), mode="replicate")
            low, high = self.dec_lo, self.dec_hi
            kernels = torch.stack((
                torch.outer(low, low), torch.outer(high, low),
                torch.outer(low, high), torch.outer(high, high),
            )).unsqueeze(1)
            return functional.conv2d(value, kernels, stride=2)

    class WaveletAdapter(nn.Module):
        def __init__(self, treatment: str):
            super().__init__()
            self.directional = treatment == "directional_type"
            self.dwt = HaarDWT(learnable=treatment != "fixed_haar")
            self.spatial_down = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.PixelUnshuffle(2),
            )
            if self.directional:
                self.ll = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GELU())
                self.lh_pre = nn.Sequential(
                    nn.Conv2d(1, 16, (1, 3), padding=(0, 1)), nn.GELU())
                self.hl_pre = nn.Sequential(
                    nn.Conv2d(1, 16, (3, 1), padding=(1, 0)), nn.GELU())
                self.hh_pre = nn.Sequential(nn.Tanh(), nn.Conv2d(1, 16, 1))
                self.lh_direction = self.direction_conv(torch.tensor([
                    [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0]]))
                self.hl_direction = self.direction_conv(torch.tensor([
                    [1.0, 0.0, -1.0], [1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0]]))
                self.hh_direction = self.direction_conv(torch.tensor([
                    [0.0, 1.0, 0.0], [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0]]))
                self.direction_gate = nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Sigmoid())
                self.line_head = nn.Conv2d(64, 1, 1)
                self.blot_head = nn.Conv2d(64, 1, 1)
            else:
                self.subband_gate = nn.Sequential(
                    nn.Conv2d(4, 4, 3, padding=1, groups=4),
                    nn.GroupNorm(4, 4), nn.GELU(),
                    nn.Conv2d(4, 64, 1), nn.Sigmoid(),
                )
            self.refine = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, groups=64),
                nn.Conv2d(64, 64, 1), nn.GELU(),
            )
            self.fuse = nn.Conv2d(64, 64, 1, bias=False)
            nn.init.zeros_(self.fuse.weight)

        @staticmethod
        def direction_conv(kernel):
            layer = nn.Conv2d(
                16, 16, 3, padding=1, groups=16, bias=False)
            with torch.no_grad():
                layer.weight.copy_(kernel.view(1, 1, 3, 3).repeat(16, 1, 1, 1))
            return layer

        def forward(self, gray):
            pad_bottom = int(gray.shape[-2] % 2)
            pad_right = int(gray.shape[-1] % 2)
            spatial_input = gray
            if pad_bottom or pad_right:
                spatial_input = functional.pad(
                    gray, (0, pad_right, 0, pad_bottom), mode="replicate")
            spatial = self.spatial_down(spatial_input)
            bands = self.dwt(gray)
            if self.directional:
                ll, lh, hl, hh = bands.chunk(4, dim=1)
                direction = torch.cat((
                    self.ll(ll), self.lh_direction(self.lh_pre(lh)),
                    self.hl_direction(self.hl_pre(hl)),
                    self.hh_direction(self.hh_pre(hh)),
                ), dim=1)
                gate = self.direction_gate(direction)
                auxiliary = direction
            else:
                gate = self.subband_gate(bands)
                auxiliary = None
            feature = self.refine(spatial * (1.0 + gate))
            return self.fuse(feature), auxiliary

    class WaveletUNetPlusPlus(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.adapter = WaveletAdapter(variant)
            self.wavelet_variant = variant

        def forward(self, gray):
            rgb = ((gray + 1.0) * 0.5).repeat(1, 3, 1, 1)
            rgb = (rgb - self.core.mean) / self.core.std
            x0_0 = self.core.encoder.relu(
                self.core.encoder.bn1(self.core.encoder.conv1(rgb)))
            injection, auxiliary = self.adapter(gray)
            injection = functional.interpolate(
                injection, size=x0_0.shape[-2:], mode="bilinear",
                align_corners=False)
            x0_0 = x0_0 + injection
            x1_0 = self.core.encoder.layer1(self.core.encoder.maxpool(x0_0))
            x2_0 = self.core.encoder.layer2(x1_0)
            x3_0 = self.core.encoder.layer3(x2_0)
            x4_0 = self.core.encoder.layer4(x3_0)
            up = self.core.up
            x3_1 = self.core.x3_1(torch.cat((x3_0, up(x4_0, x3_0)), 1))
            x2_1 = self.core.x2_1(torch.cat((x2_0, up(x3_0, x2_0)), 1))
            x1_1 = self.core.x1_1(torch.cat((x1_0, up(x2_0, x1_0)), 1))
            x0_1 = self.core.x0_1(torch.cat((x0_0, up(x1_0, x0_0)), 1))
            x2_2 = self.core.x2_2(torch.cat(
                (x2_0, x2_1, up(x3_1, x2_0)), 1))
            x1_2 = self.core.x1_2(torch.cat(
                (x1_0, x1_1, up(x2_1, x1_0)), 1))
            x0_2 = self.core.x0_2(torch.cat(
                (x0_0, x0_1, up(x1_1, x0_0)), 1))
            x1_3 = self.core.x1_3(torch.cat(
                (x1_0, x1_1, x1_2, up(x2_2, x1_0)), 1))
            x0_3 = self.core.x0_3(torch.cat(
                (x0_0, x0_1, x0_2, up(x1_2, x0_0)), 1))
            x0_4 = self.core.x0_4(torch.cat(
                (x0_0, x0_1, x0_2, x0_3, up(x1_3, x0_0)), 1))
            logits = functional.interpolate(
                self.core.head(x0_4), size=gray.shape[-2:], mode="bilinear",
                align_corners=False)
            if auxiliary is None:
                return logits
            line_logits = functional.interpolate(
                self.adapter.line_head(auxiliary), size=gray.shape[-2:],
                mode="bilinear", align_corners=False)
            blot_logits = functional.interpolate(
                self.adapter.blot_head(auxiliary), size=gray.shape[-2:],
                mode="bilinear", align_corners=False)
            return logits, line_logits, blot_logits

    return WaveletUNetPlusPlus()


def main_logits(output):
    return output[0] if isinstance(output, tuple) else output


def type_targets(target, skeleton, dilation: int):
    from torch.nn import functional as functional

    support = functional.max_pool2d(
        skeleton, kernel_size=dilation, stride=1, padding=dilation // 2)
    line = target * (support > 0).to(target.dtype)
    blot = target * (line <= 0).to(target.dtype)
    return line, blot


def auxiliary_mask_loss(logits, target):
    import torch
    from torch.nn import functional as functional

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + target.sum(dimensions) + 1.0
    present = target.sum(dimensions) > 0
    dice = torch.zeros((), device=logits.device)
    if bool(present.any()):
        dice = (1.0 - numerator[present] / denominator[present]).mean()
    return functional.binary_cross_entropy_with_logits(logits, target) + dice


def predict(ablation, model, rows: Sequence[dict[str, Any]],
            batch_size: int, device):
    import torch

    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            image = np.stack([
                row["image"] for row in rows[start:start + batch_size]
            ]).astype(np.float32)
            gray = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            probabilities.append(torch.sigmoid(
                main_logits(model(gray))).cpu().numpy()[:, 0])
    model.train()
    return np.concatenate(probabilities), np.stack(
        [row["mask"] for row in rows]) / 255.0


def train_once(corrective, ablation, base, extension,
               fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path,
               source: dict[str, Any]):
    import torch

    if seed != 42 or not torch.cuda.is_available():
        raise Experiment376Error("formal wavelet training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    corrective.tag_materialised_corrected_rows(fit_positive, fit_clean, args)
    model = make_wavelet_model(
        ablation, args.resnet34_weights,
        source["checkpoint"]["state_dict"], args.wavelet_variant).to(device)
    if args.wavelet_variant == "control_continue":
        parameter_groups = [{"params": model.parameters(), "lr": args.learning_rate}]
    else:
        parameter_groups = [
            {"params": model.core.parameters(), "lr": args.learning_rate},
            {"params": model.adapter.parameters(),
             "lr": args.adapter_learning_rate},
        ]
    optimiser = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay)
    rng = random.Random(seed)
    partitions = {
        "corrected_positive": [row for row in fit_positive if row["corrected105"]],
        "original_positive": [row for row in fit_positive if not row["corrected105"]],
        "corrected_clean": [row for row in fit_clean if row["corrected105"]],
        "original_clean": [row for row in fit_clean if not row["corrected105"]],
    }
    expected = {"corrected_positive": 77, "original_positive": 718,
                "corrected_clean": 3, "original_clean": 2382}
    if {name: len(rows) for name, rows in partitions.items()} != expected:
        raise Experiment376Error("wavelet fitting partition changed")
    cycles = {name: base.ShuffledCycle(rows, rng)
              for name, rows in partitions.items()}
    recipe = base.RECIPES[args.recipe]
    best_key = best_state = best_points = None
    best_step = -1
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rows = corrective.correction_batch(
            base, fit_positive, fit_clean, rng, step, args, cycles)
        images, masks, skeletons = [], [], []
        for row in rows:
            image, mask, skeleton, _ = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], None, rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        gray = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
        target = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
        output = model(gray)
        logits = main_logits(output)
        standard, skeleton_loss, neighbourhood = corrective.standard_loss_terms(
            extension, logits, target, skeleton, args)
        type_loss = torch.zeros((), device=device)
        if args.wavelet_variant == "directional_type":
            line_target, blot_target = type_targets(
                target, skeleton, args.line_target_dilation)
            type_loss = auxiliary_mask_loss(output[1], line_target) \
                + auxiliary_mask_loss(output[2], blot_target)
        loss = standard + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood \
            + args.type_aux_weight * type_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard.detach().cpu()),
            "type_auxiliary": float(type_loss.detach().cpu()),
        }
        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        positive_probability, positive_target = predict(
            ablation, model, validation_positive, args.batch_size, device)
        clean_probability, _ = predict(
            ablation, model, validation_clean, args.batch_size, device)
        points = base.choose_operating_points(
            positive_probability, positive_target, clean_probability,
            args.max_clean_fp,
            base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
        primary = points["localization"]
        key = (primary["iou"], primary["macro_iou"], primary["recall"],
               -primary["threshold"])
        if step >= start_select and (best_key is None or key > best_key):
            best_key, best_step, best_points = key, step, points
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(json.dumps({
            "best_step": best_step, "event": "validation", "seed": seed,
            "step": step, "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
            "variant": args.wavelet_variant,
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Experiment376Error("wavelet training produced no checkpoint")
    model.load_state_dict(best_state)
    positive_probability, positive_target = predict(
        ablation, model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        ablation, model, validation_clean, args.batch_size, device)
    points = base.choose_operating_points(
        positive_probability, positive_target, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": {
            "adapter": None if args.wavelet_variant == "control_continue"
                else args.wavelet_variant,
            "base_variant": "unetpp_resnet34",
            "input_channels": 1,
            "line_target_dilation": args.line_target_dilation,
            "variant": args.wavelet_variant,
        },
        "best_step": best_step,
        "data_arm": "new1000_plus_safe80",
        "operating_points": points,
        "seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "state_dict": model.state_dict(),
        "variant": args.wavelet_variant,
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=positive_probability.astype(np.float16),
        targets=(positive_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "last_losses": last_losses,
        "seed": seed,
        "validation": points,
        "variant": args.wavelet_variant,
    }, {"clean": clean_probability, "positive": positive_probability}


def finalise(base, staging: Path, args: argparse.Namespace,
             source: dict[str, Any]) -> dict[str, Any]:
    summary_path = staging / "summary.json"
    manifest_path = staging / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    data = summary.get("data", {})
    if data.get("fit_positive") != 795 \
            or data.get("total_fit_clean") != 2385 \
            or data.get("writer_overlap_fit_validation") != 0:
        raise Experiment376Error("wavelet data contract changed")
    summary["schema_version"] = SCHEMA
    summary["variant"] = args.wavelet_variant
    summary["protocol"].update({
        "adapter_learning_rate": args.adapter_learning_rate,
        "line_target_dilation": args.line_target_dilation,
        "seed_policy": "single fixed seed 42",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
        "type_aux_weight": args.type_aux_weight,
        "wavelet_variant": args.wavelet_variant,
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
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "variant": args.wavelet_variant,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(corrective, base, runner, full) -> argparse.ArgumentParser:
    parser = corrective.build_parser(base, runner, full)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--wavelet-variant", required=True, choices=VARIANTS)
    parser.add_argument("--adapter-learning-rate", type=float, default=3e-4)
    parser.add_argument("--type-aux-weight", type=float, default=0.1)
    parser.add_argument("--line-target-dilation", type=int, default=13)
    return parser


def run(args, corrective, ablation, base, extension, runner, full):
    validate_args(args)
    source = verify_source(args.source_checkpoint)
    full.configure_base(base)
    corrective.patch_prepare_with_source_flags(ablation, base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(corrective, ablation, b, e, fp, fc, vp, vc, a, seed,
                   staging, source)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment376Error(f"create-only output exists: {output}")
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
    corrective = load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for376", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for376", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for376", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for376", ablation.FULL1000_SHA256)
    args = build_parser(corrective, base, runner, full).parse_args()
    try:
        summary = run(
            args, corrective, ablation, base, extension, runner, full)
    except (Experiment376Error, corrective.Experiment374Error,
            ablation.Ablation371Error, base.LocalizerError,
            runner.MechanismError, full.Full1000Error) as exc:
        raise SystemExit(f"experiment 376 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"], "ensemble": summary.get("ensemble", {}),
        "variant": args.wavelet_variant,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

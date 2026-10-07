#!/usr/bin/env python3
"""Fair paper-baseline suite for real handwritten revision segmentation.

Every arm uses the experiment-418 latest reviewed-gold preparation, the same
writer-disjoint development split, the same batch composition/augmentations,
and the same evaluator.  This file intentionally never materialises M6 or any
reference/test split.  Architecture arms change only the network; loss arms
keep the established UNet++--ResNet34 architecture fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TYPEPILOT_PATH = ROOT / "scripts/418_revision_typepilot300_factorial_seed42.py"
TYPEPILOT_SHA256 = "ad88c2542a11a45c2770ddd7f589dab2eb9d8b8fe0a987cca445d272b189275f"
SCHEMA = "hme-revision-paper-baselines-dev-v1"

ARCHITECTURE_ARMS = (
    "unet_resnet34",
    "unetpp_resnet34",
    "deeplabv3p_resnet50",
    "segformer_b0",
    "segformer_b2",
    "hrnet_w18",
    "gated_scnn_resnet50",
    "dru_resnet34",
    "jtfn_resnet34",
    "mask2former_r50",
    "semantic_fpn_pointrend",
)
LOSS_ARMS = (
    "unetpp_cldice",
    "unetpp_skeleton_recall",
    "unetpp_scnp",
    "unetpp_boundary",
)
ARMS = ARCHITECTURE_ARMS + LOSS_ARMS
SEEDS = (42, 2354, 3407)
LOSS_VARIANT = {
    "unetpp_cldice": "cldice",
    "unetpp_skeleton_recall": "skeleton_recall",
    "unetpp_scnp": "scnp",
    "unetpp_boundary": "boundary",
}


class Baseline422Error(RuntimeError):
    """Raised when the frozen development-baseline contract changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_typepilot():
    actual = digest_file(TYPEPILOT_PATH)
    if actual != TYPEPILOT_SHA256:
        raise Baseline422Error(
            f"experiment-418 dependency digest mismatch: {actual}")
    specification = importlib.util.spec_from_file_location(
        "hme418_for422", TYPEPILOT_PATH)
    if specification is None or specification.loader is None:
        raise Baseline422Error("cannot import experiment 418")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _resnet_state(path: Path) -> dict[str, Any]:
    import torch

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise Baseline422Error(f"offline ImageNet weight missing: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_smp_resnet_encoder(encoder, path: Path) -> None:
    import torch

    state = _resnet_state(path)
    if encoder.conv1.weight.shape[1] == 1 and state["conv1.weight"].shape[1] == 3:
        state["conv1.weight"] = state["conv1.weight"].mean(1, keepdim=True)
    state = {key: value for key, value in state.items()
             if not key.startswith("fc.")}
    # SMP 0.5 overrides ``load_state_dict`` and returns None; invoke the
    # framework implementation so the audit can inspect incompatibilities.
    missing, unexpected = torch.nn.Module.load_state_dict(
        encoder, state, strict=False)
    if unexpected or any(not key.startswith("fc.") for key in missing):
        raise Baseline422Error(
            f"ResNet encoder state mismatch: missing={missing}, unexpected={unexpected}")


def _load_smp_mit_encoder(encoder, path: Path) -> None:
    import torch

    state = _resnet_state(path)
    state = {key: value for key, value in state.items()
             if not key.startswith("head.")}
    first = "patch_embed1.proj.weight"
    if encoder.patch_embed1.proj.weight.shape[1] == 1 \
            and state[first].shape[1] == 3:
        state[first] = state[first].mean(1, keepdim=True)
    missing, unexpected = torch.nn.Module.load_state_dict(
        encoder, state, strict=False)
    if unexpected or missing:
        raise Baseline422Error(
            f"MiT encoder state mismatch: missing={missing}, unexpected={unexpected}")


def _ensure_size(logits, value):
    from torch.nn import functional as functional

    if logits.shape[-2:] != value.shape[-2:]:
        logits = functional.interpolate(
            logits, size=value.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _smp_model(name: str, args: argparse.Namespace):
    import segmentation_models_pytorch as smp

    if name == "unet_resnet34":
        model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=1, classes=1)
        _load_smp_resnet_encoder(model.encoder, args.resnet34_weights)
        return model
    if name == "deeplabv3p_resnet50":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights=None,
            encoder_output_stride=8, in_channels=1, classes=1)
        _load_smp_resnet_encoder(model.encoder, args.resnet50_weights)
        return model
    if name == "segformer_b0":
        model = smp.Segformer(
            encoder_name="mit_b0", encoder_weights=None,
            in_channels=1, classes=1)
        _load_smp_mit_encoder(model.encoder, args.segformer_b0_weights)
        return model
    if name == "segformer_b2":
        model = smp.Segformer(
            encoder_name="mit_b2", encoder_weights=None,
            in_channels=1, classes=1)
        _load_smp_mit_encoder(model.encoder, args.segformer_b2_weights)
        return model
    raise Baseline422Error(f"unknown SMP arm: {name}")


def _make_hrnet(args: argparse.Namespace):
    import timm
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class HRNetSegmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(
                "hrnet_w18", pretrained=False, features_only=True)
            timm.models.load_checkpoint(
                self.encoder, str(args.hrnet_w18_weights), strict=False)
            channels = self.encoder.feature_info.channels()
            self.projections = nn.ModuleList([
                nn.Sequential(nn.Conv2d(channel, 64, 1, bias=False),
                              nn.BatchNorm2d(64), nn.ReLU(inplace=True))
                for channel in channels
            ])
            self.head = nn.Sequential(
                nn.Conv2d(64 * len(channels), 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
            )

        def forward(self, value):
            features = self.encoder(value.repeat(1, 3, 1, 1))
            target = features[0].shape[-2:]
            merged = torch.cat([
                functional.interpolate(projection(feature), size=target,
                                       mode="bilinear", align_corners=False)
                for projection, feature in zip(self.projections, features)
            ], dim=1)
            return _ensure_size(self.head(merged), value)

    return HRNetSegmenter()


def _make_gated_scnn(args: argparse.Namespace):
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    class GatedSCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.semantic = smp.DeepLabV3Plus(
                encoder_name="resnet50", encoder_weights=None,
                encoder_output_stride=8, in_channels=1, classes=1)
            _load_smp_resnet_encoder(
                self.semantic.encoder, args.resnet50_weights)
            self.shape = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1),
            )
            self.gate = nn.Sequential(nn.Conv2d(2, 16, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 1, 1), nn.Sigmoid())
            self.fuse = nn.Conv2d(2, 1, 1)

        def forward(self, value):
            semantic = _ensure_size(self.semantic(value), value)
            shape = self.shape(value)
            shape = shape * self.gate(torch.cat((semantic, shape), dim=1))
            return self.fuse(torch.cat((semantic, shape), dim=1))

    return GatedSCNN()


def _make_dru(args: argparse.Namespace):
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    class DRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = smp.Unet(
                encoder_name="resnet34", encoder_weights=None,
                in_channels=2, classes=1)
            state = _resnet_state(args.resnet34_weights)
            conv = state["conv1.weight"]
            state["conv1.weight"] = torch.cat(
                (conv.mean(1, keepdim=True), torch.zeros_like(conv[:, :1])), 1)
            state = {key: value for key, value in state.items()
                     if not key.startswith("fc.")}
            self.core.encoder.load_state_dict(state, strict=False)
            self.steps = 3

        def forward(self, value):
            previous = torch.zeros_like(value)
            logits = previous
            for _ in range(self.steps):
                logits = _ensure_size(
                    self.core(torch.cat((value, torch.sigmoid(previous)), 1)),
                    value)
                previous = logits
            return logits

    return DRU()


def _make_jtfn(args: argparse.Namespace):
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    class JTFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.local = smp.Unet(
                encoder_name="resnet34", encoder_weights=None,
                in_channels=1, classes=1)
            self.context = smp.FPN(
                encoder_name="resnet34", encoder_weights=None,
                in_channels=1, classes=1)
            _load_smp_resnet_encoder(
                self.local.encoder, args.resnet34_weights)
            _load_smp_resnet_encoder(
                self.context.encoder, args.resnet34_weights)
            self.fuse = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )

        def forward(self, value):
            local = _ensure_size(self.local(value), value)
            context = _ensure_size(self.context(value), value)
            return self.fuse(torch.cat((value, local, context), 1))

    return JTFN()


def _make_mask2former(args: argparse.Namespace):
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as functional

    class Mask2FormerSemantic(nn.Module):
        """ImageNet-R50 pixel decoder plus query-mask transformer decoder."""

        def __init__(self):
            super().__init__()
            encoder = torchvision.models.resnet50(weights=None)
            encoder.load_state_dict(_resnet_state(args.resnet50_weights), strict=True)
            self.encoder = encoder
            self.register_buffer("mean", torch.tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.project = nn.ModuleList([
                nn.Conv2d(channel, 128, 1) for channel in (256, 512, 1024, 2048)
            ])
            layer = nn.TransformerDecoderLayer(
                d_model=128, nhead=8, dim_feedforward=512,
                dropout=0.0, batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, num_layers=3)
            self.queries = nn.Parameter(torch.randn(16, 128) * 0.02)
            self.class_head = nn.Linear(128, 1)
            self.mask_head = nn.Linear(128, 128)
            self.pixel_head = nn.Sequential(
                nn.Conv2d(128 * 4, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1),
            )

        def forward(self, value):
            rgb = ((value + 1.0) * 0.5).repeat(1, 3, 1, 1)
            rgb = (rgb - self.mean) / self.std
            x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(rgb)))
            x = self.encoder.maxpool(x)
            features = []
            for layer in (self.encoder.layer1, self.encoder.layer2,
                          self.encoder.layer3, self.encoder.layer4):
                x = layer(x)
                features.append(x)
            target = features[0].shape[-2:]
            pixel = self.pixel_head(torch.cat([
                functional.interpolate(projection(feature), size=target,
                                       mode="bilinear", align_corners=False)
                for projection, feature in zip(self.project, features)
            ], 1))
            memory = pixel.flatten(2).transpose(1, 2)
            query = self.queries.unsqueeze(0).expand(len(value), -1, -1)
            query = self.decoder(query, memory)
            masks = torch.einsum(
                "bqc,bchw->bqhw", self.mask_head(query), pixel)
            classes = self.class_head(query).squeeze(-1)[..., None, None]
            logits = torch.logsumexp(classes + masks, dim=1, keepdim=True) \
                - math.log(masks.shape[1])
            return _ensure_size(logits, value)

    return Mask2FormerSemantic()


def _make_pointrend(args: argparse.Namespace):
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    class PointRendSemantic(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse = smp.FPN(
                encoder_name="resnet50", encoder_weights=None,
                in_channels=1, classes=1)
            _load_smp_resnet_encoder(
                self.coarse.encoder, args.resnet50_weights)
            self.refine = nn.Sequential(
                nn.Conv2d(3, 64, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )

        def forward(self, value):
            coarse = _ensure_size(self.coarse(value), value)
            uncertainty = -torch.abs(torch.sigmoid(coarse) - 0.5)
            residual = self.refine(torch.cat((value, coarse, uncertainty), 1))
            # Differentiable dense PointRend equivalent during training; the
            # uncertain-point protocol is declared separately in the report.
            return coarse + residual

    return PointRendSemantic()


def build_model(arm: str, args: argparse.Namespace, ablation):
    if arm.startswith("unetpp_") or arm == "unetpp_resnet34":
        return ablation.make_unetpp_resnet34(args.resnet34_weights)
    if arm in ("unet_resnet34", "deeplabv3p_resnet50",
               "segformer_b0", "segformer_b2"):
        return _smp_model(arm, args)
    if arm == "hrnet_w18":
        return _make_hrnet(args)
    if arm == "gated_scnn_resnet50":
        return _make_gated_scnn(args)
    if arm == "dru_resnet34":
        return _make_dru(args)
    if arm == "jtfn_resnet34":
        return _make_jtfn(args)
    if arm == "mask2former_r50":
        return _make_mask2former(args)
    if arm == "semantic_fpn_pointrend":
        return _make_pointrend(args)
    raise Baseline422Error(f"unknown baseline arm: {arm}")


def _soft_skeleton(value, iterations: int = 5):
    from torch.nn import functional as functional

    output = functional.relu(value - functional.max_pool2d(
        1.0 - value, 3, 1, 1))
    opened = functional.max_pool2d(
        functional.max_pool2d(value, 3, 1, 1) * -1.0, 3, 1, 1) * -1.0
    output = functional.relu(value - opened)
    current = value
    for _ in range(iterations):
        eroded = -functional.max_pool2d(-current, 3, 1, 1)
        opened = functional.max_pool2d(eroded, 3, 1, 1)
        delta = functional.relu(eroded - opened)
        output = output + functional.relu(delta - output * delta)
        current = eroded
    return output


def _boundary_target(target):
    from torch.nn import functional as functional

    eroded = -functional.max_pool2d(-target, 3, 1, 1)
    return (target - eroded).clamp(0, 1)


def loss_terms(logits, target, skeleton, loss_variant: str, extension):
    import torch
    from torch.nn import functional as functional

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    present = target.sum(dimensions) > 0
    numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + target.sum(dimensions) + 1.0
    dice = (1.0 - numerator[present] / denominator[present]).mean() \
        if bool(present.any()) else torch.zeros((), device=logits.device)
    standard = functional.binary_cross_entropy_with_logits(logits, target) + dice
    auxiliary = torch.zeros((), device=logits.device)
    if loss_variant == "cldice":
        prediction_skeleton = _soft_skeleton(probability)
        target_skeleton = _soft_skeleton(target)
        tprec = ((prediction_skeleton * target).sum() + 1.0) \
            / (prediction_skeleton.sum() + 1.0)
        tsens = ((target_skeleton * probability).sum() + 1.0) \
            / (target_skeleton.sum() + 1.0)
        auxiliary = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-6)
    elif loss_variant == "skeleton_recall":
        valid = skeleton.sum(dimensions) > 0
        if bool(valid.any()):
            covered = (probability * skeleton).sum(dimensions)
            auxiliary = (1.0 - (covered[valid] + 1.0)
                         / (skeleton.sum(dimensions)[valid] + 1.0)).mean()
    elif loss_variant == "scnp":
        auxiliary = 0.5 * extension.isolated_false_positive_proxy_loss(
            logits, target, 3)
        valid = skeleton.sum(dimensions) > 0
        if bool(valid.any()):
            covered = (probability * skeleton).sum(dimensions)
            auxiliary = auxiliary + 0.5 * (1.0 - (covered[valid] + 1.0)
                / (skeleton.sum(dimensions)[valid] + 1.0)).mean()
    elif loss_variant == "boundary":
        auxiliary = functional.binary_cross_entropy_with_logits(
            logits, _boundary_target(target))
    elif loss_variant != "standard":
        raise Baseline422Error(f"unknown loss variant: {loss_variant}")
    return standard + 0.20 * auxiliary, standard, auxiliary


def predict(model, rows: Sequence[dict[str, Any]], batch_size: int, device):
    import torch

    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            image = np.stack([row["image"] for row in chunk]).astype(np.float32)
            value = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            probabilities.append(torch.sigmoid(
                _ensure_size(model(value), value)).cpu().numpy()[:, 0])
    model.train()
    return np.concatenate(probabilities), np.stack(
        [row["mask"] for row in rows]).astype(np.float32) / 255.0


def train_once(ablation, base, extension,
               fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path):
    import torch

    if seed not in SEEDS or not torch.cuda.is_available():
        raise Baseline422Error("formal baselines require CUDA and a declared seed")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(args.variant, args, ablation).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(seed)
    positive_cycle = base.ShuffledCycle(fit_positive, rng)
    clean_cycle = base.ShuffledCycle(fit_clean, rng)
    recipe = base.RECIPES[args.recipe]
    loss_variant = LOSS_VARIANT.get(args.variant, "standard")
    best_key = best_state = best_points = None
    best_step = -1
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rows, _, _ = base.take_balanced_batch(
            positive_cycle, clean_cycle, args.batch_size, args.negative_ratio)
        rng.shuffle(rows)
        images, masks, skeletons = [], [], []
        for row in rows:
            image, mask, skeleton, _ = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], None, rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        value = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
        target = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
        logits = _ensure_size(model(value), value)
        loss, standard, auxiliary = loss_terms(
            logits, target, skeleton, loss_variant, extension)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        last_losses = {
            "auxiliary": float(auxiliary.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "standard": float(standard.detach().cpu()),
        }
        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        positive_probability, target_array = predict(
            model, validation_positive, args.batch_size, device)
        clean_probability, _ = predict(
            model, validation_clean, args.batch_size, device)
        points = base.choose_operating_points(
            positive_probability, target_array, clean_probability,
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
            "arm": args.variant, "best_step": best_step,
            "event": "validation", "seed": seed, "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Baseline422Error("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    positive_probability, target_array = predict(
        model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        model, validation_clean, args.batch_size, device)
    points = base.choose_operating_points(
        positive_probability, target_array, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": args.variant,
        "best_step": best_step,
        "data_arm": "trusted_gold_plus_active300_plus_typepilot300",
        "loss_variant": loss_variant,
        "operating_points": points,
        "seed": seed,
        "state_dict": model.state_dict(),
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=positive_probability.astype(np.float16),
        targets=(target_array * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "data_counts": {
            "fit_clean": len(fit_clean),
            "fit_positive": len(fit_positive),
            "validation_clean": len(validation_clean),
            "validation_positive": len(validation_positive),
        },
        "last_losses": last_losses,
        "seed": seed,
        "validation": points,
    }, {"clean": clean_probability, "positive": positive_probability}


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "segmentation-models-pytorch",
                 "timm", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def validate_protocol(args: argparse.Namespace) -> None:
    if args.variant not in ARMS or args.batch_size != 8 \
            or args.negative_ratio != 1 \
            or args.validation_interval != 500 \
            or not math.isclose(args.selection_warmup_fraction, 0.5) \
            or args.optimizer_steps != 28_300 \
            or not math.isclose(args.learning_rate, 3e-4):
        raise Baseline422Error("fair baseline protocol changed")
    if args.seed not in SEEDS:
        raise Baseline422Error("each create-only run must use one declared seed")


def build_parser(typepilot):
    parser = typepilot.build_parser()
    parser.description = __doc__.splitlines()[0]
    action = next(item for item in parser._actions if item.dest == "variant")
    action.choices = ARMS
    parser.add_argument("--resnet34-weights", required=True, type=Path)
    parser.add_argument("--resnet50-weights", required=True, type=Path)
    parser.add_argument("--segformer-b0-weights", required=True, type=Path)
    parser.add_argument("--segformer-b2-weights", required=True, type=Path)
    parser.add_argument("--hrnet-w18-weights", required=True, type=Path)
    parser.add_argument("--optimizer-steps", type=int, default=28_300)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--selection-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-clean-fp", type=float, default=0.15)
    parser.add_argument("--min-fire-pixels", type=int, default=8)
    parser.add_argument("--component-match-iou", type=float, default=0.10)
    return parser


def run(args, typepilot):
    validate_protocol(args)
    two_type = typepilot.load_two_type()
    type410, data, _source, exclusions, cleaning, type_audit, _initial = \
        two_type.prepare_data(args)
    new_positive, new_clean, new_audit = typepilot.load_new_rows(
        two_type, data, args)
    data["fit_positive"].extend(new_positive)
    data["fit_clean"].extend(new_clean)
    validation_writers = {str(row["group_uuid"]) for row in
                          data["validation_positive"] + data["validation_clean"]}
    fit_writers = {str(row["group_uuid"]) for row in
                   data["fit_positive"] + data["fit_clean"]}
    if fit_writers & validation_writers:
        raise Baseline422Error("writer isolation failed after latest gold")
    if (len(data["fit_positive"]), len(data["fit_clean"]),
            len(data["validation_positive"]), len(data["validation_clean"])) \
            != (1515, 1319, 126, 379):
        raise Baseline422Error("latest reviewed-gold sample counts changed")
    args.recipe = data["run_args"].recipe
    if args.validate_only:
        return {
            "arm": args.variant,
            "data": {
                "fit_clean": len(data["fit_clean"]),
                "fit_positive": len(data["fit_positive"]),
                "validation_clean": len(data["validation_clean"]),
                "validation_positive": len(data["validation_positive"]),
                "writer_overlap_fit_validation": 0,
            },
            "ensemble": {},
        }
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Baseline422Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result, probabilities = train_once(
            data["ablation"], data["base"], data["extension"],
            data["fit_positive"], data["fit_clean"],
            data["validation_positive"], data["validation_clean"],
            args, args.seed, staging)
        target = np.stack([
            row["mask"] for row in data["validation_positive"]
        ]).astype(np.float32) / 255.0
        points = data["base"].choose_operating_points(
            probabilities["positive"], target, probabilities["clean"],
            args.max_clean_fp,
            data["base"].scaled_min_fire_pixels(
                args.recipe, args.min_fire_pixels))
        threshold = points["localization"]["threshold"]
        all_probability = np.concatenate(
            (probabilities["positive"], probabilities["clean"]), 0)
        all_target = np.concatenate(
            (target, np.zeros_like(probabilities["clean"])), 0)
        component = data["extension"].component_metrics(
            all_probability, all_target, threshold,
            connectivity=8, match_iou=args.component_match_iou)
        boundary = data["base"].boundary_f1(
            probabilities["positive"], target, threshold,
            tolerance=max(1, int(round(
                3 * data["base"].RECIPES[args.recipe]["crop"][1] / 64))))
        split_rows = []
        for name, rows in (
            ("fit_positive", data["fit_positive"]),
            ("fit_clean", data["fit_clean"]),
            ("validation_positive", data["validation_positive"]),
            ("validation_clean", data["validation_clean"]),
        ):
            split_rows.extend({
                "group_uuid": row["group_uuid"], "line_id": row["line_id"],
                "source": row["source"], "split": name,
            } for row in rows)
        data["base"].write_jsonl(staging / "split_manifest.jsonl", sorted(
            split_rows, key=lambda row: (row["split"], row["line_id"])))
        data["base"].write_json(staging / "validation_key.json", [{
            "group_uuid": row["group_uuid"], "image_id": row["image_id"],
            "line_id": row["line_id"], "source": row["source"],
            "types": row.get("types", []),
        } for row in data["validation_positive"]])
        summary = {
            "arm": args.variant,
            "baseline_family": "architecture"
                if args.variant in ARCHITECTURE_ARMS else "loss",
            "cleaning": cleaning,
            "data": {
                "fit_clean": len(data["fit_clean"]),
                "fit_positive": len(data["fit_positive"]),
                "latest_typepilot300": new_audit,
                "validation_clean": len(data["validation_clean"]),
                "validation_positive": len(data["validation_positive"]),
                "writer_overlap_fit_validation": 0,
            },
            "ensemble": {"boundary_f1": boundary, "component": component,
                         "operating_points": points},
            "per_seed": [result],
            "protocol": {
                "architecture_arm": args.variant,
                "augmentation": "identical latest-gold paired affine",
                "batch_size": args.batch_size,
                "data_arm": "trusted_gold_plus_active300_plus_typepilot300",
                "loss_variant": LOSS_VARIANT.get(
                    args.variant, "standard_bce_dice"),
                "model_selection": "same writer-disjoint development split; no test",
                "negative_ratio": args.negative_ratio,
                "optimizer": "AdamW",
                "optimizer_steps": args.optimizer_steps,
                "package_versions": package_versions(),
                "paper_baseline_runner_sha256": digest_file(
                    Path(__file__).resolve()),
                "seed": args.seed,
            },
            "schema_version": SCHEMA,
            "type_targets_prior": type_audit,
        }
        data["base"].write_json(staging / "summary.json", summary)
        data["base"].write_json(staging / "run_manifest.json", {
            "arm": args.variant,
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_arm": "trusted_gold_plus_active300_plus_typepilot300",
            "m6_frozen_test_touched": False,
            "package_versions": package_versions(),
            "pretrained_weight_sha256": {
                "hrnet_w18": digest_file(args.hrnet_w18_weights),
                "resnet34": digest_file(args.resnet34_weights),
                "resnet50": digest_file(args.resnet50_weights),
                "segformer_b0": digest_file(args.segformer_b0_weights),
                "segformer_b2": digest_file(args.segformer_b2_weights),
            },
            "reference_evaluation_materialised": False,
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "seed": args.seed,
            "split_manifest_sha256": digest_file(
                staging / "split_manifest.jsonl"),
        })
        data["base"].write_json(staging / "complete.json", {
            "schema_version": SCHEMA, "status": "complete"})
        data["base"].write_recursive_sha256s(staging)
        staging.rename(output)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return summary


def main() -> int:
    typepilot = load_typepilot()
    parser = build_parser(typepilot)
    args = parser.parse_args()
    try:
        summary = run(args, typepilot)
    except (Baseline422Error, typepilot.Experiment418Error, RuntimeError) as exc:
        raise SystemExit(f"experiment 422 refused: {exc}") from exc
    print(json.dumps({
        "arm": args.variant,
        "data": summary["data"],
        "ensemble": summary.get("ensemble", {}),
        "status": "complete" if not args.validate_only else "validated",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

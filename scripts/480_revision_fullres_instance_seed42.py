#!/usr/bin/env python3
"""Full-resolution instance-mask ablations for handwritten revisions.

Experiment 480 fixes the resolution, duplicate-loss and operating-point
confounds identified after experiment 478.  It keeps the MiT-B0 encoder,
SegFormer feature decoder, query count and query decoder fixed, while exposing
three independent switches:

* stride-4 versus a lightweight full-resolution mask pixel decoder;
* legacy, unmatched-query-only, or disabled duplicate suppression;
* BCE+Dice, BCE+soft-IoU, or BCE+Dice+clDice matched-mask supervision.

The query refiner always runs at stride 2 when the full-resolution decoder is
enabled; only its residual is upsampled.  Instance mask and score thresholds
are selected on the development split, while the factory 0.5/0.5 result is
retained beside it.  Only seed 42 is admitted for structural screening.
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
SCHEMA = "hme-revision-fullres-instance-seed42-v1"

SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236
MATCH_IOUS = (0.10, 0.25, 0.50)
CONNECTIVITY = 8

QUERY_DIMENSION = 160
QUERY_LAYERS = 4
QUERY_COUNT = 16
ATTENTION_STRIDE = 8
DENSE_CHANNELS = 160
MASK_DIMENSION = 8
REFINE_CHANNELS = 16

PIXEL_DECODERS = ("stride4", "fullres")
DUPLICATE_MODES = ("legacy", "unmatched", "off")
MASK_LOSSES = ("dice", "soft_iou", "cldice")
MASK_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
SCORE_THRESHOLDS = (0.45, 0.55, 0.65, 0.75, 0.85)


class FullResolution480Error(RuntimeError):
    """Raised when the locked create-only protocol is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise FullResolution480Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def soft_skeleton(value: Any, iterations: int = 5) -> Any:
    """Differentiable morphology used by the published clDice baseline."""
    from torch.nn import functional

    eroded = -functional.max_pool2d(-value, 3, 1, 1)
    opened = functional.max_pool2d(eroded, 3, 1, 1)
    output = functional.relu(value - opened)
    current = value
    for _ in range(iterations):
        eroded = -functional.max_pool2d(-current, 3, 1, 1)
        opened = functional.max_pool2d(eroded, 3, 1, 1)
        delta = functional.relu(eroded - opened)
        output = output + functional.relu(delta - output * delta)
        current = eroded
    return output


def matched_mask_loss(selected: Any, gold: Any, mode: str,
                      cldice_weight: float) -> Any:
    """Matched-query loss; Hungarian assignment itself remains unchanged."""
    import torch
    from torch.nn import functional

    probability = selected.sigmoid()
    dimensions = tuple(range(1, selected.ndim))
    if mode == "dice" or mode == "cldice":
        numerator = 2.0 * (probability * gold).sum(dimensions) + 1.0
        denominator = (probability.sum(dimensions)
                       + gold.sum(dimensions) + 1.0)
        region = (1.0 - numerator / denominator).mean()
    elif mode == "soft_iou":
        intersection = (probability * gold).sum(dimensions)
        union = (probability + gold - probability * gold).sum(dimensions)
        region = (1.0 - (intersection + 1.0) / (union + 1.0)).mean()
    else:
        raise FullResolution480Error(f"unknown mask loss: {mode}")
    loss = functional.binary_cross_entropy_with_logits(selected, gold) + region
    if mode == "cldice":
        prediction_skeleton = soft_skeleton(probability)
        target_skeleton = soft_skeleton(gold)
        tprec = ((prediction_skeleton * gold).sum(dimensions) + 1.0) \
            / (prediction_skeleton.sum(dimensions) + 1.0)
        tsens = ((target_skeleton * probability).sum(dimensions) + 1.0) \
            / (target_skeleton.sum(dimensions) + 1.0)
        cldice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-6)
        loss = loss + cldice_weight * cldice.mean()
    return loss


def instance_loss(probe: Any, mask_logits: Any, objectness: Any,
                  batch_targets: Sequence[Sequence[Any]], mode: str,
                  cldice_weight: float) -> tuple[Any, Any]:
    import torch
    from torch.nn import functional

    mask_total = torch.zeros((), device=mask_logits.device)
    object_total = torch.zeros((), device=mask_logits.device)
    for index, targets in enumerate(batch_targets):
        rows, columns = probe.hungarian_match(
            mask_logits[index], objectness[index], targets)
        label = torch.zeros(mask_logits.shape[1], device=mask_logits.device)
        if rows:
            row_index = torch.as_tensor(rows, device=label.device)
            label[row_index] = 1.0
            selected = mask_logits[index][rows]
            gold = torch.stack([targets[column] for column in columns]).to(
                selected.dtype)
            mask_total = mask_total + matched_mask_loss(
                selected, gold, mode, cldice_weight)
        object_total = object_total + functional.binary_cross_entropy_with_logits(
            objectness[index], label)
    count = mask_logits.shape[0]
    return mask_total / count, object_total / count


def duplicate_loss(probe: Any, fusion: Any, mask_logits: Any,
                   objectness: Any,
                   batch_targets: Sequence[Sequence[Any]], mode: str) -> Any:
    """Suppress duplicates globally, only among unmatched queries, or not at all."""
    import torch

    if mode == "legacy":
        return fusion.duplicate_query_loss(mask_logits, objectness)
    if mode == "off":
        return mask_logits.sum() * 0.0
    if mode != "unmatched":
        raise FullResolution480Error(f"unknown duplicate mode: {mode}")

    losses = []
    for index, targets in enumerate(batch_targets):
        matched, _columns = probe.hungarian_match(
            mask_logits[index], objectness[index], targets)
        matched_set = set(matched)
        unmatched = [query for query in range(mask_logits.shape[1])
                     if query not in matched_set]
        if len(unmatched) < 2:
            continue
        probability = mask_logits[index, unmatched].sigmoid().flatten(1)
        weighted = probability * objectness[index, unmatched].sigmoid()[:, None]
        intersection = weighted @ weighted.t()
        areas = weighted.sum(dim=1)
        dice = (2.0 * intersection + 1e-5) / (
            areas[:, None] + areas[None, :] + 1e-5)
        keep = torch.triu(torch.ones_like(dice, dtype=torch.bool), diagonal=1)
        losses.append(dice[keep].mean())
    return torch.stack(losses).mean() if losses else mask_logits.sum() * 0.0


def build_predicted_instances(masks: np.ndarray, scores: np.ndarray,
                              mask_threshold: float,
                              score_threshold: float) -> list[list[np.ndarray]]:
    predicted = []
    for row_masks, row_scores in zip(masks, scores):
        row = []
        for mask, score in zip(row_masks, row_scores):
            binary = mask >= mask_threshold
            if score >= score_threshold and binary.any():
                row.append(binary)
        predicted.append(row)
    return predicted


def metrics_with_fp(probe: Any, predicted: Sequence[Sequence[np.ndarray]],
                    truths: Sequence[Sequence[np.ndarray]],
                    match_iou: float, line_count: int) -> dict[str, Any]:
    metrics = probe.instance_component_metrics(predicted, truths, match_iou)
    metrics["fp_components_per_line"] = \
        metrics["false_positive"] / line_count
    return metrics


def evaluate(probe: Any, extensions: Any, union: np.ndarray,
             masks: np.ndarray, scores: np.ndarray, rows: np.ndarray,
             positive_count: int, gold_masks: Sequence[np.ndarray],
             args: argparse.Namespace, joint: Any) -> dict[str, Any]:
    """Select instance thresholds on dev and retain the factory result."""
    truth = np.concatenate((
        np.stack(gold_masks),
        np.zeros((union.shape[0] - positive_count, *union.shape[1:]),
                 dtype=np.float32)), axis=0)
    best_pixel = None
    for threshold in joint.THRESHOLDS:
        predicted_union = union >= threshold
        intersection = float(np.logical_and(predicted_union, truth > 0.5).sum())
        denominator = float(np.logical_or(predicted_union, truth > 0.5).sum())
        iou = intersection / denominator if denominator else 0.0
        candidate = (iou, -threshold)
        if best_pixel is None or candidate > best_pixel[0]:
            best_pixel = (candidate, threshold, iou)
    assert best_pixel is not None
    pixel_threshold, pixel_iou = best_pixel[1], best_pixel[2]

    result: dict[str, Any] = {
        "threshold": pixel_threshold,
        "pixel_iou": pixel_iou,
        "union": {},
    }
    for match_iou in MATCH_IOUS:
        metrics = extensions.component_metrics(
            union, truth, pixel_threshold, connectivity=CONNECTIVITY,
            match_iou=match_iou)
        result["union"][f"{match_iou:.2f}"] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }

    truths = [probe.gold_instances(row) for row in truth]
    factory_predicted = build_predicted_instances(
        masks, scores, args.mask_threshold, args.score_threshold)
    factory = {
        f"{match_iou:.2f}": metrics_with_fp(
            probe, factory_predicted, truths, match_iou, union.shape[0])
        for match_iou in MATCH_IOUS
    }

    best = None
    for mask_threshold in MASK_THRESHOLDS:
        for score_threshold in SCORE_THRESHOLDS:
            predicted = build_predicted_instances(
                masks, scores, mask_threshold, score_threshold)
            primary = metrics_with_fp(
                probe, predicted, truths, 0.50, union.shape[0])
            key = (primary["f1"], primary["recall"], primary["precision"],
                   -primary["false_positive"], mask_threshold,
                   score_threshold)
            if best is None or key > best[0]:
                best = (key, mask_threshold, score_threshold, predicted)
    assert best is not None
    selected = {
        f"{match_iou:.2f}": metrics_with_fp(
            probe, best[3], truths, match_iou, union.shape[0])
        for match_iou in MATCH_IOUS
    }
    result.update({
        # Compatibility alias: checkpoint selection uses the selected dev point.
        "instance": selected,
        "instance_factory": factory,
        "instance_selected": selected,
        "instance_operating_point": {
            "factory": {
                "mask_threshold": args.mask_threshold,
                "score_threshold": args.score_threshold,
            },
            "grid": {
                "mask_thresholds": list(MASK_THRESHOLDS),
                "score_thresholds": list(SCORE_THRESHOLDS),
            },
            "selected": {
                "mask_threshold": best[1],
                "score_threshold": best[2],
            },
            "selection_metric": "instance_f1_match_iou_0.50",
        },
        "predicted_instances": int(sum(len(item) for item in best[3])),
        "predicted_instances_factory": int(
            sum(len(item) for item in factory_predicted)),
        "gold_instances": int(sum(len(item) for item in truths)),
    })

    row_labels = np.concatenate((
        np.ones(positive_count, dtype=np.float32),
        np.zeros(union.shape[0] - positive_count, dtype=np.float32)))
    result["row"] = {
        "balanced_accuracy_head": None,
        "balanced_accuracy_from_instances": joint.balanced_accuracy(
            scores.max(axis=1), row_labels),
    }
    return result


def build_model(baseline: Any, fusion: Any, args: argparse.Namespace,
                holder: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional

    segmenter = baseline._smp_model("segformer_b0", args)
    full_resolution = args.pixel_decoder == "fullres"

    class ConvNormAct(nn.Sequential):
        def __init__(self, inputs: int, outputs: int):
            super().__init__(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))

    class FullResolutionPixelDecoder(nn.Module):
        """Recover H/2 and H detail absent from MiT-B0's stride-4 stem."""

        def __init__(self):
            super().__init__()
            self.skip_full = ConvNormAct(1, 32)
            self.skip_half = ConvNormAct(1, 64)
            self.reduce_half = nn.Conv2d(DENSE_CHANNELS, 64, 1, bias=False)
            self.refine_half = ConvNormAct(64, 64)
            self.reduce_full = nn.Conv2d(64, 32, 1, bias=False)
            self.refine_full = ConvNormAct(32, 32)
            self.output = nn.Conv2d(32, MASK_DIMENSION, 1)

        def forward(self, quarter: Any, encoder_input: Any) -> Any:
            half_size = (encoder_input.shape[-2] // 2,
                         encoder_input.shape[-1] // 2)
            half = functional.interpolate(
                quarter, size=half_size, mode="bilinear", align_corners=False)
            half = self.reduce_half(half) + self.skip_half(
                functional.avg_pool2d(encoder_input, 2, 2))
            half = self.refine_half(half)
            full = functional.interpolate(
                half, size=encoder_input.shape[-2:], mode="bilinear",
                align_corners=False)
            full = self.reduce_full(full) + self.skip_full(encoder_input)
            return self.output(self.refine_full(full))

    class CrossOnlyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_query = nn.LayerNorm(QUERY_DIMENSION)
            self.cross = nn.MultiheadAttention(
                QUERY_DIMENSION, 4, batch_first=True)
            self.norm_feed = nn.LayerNorm(QUERY_DIMENSION)
            self.feed = nn.Sequential(
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION * 4),
                nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIMENSION * 4, QUERY_DIMENSION))

        def forward(self, queries: Any, memory: Any) -> Any:
            normed = self.norm_query(queries)
            attended, _ = self.cross(normed, memory, memory, need_weights=False)
            queries = queries + attended
            return queries + self.feed(self.norm_feed(queries))

    class CrossOnlyStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                CrossOnlyLayer() for _ in range(QUERY_LAYERS))

        def forward(self, queries: Any, memory: Any) -> Any:
            for layer in self.layers:
                queries = layer(queries, memory)
            return queries

    class QueryRefiner(nn.Module):
        """Query-conditioned local correction, fullres runs at stride 2."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(4, REFINE_CHANNELS, 3, padding=1, bias=False),
                nn.BatchNorm2d(REFINE_CHANNELS), nn.ReLU(inplace=True),
                nn.Conv2d(REFINE_CHANNELS, REFINE_CHANNELS, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(REFINE_CHANNELS), nn.ReLU(inplace=True))
            self.film = nn.Linear(QUERY_DIMENSION, REFINE_CHANNELS * 2)
            self.output = nn.Conv2d(REFINE_CHANNELS, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, base_logits: Any, dense_probability: Any,
                    ink: Any, queries: Any) -> Any:
            batch, count, height, width = base_logits.shape
            base_probability = base_logits.sigmoid()
            value = torch.stack((
                base_probability,
                dense_probability.expand(-1, count, -1, -1),
                (base_probability - dense_probability).abs(),
                ink.expand(-1, count, -1, -1),
            ), dim=2).reshape(batch * count, 4, height, width)
            features = self.features(value)
            scale, bias = self.film(queries).chunk(2, dim=-1)
            scale = scale.reshape(batch * count, REFINE_CHANNELS, 1, 1)
            bias = bias.reshape(batch * count, REFINE_CHANNELS, 1, 1)
            features = features * (1.0 + scale.tanh()) + bias
            delta = self.output(functional.relu(features)).reshape(
                batch, count, height, width)
            support = functional.max_pool2d(
                base_probability.detach(), 9, stride=1, padding=4)
            return base_logits + delta * support

    class FullResolutionDenseInstanceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = segmenter.encoder
            self.dense_decoder = segmenter.decoder
            decoder_channels = int(segmenter.segmentation_head[0].in_channels)
            self.shared_projection = nn.Sequential(
                nn.Conv2d(decoder_channels, DENSE_CHANNELS, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(DENSE_CHANNELS), nn.ReLU(inplace=True))
            self.pixel_decoder = FullResolutionPixelDecoder() \
                if full_resolution else None
            feature_channels = MASK_DIMENSION if full_resolution \
                else DENSE_CHANNELS
            self.dense_logit = nn.Conv2d(feature_channels, 1, 1)
            self.dense_feature_bias = nn.Conv2d(
                1, QUERY_DIMENSION, 1, bias=False)
            self.dense_gate_scale = nn.Parameter(torch.tensor(-2.0))
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1))

            self.queries = nn.Parameter(
                torch.randn(QUERY_COUNT, QUERY_DIMENSION) * 0.02)
            self.decoder_stack = CrossOnlyStack()
            self.mask_embed = nn.Sequential(
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
                nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIMENSION, feature_channels))
            self.objectness = nn.Linear(QUERY_DIMENSION, 1)
            self.quality = nn.Linear(QUERY_DIMENSION, 1)
            self.refiner = QueryRefiner()

            self.last_dense_logits = None
            self.last_fused_logits = None
            self.last_query_union = None
            self.last_raw_objectness = None
            self.last_quality_logits = None

        def forward(self, value: Any) -> tuple[Any, Any, None]:
            features = self.encoder(value)
            quarter = self.shared_projection(self.dense_decoder(features))
            mask_pixels = self.pixel_decoder(quarter, features[0]) \
                if self.pixel_decoder is not None else quarter
            dense_native = self.dense_logit(mask_pixels)
            dense_full = dense_native if full_resolution else \
                functional.interpolate(
                    dense_native, size=value.shape[-2:], mode="bilinear",
                    align_corners=False)

            dense_for_queries = functional.interpolate(
                dense_full.sigmoid(), size=quarter.shape[-2:], mode="bilinear",
                align_corners=False)
            scale = self.dense_gate_scale.sigmoid()
            instance_pixels = quarter * (1.0 + scale * dense_for_queries) \
                + self.dense_feature_bias(dense_for_queries)
            memory_map = functional.avg_pool2d(
                instance_pixels, ATTENTION_STRIDE, ATTENTION_STRIDE)
            memory = memory_map.flatten(2).transpose(1, 2)
            queries = self.queries.unsqueeze(0).expand(value.shape[0], -1, -1)
            queries = self.decoder_stack(queries, memory)

            base_masks = torch.einsum(
                "bqd,bdhw->bqhw", self.mask_embed(queries), mask_pixels)
            ink_full = (1.0 - ((value + 1.0) * 0.5)).clamp(0.0, 1.0)
            if full_resolution:
                half_size = (value.shape[-2] // 2, value.shape[-1] // 2)
                base_half = functional.interpolate(
                    base_masks, size=half_size, mode="bilinear",
                    align_corners=False)
                dense_half = functional.interpolate(
                    dense_full.sigmoid(), size=half_size, mode="bilinear",
                    align_corners=False)
                ink_half = functional.interpolate(
                    ink_full, size=half_size, mode="bilinear",
                    align_corners=False)
                refined_half = self.refiner(
                    base_half, dense_half, ink_half, queries)
                delta = refined_half - base_half
                masks = base_masks + functional.interpolate(
                    delta, size=value.shape[-2:], mode="bilinear",
                    align_corners=False)
            else:
                ink_native = functional.interpolate(
                    ink_full, size=base_masks.shape[-2:], mode="bilinear",
                    align_corners=False)
                masks = self.refiner(
                    base_masks, dense_native.sigmoid(), ink_native, queries)

            raw_objectness = self.objectness(queries).squeeze(-1)
            quality_logits = self.quality(queries).squeeze(-1)
            combined = (raw_objectness.sigmoid()
                        * quality_logits.sigmoid()).clamp_min(1e-8).sqrt()
            final_objectness = fusion.probability_to_logit(combined)
            query_union = fusion.query_union_probability(
                masks, final_objectness, value.shape[-2:], "soft_or")
            dense_probability_full = dense_full.sigmoid()
            disagreement = (dense_probability_full - query_union).abs()
            gate = self.fusion_gate(torch.cat((
                dense_probability_full, query_union, disagreement, ink_full),
                dim=1)).sigmoid()
            fused_probability = gate * dense_probability_full \
                + (1.0 - gate) * query_union

            self.last_dense_logits = dense_full
            self.last_fused_logits = fusion.probability_to_logit(
                fused_probability)
            self.last_query_union = query_union
            self.last_raw_objectness = raw_objectness
            self.last_quality_logits = quality_logits
            return masks, final_objectness, None

    model = FullResolutionDenseInstanceModel()
    holder.update({
        "model": model,
        "configuration": {
            "backbone": "mit_b0_shared_segformer_all_mlp",
            "dense_query_coupling_stride": 4,
            "duplicate_suppression": args.duplicate_mode,
            "instance_head": {
                "dimension": QUERY_DIMENSION,
                "layers": QUERY_LAYERS,
                "queries": QUERY_COUNT,
                "self_attention": False,
            },
            "mask_dimension": MASK_DIMENSION if full_resolution
                else DENSE_CHANNELS,
            "mask_loss": args.mask_loss,
            "mask_output_stride": 1 if full_resolution else 4,
            "pixel_decoder": args.pixel_decoder,
            "query_refiner_resolution": "stride2_delta_upsampled"
                if full_resolution else "stride4",
            "skip_source": "encoder_input_learned_stem; MiT-B0 exposes no "
                           "learned stride-1/2 feature maps",
        },
    })
    return model


def make_loss_adapter(holder: Mapping[str, Any], args: argparse.Namespace,
                      fusion: Any):
    from torch.nn import functional

    def loss(probe: Any, mask_logits: Any, objectness: Any,
             batch_targets: Sequence[Sequence[Any]]) -> tuple[Any, Any]:
        model = holder["model"]
        mask_part, objectness_part = instance_loss(
            probe, mask_logits, model.last_raw_objectness, batch_targets,
            args.mask_loss, args.cldice_weight)
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
        duplicate = duplicate_loss(
            probe, fusion, mask_logits, objectness, batch_targets,
            args.duplicate_mode)
        quality = fusion.quality_target_loss(
            probe, mask_logits, model.last_raw_objectness,
            model.last_quality_logits, batch_targets)
        holder["last_auxiliary_losses"] = {
            "consistency": float(consistency.detach()),
            "dense": float(dense_part.detach()),
            "duplicate": float(duplicate.detach()),
            "fused": float(fused_part.detach()),
            "quality": float(quality.detach()),
        }
        return (
            mask_part
            + args.dense_loss_weight * dense_part
            + args.fused_loss_weight * fused_part
            + args.consistency_weight * consistency
            + args.duplicate_weight * duplicate
            + args.quality_weight * quality,
            objectness_part,
        )

    return loss


def predict(model: Any, rows: Sequence[Mapping[str, Any]], batch_size: int,
            device: Any, _arm: str):
    import torch
    from torch.nn import functional

    model.eval()
    unions, mask_chunks, score_chunks, row_chunks = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            image = np.stack([row["image"] for row in chunk]).astype(np.float32)
            value = torch.from_numpy(
                image[:, None] / 127.5 - 1.0).to(device)
            mask_logits, objectness, _row = model(value)
            # Full-resolution arms already return HxW masks; the required
            # pixel-decoder-off control returns stride-4 masks and must follow
            # the same final bilinear reconstruction as experiment 478.
            if mask_logits.shape[-2:] != value.shape[-2:]:
                mask_logits = functional.interpolate(
                    mask_logits, size=value.shape[-2:], mode="bilinear",
                    align_corners=False)
            unions.append(model.last_fused_logits.sigmoid().cpu().numpy()[:, 0])
            mask_chunks.append(mask_logits.sigmoid().cpu().numpy())
            score_chunks.append(objectness.sigmoid().cpu().numpy())
            row_chunks.append(np.zeros(len(chunk), dtype=np.float32))
    model.train()
    return (np.concatenate(unions, axis=0),
            np.concatenate(mask_chunks, axis=0),
            np.concatenate(score_chunks, axis=0),
            np.concatenate(row_chunks, axis=0))


def parser(joint: Any, baseline: Any, typepilot: Any,
           argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--pixel-decoder", choices=PIXEL_DECODERS, required=True)
    base.add_argument("--duplicate-mode", choices=DUPLICATE_MODES, required=True)
    base.add_argument("--mask-loss", choices=MASK_LOSSES, required=True)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    base.add_argument("--duplicate-weight", type=float, default=0.05)
    base.add_argument("--quality-weight", type=float, default=0.5)
    base.add_argument("--cldice-weight", type=float, default=0.20)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == "--arm" or value.startswith("--arm=")
           for value in arguments):
        raise FullResolution480Error("experiment 480 fixes --arm to instance")
    return base.parse_args([*arguments, "--arm", "instance"])


def finalise(engine_output: Path, final_output: Path,
             args: argparse.Namespace, holder: Mapping[str, Any],
             dependencies: Mapping[str, Any]) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise FullResolution480Error(
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
        raise FullResolution480Error("development gold-instance count changed")
    protocol = {
        "cldice_weight": args.cldice_weight,
        "consistency_weight": args.consistency_weight,
        "dense_loss_weight": args.dense_loss_weight,
        "duplicate_mode": args.duplicate_mode,
        "duplicate_weight": args.duplicate_weight,
        "factory_instance_workpoint": {
            "mask_threshold": args.mask_threshold,
            "score_threshold": args.score_threshold,
        },
        "frozen_test_touched": False,
        "fused_loss_weight": args.fused_loss_weight,
        "instance_workpoint_grid": {
            "mask_thresholds": list(MASK_THRESHOLDS),
            "score_thresholds": list(SCORE_THRESHOLDS),
        },
        "m6_or_reference_materialised": False,
        "mask_loss": args.mask_loss,
        "pixel_decoder": args.pixel_decoder,
        "primary_metric": "instance_f1_match_iou_0.50",
        "quality_weight": args.quality_weight,
        "query_refiner_memory_strategy": "stride2_delta_upsampled"
            if args.pixel_decoder == "fullres" else "stride4",
        "selected_instance_workpoint": validation[
            "instance_operating_point"]["selected"],
        "writer_overlap_fit_validation": 0,
    }
    summary.update({
        "architecture": holder["configuration"],
        "method": "full_resolution_instance_mask_ablation",
        "schema_version": SCHEMA,
    })
    summary["protocol"].update(protocol)
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "architecture": holder["configuration"],
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
        "architecture": holder["configuration"],
        "schema_version": "hme-revision-fullres-instance-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"fullres_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "duplicate_mode": args.duplicate_mode,
        "mask_loss": args.mask_loss,
        "pixel_decoder": args.pixel_decoder,
        "seed": args.seed,
        "status": "complete",
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for480_fullres")
    baseline = load_module(BASELINE_PATH, "hme422_for480_fullres")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for480_fullres")
    fusion = load_module(FUSION_PATH, "hme478_for480_fullres")
    args = parser(joint, baseline, typepilot, argv)
    if args.seed != SEED:
        raise SystemExit("480 fullres refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"480 fullres refused: output already exists: {final_output}")

    holder: dict[str, Any] = {}
    joint.build_model = lambda _baseline, _ablation, run_args: \
        build_model(baseline, fusion, run_args, holder)
    joint.instance_loss = make_loss_adapter(holder, args, fusion)
    joint.predict = predict
    joint.evaluate = lambda probe, extensions, union, masks, scores, rows, \
            positive_count, gold_masks, run_args: evaluate(
                probe, extensions, union, masks, scores, rows,
                positive_count, gold_masks, run_args, joint)
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("480 fullres refused: sample counts changed")
        print(json.dumps({
            "counts": result["counts"],
            "frozen_test_touched": False,
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
    print(json.dumps({
        "instance_f1_050_factory":
            validation["instance_factory"]["0.50"]["f1"],
        "instance_f1_050_selected":
            validation["instance_selected"]["0.50"]["f1"],
        "output_dir": str(final_output),
        "pixel_iou": validation["pixel_iou"],
        "seed": args.seed,
        "selected_workpoint":
            validation["instance_operating_point"]["selected"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Spatially grounded query-mask ablations on experiment 480's best model.

Five single-change arms test missing spatial/instance mechanisms: 2-D memory
position encoding, query-relative mask coordinates with centroid supervision,
a CondInst-style dynamic mask residual, query self-attention, and horizontal
position-indexed matching.  Three additional arms cross self-attention with
the first three mechanisms.  Architecture, data and all non-target losses are
otherwise locked to experiment 480 ``fullres_dup_off``.

The primary metric is instance F1 at IoU>=.50.  Every validation record keeps
factory and development-selected workpoints, area-bin recall, oracle recall,
and recall/oracle recall stratified by one versus multiple gold instances on
the line.  Only seed 42 is admitted for this structural screen.
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
import types
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
SIZE_PATH = ROOT / "scripts/481_revision_size_separation_seed42.py"
FAST_LOOP_PATH = ROOT / "scripts/483_revision_fast_train_loop.py"
SCHEMA = "hme-revision-spatial-query-seed42-v1"

SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236
EXPECTED_AREA_BIN_COUNTS = (75, 73, 88)
EXPECTED_GOLD_COUNT_STRATA = (60, 176)
BASELINE_OUTPUT = \
    "work/revision_fullres_instance_seed42_v1/fullres_dup_off"

QUERY_DIMENSION = 160
QUERY_COUNT = 16
MASK_DIMENSION = 8
DYNAMIC_LAYERS = 3

ARMS = (
    "memory_position",
    "relative_coordinates",
    "dynamic_condinst",
    "self_attention",
    "position_indexed_queries",
    "memory_position_self_attention",
    "relative_coordinates_self_attention",
    "dynamic_condinst_self_attention",
)


class SpatialQuery482Error(RuntimeError):
    """Raised when experiment 482's locked protocol is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SpatialQuery482Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def arm_switches(arm: str) -> dict[str, bool]:
    if arm not in ARMS:
        raise SpatialQuery482Error(f"unknown arm: {arm}")
    return {
        "memory_position": arm.startswith("memory_position"),
        "relative_coordinates": arm.startswith("relative_coordinates"),
        "dynamic_condinst": arm.startswith("dynamic_condinst"),
        "self_attention": arm == "self_attention"
            or arm.endswith("_self_attention"),
        "position_indexed_queries": arm == "position_indexed_queries",
    }


def dynamic_parameter_count(channels: int = MASK_DIMENSION,
                            layers: int = DYNAMIC_LAYERS) -> int:
    total = 0
    inputs = channels + 2
    for layer in range(layers):
        outputs = 1 if layer == layers - 1 else channels
        total += inputs * outputs + outputs
        inputs = outputs
    return total


def sine_position_encoding(height: int, width: int, dimension: int,
                           device: Any, dtype: Any) -> Any:
    """Parameter-free DETR-style 2-D sine/cosine encoding in BCHW form."""
    import math
    import torch

    if dimension % 4:
        raise SpatialQuery482Error("2-D sine position dimension must divide by 4")
    axis_dimension = dimension // 2
    frequency_count = axis_dimension // 2
    frequencies = torch.arange(
        frequency_count, device=device, dtype=torch.float32)
    frequencies = 10000.0 ** (2.0 * frequencies / axis_dimension)
    y = torch.linspace(0.0, 2.0 * math.pi, height,
                       device=device, dtype=torch.float32)[:, None]
    x = torch.linspace(0.0, 2.0 * math.pi, width,
                       device=device, dtype=torch.float32)[:, None]
    y_embedding = torch.stack(
        ((y / frequencies).sin(), (y / frequencies).cos()), dim=-1
    ).flatten(1).t()[:, :, None].expand(-1, -1, width)
    x_embedding = torch.stack(
        ((x / frequencies).sin(), (x / frequencies).cos()), dim=-1
    ).flatten(1).t()[:, None, :].expand(-1, height, -1)
    return torch.cat((y_embedding, x_embedding), dim=0)[None].to(dtype=dtype)


def coordinate_grid(height: int, width: int, device: Any, dtype: Any,
                    *, signed: bool) -> tuple[Any, Any]:
    import torch

    lower, upper = (-1.0, 1.0) if signed else (0.0, 1.0)
    y = torch.linspace(lower, upper, height, device=device, dtype=dtype)
    x = torch.linspace(lower, upper, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def relative_coordinate_masks(mask_pixels: Any, queries: Any,
                              centres: Any, mask_embed: Any) -> Any:
    """Linear masks over appearance plus query-relative x/y coordinates."""
    batch, channels, height, width = mask_pixels.shape
    count = queries.shape[1]
    grid_x, grid_y = coordinate_grid(
        height, width, mask_pixels.device, mask_pixels.dtype, signed=False)
    relative_x = grid_x[None, None] - centres[:, :, 0, None, None]
    relative_y = grid_y[None, None] - centres[:, :, 1, None, None]
    appearance = mask_pixels[:, None].expand(
        batch, count, channels, height, width)
    augmented = torch_cat((appearance, relative_x[:, :, None],
                           relative_y[:, :, None]), dim=2)
    weights = mask_embed(queries)
    return (weights[:, :, :, None, None] * augmented).sum(dim=2)


def torch_cat(values: Sequence[Any], dim: int) -> Any:
    import torch

    return torch.cat(tuple(values), dim=dim)


def dynamic_masks(mask_pixels: Any, queries: Any, controller: Any) -> Any:
    """Three-layer grouped 1x1 CondInst residual with absolute x/y channels."""
    from torch.nn import functional

    batch, channels, height, width = mask_pixels.shape
    count = queries.shape[1]
    grid_x, grid_y = coordinate_grid(
        height, width, mask_pixels.device, mask_pixels.dtype, signed=True)
    grid = torch_cat((grid_x[None, None], grid_y[None, None]), dim=1)
    grid = grid.expand(batch, -1, -1, -1)
    stacked = torch_cat((mask_pixels, grid), dim=1)
    value = stacked[:, None].expand(-1, count, -1, -1, -1).reshape(
        batch * count, channels + 2, height, width)
    parameters = controller(queries).reshape(batch * count, -1)
    offset = 0
    inputs = channels + 2
    for layer in range(DYNAMIC_LAYERS):
        outputs = 1 if layer == DYNAMIC_LAYERS - 1 else channels
        weight = parameters[:, offset:offset + inputs * outputs]
        offset += inputs * outputs
        bias = parameters[:, offset:offset + outputs]
        offset += outputs
        weight = weight.reshape(batch * count * outputs, inputs, 1, 1)
        value = functional.conv2d(
            value.reshape(1, -1, height, width), weight,
            bias.reshape(-1), groups=batch * count)
        value = value.reshape(batch * count, outputs, height, width)
        if layer < DYNAMIC_LAYERS - 1:
            value = functional.relu(value)
        inputs = outputs
    if offset != parameters.shape[1]:
        raise SpatialQuery482Error("dynamic parameter layout did not close")
    return value.reshape(batch, count, height, width)


def gold_centres(targets: Sequence[Any], reference: Any) -> Any:
    import torch

    height, width = reference.shape[-2:]
    centres = []
    for target in targets:
        y, x = torch.nonzero(target >= 0.5, as_tuple=True)
        if x.numel() == 0:
            centres.append(torch.zeros(
                2, device=reference.device, dtype=reference.dtype))
        else:
            centres.append(torch.stack((
                (x.to(reference.dtype).mean() + 0.5) / width,
                (y.to(reference.dtype).mean() + 0.5) / height,
            )))
    return torch.stack(centres) if centres else torch.empty(
        (0, 2), device=reference.device, dtype=reference.dtype)


def spatial_hungarian_match(probe: Any, mask_logits: Any, objectness: Any,
                            targets: Sequence[Any], enabled: bool,
                            location_weight: float) -> tuple[list[int], list[int]]:
    if not enabled:
        return probe.hungarian_match(mask_logits, objectness, targets)
    if not targets:
        return [], []
    import torch
    from scipy.optimize import linear_sum_assignment

    probability = mask_logits.sigmoid().flatten(1)
    gold = torch.stack(targets).flatten(1).to(probability.dtype)
    numerator = 2.0 * probability @ gold.t()
    denominator = probability.sum(1)[:, None] + gold.sum(1)[None, :]
    dice_cost = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    class_cost = -torch.log_softmax(
        torch.stack((objectness, -objectness), dim=1), dim=1
    )[:, 0][:, None]
    target_centres = gold_centres(targets, mask_logits)[:, 0]
    query_centres = (torch.arange(
        mask_logits.shape[0], device=mask_logits.device,
        dtype=mask_logits.dtype) + 0.5) / mask_logits.shape[0]
    position_cost = (query_centres[:, None] - target_centres[None, :]).abs()
    cost = (dice_cost + class_cost + location_weight * position_cost) \
        .detach().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    return list(rows), list(columns)


def build_model(baseline: Any, fusion: Any, fullres: Any,
                args: argparse.Namespace, holder: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional

    switches = arm_switches(args.spatial_arm)
    model = fullres.build_model(baseline, fusion, args, holder)

    class SelfCrossLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_self = nn.LayerNorm(QUERY_DIMENSION)
            self.self_attention = nn.MultiheadAttention(
                QUERY_DIMENSION, 4, batch_first=True)
            self.norm_cross = nn.LayerNorm(QUERY_DIMENSION)
            self.cross_attention = nn.MultiheadAttention(
                QUERY_DIMENSION, 4, batch_first=True)
            self.norm_feed = nn.LayerNorm(QUERY_DIMENSION)
            self.feed = nn.Sequential(
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION * 4),
                nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIMENSION * 4, QUERY_DIMENSION))

        def forward(self, queries: Any, memory: Any) -> Any:
            normalised = self.norm_self(queries)
            attended, _ = self.self_attention(
                normalised, normalised, normalised, need_weights=False)
            queries = queries + attended
            normalised = self.norm_cross(queries)
            attended, _ = self.cross_attention(
                normalised, memory, memory, need_weights=False)
            queries = queries + attended
            return queries + self.feed(self.norm_feed(queries))

    class SelfCrossStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                SelfCrossLayer() for _ in range(4))

        def forward(self, queries: Any, memory: Any) -> Any:
            for layer in self.layers:
                queries = layer(queries, memory)
            return queries

    if switches["self_attention"]:
        model.decoder_stack = SelfCrossStack()
    if switches["relative_coordinates"]:
        model.center_head = nn.Linear(QUERY_DIMENSION, 2)
        model.mask_embed = nn.Sequential(
            nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
            nn.ReLU(inplace=True),
            nn.Linear(QUERY_DIMENSION, MASK_DIMENSION + 2))
    else:
        model.center_head = None
    if switches["dynamic_condinst"]:
        model.dynamic_controller = nn.Linear(
            QUERY_DIMENSION, dynamic_parameter_count())
        nn.init.zeros_(model.dynamic_controller.weight)
        nn.init.zeros_(model.dynamic_controller.bias)
    else:
        model.dynamic_controller = None
    model.last_predicted_centres = None

    def spatial_forward(self: Any, value: Any) -> tuple[Any, Any, None]:
        features = self.encoder(value)
        quarter = self.shared_projection(self.dense_decoder(features))
        mask_pixels = self.pixel_decoder(quarter, features[0])
        dense_full = self.dense_logit(mask_pixels)
        dense_for_queries = functional.interpolate(
            dense_full.sigmoid(), size=quarter.shape[-2:], mode="bilinear",
            align_corners=False)
        scale = self.dense_gate_scale.sigmoid()
        instance_pixels = quarter * (1.0 + scale * dense_for_queries) \
            + self.dense_feature_bias(dense_for_queries)
        memory_map = functional.avg_pool2d(
            instance_pixels, fullres.ATTENTION_STRIDE,
            fullres.ATTENTION_STRIDE)
        if switches["memory_position"]:
            memory_map = memory_map + sine_position_encoding(
                memory_map.shape[-2], memory_map.shape[-1],
                memory_map.shape[1], memory_map.device, memory_map.dtype)
        memory = memory_map.flatten(2).transpose(1, 2)
        queries = self.queries.unsqueeze(0).expand(value.shape[0], -1, -1)
        queries = self.decoder_stack(queries, memory)

        if switches["relative_coordinates"]:
            centres = self.center_head(queries).sigmoid()
            base_masks = relative_coordinate_masks(
                mask_pixels, queries, centres, self.mask_embed)
            self.last_predicted_centres = centres
        else:
            base_masks = torch.einsum(
                "bqd,bdhw->bqhw", self.mask_embed(queries), mask_pixels)
            self.last_predicted_centres = None
        if switches["dynamic_condinst"]:
            base_masks = base_masks + dynamic_masks(
                mask_pixels, queries, self.dynamic_controller)

        ink_full = (1.0 - ((value + 1.0) * 0.5)).clamp(0.0, 1.0)
        half_size = (value.shape[-2] // 2, value.shape[-1] // 2)
        base_half = functional.interpolate(
            base_masks, size=half_size, mode="bilinear", align_corners=False)
        dense_half = functional.interpolate(
            dense_full.sigmoid(), size=half_size, mode="bilinear",
            align_corners=False)
        ink_half = functional.interpolate(
            ink_full, size=half_size, mode="bilinear", align_corners=False)
        refined_half = self.refiner(base_half, dense_half, ink_half, queries)
        masks = base_masks + functional.interpolate(
            refined_half - base_half, size=value.shape[-2:], mode="bilinear",
            align_corners=False)

        raw_objectness = self.objectness(queries).squeeze(-1)
        quality_logits = self.quality(queries).squeeze(-1)
        combined = (raw_objectness.sigmoid()
                    * quality_logits.sigmoid()).clamp_min(1e-8).sqrt()
        final_objectness = fusion.probability_to_logit(combined)
        query_union = fusion.query_union_probability(
            masks, final_objectness, value.shape[-2:], "soft_or")
        dense_probability = dense_full.sigmoid()
        disagreement = (dense_probability - query_union).abs()
        gate = self.fusion_gate(torch_cat((
            dense_probability, query_union, disagreement, ink_full),
            dim=1)).sigmoid()
        fused_probability = gate * dense_probability \
            + (1.0 - gate) * query_union
        self.last_dense_logits = dense_full
        self.last_fused_logits = fusion.probability_to_logit(fused_probability)
        self.last_query_union = query_union
        self.last_raw_objectness = raw_objectness
        self.last_quality_logits = quality_logits
        return masks, final_objectness, None

    model.forward = types.MethodType(spatial_forward, model)
    holder["configuration"].update({
        "spatial_arm": args.spatial_arm,
        "memory_position_encoding": "2d_sine"
            if switches["memory_position"] else None,
        "relative_coordinate_masks": switches["relative_coordinates"],
        "centroid_head": switches["relative_coordinates"],
        "dynamic_mask_head": "linear_plus_condinst_residual"
            if switches["dynamic_condinst"] else None,
        "dynamic_parameters_per_query": dynamic_parameter_count()
            if switches["dynamic_condinst"] else 0,
        "query_self_attention": switches["self_attention"],
        "position_indexed_matching": switches["position_indexed_queries"],
    })
    return model


def standard_instance_losses(probe: Any, mask_logits: Any, objectness: Any,
                             batch_targets: Sequence[Sequence[Any]],
                             args: argparse.Namespace,
                             predicted_centres: Any) -> tuple[Any, Any, Any]:
    import torch
    from torch.nn import functional

    position_indexed = arm_switches(args.spatial_arm)[
        "position_indexed_queries"]
    mask_total = torch.zeros((), device=mask_logits.device)
    object_total = torch.zeros((), device=mask_logits.device)
    centre_total = torch.zeros((), device=mask_logits.device)
    for index, targets in enumerate(batch_targets):
        rows, columns = spatial_hungarian_match(
            probe, mask_logits[index], objectness[index], targets,
            position_indexed, args.location_cost_weight)
        label = torch.zeros(mask_logits.shape[1], device=mask_logits.device)
        if rows:
            row_index = torch.as_tensor(rows, device=label.device)
            label[row_index] = 1.0
            selected = mask_logits[index][rows]
            gold = torch.stack([targets[column] for column in columns]).to(
                selected.dtype)
            probability = selected.sigmoid()
            dimensions = tuple(range(1, probability.ndim))
            numerator = 2.0 * (probability * gold).sum(dimensions) + 1.0
            denominator = probability.sum(dimensions) + gold.sum(dimensions) + 1.0
            mask_total = mask_total + functional.binary_cross_entropy_with_logits(
                selected, gold) + (1.0 - numerator / denominator).mean()
            if predicted_centres is not None:
                targets_xy = gold_centres(
                    [targets[column] for column in columns], selected)
                centre_total = centre_total + functional.smooth_l1_loss(
                    predicted_centres[index, rows], targets_xy)
        object_total = object_total + functional.binary_cross_entropy_with_logits(
            objectness[index], label)
    count = mask_logits.shape[0]
    return mask_total / count, object_total / count, centre_total / count


def quality_target_loss(probe: Any, mask_logits: Any, raw_objectness: Any,
                        quality_logits: Any,
                        batch_targets: Sequence[Sequence[Any]],
                        args: argparse.Namespace) -> Any:
    import torch
    from torch.nn import functional

    position_indexed = arm_switches(args.spatial_arm)[
        "position_indexed_queries"]
    total = torch.zeros((), device=mask_logits.device)
    for index, targets in enumerate(batch_targets):
        label = torch.zeros_like(quality_logits[index])
        if targets:
            rows, columns = spatial_hungarian_match(
                probe, mask_logits[index], raw_objectness[index], targets,
                position_indexed, args.location_cost_weight)
            if rows:
                selected = mask_logits[index][rows].sigmoid()
                gold = torch.stack([targets[column]
                                    for column in columns]).to(selected.dtype)
                intersection = (selected * gold).flatten(1).sum(1)
                union = (selected + gold - selected * gold).flatten(1).sum(1)
                soft_iou = ((intersection + 1.0) / (union + 1.0)).detach()
                label[torch.as_tensor(rows, device=label.device)] = soft_iou
        total = total + functional.binary_cross_entropy_with_logits(
            quality_logits[index], label)
    return total / mask_logits.shape[0]


def make_loss_adapter(holder: Mapping[str, Any], args: argparse.Namespace,
                      fusion: Any):
    from torch.nn import functional

    def loss(probe: Any, mask_logits: Any, objectness: Any,
             batch_targets: Sequence[Sequence[Any]]) -> tuple[Any, Any]:
        model = holder["model"]
        mask_part, objectness_part, centre = standard_instance_losses(
            probe, mask_logits, model.last_raw_objectness, batch_targets,
            args, model.last_predicted_centres)
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
        quality = quality_target_loss(
            probe, mask_logits, model.last_raw_objectness,
            model.last_quality_logits, batch_targets, args)
        holder["last_auxiliary_losses"] = {
            "centre": float(centre.detach()),
            "consistency": float(consistency.detach()),
            "dense": float(dense_part.detach()),
            "fused": float(fused_part.detach()),
            "quality": float(quality.detach()),
        }
        return (
            mask_part
            + args.dense_loss_weight * dense_part
            + args.fused_loss_weight * fused_part
            + args.consistency_weight * consistency
            + args.quality_weight * quality
            + args.centre_loss_weight * centre,
            objectness_part,
        )

    return loss


def gold_count_stratified_recall(
        predicted: Sequence[Sequence[np.ndarray]],
        truths: Sequence[Sequence[np.ndarray]], sizeprobe: Any,
        match_iou: float = 0.50) -> dict[str, Any]:
    result = {
        "one_gold": {"gold": 0, "true_positive": 0},
        "multiple_gold": {"gold": 0, "true_positive": 0},
    }
    for row_predicted, row_truths in zip(predicted, truths):
        if not row_truths:
            continue
        key = "one_gold" if len(row_truths) == 1 else "multiple_gold"
        matched = sizeprobe.greedy_matched_truths(
            row_predicted, row_truths, match_iou)
        result[key]["gold"] += len(row_truths)
        result[key]["true_positive"] += len(matched)
    for values in result.values():
        values["recall"] = values["true_positive"] / values["gold"] \
            if values["gold"] else 0.0
    return result


def gold_count_stratified_oracle(masks: np.ndarray,
                                 truths: Sequence[Sequence[np.ndarray]],
                                 mask_threshold: float,
                                 match_iou: float = 0.50) -> dict[str, Any]:
    result = {
        "one_gold": {"gold": 0, "recovered": 0},
        "multiple_gold": {"gold": 0, "recovered": 0},
    }
    for row_masks, row_truths in zip(masks, truths):
        if not row_truths:
            continue
        key = "one_gold" if len(row_truths) == 1 else "multiple_gold"
        binary_queries = row_masks >= mask_threshold
        query_areas = binary_queries.reshape(binary_queries.shape[0], -1).sum(1)
        for truth in row_truths:
            truth_area = int(truth.sum())
            best = 0.0
            for query, query_area in zip(binary_queries, query_areas):
                intersection = int(np.logical_and(query, truth).sum())
                union = int(query_area) + truth_area - intersection
                best = max(best, intersection / union if union else 0.0)
            result[key]["gold"] += 1
            result[key]["recovered"] += int(best >= match_iou)
    for values in result.values():
        values["recall"] = values["recovered"] / values["gold"] \
            if values["gold"] else 0.0
    return result


def evaluate_with_diagnostics(probe: Any, extensions: Any, union: np.ndarray,
                              masks: np.ndarray, scores: np.ndarray,
                              rows: np.ndarray, positive_count: int,
                              gold_masks: Sequence[np.ndarray],
                              args: argparse.Namespace, joint: Any,
                              fullres: Any, sizeprobe: Any) -> dict[str, Any]:
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
    area = sizeprobe.area_binned_recall(selected, truths)
    area_counts = tuple(area[name]["gold"]
                        for name, _lower, _upper in sizeprobe.AREA_BINS)
    if area_counts != EXPECTED_AREA_BIN_COUNTS:
        raise SpatialQuery482Error(f"gold area-bin counts changed: {area_counts}")
    count_recall = gold_count_stratified_recall(selected, truths, sizeprobe)
    count_strata = (count_recall["one_gold"]["gold"],
                    count_recall["multiple_gold"]["gold"])
    if count_strata != EXPECTED_GOLD_COUNT_STRATA:
        raise SpatialQuery482Error(
            f"gold count strata changed: {count_strata}")
    result["instance_diagnostics"] = {
        "area_binned_recall_050": area,
        "gold_count_stratified_recall_050": count_recall,
        "gold_count_stratified_oracle_recall_050":
            gold_count_stratified_oracle(
                masks, truths, selected_point["mask_threshold"]),
        "oracle_recall_050": sizeprobe.oracle_recall(
            masks, truths, selected_point["mask_threshold"]),
        "oracle_recall_050_factory_mask_threshold": sizeprobe.oracle_recall(
            masks, truths, args.mask_threshold),
    }
    return result


def parser(joint: Any, baseline: Any, typepilot: Any,
           argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--spatial-arm", choices=ARMS, required=True)
    base.add_argument("--centre-loss-weight", type=float, default=1.0)
    base.add_argument("--location-cost-weight", type=float, default=1.0)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    base.add_argument("--quality-weight", type=float, default=0.5)
    # Off by default.  Experiments 480 and 481 never call
    # use_deterministic_algorithms, so an arm that does cannot be compared with
    # their readings without an uncontrolled variable, and experiment 446
    # already measured strict mode at 2.2x the cost.
    base.add_argument("--strict-determinism", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    forbidden = ("--arm", "--pixel-decoder", "--duplicate-mode", "--mask-loss")
    if any(any(value == flag or value.startswith(flag + "=")
               for flag in forbidden) for value in arguments):
        raise SpatialQuery482Error(
            "experiment 482 locks the experiment-480 architecture")
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
        raise SpatialQuery482Error(f"create-only output exists: {final_output}")
    for name in ("SHA256SUMS", "complete.json"):
        path = engine_output / name
        if path.exists():
            path.unlink()
    summary_path = engine_output / "summary.json"
    manifest_path = engine_output / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = summary["per_seed"][0]["validation"]
    if int(validation["gold_instances"]) != EXPECTED_GOLD_INSTANCES:
        raise SpatialQuery482Error("development gold-instance count changed")
    if "instance_diagnostics" not in validation:
        raise SpatialQuery482Error("required instance diagnostics are absent")
    switches = arm_switches(args.spatial_arm)
    protocol = {
        "baseline_output": BASELINE_OUTPUT,
        "centre_loss_weight": args.centre_loss_weight
            if switches["relative_coordinates"] else 0.0,
        "factory_instance_workpoint": {
            "mask_threshold": args.mask_threshold,
            "score_threshold": args.score_threshold,
        },
        "frozen_test_touched": False,
        "gold_area_bins_pixels": [300, 900],
        "gold_count_strata": ["one_gold", "multiple_gold"],
        "location_cost_weight": args.location_cost_weight
            if switches["position_indexed_queries"] else 0.0,
        "m6_or_reference_materialised": False,
        "primary_metric": "instance_f1_match_iou_0.50",
        "selected_instance_workpoint": validation[
            "instance_operating_point"]["selected"],
        "spatial_arm": args.spatial_arm,
        "training_loop": "fast_bit_exact_483",
        "determinism": "strict" if args.strict_determinism else "cudnn",
        "cublas_workspace_config":
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if args.strict_determinism else None,
        "training_loop_optimisations": [
            "numpy_target_components",
            "batched_component_upload",
            "validation_only_loss_materialisation",
            "depth_one_prefetch",
            "sparse_early_validation",
        ],
        "validation_schedule": {
            "early_interval": args.validation_interval * 4,
            "early_rule": "step < round(optimizer_steps * 0.5)",
            "late_interval": args.validation_interval,
        },
        "writer_overlap_fit_validation": 0,
    }
    summary.update({
        "architecture": holder["configuration"],
        "method": "spatially_grounded_revision_queries",
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
        "schema_version": "hme-revision-spatial-query-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"spatial_query_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "seed": args.seed,
        "spatial_arm": args.spatial_arm,
        "status": "complete",
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for482_spatial")
    baseline = load_module(BASELINE_PATH, "hme422_for482_spatial")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for482_spatial")
    fusion = load_module(FUSION_PATH, "hme478_for482_spatial")
    fullres = load_module(FULLRES_PATH, "hme480_for482_spatial")
    sizeprobe = load_module(SIZE_PATH, "hme481_for482_spatial")
    fastloop = load_module(FAST_LOOP_PATH, "hme483_for482_spatial")
    args = parser(joint, baseline, typepilot, argv)
    fastloop.set_strict_determinism(args.strict_determinism)
    if args.seed != SEED:
        raise SystemExit("482 spatial probe refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"482 spatial probe refused: output already exists: {final_output}")

    holder: dict[str, Any] = {}
    joint.build_model = lambda _baseline, _ablation, run_args: \
        build_model(baseline, fusion, fullres, run_args, holder)
    joint.instance_loss = make_loss_adapter(holder, args, fusion)
    joint.predict = fullres.predict
    joint.evaluate = lambda probe, extensions, union, masks, scores, rows, \
            positive_count, gold_masks, run_args: evaluate_with_diagnostics(
                probe, extensions, union, masks, scores, rows,
                positive_count, gold_masks, run_args, joint, fullres,
                sizeprobe)
    joint.train_once = lambda run_baseline, probe, extensions, data, \
            run_args, staging: fastloop.fast_train_once(
                joint, run_baseline, probe, extensions, data, run_args,
                staging)
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("482 spatial probe refused: sample counts changed")
        print(json.dumps({
            "counts": result["counts"],
            "frozen_test_touched": False,
            "seed": args.seed,
            "spatial_arm": args.spatial_arm,
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
            "fastloop": fastloop,
            "joint": joint,
            "sizeprobe": sizeprobe,
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
        "gold_count_stratified_oracle_recall_050":
            diagnostics["gold_count_stratified_oracle_recall_050"],
        "gold_count_stratified_recall_050":
            diagnostics["gold_count_stratified_recall_050"],
        "instance_f1_050_factory":
            validation["instance_factory"]["0.50"]["f1"],
        "instance_f1_050_selected":
            validation["instance_selected"]["0.50"]["f1"],
        "oracle_recall_050": diagnostics["oracle_recall_050"]["recall"],
        "output_dir": str(final_output),
        "pixel_iou": validation["pixel_iou"],
        "seed": args.seed,
        "spatial_arm": args.spatial_arm,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

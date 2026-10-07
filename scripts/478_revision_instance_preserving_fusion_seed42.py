#!/usr/bin/env python3
"""Seed-42 ablation of instance-preserving dense--query refinement.

This create-only experiment keeps the data, split, optimisation recipe and
MiT-B0/SegFormer dense-coupled control fixed.  It tests four nested changes:

* ``query_refine``: refine every query mask separately with dense evidence;
* ``soft_union``: add a probabilistic query union and duplicate suppression;
* ``asym_consistency``: use the better supervised branch as a detached teacher;
* ``full_quality``: combine all changes and predict per-query mask quality.

Only seed 42 is admitted.  Frozen test, M6 and reference material are never
requested.  Experiment 477 remains the immutable control.
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
CONTROL_PATH = ROOT / "scripts/477_revision_sdir_seed42.py"
SCHEMA = "hme-revision-instance-preserving-fusion-seed42-v1"

VARIANTS = (
    "query_refine",
    "soft_union",
    "asym_consistency",
    "full_quality",
)
SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236

QUERY_DIMENSION = 160
QUERY_LAYERS = 4
QUERY_COUNT = 16
ATTENTION_STRIDE = 8
DENSE_CHANNELS = 160
REFINE_CHANNELS = 16


class Fusion478Error(RuntimeError):
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
        raise Fusion478Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def probability_to_logit(probability: Any, epsilon: float = 1e-5) -> Any:
    value = probability.clamp(epsilon, 1.0 - epsilon)
    return (value / (1.0 - value)).log()


def query_union_probability(mask_logits: Any, objectness: Any,
                            target_size: Sequence[int], mode: str) -> Any:
    """Return a differentiable max or true probabilistic noisy-OR union."""
    from torch.nn import functional

    masks = functional.interpolate(
        mask_logits, size=tuple(target_size), mode="bilinear",
        align_corners=False).sigmoid()
    weighted = (masks * objectness.sigmoid()[:, :, None, None]).clamp(
        0.0, 1.0 - 1e-5)
    if mode == "max":
        return weighted.amax(dim=1, keepdim=True)
    if mode == "soft_or":
        return 1.0 - torch_log_complement_sum(weighted)
    raise Fusion478Error(f"unknown query union: {mode}")


def torch_log_complement_sum(weighted: Any) -> Any:
    """Stable product of query-wise complements, kept separate for testing."""
    import torch

    return torch.exp(torch.log1p(-weighted).sum(dim=1, keepdim=True))


def union_targets(batch_targets: Sequence[Sequence[Any]], reference: Any) -> Any:
    import torch

    result = []
    height, width = reference.shape[-2:]
    for targets in batch_targets:
        if targets:
            result.append(torch.stack([
                target.to(device=reference.device, dtype=reference.dtype)
                for target in targets
            ]).amax(dim=0, keepdim=True))
        else:
            result.append(torch.zeros(
                (1, height, width), device=reference.device,
                dtype=reference.dtype))
    return torch.stack(result)


def binary_mask_loss(logits: Any, target: Any) -> Any:
    from torch.nn import functional

    probability = logits.sigmoid()
    numerator = 2.0 * (probability * target).flatten(1).sum(1) + 1.0
    denominator = (probability.flatten(1).sum(1)
                   + target.flatten(1).sum(1) + 1.0)
    return functional.binary_cross_entropy_with_logits(logits, target) \
        + (1.0 - numerator / denominator).mean()


def asymmetric_better_branch_consistency(dense_probability: Any,
                                         query_probability: Any,
                                         target: Any) -> Any:
    """Let the branch closer to the gold pixel teach the weaker branch."""
    from torch.nn import functional

    dense_error = (dense_probability.detach() - target).abs()
    query_error = (query_probability.detach() - target).abs()
    dense_is_teacher = dense_error <= query_error
    query_student = functional.smooth_l1_loss(
        query_probability, dense_probability.detach(), reduction="none")
    dense_student = functional.smooth_l1_loss(
        dense_probability, query_probability.detach(), reduction="none")
    return query_student.where(dense_is_teacher, dense_student).mean()


def duplicate_query_loss(mask_logits: Any, objectness: Any) -> Any:
    """Penalise two confident queries explaining the same pixels."""
    import torch

    probability = mask_logits.sigmoid().flatten(2)
    weighted = probability * objectness.sigmoid()[:, :, None]
    intersection = torch.einsum("bqn,bkn->bqk", weighted, weighted)
    areas = weighted.sum(dim=2)
    dice = (2.0 * intersection + 1e-5) / (
        areas[:, :, None] + areas[:, None, :] + 1e-5)
    count = dice.shape[1]
    if count < 2:
        return dice.sum() * 0.0
    keep = torch.triu(torch.ones(
        (count, count), device=dice.device, dtype=torch.bool), diagonal=1)
    return dice[:, keep].mean()


def quality_target_loss(probe: Any, mask_logits: Any, raw_objectness: Any,
                        quality_logits: Any,
                        batch_targets: Sequence[Sequence[Any]]) -> Any:
    """Train query quality with its detached soft IoU after matching."""
    import torch
    from torch.nn import functional

    total = torch.zeros((), device=mask_logits.device)
    for index, targets in enumerate(batch_targets):
        label = torch.zeros_like(quality_logits[index])
        if targets:
            rows, columns = probe.hungarian_match(
                mask_logits[index], raw_objectness[index], targets)
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


def build_model(baseline: Any, args: argparse.Namespace,
                holder: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional

    segmenter = baseline._smp_model("segformer_b0", args)
    union_mode = "soft_or" if args.refinement_variant in (
        "soft_union", "full_quality") else "max"
    use_asymmetric = args.refinement_variant in (
        "asym_consistency", "full_quality")
    use_quality = args.refinement_variant == "full_quality"
    use_duplicate = args.refinement_variant in (
        "soft_union", "full_quality")

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
                nn.Linear(QUERY_DIMENSION * 4, QUERY_DIMENSION),
            )

        def forward(self, queries, memory):
            normed = self.norm_query(queries)
            attended, _ = self.cross(
                normed, memory, memory, need_weights=False)
            queries = queries + attended
            return queries + self.feed(self.norm_feed(queries))

    class CrossOnlyStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                CrossOnlyLayer() for _ in range(QUERY_LAYERS))

        def forward(self, queries, memory):
            for layer in self.layers:
                queries = layer(queries, memory)
            return queries

    class QueryRefiner(nn.Module):
        """Shared local refiner with query-conditioned FiLM parameters."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(4, REFINE_CHANNELS, 3, padding=1, bias=False),
                nn.BatchNorm2d(REFINE_CHANNELS), nn.ReLU(inplace=True),
                nn.Conv2d(REFINE_CHANNELS, REFINE_CHANNELS, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(REFINE_CHANNELS), nn.ReLU(inplace=True),
            )
            self.film = nn.Linear(QUERY_DIMENSION, REFINE_CHANNELS * 2)
            self.output = nn.Conv2d(REFINE_CHANNELS, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, base_logits, dense_probability, ink, queries):
            batch, count, height, width = base_logits.shape
            base_probability = base_logits.sigmoid()
            dense = dense_probability.expand(-1, count, -1, -1)
            ink_query = ink.expand(-1, count, -1, -1)
            value = torch.stack((
                base_probability,
                dense,
                (base_probability - dense).abs(),
                ink_query,
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

    class RefinedDenseInstanceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = segmenter.encoder
            self.dense_decoder = segmenter.decoder
            decoder_channels = int(segmenter.segmentation_head[0].in_channels)
            self.shared_projection = nn.Sequential(
                nn.Conv2d(decoder_channels, DENSE_CHANNELS, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(DENSE_CHANNELS), nn.ReLU(inplace=True),
            )
            self.dense_logit = nn.Conv2d(DENSE_CHANNELS, 1, 1)
            self.dense_feature_bias = nn.Conv2d(
                1, QUERY_DIMENSION, 1, bias=False)
            self.dense_gate_scale = nn.Parameter(torch.tensor(-2.0))
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

            self.queries = nn.Parameter(
                torch.randn(QUERY_COUNT, QUERY_DIMENSION) * 0.02)
            self.decoder_stack = CrossOnlyStack()
            self.mask_embed = nn.Sequential(
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
                nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
            )
            self.objectness = nn.Linear(QUERY_DIMENSION, 1)
            self.quality = nn.Linear(QUERY_DIMENSION, 1) \
                if use_quality else None
            self.refiner = QueryRefiner()

            self.last_dense_logits = None
            self.last_fused_logits = None
            self.last_query_union = None
            self.last_raw_objectness = None
            self.last_quality_logits = None

        def forward(self, value):
            features = self.encoder(value)
            pixels = self.shared_projection(self.dense_decoder(features))
            dense_quarter = self.dense_logit(pixels)
            dense_full = functional.interpolate(
                dense_quarter, size=value.shape[-2:], mode="bilinear",
                align_corners=False)

            dense_probability = dense_quarter.sigmoid()
            scale = self.dense_gate_scale.sigmoid()
            instance_pixels = pixels * (1.0 + scale * dense_probability) \
                + self.dense_feature_bias(dense_probability)
            memory_map = functional.avg_pool2d(
                instance_pixels, ATTENTION_STRIDE, ATTENTION_STRIDE)
            memory = memory_map.flatten(2).transpose(1, 2)
            queries = self.queries.unsqueeze(0).expand(
                value.shape[0], -1, -1)
            queries = self.decoder_stack(queries, memory)

            base_masks = torch.einsum(
                "bqd,bdhw->bqhw", self.mask_embed(queries), instance_pixels)
            ink = functional.interpolate(
                (1.0 - ((value + 1.0) * 0.5)).clamp(0.0, 1.0),
                size=base_masks.shape[-2:], mode="bilinear",
                align_corners=False)
            masks = self.refiner(
                base_masks, dense_probability, ink, queries)

            raw_objectness = self.objectness(queries).squeeze(-1)
            quality_logits = self.quality(queries).squeeze(-1) \
                if self.quality is not None else None
            if quality_logits is None:
                final_objectness = raw_objectness
            else:
                combined = (raw_objectness.sigmoid()
                            * quality_logits.sigmoid()).clamp_min(1e-8).sqrt()
                final_objectness = probability_to_logit(combined)

            query_union = query_union_probability(
                masks, final_objectness, value.shape[-2:], union_mode)
            dense_probability_full = dense_full.sigmoid()
            disagreement = (dense_probability_full - query_union).abs()
            ink_full = (1.0 - ((value + 1.0) * 0.5)).clamp(0.0, 1.0)
            gate = self.fusion_gate(torch.cat((
                dense_probability_full, query_union, disagreement, ink_full),
                dim=1)).sigmoid()
            fused_probability = gate * dense_probability_full \
                + (1.0 - gate) * query_union

            self.last_dense_logits = dense_full
            self.last_fused_logits = probability_to_logit(fused_probability)
            self.last_query_union = query_union
            self.last_raw_objectness = raw_objectness
            self.last_quality_logits = quality_logits
            return masks, final_objectness, None

    model = RefinedDenseInstanceModel()
    holder.update({
        "model": model,
        "configuration": {
            "backbone": "mit_b0_shared_segformer_all_mlp",
            "coupling": "instance_preserving_dense_query",
            "instance_head": {
                "dimension": QUERY_DIMENSION,
                "layers": QUERY_LAYERS,
                "queries": QUERY_COUNT,
                "self_attention": False,
            },
            "query_refiner": "local_film_residual_with_support_gate",
            "query_union": union_mode,
            "asymmetric_consistency": use_asymmetric,
            "duplicate_suppression": use_duplicate,
            "quality_head": use_quality,
            "variant": args.refinement_variant,
        },
    })
    return model


def make_loss_adapter(original_instance_loss: Any,
                      holder: Mapping[str, Any], args: argparse.Namespace):
    from torch.nn import functional

    def loss(probe, mask_logits, objectness, batch_targets):
        model = holder["model"]
        score_for_binary_loss = model.last_raw_objectness \
            if model.last_quality_logits is not None else objectness
        instance_part, objectness_part = original_instance_loss(
            probe, mask_logits, score_for_binary_loss, batch_targets)
        dense_logits = functional.interpolate(
            model.last_dense_logits, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        fused_logits = functional.interpolate(
            model.last_fused_logits, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        query_union = functional.interpolate(
            model.last_query_union, size=mask_logits.shape[-2:],
            mode="bilinear", align_corners=False)
        target = union_targets(batch_targets, dense_logits)
        dense_part = binary_mask_loss(dense_logits, target)
        fused_part = binary_mask_loss(fused_logits, target)
        if args.refinement_variant in ("asym_consistency", "full_quality"):
            consistency = asymmetric_better_branch_consistency(
                dense_logits.sigmoid(), query_union, target)
        else:
            consistency = functional.smooth_l1_loss(
                dense_logits.sigmoid(), query_union)

        duplicate = mask_logits.sum() * 0.0
        if args.refinement_variant in ("soft_union", "full_quality"):
            duplicate = duplicate_query_loss(mask_logits, objectness)
        quality = mask_logits.sum() * 0.0
        if model.last_quality_logits is not None:
            quality = quality_target_loss(
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
            instance_part
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
            image = np.stack([row["image"] for row in chunk]).astype(
                np.float32)
            value = torch.from_numpy(
                image[:, None] / 127.5 - 1.0).to(device)
            mask_logits, objectness, _row = model(value)
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
           argv: Sequence[str] | None = None):
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--refinement-variant", choices=VARIANTS, required=True)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    base.add_argument("--duplicate-weight", type=float, default=0.05)
    base.add_argument("--quality-weight", type=float, default=0.5)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == "--arm" or value.startswith("--arm=")
           for value in arguments):
        raise Fusion478Error("the refinement probe fixes --arm to instance")
    return base.parse_args([*arguments, "--arm", "instance"])


def finalise(engine_output: Path, final_output: Path,
             args: argparse.Namespace, holder: Mapping[str, Any],
             dependencies: Mapping[str, Any]) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise Fusion478Error(f"create-only output exists: {final_output}")
    for name in ("SHA256SUMS", "complete.json"):
        path = engine_output / name
        if path.exists():
            path.unlink()
    summary_path = engine_output / "summary.json"
    manifest_path = engine_output / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = summary["per_seed"][0]["validation"]
    if int(validation["gold_instances"]) != EXPECTED_GOLD_INSTANCES:
        raise Fusion478Error("development gold-instance count changed")
    summary.update({
        "architecture": holder["configuration"],
        "method": "instance_preserving_dense_query_fusion",
        "schema_version": SCHEMA,
    })
    summary["protocol"].update({
        "control_experiment": "revision_sdir_seed42_v1/dense_coupled",
        "consistency_weight": args.consistency_weight,
        "dense_loss_weight": args.dense_loss_weight,
        "duplicate_weight": args.duplicate_weight,
        "frozen_test_touched": False,
        "fused_loss_weight": args.fused_loss_weight,
        "m6_or_reference_materialised": False,
        "output_contract": {
            "components": "refined_instance_query_masks",
            "pixel_union": "learned_dense_refined_instance_fusion",
        },
        "quality_weight": args.quality_weight,
        "refinement_variant": args.refinement_variant,
        "writer_overlap_fit_validation": 0,
    })
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
        "schema_version": "hme-revision-instance-preserving-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"refined_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "method": "instance_preserving_dense_query_fusion",
        "seed": args.seed,
        "status": "complete",
        "variant": args.refinement_variant,
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for478_refined_fusion")
    baseline = load_module(BASELINE_PATH, "hme422_for478_refined_fusion")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for478_refined_fusion")
    control = load_module(CONTROL_PATH, "hme477_for478_refined_fusion")
    args = parser(joint, baseline, typepilot, argv)
    if args.seed != SEED:
        raise SystemExit("478 refinement refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"478 refinement refused: output already exists: {final_output}")

    holder: dict[str, Any] = {}
    original_instance_loss = joint.instance_loss
    joint.build_model = lambda _baseline, _ablation, run_args: \
        build_model(baseline, run_args, holder)
    joint.instance_loss = make_loss_adapter(
        original_instance_loss, holder, args)
    joint.predict = predict
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("478 refinement refused: sample counts changed")
        print(json.dumps({
            "counts": result["counts"],
            "frozen_test_touched": False,
            "seed": args.seed,
            "validated": True,
            "variant": args.refinement_variant,
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
            "control": control,
            "joint": joint,
            "typepilot": typepilot,
        })
    except BaseException:
        if engine_output.is_dir():
            shutil.rmtree(engine_output)
        raise
    result = json.loads((final_output / "summary.json").read_text(
        encoding="utf-8"))["per_seed"][0]
    print(json.dumps({
        "instance_f1_010": result["validation"]["instance"]["0.10"]["f1"],
        "output_dir": str(final_output),
        "pixel_iou": result["validation"]["pixel_iou"],
        "seed": args.seed,
        "variant": args.refinement_variant,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Seed-42 probe of a stroke-aware dense--instance revision network.

The experiment tests one architectural hypothesis without changing data,
augmentation, optimiser, writer split, or evaluation code: a revision model
should combine SegFormer-like dense coverage with query-level instance
evidence.  Three nested arms make the contribution auditable:

* ``dense_parallel`` shares a MiT-B0 encoder between a dense mask decoder and
  the locked 160d/4-layer/cross-only instance-query head;
* ``dense_coupled`` adds differentiable dense-to-query spatial gating and a
  query-to-dense learned fusion gate;
* ``sdir_full`` additionally preserves full-resolution pen-stroke evidence in
  a shallow detail stem before the learned fusion.

Only seed 42 is admitted.  Frozen test, M6, and reference material are never
requested.  The existing model-2 and SegFormer-B0 results remain untouched.
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
SCHEMA = "hme-revision-sdir-seed42-v1"

VARIANTS = ("dense_parallel", "dense_coupled", "sdir_full")
SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236

QUERY_DIMENSION = 160
QUERY_LAYERS = 4
QUERY_COUNT = 16
ATTENTION_STRIDE = 8
DENSE_CHANNELS = 160
STROKE_CHANNELS = 24
STROKE_QUARTER_CHANNELS = 48


class SDIRError(RuntimeError):
    """Raised when the create-only SDIR protocol is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SDIRError(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def probability_to_logit(probability: Any, epsilon: float = 1e-5) -> Any:
    value = probability.clamp(epsilon, 1.0 - epsilon)
    return (value / (1.0 - value)).log()


def query_union_probability(mask_logits: Any, objectness: Any,
                            target_size: Sequence[int]) -> Any:
    """Differentiable noisy-OR-style union used by coupling and inference."""
    from torch.nn import functional

    masks = functional.interpolate(
        mask_logits, size=tuple(target_size), mode="bilinear",
        align_corners=False).sigmoid()
    scores = objectness.sigmoid()[:, :, None, None]
    return (masks * scores).amax(dim=1, keepdim=True)


def union_targets(batch_targets: Sequence[Sequence[Any]], reference: Any) -> Any:
    """Convert instance targets into one dense target without changing labels."""
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
    """BCE plus soft Dice, matching the established positive mask objective."""
    from torch.nn import functional

    probability = logits.sigmoid()
    numerator = 2.0 * (probability * target).flatten(1).sum(1) + 1.0
    denominator = (probability.flatten(1).sum(1)
                   + target.flatten(1).sum(1) + 1.0)
    return functional.binary_cross_entropy_with_logits(logits, target) \
        + (1.0 - numerator / denominator).mean()


def build_sdir_model(baseline: Any, args: argparse.Namespace,
                     holder: dict[str, Any]) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional

    segmenter = baseline._smp_model("segformer_b0", args)
    variant = args.sdir_variant
    coupled = variant in ("dense_coupled", "sdir_full")
    stroke_aware = variant == "sdir_full"

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
            attended, _ = self.cross(normed, memory, memory,
                                     need_weights=False)
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

    class SDIRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = variant
            self.encoder = segmenter.encoder
            self.dense_decoder = segmenter.decoder
            dense_decoder_channels = int(
                segmenter.segmentation_head[0].in_channels)

            if stroke_aware:
                self.stroke_full = nn.Sequential(
                    nn.Conv2d(1, STROKE_CHANNELS, 3, padding=1, bias=False),
                    nn.BatchNorm2d(STROKE_CHANNELS), nn.ReLU(inplace=True),
                    nn.Conv2d(STROKE_CHANNELS, STROKE_CHANNELS, 3,
                              padding=1, bias=False),
                    nn.BatchNorm2d(STROKE_CHANNELS), nn.ReLU(inplace=True),
                )
                self.stroke_quarter = nn.Sequential(
                    nn.Conv2d(STROKE_CHANNELS, 32, 3, stride=2,
                              padding=1, bias=False),
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, STROKE_QUARTER_CHANNELS, 3, stride=2,
                              padding=1, bias=False),
                    nn.BatchNorm2d(STROKE_QUARTER_CHANNELS),
                    nn.ReLU(inplace=True),
                )
                self.stroke_logit = nn.Conv2d(STROKE_CHANNELS, 1, 1)
                dense_decoder_channels += STROKE_QUARTER_CHANNELS
            else:
                self.stroke_full = None
                self.stroke_quarter = None
                self.stroke_logit = None

            self.shared_projection = nn.Sequential(
                nn.Conv2d(dense_decoder_channels, DENSE_CHANNELS, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(DENSE_CHANNELS), nn.ReLU(inplace=True),
            )
            self.dense_logit = nn.Conv2d(DENSE_CHANNELS, 1, 1)

            if coupled:
                self.dense_feature_bias = nn.Conv2d(
                    1, QUERY_DIMENSION, 1, bias=False)
                self.dense_gate_scale = nn.Parameter(torch.tensor(-2.0))
                self.fusion_gate = nn.Sequential(
                    nn.Conv2d(4, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1),
                )
            else:
                self.dense_feature_bias = None
                self.register_parameter("dense_gate_scale", None)
                self.fusion_gate = None

            self.queries = nn.Parameter(
                torch.randn(QUERY_COUNT, QUERY_DIMENSION) * 0.02)
            self.decoder_stack = CrossOnlyStack()
            self.mask_embed = nn.Sequential(
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
                nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIMENSION, QUERY_DIMENSION),
            )
            self.objectness = nn.Linear(QUERY_DIMENSION, 1)

            self.last_dense_logits = None
            self.last_fused_logits = None
            self.last_query_union = None
            self.last_fusion_gate = None

        def forward(self, value):
            features = self.encoder(value)
            decoded = self.dense_decoder(features)
            full_stroke = None
            if self.stroke_full is not None:
                full_stroke = self.stroke_full(value)
                quarter_stroke = self.stroke_quarter(full_stroke)
                if quarter_stroke.shape[-2:] != decoded.shape[-2:]:
                    quarter_stroke = functional.interpolate(
                        quarter_stroke, size=decoded.shape[-2:],
                        mode="bilinear", align_corners=False)
                decoded = torch.cat((decoded, quarter_stroke), dim=1)

            pixels = self.shared_projection(decoded)
            dense_quarter = self.dense_logit(pixels)
            dense_full = functional.interpolate(
                dense_quarter, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
            if full_stroke is not None:
                dense_full = dense_full + self.stroke_logit(full_stroke)

            instance_pixels = pixels
            if self.dense_feature_bias is not None:
                dense_probability = functional.interpolate(
                    dense_full.sigmoid(), size=pixels.shape[-2:],
                    mode="bilinear", align_corners=False)
                scale = self.dense_gate_scale.sigmoid()
                instance_pixels = pixels * (1.0 + scale * dense_probability) \
                    + self.dense_feature_bias(dense_probability)

            memory_map = functional.avg_pool2d(
                instance_pixels, ATTENTION_STRIDE, ATTENTION_STRIDE)
            memory = memory_map.flatten(2).transpose(1, 2)
            queries = self.queries.unsqueeze(0).expand(
                value.shape[0], -1, -1)
            queries = self.decoder_stack(queries, memory)
            masks = torch.einsum(
                "bqd,bdhw->bqhw", self.mask_embed(queries),
                instance_pixels)
            objectness = self.objectness(queries).squeeze(-1)
            query_union = query_union_probability(
                masks, objectness, value.shape[-2:])

            if self.fusion_gate is None:
                fused_probability = dense_full.sigmoid()
                gate = torch.ones_like(fused_probability)
            else:
                dense_probability = dense_full.sigmoid()
                disagreement = (dense_probability - query_union).abs()
                ink = (1.0 - ((value + 1.0) * 0.5)).clamp(0.0, 1.0)
                gate = self.fusion_gate(torch.cat((
                    dense_probability, query_union, disagreement, ink), dim=1)
                ).sigmoid()
                fused_probability = gate * dense_probability \
                    + (1.0 - gate) * query_union

            self.last_dense_logits = dense_full
            self.last_fused_logits = probability_to_logit(fused_probability)
            self.last_query_union = query_union
            self.last_fusion_gate = gate
            return masks, objectness, None

    model = SDIRModel()
    holder.update({
        "model": model,
        "configuration": {
            "coupled": coupled,
            "dense_decoder": "segformer_all_mlp",
            "encoder": "mit_b0_shared",
            "fusion": "learned_dense_query_disagreement_ink_gate"
                if coupled else "dense_only",
            "instance_head": {
                "dimension": QUERY_DIMENSION,
                "layers": QUERY_LAYERS,
                "queries": QUERY_COUNT,
                "self_attention": False,
            },
            "stroke_full_resolution": stroke_aware,
            "variant": variant,
        },
    })
    return model


def make_loss_adapter(original_instance_loss: Any,
                      holder: Mapping[str, Any], args: argparse.Namespace):
    from torch.nn import functional

    def loss(probe, mask_logits, objectness, batch_targets):
        instance_part, objectness_part = original_instance_loss(
            probe, mask_logits, objectness, batch_targets)
        model = holder["model"]
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
        consistency = functional.smooth_l1_loss(
            dense_logits.sigmoid(), query_union)
        holder["last_auxiliary_losses"] = {
            "consistency": float(consistency.detach()),
            "dense": float(dense_part.detach()),
            "fused": float(fused_part.detach()),
        }
        return (
            instance_part
            + args.dense_loss_weight * dense_part
            + args.fused_loss_weight * fused_part
            + args.consistency_weight * consistency,
            objectness_part,
        )

    return loss


def predict_sdir(model: Any, rows: Sequence[Mapping[str, Any]],
                 batch_size: int, device: Any, _arm: str):
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
    base.add_argument("--sdir-variant", choices=VARIANTS, required=True)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == "--arm" or value.startswith("--arm=")
           for value in arguments):
        raise SDIRError("the SDIR probe fixes --arm to instance")
    return base.parse_args([*arguments, "--arm", "instance"])


def finalise(engine_output: Path, final_output: Path,
             args: argparse.Namespace, holder: Mapping[str, Any],
             dependencies: Mapping[str, Any]) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise SDIRError(f"create-only output exists: {final_output}")
    for name in ("SHA256SUMS", "complete.json"):
        path = engine_output / name
        if path.exists():
            path.unlink()
    summary_path = engine_output / "summary.json"
    manifest_path = engine_output / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = summary["per_seed"][0]["validation"]
    if int(validation["gold_instances"]) != EXPECTED_GOLD_INSTANCES:
        raise SDIRError("development gold-instance count changed")
    summary.update({
        "architecture": holder["configuration"],
        "method": "sdir",
        "schema_version": SCHEMA,
    })
    summary["protocol"].update({
        "backbone": "mit_b0_shared",
        "consistency_weight": args.consistency_weight,
        "dense_loss_weight": args.dense_loss_weight,
        "frozen_test_touched": False,
        "fused_loss_weight": args.fused_loss_weight,
        "m6_or_reference_materialised": False,
        "output_contract": {
            "components": "instance_query_masks",
            "pixel_union": "learned_dense_instance_fusion",
        },
        "sdir_variant": args.sdir_variant,
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
        "schema_version": "hme-revision-sdir-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"sdir_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "method": "sdir",
        "seed": args.seed,
        "status": "complete",
        "variant": args.sdir_variant,
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for477_sdir")
    baseline = load_module(BASELINE_PATH, "hme422_for477_sdir")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for477_sdir")
    args = parser(joint, baseline, typepilot, argv)
    if args.seed != SEED:
        raise SystemExit("477 SDIR refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"477 SDIR refused: output already exists: {final_output}")

    holder: dict[str, Any] = {}
    original_instance_loss = joint.instance_loss
    joint.build_model = lambda _baseline, _ablation, run_args: \
        build_sdir_model(baseline, run_args, holder)
    joint.instance_loss = make_loss_adapter(
        original_instance_loss, holder, args)
    joint.predict = predict_sdir
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("477 SDIR refused: sample counts changed")
        print(json.dumps({
            "counts": result["counts"],
            "frozen_test_touched": False,
            "seed": args.seed,
            "validated": True,
            "variant": args.sdir_variant,
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
        "instance_f1_050": result["validation"]["instance"]["0.50"]["f1"],
        "output_dir": str(final_output),
        "pixel_iou": result["validation"]["pixel_iou"],
        "seed": args.seed,
        "variant": args.sdir_variant,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fix the instance head's mask decoder, then let a row head share its trunk.

Two findings drive this.  First, the instance head's recall deficit at
IoU>=0.50 survived every threshold rule across 88 grid points and two
calibration families, while at IoU>=0.10 it was level with the semantic
control, so its masks are the wrong shape rather than the wrong size.  Its mask
is a dot product between a query vector and the pixel features -- a linear
function of them -- which is far weaker than the convolutional decoder the
semantic arm ends in.  Second, the row classifier and the localiser are
separate models that share no features, although "this line has a revision" and
"some instance exists here" are the same fact at two granularities.

Three arms, one trunk, one protocol:

* ``instance`` -- experiment 442's head, reproduced here as the control;
* ``dynamic``  -- the query predicts the weights of a small convolution applied
  to the pixel features instead of a single dot product, so the mask stops
  being linear in them;
* ``joint``    -- ``dynamic`` plus a row head on the same trunk, tied to the
  instance head by an aggregation loss: the row probability must agree with the
  strongest objectness on that line.

Row labels cost nothing here: every training line is already either a revision
line or an adjudicated clean one, so the row head needs no new annotation.

The script deliberately does not: introduce new training data, change the
backbone or its ImageNet initialisation, alter the augmentation or the batch
composition, move any threshold, or read M6, reference or frozen-test material.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts/422_revision_paper_baselines.py"
BASELINE_SHA256 = "a566110f4a629ba038c97b895ee0d442f9d445cadb7fa961f048175cddb2a8b9"
PROBE_PATH = ROOT / "scripts/442_revision_instance_head_probe.py"
PROBE_FUNCTIONS = ("gold_instances", "instance_component_metrics",
                   "hungarian_match", "union_instances")
PROBE_SHA256 = "e7640d9cb425d22c3f739e25ba11c8781ed6ec8af4a71f8d27ad1c42804874da"

SCHEMA = "hme-revision-joint-head-probe-v1"

ARMS = ("instance", "dynamic", "joint")
SEEDS = (42, 2354, 3407)

EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_TRUE_COMPONENTS = 236
CONNECTIVITY = 8
MATCH_IOUS = (0.10, 0.25, 0.50)

NUM_QUERIES = 16
QUERY_DIM = 96
ATTENTION_STRIDE = 8

# The dynamic mask branch runs at half resolution: a 6-pixel stroke is still
# three pixels there, and full resolution for sixteen queries does not fit.
DYNAMIC_STRIDE = 2
DYNAMIC_CHANNELS = 8
DYNAMIC_LAYERS = 3

THRESHOLDS = tuple(round(0.02 * step, 2) for step in range(1, 50))


class JointProbeError(RuntimeError):
    """Raised when the joint-head contract is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected: str, *, enforce: bool) -> Any:
    actual = digest_file(path)
    if enforce and actual != expected:
        raise JointProbeError(f"{path.name} digest mismatch: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise JointProbeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = actual
    return module


def probe_digest(module: Any) -> str:
    """Digest only the shared pieces this script reuses."""
    import inspect

    blob = "".join(inspect.getsource(getattr(module, name))
                   for name in PROBE_FUNCTIONS)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# model


def dynamic_parameter_count() -> int:
    """Weights and biases of the per-query convolution, laid out end to end."""
    total = 0
    inputs = DYNAMIC_CHANNELS + 2          # features plus relative coordinates
    for layer in range(DYNAMIC_LAYERS):
        outputs = 1 if layer == DYNAMIC_LAYERS - 1 else DYNAMIC_CHANNELS
        total += inputs * outputs + outputs
        inputs = outputs
    return total


def build_model(baseline: Any, ablation: Any, args: argparse.Namespace):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    backbone = ablation.make_unetpp_resnet34(args.resnet34_weights)

    class JointHead(nn.Module):
        def __init__(self, model, arm: str):
            super().__init__()
            channels = model.head.in_channels
            model.head = nn.Identity()
            self.backbone = model
            self.arm = arm
            self.project = nn.Conv2d(channels, QUERY_DIM, 1)
            self.queries = nn.Parameter(torch.randn(NUM_QUERIES, QUERY_DIM) * 0.02)
            layer = nn.TransformerDecoderLayer(
                d_model=QUERY_DIM, nhead=4, dim_feedforward=QUERY_DIM * 4,
                dropout=0.0, batch_first=True, norm_first=True)
            self.decoder_stack = nn.TransformerDecoder(layer, num_layers=2)
            self.mask_embed = nn.Sequential(
                nn.Linear(QUERY_DIM, QUERY_DIM), nn.ReLU(inplace=True),
                nn.Linear(QUERY_DIM, QUERY_DIM))
            self.objectness = nn.Linear(QUERY_DIM, 1)
            if arm in ("dynamic", "joint"):
                self.mask_features = nn.Conv2d(QUERY_DIM, DYNAMIC_CHANNELS, 1)
                self.controller = nn.Linear(QUERY_DIM, dynamic_parameter_count())
                nn.init.zeros_(self.controller.weight)
                nn.init.zeros_(self.controller.bias)
            if arm == "joint":
                self.row_head = nn.Sequential(
                    nn.Linear(QUERY_DIM, QUERY_DIM), nn.ReLU(inplace=True),
                    nn.Linear(QUERY_DIM, 1))

        def dynamic_masks(self, queries, pixels):
            """Apply each query's own small convolution to the pixel features."""
            batch, count, _ = queries.shape
            features = self.mask_features(pixels)
            features = functional.avg_pool2d(features, DYNAMIC_STRIDE,
                                             DYNAMIC_STRIDE)
            height, width = features.shape[-2:]
            ys = torch.linspace(-1.0, 1.0, height, device=features.device)
            xs = torch.linspace(-1.0, 1.0, width, device=features.device)
            grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"))
            grid = grid[None].expand(batch, -1, -1, -1)
            stacked = torch.cat((features, grid), 1)
            stacked = stacked[:, None].expand(-1, count, -1, -1, -1)
            value = stacked.reshape(batch * count, DYNAMIC_CHANNELS + 2,
                                    height, width)

            parameters = self.controller(queries).reshape(batch * count, -1)
            offset = 0
            inputs = DYNAMIC_CHANNELS + 2
            for layer in range(DYNAMIC_LAYERS):
                outputs = 1 if layer == DYNAMIC_LAYERS - 1 else DYNAMIC_CHANNELS
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
            return value.reshape(batch, count, height, width)

        def forward(self, value):
            pixels = self.project(self.backbone(value))
            coarse = functional.avg_pool2d(pixels, ATTENTION_STRIDE,
                                           ATTENTION_STRIDE)
            memory = coarse.flatten(2).transpose(1, 2)
            queries = self.queries.unsqueeze(0).expand(coarse.shape[0], -1, -1)
            queries = self.decoder_stack(queries, memory)
            masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(queries), pixels)
            if self.arm in ("dynamic", "joint"):
                # A residual on the linear mask, so the arm is a strict
                # extension: the controller starts at zero and adds nothing.
                residual = self.dynamic_masks(queries, pixels)
                residual = functional.interpolate(
                    residual, size=masks.shape[-2:], mode="bilinear",
                    align_corners=False)
                masks = masks + residual
            row = None
            if self.arm == "joint":
                row = self.row_head(queries.mean(1)).squeeze(-1)
            return masks, self.objectness(queries).squeeze(-1), row

    return JointHead(backbone, args.arm)


# --------------------------------------------------------------------------
# losses


def instance_loss(probe: Any, mask_logits, objectness, batch_targets):
    import torch
    from torch.nn import functional as functional

    mask_total = torch.zeros((), device=mask_logits.device)
    object_total = torch.zeros((), device=mask_logits.device)
    for index in range(mask_logits.shape[0]):
        targets = batch_targets[index]
        rows, columns = probe.hungarian_match(
            mask_logits[index], objectness[index], targets)
        label = torch.zeros(mask_logits.shape[1], device=mask_logits.device)
        if rows:
            label[torch.as_tensor(rows, device=label.device)] = 1.0
            selected = mask_logits[index][rows]
            gold = torch.stack([targets[column] for column in columns]).to(
                selected.dtype)
            probability = torch.sigmoid(selected)
            numerator = 2.0 * (probability * gold).flatten(1).sum(1) + 1.0
            denominator = (probability.flatten(1).sum(1)
                           + gold.flatten(1).sum(1) + 1.0)
            mask_total = mask_total + \
                functional.binary_cross_entropy_with_logits(selected, gold) + \
                (1.0 - numerator / denominator).mean()
        object_total = object_total + functional.binary_cross_entropy_with_logits(
            objectness[index], label)
    count = mask_logits.shape[0]
    return mask_total / count, object_total / count


def row_losses(row_logits, objectness, row_labels):
    """Supervise the row head, and tie it to the strongest instance.

    "This line has a revision" and "some instance exists here" are the same
    fact; the agreement term is what stops the two heads from drifting apart.
    """
    import torch
    from torch.nn import functional as functional

    supervised = functional.binary_cross_entropy_with_logits(
        row_logits, row_labels)
    aggregated = objectness.max(dim=1).values
    agreement = functional.mse_loss(torch.sigmoid(row_logits),
                                    torch.sigmoid(aggregated))
    return supervised, agreement


# --------------------------------------------------------------------------
# prediction and evaluation


def predict(model, rows, batch_size: int, device, arm: str):
    import torch

    model.eval()
    unions, mask_chunks, score_chunks, row_chunks = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            image = np.stack([row["image"] for row in chunk]).astype(np.float32)
            value = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            mask_logits, objectness, row = model(value)
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
            masks = torch.sigmoid(mask_logits)
            scores = torch.sigmoid(objectness)
            unions.append((masks * scores[:, :, None, None]).amax(1).cpu().numpy())
            mask_chunks.append(masks.cpu().numpy())
            score_chunks.append(scores.cpu().numpy())
            row_chunks.append(torch.sigmoid(row).cpu().numpy()
                              if row is not None
                              else np.zeros(len(chunk), dtype=np.float32))
    model.train()
    return (np.concatenate(unions, 0), np.concatenate(mask_chunks, 0),
            np.concatenate(score_chunks, 0), np.concatenate(row_chunks, 0))


def balanced_accuracy(probabilities, labels, threshold: float = 0.5) -> float:
    predicted = probabilities >= threshold
    positive = labels > 0.5
    if not positive.any() or positive.all():
        return 0.0
    tpr = float((predicted & positive).sum()) / float(positive.sum())
    tnr = float((~predicted & ~positive).sum()) / float((~positive).sum())
    return 0.5 * (tpr + tnr)


def evaluate(probe: Any, extensions: Any, union, masks, scores, rows,
             positive_count: int, gold_masks, args) -> dict[str, Any]:
    truth = np.concatenate(
        (np.stack(gold_masks),
         np.zeros((union.shape[0] - positive_count, *union.shape[1:]),
                  dtype=np.float32)), 0)
    best = None
    for threshold in THRESHOLDS:
        predicted = union >= threshold
        intersection = float(np.logical_and(predicted, truth > 0.5).sum())
        denominator = float(np.logical_or(predicted, truth > 0.5).sum())
        iou = intersection / denominator if denominator else 0.0
        if best is None or iou > best[1]:
            best = (threshold, iou)
    threshold = best[0]

    result = {"threshold": threshold, "pixel_iou": best[1],
              "union": {}, "instance": {}}
    for match_iou in MATCH_IOUS:
        metrics = extensions.component_metrics(
            union, truth, threshold, connectivity=CONNECTIVITY,
            match_iou=match_iou)
        result["union"][f"{match_iou:.2f}"] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"])}

    truths = [probe.gold_instances(row) for row in truth]
    predicted = []
    for row_masks, row_scores in zip(masks, scores):
        predicted.append([mask >= args.mask_threshold
                          for mask, score in zip(row_masks, row_scores)
                          if score >= args.score_threshold
                          and (mask >= args.mask_threshold).any()])
    for match_iou in MATCH_IOUS:
        metrics = probe.instance_component_metrics(predicted, truths, match_iou)
        metrics["fp_components_per_line"] = \
            metrics["false_positive"] / union.shape[0]
        result["instance"][f"{match_iou:.2f}"] = metrics
    result["predicted_instances"] = int(sum(len(item) for item in predicted))
    result["gold_instances"] = int(sum(len(item) for item in truths))

    row_labels = np.concatenate((np.ones(positive_count, dtype=np.float32),
                                 np.zeros(union.shape[0] - positive_count,
                                          dtype=np.float32)))
    result["row"] = {
        "balanced_accuracy_head": balanced_accuracy(rows, row_labels)
        if args.arm == "joint" else None,
        # Available for every arm: the strongest instance is itself a row
        # detector, which is what the agreement loss ties the row head to.
        "balanced_accuracy_from_instances": balanced_accuracy(
            scores.max(axis=1), row_labels),
    }
    return result


# --------------------------------------------------------------------------
# training


def learning_rate_at(step: int, args) -> float:
    if step <= args.warmup_steps:
        return args.learning_rate * step / max(1, args.warmup_steps)
    progress = ((step - args.warmup_steps)
                / max(1, args.optimizer_steps - args.warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return args.learning_rate * (
        0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def agreement_weight(step: int, args) -> float:
    """Ramp the agreement term in, so it cannot dominate an untrained model."""
    if args.arm != "joint":
        return 0.0
    ramp = min(1.0, step / max(1, args.agreement_ramp_steps))
    return args.agreement_weight * ramp


def train_once(baseline, probe, extensions, data, args, staging: Path):
    import torch

    seed = args.seed
    if seed not in SEEDS or not torch.cuda.is_available():
        raise JointProbeError("the probe requires CUDA and a declared seed")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base, extension, ablation = data["base"], data["extension"], data["ablation"]
    model = build_model(baseline, ablation, args).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    rng = random.Random(seed)
    positive_cycle = base.ShuffledCycle(data["fit_positive"], rng)
    clean_cycle = base.ShuffledCycle(data["fit_clean"], rng)

    validation_rows = data["validation_positive"] + data["validation_clean"]
    positive_count = len(data["validation_positive"])
    gold_masks = [row["mask"].astype(np.float32) / 255.0
                  for row in data["validation_positive"]]

    curve, best_key, best_state, best_step = [], None, None, -1
    start_select = int(round(args.optimizer_steps * 0.5))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rate = learning_rate_at(step, args)
        for group in optimiser.param_groups:
            group["lr"] = rate
        rows, positive_in_batch, _clean = base.take_balanced_batch(
            positive_cycle, clean_cycle, args.batch_size, args.negative_ratio)
        # Row labels come free: the batch builder already knows which rows are
        # revision rows and which are adjudicated clean.
        labels = np.concatenate(
            (np.ones(positive_in_batch, dtype=np.float32),
             np.zeros(len(rows) - positive_in_batch, dtype=np.float32)))
        order = list(range(len(rows)))
        rng.shuffle(order)
        rows = [rows[index] for index in order]
        labels = labels[order]

        images, masks = [], []
        for row in rows:
            image, mask, _skeleton, _aux = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], None, rng)
            images.append(image)
            masks.append(mask)
        value = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0).to(device)
        target = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        row_labels = torch.from_numpy(labels).to(device)

        mask_logits, objectness, row_logits = model(value)
        mask_logits = torch.nn.functional.interpolate(
            mask_logits, size=value.shape[-2:], mode="bilinear",
            align_corners=False)
        batch_targets = [
            [torch.from_numpy(piece.astype(np.float32)).to(device)
             for piece in probe.gold_instances(
                 target[index, 0].detach().cpu().numpy())]
            for index in range(len(rows))]
        mask_part, object_part = instance_loss(
            probe, mask_logits, objectness, batch_targets)
        loss = mask_part + args.objectness_weight * object_part
        supervised = agreement = torch.zeros((), device=device)
        if args.arm == "joint":
            supervised, agreement = row_losses(
                row_logits, objectness, row_labels)
            loss = loss + args.row_weight * supervised \
                + agreement_weight(step, args) * agreement
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        last_losses = {"loss": float(loss.detach()),
                       "mask": float(mask_part.detach()),
                       "objectness": float(object_part.detach()),
                       "row": float(supervised.detach()),
                       "agreement": float(agreement.detach())}

        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        union, masks_out, scores_out, rows_out = predict(
            model, validation_rows, args.batch_size, device, args.arm)
        reading = evaluate(probe, extensions, union, masks_out, scores_out,
                           rows_out, positive_count, gold_masks, args)
        entry = {"step": step, "learning_rate": rate,
                 "pixel_iou": reading["pixel_iou"],
                 "instance_f1_050": reading["instance"]["0.50"]["f1"],
                 "instance_recall_050": reading["instance"]["0.50"]["recall"],
                 "row_bacc": reading["row"]["balanced_accuracy_head"],
                 "loss": last_losses["loss"]}
        curve.append(entry)
        key = (reading["instance"]["0.50"]["f1"], reading["pixel_iou"])
        if step >= start_select and (best_key is None or key > best_key):
            best_key, best_step = key, step
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        print(json.dumps({"arm": args.arm, "event": "validation", "seed": seed,
                          "step": step, "best_step": best_step,
                          "instance_f1_050": round(entry["instance_f1_050"], 6),
                          "instance_recall_050":
                              round(entry["instance_recall_050"], 6)},
                         sort_keys=True), flush=True)

    if best_state is None:
        raise JointProbeError("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    union, masks_out, scores_out, rows_out = predict(
        model, validation_rows, args.batch_size, device, args.arm)
    final = evaluate(probe, extensions, union, masks_out, scores_out, rows_out,
                     positive_count, gold_masks, args)
    binary = masks_out >= args.mask_threshold
    np.savez_compressed(
        staging / f"validation_instances_seed{seed}.npz",
        packed_masks=np.packbits(binary, axis=None),
        mask_shape=np.asarray(binary.shape, dtype=np.int64),
        scores=scores_out.astype(np.float32),
        row_probabilities=rows_out.astype(np.float32),
        mask_threshold=np.asarray(args.mask_threshold, dtype=np.float32),
        positive_count=np.asarray(positive_count, dtype=np.int64),
        quantised_probabilities=np.clip(
            np.rint(masks_out * 255.0), 0, 255).astype(np.uint8))
    with (staging / f"curve_seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
        for entry in curve:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"seed": seed, "best_step": best_step, "last_losses": last_losses,
            "validation": final, "curve_length": len(curve)}


# --------------------------------------------------------------------------


def prepare(baseline, typepilot, args) -> dict[str, Any]:
    two_type = typepilot.load_two_type()
    _type410, data, *_rest = two_type.prepare_data(args)
    new_positive, new_clean, _audit = typepilot.load_new_rows(
        two_type, data, args)
    data["fit_positive"].extend(new_positive)
    data["fit_clean"].extend(new_clean)
    fit = {str(row["group_uuid"]) for row in
           data["fit_positive"] + data["fit_clean"]}
    validation = {str(row["group_uuid"]) for row in
                  data["validation_positive"] + data["validation_clean"]}
    if fit & validation:
        raise JointProbeError("writer isolation failed")
    counts = (len(data["fit_positive"]), len(data["fit_clean"]),
              len(data["validation_positive"]), len(data["validation_clean"]))
    if counts != EXPECTED_COUNTS:
        raise JointProbeError(f"sample counts changed: {counts}")
    args.recipe = data["run_args"].recipe
    return data


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def run(args, baseline, typepilot) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise JointProbeError(f"unknown arm: {args.arm}")
    if args.seed not in SEEDS:
        raise JointProbeError("each create-only run must use one declared seed")
    output = Path(args.output_dir)
    if output.exists():
        raise JointProbeError(f"output directory already exists: {output}")
    probe = load_module(PROBE_PATH, "hme442_for445", PROBE_SHA256,
                        enforce=False)
    actual = probe_digest(probe)
    if actual != PROBE_SHA256 and not args.allow_probe_drift:
        raise JointProbeError(f"experiment-442 shared-code digest mismatch: {actual}")
    extensions = load_module(
        ROOT / "scripts/357_revision_localizer_extensions.py",
        "hme357_for445", "", enforce=False)
    data = prepare(baseline, typepilot, args)
    if args.validate_only:
        return {"arm": args.arm, "validated": True,
                "dynamic_parameters": dynamic_parameter_count(),
                "counts": list(EXPECTED_COUNTS)}

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        result = train_once(baseline, probe, extensions, data, args, staging)
        summary = {
            "schema_version": SCHEMA, "arm": args.arm, "per_seed": [result],
            "protocol": {
                "backbone": "unetpp_resnet34",
                "num_queries": NUM_QUERIES,
                "dynamic_mask": args.arm in ("dynamic", "joint"),
                "dynamic_stride": DYNAMIC_STRIDE,
                "dynamic_channels": DYNAMIC_CHANNELS,
                "dynamic_layers": DYNAMIC_LAYERS,
                "dynamic_parameters": dynamic_parameter_count(),
                "row_head": args.arm == "joint",
                "row_weight": args.row_weight,
                "agreement_weight": args.agreement_weight,
                "agreement_ramp_steps": args.agreement_ramp_steps,
                "objectness_weight": args.objectness_weight,
                "schedule": "warmup then cosine to 5%",
                "warmup_steps": args.warmup_steps,
                "optimizer_steps": args.optimizer_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "negative_ratio": args.negative_ratio,
                "mask_threshold": args.mask_threshold,
                "score_threshold": args.score_threshold,
                "seed": args.seed, "recipe": args.recipe,
                "experiment_422_sha256": baseline.__source_sha256__,
                "experiment_442_shared_sha256": actual,
                "package_versions": baseline.package_versions(),
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1),
            encoding="utf-8")
        (staging / "run_manifest.json").write_text(json.dumps({
            "schema_version": SCHEMA, "create_only": True,
            "git_commit": git_commit(),
            "script_sha256": digest_file(Path(__file__).resolve()),
            "protocol": summary["protocol"],
            "environment": {"python": platform.python_version(),
                            "platform": platform.platform()},
        }, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        listing = [f"{digest_file(path)}  {path.name}"
                   for path in sorted(staging.iterdir())
                   if path.is_file() and path.name != "SHA256SUMS"]
        (staging / "SHA256SUMS").write_text("\n".join(listing) + "\n",
                                            encoding="utf-8")
        (staging / "complete.json").write_text(json.dumps(
            {"status": "complete", "arm": args.arm, "seed": args.seed},
            sort_keys=True), encoding="utf-8")
        staging.rename(output)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    final = result["validation"]
    return {"arm": args.arm, "seed": args.seed, "best_step": result["best_step"],
            "pixel_iou": final["pixel_iou"],
            "instance_f1_050": final["instance"]["0.50"]["f1"],
            "instance_recall_050": final["instance"]["0.50"]["recall"],
            "row_bacc": final["row"]["balanced_accuracy_head"],
            "row_bacc_from_instances":
                final["row"]["balanced_accuracy_from_instances"],
            "output_dir": str(output)}


def build_parser(baseline, typepilot) -> argparse.ArgumentParser:
    parser = baseline.build_parser(typepilot)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--row-weight", type=float, default=1.0)
    parser.add_argument("--agreement-weight", type=float, default=1.0)
    parser.add_argument("--agreement-ramp-steps", type=int, default=2000)
    parser.add_argument("--allow-probe-drift", action="store_true")
    return parser


def main() -> int:
    baseline = load_module(BASELINE_PATH, "hme422_for445", BASELINE_SHA256,
                           enforce=True)
    typepilot = baseline.load_typepilot()
    args = build_parser(baseline, typepilot).parse_args()
    try:
        summary = run(args, baseline, typepilot)
    except JointProbeError as exc:
        raise SystemExit("joint-head probe refused: " + str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

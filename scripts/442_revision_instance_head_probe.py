#!/usr/bin/env python3
"""Ask whether predicting revision instances beats predicting one mask.

Measurement on the current best segmenter says 56 of 236 gold revisions (23.7%)
are drawn but fused: one predicted blob swallows two revisions that sit a median
of 4 pixels apart, and the component verifier can only delete whole components,
never split one.  Semantic segmentation cannot represent "these are two", so no
threshold, schedule or capacity change can recover them.

This probe changes only the output representation and the loss.  Both arms use
the same UNet++--ResNet34 backbone, the same data, the same augmentation, the
same schedule and the same evaluator:

* ``semantic``  -- the established single-mask head, as a paired control;
* ``instance``  -- learnable queries over the decoder features, each emitting a
  mask and an objectness logit, trained through Hungarian matching against the
  connected components of the gold mask.

Both arms are scored twice: once on the union of their output, which is exactly
comparable with every experiment-422 baseline, and once on the predicted
instances themselves, which is the only reading that can show a separation
benefit.  Reporting only the second would be marking our own homework.

The script deliberately does not: introduce new training data, change the
backbone or its ImageNet initialisation, alter the augmentation or the batch
composition, select an operating threshold outside the development split
experiment 422 already used, or read M6, reference or frozen-test material.
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
EXTENSIONS_PATH = ROOT / "scripts/357_revision_localizer_extensions.py"
MATCHING_FUNCTIONS = ("_label_components", "component_metrics")
MATCHING_SHA256 = "d1017748b096d529bd05bb85e966d88dd103bab4c9c1a6d41d80a43234bae414"

SCHEMA = "hme-revision-instance-head-probe-v1"

ARMS = ("semantic", "instance")
SEEDS = (42, 2354, 3407)

EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_TRUE_COMPONENTS = 236
CONNECTIVITY = 8
MATCH_IOUS = (0.10, 0.25, 0.50)

# The gold mask never holds more than 7 connected components on a line, so
# sixteen queries cannot be the binding constraint.
NUM_QUERIES = 16
QUERY_DIM = 96
# Queries attend to a coarsened feature map; 96x768 tokens would not fit.
ATTENTION_STRIDE = 8

THRESHOLDS = tuple(round(0.02 * step, 2) for step in range(1, 50))


class InstanceProbeError(RuntimeError):
    """Raised when the instance-probe contract is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected: str, *, enforce: bool) -> Any:
    actual = digest_file(path)
    if enforce and actual != expected:
        raise InstanceProbeError(f"{path.name} digest mismatch: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise InstanceProbeError(f"cannot import {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = actual
    return module


def matching_digest(module: Any) -> str:
    import inspect

    blob = "".join(inspect.getsource(getattr(module, name))
                   for name in MATCHING_FUNCTIONS)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# instances


def gold_instances(mask) -> list:
    """Connected components of one gold mask, which is what the metric counts."""
    from scipy import ndimage

    structure = np.ones((3, 3), bool)
    labels, count = ndimage.label(np.asarray(mask) > 0.5, structure=structure)
    return [(labels == index) for index in range(1, count + 1)]


def instance_component_metrics(instances: Sequence[Sequence[np.ndarray]],
                               truths: Sequence[Sequence[np.ndarray]],
                               match_iou: float) -> dict[str, Any]:
    """Match explicit predicted instances against gold components, one to one.

    Experiment 357 derives predicted components by labelling a thresholded
    mask, which re-fuses instances that touch.  This takes the instances as
    given; matching is otherwise the same rule, so the two agree exactly when
    the instances are the connected components of their own union.
    """
    true_positive = false_positive = false_negative = 0
    for predicted, truth in zip(instances, truths):
        pairs = []
        for p_index, p_mask in enumerate(predicted):
            p_area = int(p_mask.sum())
            if p_area == 0:
                continue
            for t_index, t_mask in enumerate(truth):
                intersection = int(np.logical_and(p_mask, t_mask).sum())
                if not intersection:
                    continue
                union = p_area + int(t_mask.sum()) - intersection
                iou = intersection / union
                if iou >= match_iou:
                    pairs.append((iou, p_index, t_index))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_p, used_t = set(), set()
        matched = 0
        for _iou, p_index, t_index in pairs:
            if p_index in used_p or t_index in used_t:
                continue
            used_p.add(p_index)
            used_t.add(t_index)
            matched += 1
        non_empty = sum(1 for mask in predicted if mask.sum())
        true_positive += matched
        false_positive += non_empty - matched
        false_negative += len(truth) - matched
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else 0.0)
    recall = (true_positive / (true_positive + false_negative)
              if true_positive + false_negative else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative}


# --------------------------------------------------------------------------
# model


def build_instance_model(baseline: Any, args: argparse.Namespace, ablation: Any):
    import torch
    from torch import nn

    backbone = ablation.make_unetpp_resnet34(args.resnet34_weights)

    class InstanceHead(nn.Module):
        """Queries over the shared decoder features; one mask per query.

        The backbone's 1x1 head is swapped for an identity so its forward
        returns the 64-channel decoder map at input resolution.  Nothing else
        about the backbone changes, which is what makes the semantic control a
        control.
        """

        def __init__(self, model, calibrated: bool):
            super().__init__()
            channels = model.head.in_channels
            model.head = nn.Identity()
            self.backbone = model
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
            # A per-query affine on the mask logits is a learned threshold:
            # cutting the calibrated logit at zero is cutting the raw one at
            # -bias/scale, which differs per instance.
            self.calibration = nn.Linear(QUERY_DIM, 2) if calibrated else None
            if self.calibration is not None:
                nn.init.zeros_(self.calibration.weight)
                nn.init.zeros_(self.calibration.bias)

        def forward(self, value):
            pixels = self.project(self.backbone(value))
            coarse = torch.nn.functional.avg_pool2d(
                pixels, ATTENTION_STRIDE, ATTENTION_STRIDE)
            memory = coarse.flatten(2).transpose(1, 2)
            queries = self.queries.unsqueeze(0).expand(coarse.shape[0], -1, -1)
            queries = self.decoder_stack(queries, memory)
            masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(queries), pixels)
            if self.calibration is not None:
                parameters = self.calibration(queries)
                scale = torch.nn.functional.softplus(parameters[..., 0]) + 0.5
                masks = masks * scale[..., None, None] \
                    + parameters[..., 1][..., None, None]
            return masks, self.objectness(queries).squeeze(-1)

    return InstanceHead(backbone, args.mask_calibration == "learned")


def hungarian_match(mask_logits, objectness, targets):
    """Cheapest one-to-one assignment of queries to this image's gold instances."""
    import torch
    from scipy.optimize import linear_sum_assignment

    if not targets:
        return [], []
    probability = torch.sigmoid(mask_logits)
    flat = probability.flatten(1)
    gold = torch.stack(targets).flatten(1).to(flat.dtype)
    numerator = 2.0 * flat @ gold.t()
    denominator = flat.sum(1)[:, None] + gold.sum(1)[None, :]
    dice_cost = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    class_cost = -torch.log_softmax(
        torch.stack((objectness, -objectness), 1), 1)[:, 0][:, None]
    cost = (dice_cost + class_cost).detach().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    return list(rows), list(columns)


def instance_loss(mask_logits, objectness, batch_targets):
    import torch
    from torch.nn import functional as functional

    total = torch.zeros((), device=mask_logits.device)
    object_total = torch.zeros((), device=mask_logits.device)
    for index in range(mask_logits.shape[0]):
        targets = batch_targets[index]
        rows, columns = hungarian_match(
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
            total = total + functional.binary_cross_entropy_with_logits(
                selected, gold) + (1.0 - numerator / denominator).mean()
        # Unmatched queries must learn to say "no object", which is where
        # precision enters the objective at all.
        object_total = object_total + functional.binary_cross_entropy_with_logits(
            objectness[index], label)
    count = mask_logits.shape[0]
    return total / count + object_total / count, total / count, object_total / count


# --------------------------------------------------------------------------
# prediction


def predict_semantic(baseline, model, rows, batch_size, device):
    return baseline.predict(model, rows, batch_size, device)


def predict_instance(model, rows, batch_size, device):
    """Per-row instance masks plus the union probability map."""
    import torch

    model.eval()
    unions, per_row = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            image = np.stack([row["image"] for row in chunk]).astype(np.float32)
            value = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            mask_logits, objectness = model(value)
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
            masks = torch.sigmoid(mask_logits)
            scores = torch.sigmoid(objectness)
            weighted = masks * scores[:, :, None, None]
            unions.append(weighted.amax(1).cpu().numpy())
            per_row.append((masks.cpu().numpy(), scores.cpu().numpy()))
    model.train()
    union = np.concatenate(unions, 0)
    masks = np.concatenate([item[0] for item in per_row], 0)
    scores = np.concatenate([item[1] for item in per_row], 0)
    return union, masks, scores


def instances_from_prediction(masks, scores, mask_threshold: float,
                              score_threshold: float) -> list[list[np.ndarray]]:
    out = []
    for row_masks, row_scores in zip(masks, scores):
        keep = []
        for mask, score in zip(row_masks, row_scores):
            if score < score_threshold:
                continue
            binary = mask >= mask_threshold
            if binary.any():
                keep.append(binary)
        out.append(keep)
    return out


# --------------------------------------------------------------------------
# evaluation


def union_instances(extensions: Any, probability, threshold: float):
    """Connected components of a thresholded map, as the baselines are scored."""
    out = []
    for row in probability:
        labels, areas = extensions._label_components(row >= threshold, CONNECTIVITY)
        out.append([(labels == index + 1) for index in range(len(areas))])
    return out


def evaluate(extensions: Any, arm: str, union, masks, scores,
             positive_count: int, gold, args) -> dict[str, Any]:
    """Union scoring for comparability; instance scoring for the actual question."""
    truth = np.concatenate(
        (np.stack(gold["positive_masks"]),
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

    result = {"threshold": threshold, "pixel_iou": best[1], "union": {},
              "instance": {}}
    for match_iou in MATCH_IOUS:
        metrics = extensions.component_metrics(
            union, truth, threshold, connectivity=CONNECTIVITY,
            match_iou=match_iou)
        result["union"][f"{match_iou:.2f}"] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "false_positive": int(metrics["false_positive"]),
            "fp_components_per_line": float(metrics["false_positive"])
                                      / union.shape[0]}

    truths = [gold_instances(row) for row in truth]
    if arm == "instance":
        predicted = instances_from_prediction(
            masks, scores, args.mask_threshold, args.score_threshold)
    else:
        predicted = union_instances(extensions, union, threshold)
    for match_iou in MATCH_IOUS:
        metrics = instance_component_metrics(predicted, truths, match_iou)
        metrics["fp_components_per_line"] = \
            metrics["false_positive"] / union.shape[0]
        result["instance"][f"{match_iou:.2f}"] = metrics
    result["predicted_instances"] = int(sum(len(item) for item in predicted))
    result["gold_instances"] = int(sum(len(item) for item in truths))
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


def train_once(baseline, extensions, data, args, staging: Path) -> dict[str, Any]:
    import torch

    seed = args.seed
    if seed not in SEEDS or not torch.cuda.is_available():
        raise InstanceProbeError("the probe requires CUDA and a declared seed")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base, extension, ablation = data["base"], data["extension"], data["ablation"]
    if args.arm == "instance":
        model = build_instance_model(baseline, args, ablation).to(device)
    else:
        model = ablation.make_unetpp_resnet34(args.resnet34_weights).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    rng = random.Random(seed)
    positive_cycle = base.ShuffledCycle(data["fit_positive"], rng)
    clean_cycle = base.ShuffledCycle(data["fit_clean"], rng)

    validation_rows = data["validation_positive"] + data["validation_clean"]
    positive_count = len(data["validation_positive"])
    gold = {"positive_masks": [row["mask"].astype(np.float32) / 255.0
                               for row in data["validation_positive"]]}

    curve = []
    best_key = best_state = None
    best_step = -1
    start_select = int(round(args.optimizer_steps * 0.5))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rate = learning_rate_at(step, args)
        for group in optimiser.param_groups:
            group["lr"] = rate
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

        if args.arm == "instance":
            mask_logits, objectness = model(value)
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
            batch_targets = []
            for index in range(len(rows)):
                pieces = gold_instances(target[index, 0].detach().cpu().numpy())
                batch_targets.append([torch.from_numpy(piece.astype(np.float32)
                                                       ).to(device)
                                      for piece in pieces])
            _total, mask_part, object_part = instance_loss(
                mask_logits, objectness, batch_targets)
            loss = mask_part + args.objectness_weight * object_part
            last_losses = {"loss": float(loss.detach()),
                           "mask": float(mask_part.detach()),
                           "objectness": float(object_part.detach())}
        else:
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
            logits = baseline._ensure_size(model(value), value)
            loss, standard, auxiliary = baseline.loss_terms(
                logits, target, skeleton, "standard", extension)
            last_losses = {"loss": float(loss.detach()),
                           "mask": float(standard.detach()),
                           "objectness": 0.0}
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()

        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        union, masks_out, scores_out = evaluate_forward(
            baseline, model, args, validation_rows, device)
        reading = evaluate(extensions, args.arm, union, masks_out, scores_out,
                           positive_count, gold, args)
        entry = {"step": step, "learning_rate": rate,
                 "pixel_iou": reading["pixel_iou"],
                 "union_f1_050": reading["union"]["0.50"]["f1"],
                 "instance_f1_050": reading["instance"]["0.50"]["f1"],
                 "loss": last_losses["loss"]}
        curve.append(entry)
        # Selection uses the reading the probe is judged on, for both arms.
        key = (reading["instance"]["0.50"]["f1"], reading["pixel_iou"])
        if step >= start_select and (best_key is None or key > best_key):
            best_key, best_step = key, step
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        print(json.dumps({"arm": args.arm, "event": "validation", "seed": seed,
                          "step": step, "best_step": best_step,
                          "instance_f1_050": round(entry["instance_f1_050"], 6),
                          "union_f1_050": round(entry["union_f1_050"], 6)},
                         sort_keys=True), flush=True)

    if best_state is None:
        raise InstanceProbeError("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    union, masks_out, scores_out = evaluate_forward(
        baseline, model, args, validation_rows, device)
    final = evaluate(extensions, args.arm, union, masks_out, scores_out,
                     positive_count, gold, args)
    np.savez_compressed(staging / f"validation_union_seed{seed}.npz",
                        union=union.astype(np.float16))
    if args.arm == "instance":
        # Per-query output, so the objectness threshold -- hard-coded until now
        # and never swept -- can be swept offline instead of by retraining.
        # Masks are packed at the mask threshold actually used; scores are kept
        # in full so the whole score axis stays open.
        binary = masks_out >= args.mask_threshold
        payload = {
            "packed_masks": np.packbits(binary, axis=None),
            "mask_shape": np.asarray(binary.shape, dtype=np.int64),
            "scores": scores_out.astype(np.float32),
            "mask_threshold": np.asarray(args.mask_threshold, dtype=np.float32),
            "positive_count": np.asarray(positive_count, dtype=np.int64),
        }
        if args.save_probabilities:
            # Quantised to a byte so every threshold rule -- fixed, per line,
            # or relative to each instance's own peak -- can be read offline
            # instead of costing a training run each.
            payload["quantised_probabilities"] = np.clip(
                np.rint(masks_out * 255.0), 0, 255).astype(np.uint8)
        np.savez_compressed(
            staging / f"validation_instances_seed{seed}.npz", **payload)
    with (staging / f"curve_seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
        for entry in curve:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"seed": seed, "best_step": best_step, "last_losses": last_losses,
            "validation": final, "curve_length": len(curve)}


def evaluate_forward(baseline, model, args, rows, device):
    if args.arm == "instance":
        return predict_instance(model, rows, args.batch_size, device)
    probability, _ = baseline.predict(model, rows, args.batch_size, device)
    return probability, None, None


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
        raise InstanceProbeError("writer isolation failed")
    counts = (len(data["fit_positive"]), len(data["fit_clean"]),
              len(data["validation_positive"]), len(data["validation_clean"]))
    if counts != EXPECTED_COUNTS:
        raise InstanceProbeError(f"sample counts changed: {counts}")
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
        raise InstanceProbeError(f"unknown arm: {args.arm}")
    if args.seed not in SEEDS:
        raise InstanceProbeError("each create-only run must use one declared seed")
    output = Path(args.output_dir)
    if output.exists():
        raise InstanceProbeError(f"output directory already exists: {output}")
    extensions = load_module(EXTENSIONS_PATH, "hme357_for442", "",
                             enforce=False)
    actual = matching_digest(extensions)
    if actual != MATCHING_SHA256 and not args.allow_matching_drift:
        raise InstanceProbeError(f"experiment-357 matching digest mismatch: {actual}")
    data = prepare(baseline, typepilot, args)
    if args.validate_only:
        return {"arm": args.arm, "validated": True,
                "counts": list(EXPECTED_COUNTS)}

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        result = train_once(baseline, extensions, data, args, staging)
        summary = {
            "schema_version": SCHEMA, "arm": args.arm, "per_seed": [result],
            "protocol": {
                "backbone": "unetpp_resnet34",
                "num_queries": NUM_QUERIES if args.arm == "instance" else None,
                "match_ious": list(MATCH_IOUS),
                "schedule": "warmup then cosine to 5%",
                "warmup_steps": args.warmup_steps,
                "optimizer_steps": args.optimizer_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "negative_ratio": args.negative_ratio,
                "seed": args.seed, "recipe": args.recipe,
                "mask_threshold": args.mask_threshold,
                "score_threshold": args.score_threshold,
                "objectness_weight": args.objectness_weight,
                "mask_calibration": args.mask_calibration,
                "experiment_422_sha256": baseline.__source_sha256__,
                "experiment_357_matching_sha256": actual,
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
            "union_f1_050": final["union"]["0.50"]["f1"],
            "instance_f1_050": final["instance"]["0.50"]["f1"],
            "predicted_instances": final["predicted_instances"],
            "output_dir": str(output)}


def build_parser(baseline, typepilot) -> argparse.ArgumentParser:
    parser = baseline.build_parser(typepilot)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--mask-calibration", choices=("none", "learned"),
                        default="none")
    parser.add_argument("--save-probabilities", action="store_true")
    return parser


def main() -> int:
    baseline = load_module(BASELINE_PATH, "hme422_for442", BASELINE_SHA256,
                           enforce=True)
    typepilot = baseline.load_typepilot()
    args = build_parser(baseline, typepilot).parse_args()
    try:
        summary = run(args, baseline, typepilot)
    except InstanceProbeError as exc:
        raise SystemExit("instance-head probe refused: " + str(exc))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

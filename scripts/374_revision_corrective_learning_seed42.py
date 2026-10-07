#!/usr/bin/env python3
"""Test corrective fine-tuning and prior-mask refinement at fixed seed 42.

Both stages start from the experiment-373 UNet++ trained on the original data.
``finetune_replay`` updates that model with one corrected positive and one
original positive per batch plus deterministic clean replay. ``prior_refiner``
freezes the source UNet++, feeds its probability map beside the image to a
lightweight residual U-Net, and learns only corrections to the source logits.
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
ABLATION_PATH = ROOT / "scripts/371_revision_gold105_ablation.py"
ABLATION_SHA256 = "5a972c1598698eea2e15b0b997b1b6a9a412a953a57e85d27333ef42c294cb94"
SOURCE_SCHEMA = "hme-revision-unetpp-original-seed42-dev-v1"
SCHEMA = "hme-revision-corrective-learning-seed42-dev-v1"
STAGES = ("finetune_replay", "prior_refiner")


class Experiment374Error(RuntimeError):
    """Raised when corrective-learning protocol or source integrity changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment374Error(f"expected JSON object: {path}")
    return value


def load_ablation():
    if digest_file(ABLATION_PATH) != ABLATION_SHA256:
        raise Experiment374Error("experiment-371 dependency digest mismatch")
    specification = importlib.util.spec_from_file_location(
        "hme371_for374", ABLATION_PATH)
    if specification is None or specification.loader is None:
        raise Experiment374Error("cannot import experiment 371")
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
        raise Experiment374Error("source checkpoint must be experiment-373 seed42")
    complete = read_json(root / "complete.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"} \
            or summary.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("schema_version") != SOURCE_SCHEMA:
        raise Experiment374Error("source experiment schema or completion changed")
    if manifest.get("corrected105_read") is not False \
            or manifest.get("m6_frozen_test_touched") is not False \
            or summary.get("data", {}).get("data_arm") != "new1000" \
            or summary.get("data", {}).get("writer_overlap_fit_validation") != 0:
        raise Experiment374Error("source experiment safety contract changed")
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in sums:
            raise Experiment374Error("source SHA256SUMS is malformed")
        sums[name] = digest
    if sums.get(path.name) != digest_file(path):
        raise Experiment374Error("source checkpoint digest mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("seed") != 42 or payload.get("data_arm") != "new1000" \
            or payload.get("training_stage") != "scratch_original1000" \
            or payload.get("architecture", {}).get("variant") != "unetpp_resnet34":
        raise Experiment374Error("source checkpoint metadata changed")
    return {
        "checkpoint": payload,
        "checkpoint_sha256": digest_file(path),
        "manifest_sha256": digest_file(root / "run_manifest.json"),
        "root": root,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.stage not in STAGES or args.mode != "dev" \
            or args.data_arm != "new1000" or args.recipe != "optimized" \
            or args.seeds != "42" or args.input_mode != "image" \
            or args.merge_radius_factor != 0 \
            or not math.isclose(args.isolated_fp_weight, 0.5):
        raise Experiment374Error("corrective-learning base protocol changed")
    expected_steps = 3000 if args.stage == "finetune_replay" else 4000
    expected_lr = 3e-5 if args.stage == "finetune_replay" else 1e-4
    if args.optimizer_steps != expected_steps \
            or args.validation_interval != 100 \
            or not math.isclose(args.selection_warmup_fraction, 0.2) \
            or not math.isclose(args.learning_rate, expected_lr):
        raise Experiment374Error("stage-specific optimisation protocol changed")
    if args.corrected_positive_per_batch != 1 \
            or args.corrected_clean_interval != 20 \
            or args.refiner_base_channels != 32 \
            or not math.isclose(args.residual_weight, 0.01):
        raise Experiment374Error("corrective sampling or refiner protocol changed")


def patch_prepare_with_source_flags(ablation, base, args) -> None:
    ablation.patch_prepare(base, args)
    prepared_with_extra = base.prepare

    def prepare(run_args):
        prepared = prepared_with_extra(run_args)
        fitting, _excluded, _qa = ablation.load_extra_rows(
            base, prepared, run_args)
        corrected_ids = {str(row["line_id"]) for row in fitting}
        if len(corrected_ids) != 80:
            raise Experiment374Error("corrected safe-row identity changed")
        # ``runner.run`` materialises rows after ``prepare`` and intentionally
        # keeps only its canonical row fields.  Preserve the identity set on
        # the run arguments, then restore the training-only flag after
        # materialisation in ``train_once``.
        run_args.corrected_fit_line_ids = tuple(sorted(corrected_ids))
        for name in ("fit_positive", "fit_clean"):
            prepared[name] = [dict(
                row, corrected105=str(row["line_id"]) in corrected_ids)
                for row in prepared[name]]
        corrected_positive = sum(
            bool(row["corrected105"]) for row in prepared["fit_positive"])
        corrected_clean = sum(
            bool(row["corrected105"]) for row in prepared["fit_clean"])
        if corrected_positive != 77 or corrected_clean != 3:
            raise Experiment374Error("corrected replay partition changed")
        return prepared

    base.prepare = prepare


def tag_materialised_corrected_rows(fit_positive: list[dict],
                                    fit_clean: list[dict],
                                    args: argparse.Namespace) -> None:
    corrected_ids = set(getattr(args, "corrected_fit_line_ids", ()))
    if len(corrected_ids) != 80:
        raise Experiment374Error("materialised corrected identity set changed")
    for row in fit_positive + fit_clean:
        row["corrected105"] = str(row["line_id"]) in corrected_ids
    counts = {
        "corrected_positive": sum(
            bool(row["corrected105"]) for row in fit_positive),
        "original_positive": sum(
            not row["corrected105"] for row in fit_positive),
        "corrected_clean": sum(
            bool(row["corrected105"]) for row in fit_clean),
        "original_clean": sum(
            not row["corrected105"] for row in fit_clean),
    }
    expected = {"corrected_positive": 77, "original_positive": 718,
                "corrected_clean": 3, "original_clean": 2382}
    if counts != expected:
        raise Experiment374Error(
            f"materialised corrective partition changed: {counts}")


def load_source_model(ablation, args, device, source: dict[str, Any]):
    model = ablation.make_unetpp_resnet34(args.resnet34_weights).to(device)
    model.load_state_dict(source["checkpoint"]["state_dict"], strict=True)
    return model


def correction_batch(base, fit_positive: list[dict], fit_clean: list[dict],
                     rng: random.Random, step: int,
                     args: argparse.Namespace, cycles: dict[str, Any]):
    positive_count = int(round(args.batch_size / (1.0 + args.negative_ratio)))
    clean_count = args.batch_size - positive_count
    if positive_count != 2 or clean_count != 6:
        raise Experiment374Error("corrective batch expects exactly 2 positive + 6 clean")
    rows = cycles["corrected_positive"].take(1) \
        + cycles["original_positive"].take(1)
    if step % args.corrected_clean_interval == 0:
        rows += cycles["corrected_clean"].take(1)
        rows += cycles["original_clean"].take_distinct(clean_count - 1)
    else:
        rows += cycles["original_clean"].take_distinct(clean_count)
    identities = [str(row["line_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise Experiment374Error("corrective batch contains duplicate lines")
    rng.shuffle(rows)
    return rows


def standard_loss_terms(extension, logits, target, skeleton,
                        args: argparse.Namespace):
    import torch
    from torch import nn

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
    denominator = probability.sum(dimensions) + target.sum(dimensions) + 1.0
    positive = target.sum(dimensions) > 0
    dice = (1.0 - numerator[positive] / denominator[positive]).mean()
    standard = nn.functional.binary_cross_entropy_with_logits(logits, target) + dice
    skeleton_sum = skeleton.sum(dimensions)
    valid_skeleton = skeleton_sum > 0
    skeleton_loss = torch.zeros((), device=logits.device)
    recipe = extension.BASE_RECIPES[args.recipe] \
        if hasattr(extension, "BASE_RECIPES") else None
    if bool(valid_skeleton.any()):
        covered = (probability * skeleton).sum(dimensions)
        skeleton_loss = (1.0 - (covered[valid_skeleton] + 1.0)
                         / (skeleton_sum[valid_skeleton] + 1.0)).mean()
    neighbourhood = extension.isolated_false_positive_proxy_loss(
        logits, target, args.neighbourhood_kernel)
    return standard, skeleton_loss, neighbourhood


def predict_stage(ablation, model, source_model, rows: Sequence[dict],
                  batch_size: int, device, stage: str):
    import torch

    if stage == "finetune_replay":
        return ablation.predict(model, rows, batch_size, device)
    probabilities = []
    model.eval()
    source_model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            image = np.stack([
                row["image"] for row in rows[start:start + batch_size]
            ]).astype(np.float32)
            gray = torch.from_numpy(image[:, None] / 127.5 - 1.0).to(device)
            prior = torch.sigmoid(ablation.main_logits(source_model(gray)))
            refiner_input = torch.cat((gray, prior * 2.0 - 1.0), dim=1)
            delta = model(refiner_input)
            prior_logit = torch.logit(prior.clamp(1e-4, 1.0 - 1e-4))
            probabilities.append(torch.sigmoid(prior_logit + delta)
                                 .cpu().numpy()[:, 0])
    model.train()
    return np.concatenate(probabilities), np.stack(
        [row["mask"] for row in rows]) / 255.0


def train_once(ablation, base, extension, fit_positive: list[dict],
               fit_clean: list[dict], validation_positive: list[dict],
               validation_clean: list[dict], args: argparse.Namespace,
               seed: int, staging: Path, source: dict[str, Any]):
    import torch

    if seed != 42 or not torch.cuda.is_available():
        raise Experiment374Error("formal corrective training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tag_materialised_corrected_rows(fit_positive, fit_clean, args)
    source_model = load_source_model(ablation, args, device, source)
    if args.stage == "finetune_replay":
        model = source_model
        source_for_prediction = None
    else:
        source_model.eval()
        for parameter in source_model.parameters():
            parameter.requires_grad_(False)
        model = extension.make_unet(2, args.refiner_base_channels).to(device)
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
        source_for_prediction = source_model

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        raise Experiment374Error("corrective fitting partition changed")
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
        rows = correction_batch(base, fit_positive, fit_clean, rng, step,
                                args, cycles)
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
        residual = torch.zeros((), device=device)
        if args.stage == "finetune_replay":
            logits = ablation.main_logits(model(gray))
        else:
            with torch.no_grad():
                prior = torch.sigmoid(ablation.main_logits(source_model(gray)))
            delta = model(torch.cat((gray, prior * 2.0 - 1.0), dim=1))
            logits = torch.logit(prior.clamp(1e-4, 1.0 - 1e-4)) + delta
            residual = delta.abs().mean()
        standard, skeleton_loss, neighbourhood = standard_loss_terms(
            extension, logits, target, skeleton, args)
        loss = standard + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood \
            + (args.residual_weight * residual
               if args.stage == "prior_refiner" else 0.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood.detach().cpu()),
            "residual": float(residual.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard.detach().cpu()),
        }
        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        positive_probability, positive_target = predict_stage(
            ablation, model, source_for_prediction, validation_positive,
            args.batch_size, device, args.stage)
        clean_probability, _ = predict_stage(
            ablation, model, source_for_prediction, validation_clean,
            args.batch_size, device, args.stage)
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
            "stage": args.stage, "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Experiment374Error("corrective training produced no checkpoint")
    model.load_state_dict(best_state)
    positive_probability, positive_target = predict_stage(
        ablation, model, source_for_prediction, validation_positive,
        args.batch_size, device, args.stage)
    clean_probability, _ = predict_stage(
        ablation, model, source_for_prediction, validation_clean,
        args.batch_size, device, args.stage)
    points = base.choose_operating_points(
        positive_probability, positive_target, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": {
            "base_channels": 48 if args.stage == "finetune_replay"
                else args.refiner_base_channels,
            "input_channels": 1 if args.stage == "finetune_replay" else 2,
            "residual_logits": args.stage == "prior_refiner",
            "variant": "unetpp_resnet34" if args.stage == "finetune_replay"
                else "image_prior_residual_unet",
        },
        "best_step": best_step,
        "data_arm": "new1000_plus_safe80",
        "operating_points": points,
        "seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "stage": args.stage,
        "state_dict": model.state_dict(),
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
        "stage": args.stage,
        "validation": points,
        "variant": args.stage,
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
        raise Experiment374Error("corrective data contract changed")
    summary["schema_version"] = SCHEMA
    summary["variant"] = args.stage
    summary["protocol"].update({
        "corrected_clean_interval": args.corrected_clean_interval,
        "corrected_positive_per_batch": args.corrected_positive_per_batch,
        "refiner_base_channels": args.refiner_base_channels,
        "residual_weight": args.residual_weight,
        "seed_policy": "single fixed seed 42",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "stage": args.stage,
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
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
        "stage": args.stage,
        "variant": args.stage,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(base, runner, full) -> argparse.ArgumentParser:
    parser = full.build_parser(base, runner)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--extra-key", required=True, type=Path)
    parser.add_argument("--extra-returned", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--resnet34-weights", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--corrected-positive-per-batch", type=int, default=1)
    parser.add_argument("--corrected-clean-interval", type=int, default=20)
    parser.add_argument("--refiner-base-channels", type=int, default=32)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    return parser


def run(args, ablation, base, extension, runner, full):
    validate_args(args)
    source = verify_source(args.source_checkpoint)
    full.configure_base(base)
    patch_prepare_with_source_flags(ablation, base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(ablation, b, e, fp, fc, vp, vc, a, seed, staging, source)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment374Error(f"create-only output exists: {output}")
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
    ablation = load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for374", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for374", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for374", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for374", ablation.FULL1000_SHA256)
    args = build_parser(base, runner, full).parse_args()
    try:
        summary = run(args, ablation, base, extension, runner, full)
    except (Experiment374Error, ablation.Ablation371Error,
            base.LocalizerError, runner.MechanismError, full.Full1000Error) as exc:
        raise SystemExit(f"experiment 374 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"], "ensemble": summary.get("ensemble", {}),
        "stage": args.stage,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

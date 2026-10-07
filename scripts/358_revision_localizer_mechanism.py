#!/usr/bin/env python3
"""Development-only mechanism ablations for revision-region localisation.

This runner deliberately reuses experiment 354's pinned data preparation while
never invoking its final/reference path.  It supports the combined gold arm on
the same writer-disjoint development split and adds, one factor at a time:
near-endpoint target bridges, a label-blind Qwen rendered-stroke prior, and an
SCNP-inspired neighbourhood loss.  Every output is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "354_revision_localizer_optimized.py"
EXTENSION_PATH = ROOT / "scripts" / "357_revision_localizer_extensions.py"
BASE_SHA256 = "73e3582b59186f97899571f832e01f983034fed4bba2269b2feb5d57fa4e2780"
SCHEMA = "hme-revision-localizer-mechanism-dev-v1"


class MechanismError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str | None = None):
    actual = digest_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise MechanismError(
            f"dependency digest mismatch for {path.name}: {actual}"
        )
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise MechanismError(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def add_extension_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--input-mode", choices=(
        "image", "image_ocr", "image_spatial"),
                        default="image")
    parser.add_argument("--ocr-source", type=Path)
    parser.add_argument("--qwen-predictions", type=Path)
    parser.add_argument("--qwen-retry", type=Path)
    parser.add_argument("--ocr-font", type=Path)
    parser.add_argument("--ocr-dropout", type=float, default=0.10)
    parser.add_argument("--spatial-cache-dir", type=Path)
    parser.add_argument("--spatial-cache-channels",
                        default="text_ink,text_skeleton")
    parser.add_argument("--merge-radius-factor", type=float, default=0.0)
    parser.add_argument("--isolated-fp-weight", type=float, default=0.0)
    parser.add_argument("--neighbourhood-kernel", type=int, default=3)
    parser.add_argument("--component-match-iou", type=float, default=0.10)
    return parser


def validate_args(args: argparse.Namespace, base) -> None:
    if args.mode != "dev":
        raise MechanismError("this mechanism runner is development-only")
    if args.recipe != "optimized":
        raise MechanismError("mechanism ablations are locked to recipe=optimized")
    if args.recipe_lock is not None or args.hard_source_dir is not None:
        raise MechanismError("reference locks and mining inputs are forbidden")
    if not args.gold_confirmed:
        raise MechanismError("the gold package must be explicitly confirmed")
    if args.optimizer_steps <= 0 or args.validation_interval <= 0:
        raise MechanismError("steps and validation interval must be positive")
    if args.merge_radius_factor < 0:
        raise MechanismError("merge radius factor cannot be negative")
    if args.isolated_fp_weight < 0:
        raise MechanismError("isolated-FP weight cannot be negative")
    if args.neighbourhood_kernel < 1 or args.neighbourhood_kernel % 2 != 1:
        raise MechanismError("neighbourhood kernel must be a positive odd integer")
    if not 0.0 <= args.ocr_dropout < 1.0:
        raise MechanismError("OCR dropout must be in [0,1)")
    if not 0.0 < args.component_match_iou <= 1.0:
        raise MechanismError("component match IoU must be in (0,1]")
    if args.input_mode == "image_ocr":
        required = (args.ocr_source, args.qwen_predictions, args.ocr_font)
        if any(path is None for path in required):
            raise MechanismError(
                "image_ocr requires --ocr-source, --qwen-predictions, and --ocr-font"
            )
        for path in required:
            if not Path(path).is_file():
                raise MechanismError(f"OCR dependency missing: {path}")
        if args.qwen_retry is not None and not args.qwen_retry.is_file():
            raise MechanismError(f"Qwen retry file missing: {args.qwen_retry}")
        if args.spatial_cache_dir is not None:
            raise MechanismError("image_ocr cannot also consume a spatial cache")
    elif args.input_mode == "image_spatial":
        if args.spatial_cache_dir is None or not args.spatial_cache_dir.is_dir():
            raise MechanismError(
                "image_spatial requires --spatial-cache-dir")
        spatial_channel_names(args)
        if any(value is not None for value in (
                args.ocr_source, args.qwen_predictions,
                args.qwen_retry, args.ocr_font)):
            raise MechanismError(
                "image_spatial cannot also consume Qwen-render inputs")
    elif any(value is not None for value in (
            args.ocr_source, args.qwen_predictions, args.qwen_retry,
            args.ocr_font, args.spatial_cache_dir)):
        raise MechanismError("auxiliary inputs require a matching input mode")
    if base.RECIPES[args.recipe]["augment"] != "affine":
        raise MechanismError("the extension requires the affine augmentation recipe")


def spatial_channel_names(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(value.strip() for value in
                  str(args.spatial_cache_channels).split(",") if value.strip())
    allowed = {"text_prob", "text_ink", "text_skeleton"}
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise MechanismError(
            "spatial cache channels must be unique names from "
            "text_prob,text_ink,text_skeleton")
    return names


def auxiliary_channel_count(args: argparse.Namespace) -> int:
    if args.input_mode == "image":
        return 0
    if args.input_mode == "image_ocr":
        return 1
    return len(spatial_channel_names(args))


def bridged_rows(rows: list[dict], extension, radius_factor: float):
    output, total_bridges, total_length = [], 0, 0.0
    identities = []
    for row in rows:
        clone = dict(row)
        marks, audit = extension.bridge_near_line_endpoints(
            row["marks"], radius_factor=radius_factor,
            default_stroke_width=3.0,
        )
        clone["marks"] = marks
        output.append(clone)
        total_bridges += int(audit["bridges_added"])
        total_length += float(audit["bridge_length"])
        identities.append({
            "line_id": str(row["line_id"]),
            "bridges_added": int(audit["bridges_added"]),
            "bridge_length": round(float(audit["bridge_length"]), 8),
        })
    encoded = "".join(json.dumps(value, sort_keys=True, separators=(",", ":"))
                      for value in identities)
    return output, {
        "bridge_length": total_length,
        "bridges_added": total_bridges,
        "identity_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "radius_factor": radius_factor,
        "rows": len(rows),
    }


def attach_ocr(rows: list[dict], lookup: dict[str, dict], extension,
               args: argparse.Namespace, crop: tuple[int, int]) -> dict[str, Any]:
    width, height = crop
    missing, invalid, attached = [], 0, 0
    rendered_sha = hashlib.sha256()
    for row in rows:
        record = lookup.get(str(row["line_id"]))
        if record is None:
            missing.append(str(row["line_id"]))
            continue
        valid = bool(record.get("valid"))
        if not valid:
            invalid += 1
        channel = extension.render_ocr_channel(
            str(record.get("text", "")), args.ocr_font, width, height,
            valid=valid,
        )
        row["aux_channels"] = channel[None]
        rendered_sha.update(str(row["line_id"]).encode("utf-8"))
        rendered_sha.update(channel.tobytes())
        attached += 1
    if missing:
        raise MechanismError(
            f"OCR lookup misses {len(missing)} selected rows; first={missing[:5]}"
        )
    return {
        "attached": attached,
        "invalid_or_empty": invalid,
        "rendered_channels_sha256": rendered_sha.hexdigest(),
    }


def attach_spatial_cache(rows: list[dict], cache_dir: Path,
                         channel_names: tuple[str, ...],
                         crop: tuple[int, int]) -> dict[str, Any]:
    import numpy as np

    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "cache_manifest.jsonl"
    run_manifest_path = cache_dir / "run_manifest.json"
    sums_path = cache_dir / "SHA256SUMS"
    for path in (manifest_path, run_manifest_path, sums_path):
        if path.is_symlink() or not path.is_file():
            raise MechanismError(f"spatial cache dependency missing: {path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    contract = run_manifest.get("split_contract") or {}
    if run_manifest.get("schema_version") != "hme-ppocrv6-dense-stroke-cache-v1" \
            or contract.get("labels_read") is not False \
            or contract.get("ocr_boxes_used") is not False \
            or contract.get("recognizer_used") is not False:
        raise MechanismError("spatial cache protocol is not label-free dense OCR")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (cache_dir / relative).resolve()
        if cache_dir not in path.parents or not path.is_file() \
                or digest_file(path) != expected:
            raise MechanismError(f"spatial cache checksum mismatch: {relative}")
    by_line = {}
    records = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        records.append(record)
        identities = set(record.get("requested_line_ids") or [])
        identities.update(record.get("alias_line_ids") or [])
        identities.add(record.get("representative_line_id"))
        for identity in identities:
            if not identity:
                continue
            identity = str(identity)
            previous = by_line.get(identity)
            if previous is not None and previous["crop_id"] != record["crop_id"]:
                raise MechanismError(f"spatial cache alias conflict: {identity}")
            by_line[identity] = record
    width, height = crop
    loaded_by_file, missing = {}, []
    channel_hash = hashlib.sha256()
    for row in rows:
        line_id = str(row["line_id"])
        record = by_line.get(line_id)
        if record is None:
            missing.append(line_id)
            continue
        relative = str(record["cache_file"])
        cache_path = (cache_dir / relative).resolve()
        if cache_dir not in cache_path.parents \
                or digest_file(cache_path) != record["cache_sha256"]:
            raise MechanismError(f"spatial crop checksum mismatch: {relative}")
        if relative not in loaded_by_file:
            with np.load(cache_path, allow_pickle=False) as document:
                if not set(channel_names) <= set(document.files):
                    raise MechanismError(
                        f"spatial crop lacks requested channels: {relative}")
                channels = []
                for name in channel_names:
                    value = np.asarray(document[name], dtype=np.float32)
                    if value.shape != (height, width) \
                            or not np.isfinite(value).all() \
                            or bool((value < 0).any()) or bool((value > 1).any()):
                        raise MechanismError(
                            f"invalid spatial channel {name}: {relative}")
                    channels.append(value)
                loaded_by_file[relative] = np.stack(channels)
        row["aux_channels"] = loaded_by_file[relative]
        channel_hash.update(line_id.encode("utf-8"))
        channel_hash.update(row["aux_channels"].tobytes())
    if missing:
        raise MechanismError(
            f"spatial cache misses {len(missing)} selected rows; first={missing[:5]}")
    return {
        "cache_files_loaded": len(loaded_by_file),
        "cache_manifest_sha256": digest_file(manifest_path),
        "cache_run_manifest_sha256": digest_file(run_manifest_path),
        "channel_names": list(channel_names),
        "rendered_channels_sha256": channel_hash.hexdigest(),
        "rows_attached": len(rows),
        "source_cache_files": len(records),
    }


def predict(model, rows: list[dict], batch_size: int, device, input_mode: str):
    import numpy as np
    import torch

    probabilities, targets = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            gray = np.stack([row["image"] for row in chunk]).astype(np.float32)
            channels = [gray[:, None] / 127.5 - 1.0]
            if input_mode != "image":
                auxiliary = np.stack([
                    row["aux_channels"] for row in chunk]).astype(np.float32)
                channels.append(auxiliary)
            x = torch.from_numpy(np.concatenate(channels, axis=1)).to(device)
            probabilities.append(torch.sigmoid(model(x)).cpu().numpy()[:, 0])
            if "mask" in chunk[0]:
                targets.append(np.stack([row["mask"] for row in chunk]).astype(
                    np.float32) / 255.0)
    model.train()
    probability = np.concatenate(probabilities) if probabilities else np.empty((0, 1, 1))
    target = np.concatenate(targets) if targets else None
    return probability, target


def train_once(base, extension, fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path):
    import numpy as np
    import torch
    from torch import nn

    recipe = base.RECIPES[args.recipe]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") \
            not in (":4096:8", ":16:8"):
        raise MechanismError("CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1 + auxiliary_channel_count(args)
    model = extension.make_unet(in_channels, args.base_channels).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(seed)
    # Keep the base image/geometry augmentation stream byte-for-byte comparable
    # with image-only arms.  Auxiliary-channel dropout gets its own stream.
    auxiliary_dropout_seed = seed ^ 0x3580C12
    auxiliary_rng = random.Random(auxiliary_dropout_seed)
    positive_cycle = base.ShuffledCycle(fit_positive, rng)
    clean_cycle = base.ShuffledCycle(fit_clean, rng)
    best_key, best_step, best_state, best_points = None, -1, None, None
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        chunk, _, _ = base.take_balanced_batch(
            positive_cycle, clean_cycle, args.batch_size, args.negative_ratio)
        rng.shuffle(chunk)
        images, masks, skeletons, auxes = [], [], [], []
        for row in chunk:
            image, mask, skeleton = row["image"], row["mask"], row["line_skeleton"]
            aux = row.get("aux_channels")
            image, mask, skeleton, aux = extension.paired_affine_with_aux(
                image, mask, skeleton, aux, rng)
            if aux is not None and auxiliary_rng.random() < args.ocr_dropout:
                aux = np.zeros_like(aux)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
            if aux is not None:
                auxes.append(aux)
        gray = np.stack(images).astype(np.float32)
        channels = [gray[:, None] / 127.5 - 1.0]
        if args.input_mode != "image":
            channels.append(np.stack(auxes).astype(np.float32))
        x = torch.from_numpy(np.concatenate(channels, axis=1)).to(device)
        y = torch.from_numpy(np.stack(masks).astype(np.float32) / 255.0)[:, None].to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32) / 255.0)[:, None].to(device)
        logits = model(x)
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, probability.ndim))
        numerator = 2.0 * (probability * y).sum(dimensions) + 1.0
        denominator = probability.sum(dimensions) + y.sum(dimensions) + 1.0
        positive_target = y.sum(dimensions) > 0
        dice_loss = (1.0 - numerator[positive_target]
                     / denominator[positive_target]).mean()
        standard_loss = bce(logits, y) + dice_loss
        skeleton_sum = skeleton.sum(dimensions)
        valid_skeleton = skeleton_sum > 0
        skeleton_loss = torch.zeros((), device=device)
        if recipe["skeleton_weight"] and bool(valid_skeleton.any()):
            covered = (probability * skeleton).sum(dimensions)
            skeleton_loss = (1.0 - (covered[valid_skeleton] + 1.0)
                             / (skeleton_sum[valid_skeleton] + 1.0)).mean()
        neighbourhood_loss = torch.zeros((), device=device)
        if args.isolated_fp_weight:
            neighbourhood_loss = extension.isolated_false_positive_proxy_loss(
                logits, y, args.neighbourhood_kernel)
        loss = standard_loss + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood_loss.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard_loss.detach().cpu()),
        }

        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        val_probability, val_target = predict(
            model, validation_positive, args.batch_size, device, args.input_mode)
        clean_probability, _ = predict(
            model, validation_clean, args.batch_size, device, args.input_mode)
        points = base.choose_operating_points(
            val_probability, val_target, clean_probability,
            args.max_clean_fp,
            base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels),
        )
        primary = points["localization"]
        key = (primary["iou"], primary["macro_iou"], primary["recall"],
               -primary["threshold"])
        eligible = step >= start_select
        if eligible and (best_key is None or key > best_key):
            best_key, best_step, best_points = key, step, points
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(json.dumps({
            "best_step": best_step,
            "event": "validation",
            "input_mode": args.input_mode,
            "loss": round(last_losses["loss"], 6),
            "seed": seed,
            "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 5),
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise MechanismError("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    val_probability, val_target = predict(
        model, validation_positive, args.batch_size, device, args.input_mode)
    clean_probability, _ = predict(
        model, validation_clean, args.batch_size, device, args.input_mode)
    points = base.choose_operating_points(
        val_probability, val_target, clean_probability, args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels),
    )
    torch.save({
        "architecture": {"base_channels": args.base_channels,
                         "crop": list(recipe["crop"]),
                         "in_channels": in_channels},
        "best_step": best_step,
        "data_arm": args.data_arm,
        "input_mode": args.input_mode,
        "operating_points": points,
        "seed": seed,
        "auxiliary_dropout_seed": auxiliary_dropout_seed,
        "state_dict": model.state_dict(),
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=val_probability.astype(np.float16),
        targets=(val_target * 255).astype(np.uint8),
    )
    return {
        "auxiliary_dropout_seed": auxiliary_dropout_seed,
        "best_step": best_step,
        "last_losses": last_losses,
        "seed": seed,
        "validation": points,
    }, {"clean": clean_probability, "positive": val_probability}


def run(args: argparse.Namespace, base, extension) -> dict[str, Any]:
    import numpy as np

    validate_args(args, base)
    seeds = base.parse_seeds(args.seeds)
    prepared = base.prepare(args)
    if prepared["audit"]["reference_sets_materialised"]:
        raise MechanismError("development preparation unexpectedly materialised reference data")
    if args.validate_only:
        return {
            "across_seeds": {},
            "data": prepared["audit"],
            "ensemble": {},
            "merge": {},
            "mode": "mechanism-dev-validated",
            "ocr": None,
            "schema_version": SCHEMA,
        }
    output = Path(args.output_dir).resolve()
    staging = base.atomic_staging(output)
    recipe = base.RECIPES[args.recipe]
    images_dir = Path(args.crop_images_dir)

    raw_fit_positive = prepared["fit_positive"]
    if args.merge_radius_factor:
        derived_fit_positive, merge_audit = bridged_rows(
            raw_fit_positive, extension, args.merge_radius_factor)
    else:
        derived_fit_positive = raw_fit_positive
        merge_audit = {
            "bridge_length": 0.0, "bridges_added": 0,
            "identity_sha256": None, "radius_factor": 0.0,
            "rows": len(raw_fit_positive),
        }
    fit_positive, clipped_positive = base.materialise(
        derived_fit_positive, prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    fit_clean, clipped_clean = base.materialise(
        prepared["fit_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    # Validation always keeps the original human target, so all ablations are
    # selected and compared against exactly the same gold geometry.
    validation_positive, clipped_validation_positive = base.materialise(
        prepared["validation_positive"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    validation_clean, clipped_validation_clean = base.materialise(
        prepared["validation_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])

    ocr_audit = None
    if args.input_mode == "image_ocr":
        lookup, lookup_audit = extension.build_qwen_text_lookup(
            args.ocr_source, args.qwen_predictions, args.qwen_retry)
        attachment = attach_ocr(
            fit_positive + fit_clean + validation_positive + validation_clean,
            lookup, extension, args, recipe["crop"],
        )
        ocr_audit = {**lookup_audit, **attachment}
    elif args.input_mode == "image_spatial":
        ocr_audit = attach_spatial_cache(
            fit_positive + fit_clean + validation_positive + validation_clean,
            args.spatial_cache_dir, spatial_channel_names(args), recipe["crop"],
        )

    split_rows = []
    for split_name, rows in (
        ("fit_positive", fit_positive), ("fit_clean", fit_clean),
        ("validation_positive", validation_positive),
        ("validation_clean", validation_clean),
    ):
        split_rows.extend({
            "group_uuid": row["group_uuid"], "line_id": row["line_id"],
            "source": row["source"], "split": split_name,
        } for row in rows)
    split_path = staging / "split_manifest.jsonl"
    base.write_jsonl(split_path, sorted(
        split_rows, key=lambda value: (value["split"], value["line_id"])))
    base.write_json(staging / "validation_key.json", [{
        "group_uuid": row["group_uuid"], "image_id": row["image_id"],
        "line_id": row["line_id"], "source": row["source"],
        "types": row.get("types", []),
    } for row in validation_positive])

    results, positive_total, clean_total = [], None, None
    for seed in seeds:
        result, probabilities = train_once(
            base, extension, fit_positive, fit_clean, validation_positive,
            validation_clean, args, seed, staging)
        results.append(result)
        if positive_total is None:
            positive_total = probabilities["positive"].astype(np.float64)
            clean_total = probabilities["clean"].astype(np.float64)
        else:
            positive_total += probabilities["positive"]
            clean_total += probabilities["clean"]
    positive_mean = positive_total / len(seeds)
    clean_mean = clean_total / len(seeds)
    targets = np.stack([row["mask"] for row in validation_positive]).astype(
        np.float32) / 255.0
    points = base.choose_operating_points(
        positive_mean, targets, clean_mean, args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels),
    )
    threshold = points["localization"]["threshold"]
    all_probability = np.concatenate((positive_mean, clean_mean), axis=0)
    all_targets = np.concatenate((targets, np.zeros_like(clean_mean)), axis=0)
    component = extension.component_metrics(
        all_probability, all_targets, threshold,
        connectivity=8, match_iou=args.component_match_iou,
    )
    boundary = base.boundary_f1(
        positive_mean, targets, threshold,
        tolerance=max(1, int(round(3 * recipe["crop"][1] / 64))),
    )
    np.savez_compressed(
        staging / "validation_predictions_mean.npz",
        clean_probabilities=clean_mean.astype(np.float16),
        positive_probabilities=positive_mean.astype(np.float16),
        positive_targets=(targets * 255).astype(np.uint8),
    )
    summary = {
        "across_seeds": {
            "validation_localization_iou": base.aggregate(
                results, ("validation", "localization", "iou")),
        },
        "data": prepared["audit"],
        "ensemble": {
            "boundary_f1": boundary,
            "component": component,
            "operating_points": points,
        },
        "merge": merge_audit,
        "mode": "mechanism-dev",
        "ocr": ocr_audit,
        "per_seed": results,
        "protocol": {
            "base_354_sha256": BASE_SHA256,
            "batch_size": args.batch_size,
            "component_match_iou": args.component_match_iou,
            "data_arm": args.data_arm,
            "input_mode": args.input_mode,
            "input_channels": 1 + auxiliary_channel_count(args),
            "isolated_fp_weight": args.isolated_fp_weight,
            "merge_radius_factor": args.merge_radius_factor,
            "neighbourhood_kernel": args.neighbourhood_kernel,
            "negative_ratio": args.negative_ratio,
            "ocr_dropout": args.ocr_dropout,
            "ocr_dropout_rng": "independent random.Random(seed ^ 0x3580C12)",
            "optimizer_steps": args.optimizer_steps,
            "recipe": args.recipe,
            "seeds": seeds,
            "split_manifest_sha256": digest_file(split_path),
            "threshold_selected_on": "writer-disjoint new609 validation only",
        },
        "schema_version": SCHEMA,
    }
    base.write_json(staging / "summary.json", summary)
    input_hashes = {
        "base_354": BASE_SHA256,
        "extension_357": digest_file(EXTENSION_PATH),
        "hard_negative_manifest": digest_file(args.hard_negative_manifest),
        "new_returned": digest_file(args.new_returned),
        "old_returned": digest_file(args.old_returned),
    }
    if args.input_mode == "image_ocr":
        input_hashes.update({
            "ocr_font": digest_file(args.ocr_font),
            "ocr_source": digest_file(args.ocr_source),
            "qwen_predictions": digest_file(args.qwen_predictions),
            "qwen_retry": None if args.qwen_retry is None else
                digest_file(args.qwen_retry),
        })
    elif args.input_mode == "image_spatial":
        input_hashes.update({
            "spatial_cache_manifest": digest_file(
                args.spatial_cache_dir / "cache_manifest.jsonl"),
            "spatial_cache_run_manifest": digest_file(
                args.spatial_cache_dir / "run_manifest.json"),
            "spatial_cache_sha256s": digest_file(
                args.spatial_cache_dir / "SHA256SUMS"),
        })
    base.write_json(staging / "run_manifest.json", {
        "clipped_coordinate_values": clipped_positive + clipped_clean
            + clipped_validation_positive + clipped_validation_clean,
        "command": [sys.executable, *sys.argv],
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_arm": args.data_arm,
        "evaluation_fold_trained_on": False,
        "git_revision": base.git_revision(),
        "gold_confirmation": True,
        "input_dependency_hashes": prepared["input_dependency_hashes"],
        "input_hashes": input_hashes,
        "m6_frozen_test_touched": False,
        "mode": "mechanism-dev",
        "network_used": False,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "split_manifest_sha256": digest_file(split_path),
    })
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    staging.rename(output)
    return summary


def main() -> int:
    base = load_module(BASE_PATH, "hme354_for358", BASE_SHA256)
    extension = load_module(EXTENSION_PATH, "hme357_for358")
    parser = add_extension_arguments(base.build_parser())
    args = parser.parse_args()
    try:
        summary = run(args, base, extension)
    except (MechanismError, base.LocalizerError) as exc:
        raise SystemExit(f"experiment 358 refused: {exc}") from exc
    print(json.dumps({
        "across_seeds": summary["across_seeds"],
        "ensemble": summary["ensemble"],
        "merge": summary["merge"],
        "ocr": summary["ocr"],
        "status": summary["mode"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

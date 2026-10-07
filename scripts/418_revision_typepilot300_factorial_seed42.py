#!/usr/bin/env python3
"""Factorial study for the newly reviewed type-pilot 300 at fixed seed 42.

The four arms cross current-best fine-tuning versus all-gold re-training with
automatic morphology targets versus human line/dense targets.  ``overwrite``
is merged into the dense target because it remains too rare for a standalone
expert.  The cleaned writer-disjoint development set from experiments 410/416
is reused verbatim; frozen test/reference material is never materialised.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TWO_TYPE_PATH = ROOT / "scripts/416_revision_two_type_architecture_seed42.py"
TWO_TYPE_SHA256 = "af7505c8174bfa1cbb1abf5fcf8d0f1931e6becb1e8bfb0cb486d10d3137ea58"
SCHEMA = "hme-revision-typepilot300-factorial-seed42-dev-v1"
SEED = 42
VARIANTS = {
    "finetune_auto": {"stage": "finetune", "target": "auto", "steps": 4000},
    "finetune_human": {"stage": "finetune", "target": "human", "steps": 4000},
    "scratch_auto": {"stage": "scratch", "target": "auto", "steps": 6000},
    "scratch_human": {"stage": "scratch", "target": "human", "steps": 6000},
}


class Experiment418Error(RuntimeError):
    """Raised when a data, provenance, or create-only invariant changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_two_type():
    if digest_file(TWO_TYPE_PATH) != TWO_TYPE_SHA256:
        raise Experiment418Error("experiment-416 dependency digest mismatch")
    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "hme416_for418", TWO_TYPE_PATH)
    if specification is None or specification.loader is None:
        raise Experiment418Error("cannot import experiment 416")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment418Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment418Error(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment418Error(f"invalid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise Experiment418Error(f"JSONL contains a non-object: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def canonical_mark(mark: Mapping[str, Any]) -> str:
    ignored = {"proposed", "weak", "component_score"}
    value = {}
    for key in sorted(mark):
        if key in ignored:
            continue
        item = mark[key]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            item = round(float(item), 6)
        value[key] = item
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def correction_audit(key_rows: Sequence[Mapping[str, Any]],
                     audit_rows: Sequence[Mapping[str, Any]],
                     returned: Mapping[str, Any]) -> dict[str, Any]:
    audits = {str(row["id"]): row for row in audit_rows}
    records = returned.get("records")
    if not isinstance(records, dict):
        raise Experiment418Error("returned annotation records are absent")
    counts = collections.Counter()
    changed_proposed = 0
    for key in key_rows:
        record_id = str(key["id"])
        original = list(audits[record_id].get("final_marks") or [])
        final = list(records[record_id].get("marks") or [])
        original_counter = collections.Counter(map(canonical_mark, original))
        proposed = [mark for mark in final if bool(mark.get("proposed"))]
        proposed_counter = collections.Counter(map(canonical_mark, proposed))
        unchanged = sum((original_counter & proposed_counter).values())
        changed_proposed += len(proposed) - unchanged
        counts.update({
            "shown": len(original),
            "unchanged_kept": unchanged,
            "deleted_or_replaced": len(original) - unchanged,
            "human_added": sum(not bool(mark.get("proposed")) for mark in final),
            "final_marks": len(final),
        })
    return {**dict(counts), "changed_proposed_geometry": changed_proposed}


def load_new_rows(two_type: Any, data: dict[str, Any],
                  args: argparse.Namespace):
    base = data["base"]
    key_path = args.new_key.expanduser().resolve()
    returned_path = args.new_returned.expanduser().resolve()
    audit_path = args.new_audit.expanduser().resolve()
    manifest_path = args.new_manifest.expanduser().resolve()
    key_rows = sorted(read_jsonl(key_path), key=lambda row: int(row["index"]))
    audit_rows = read_jsonl(audit_path)
    returned = read_json(returned_path)
    manifest = read_json(manifest_path)
    if len(key_rows) != 300 or len(audit_rows) != 300 \
            or len(returned.get("records") or {}) != 300 \
            or [int(row["index"]) for row in key_rows] != list(range(300)):
        raise Experiment418Error("new type-pilot package is not exactly 300 rows")
    identifiers = {str(row["id"]) for row in key_rows}
    if identifiers != {str(row["id"]) for row in audit_rows} \
            or identifiers != set(returned["records"]):
        raise Experiment418Error("new package ID coverage differs")
    fingerprint = str(manifest.get("page_fingerprint"))
    if fingerprint != str(returned.get("page_fingerprint")) \
            or manifest.get("sampled") != 300 \
            or manifest.get("frozen_or_protected_data_touched") is not False \
            or manifest.get("evaluation_writers_touched") is not False:
        raise Experiment418Error("new package safety envelope differs")

    segmenter = base.load_module(
        base.SEGMENTER_SCRIPT, "hme344_for418", base.SEGMENTER_SHA256)
    catalogue_rows, catalogue = base.load_catalogue(segmenter)
    evaluation_package = base.load_module(
        base.EVAL_PACKAGE_SCRIPT, "hme325_for418", base.EVAL_PACKAGE_SHA256)
    protected_images = {
        str(row["image_id"]) for row in catalogue_rows
        if set(row.get("data_roles") or [])
        & {"fixed_development_only", "fixed_external_test_only"}
    }
    pinned = evaluation_package.collect_test_identifiers()
    rows, sanitation = base.read_package(
        key_path, returned_path, name="typepilot300",
        expected_key_sha=digest_file(key_path),
        expected_returned_sha=digest_file(returned_path),
        expected_fingerprint=fingerprint, expected_records=300,
        protected_images=protected_images, pinned_test_identifiers=pinned)
    skipped_protected = int(
        sanitation["protected_image_records_not_semantically_read"])
    skipped_pinned = int(
        sanitation["pinned_test_records_not_semantically_read"])
    if sanitation["safe_records"] + skipped_protected + skipped_pinned != 300:
        raise Experiment418Error("new package protection accounting differs")

    validation_writers = {
        str(row["group_uuid"]) for row in
        data["validation_positive"] + data["validation_clean"]
    }
    fit_rows = [row for row in rows
                if str(row["group_uuid"]) not in validation_writers]
    excluded = [row for row in rows
                if str(row["group_uuid"]) in validation_writers]
    existing_ids = {str(row["line_id"]) for row in
                    data["fit_positive"] + data["fit_clean"]
                    + data["validation_positive"] + data["validation_clean"]}
    if {str(row["line_id"]) for row in fit_rows} & existing_ids:
        raise Experiment418Error("new type-pilot rows overlap an existing split")

    base_protocol = segmenter.load_region().load_matched().load_gap().load_base()
    fold_images = Path(data["run_args"].fold_images_dir
                       or data["run_args"].crop_images_dir)
    training_rows, evaluation_rows = base_protocol._validate_training_inputs(
        Path(data["run_args"].input_root), fold_images)
    fold_lookup = {str(row["line_id"]): row
                   for row in list(training_rows) + list(evaluation_rows)}
    recipe = base.RECIPES[data["run_args"].recipe]
    images_dir = Path(data["run_args"].crop_images_dir)

    def materialise(values):
        return base.materialise(
            values, segmenter, base_protocol, catalogue, images_dir,
            recipe["crop"], fold_lookup)

    positive_raw = [row for row in fit_rows if row["marks"]]
    clean_raw = [row for row in fit_rows if not row["marks"]]
    positive, clipped_positive = materialise(positive_raw)
    clean, clipped_clean = materialise(clean_raw)

    raw_lookup = {str(row["line_id"]): row for row in positive_raw}
    masks_by_type: dict[str, dict[str, np.ndarray]] = {}
    clipped_types = {}
    for mark_type in ("line", "blot", "overwrite"):
        typed = [{**row, "marks": [dict(mark) for mark in row["marks"]
                                    if str(mark["type"]) == mark_type]}
                 for row in positive_raw]
        rendered, clipped = materialise(typed)
        masks_by_type[mark_type] = {
            str(row["line_id"]): np.asarray(row["mask"], dtype=np.uint8)
            for row in rendered}
        clipped_types[mark_type] = int(clipped)
    mismatch = 0
    for row in positive:
        line_id = str(row["line_id"])
        stack = np.stack([masks_by_type[name][line_id]
                          for name in ("line", "blot", "overwrite")])
        stack, difference = two_type.reconcile_type_masks(
            stack, np.asarray(row["mask"] > 0, dtype=bool))
        mismatch += int(difference)
        row["human_type_masks"] = stack
        row["typepilot300"] = True
        row["active300"] = True
        row["hard_negative_mask"] = np.zeros_like(row["mask"], dtype=np.uint8)
    zero = np.zeros((3, *positive[0]["mask"].shape), dtype=np.uint8)
    for row in clean:
        row["human_type_masks"] = zero.copy()
        row["typepilot300"] = True
        row["active300"] = True
        row["hard_negative_mask"] = np.zeros_like(row["mask"], dtype=np.uint8)
    for row in data["fit_positive"] + data["fit_clean"]:
        row["typepilot300"] = False

    corrections = correction_audit(key_rows, audit_rows, returned)
    audit = {
        "corrections": corrections,
        "excluded_validation_writer_rows": len(excluded),
        "fitting_clean": len(clean),
        "fitting_positive": len(positive),
        "fitting_rows": len(fit_rows),
        "fitting_writers": len({str(row["group_uuid"]) for row in fit_rows}),
        "input_sha256": {
            "audit": digest_file(audit_path), "key": digest_file(key_path),
            "manifest": digest_file(manifest_path),
            "returned": digest_file(returned_path),
        },
        "sanitation": sanitation,
        "type_clipped_coordinate_values": clipped_types,
        "union_mismatch_pixels_before_reconciliation": mismatch,
        "union_mismatch_pixels_after_reconciliation": 0,
        "union_clipped_coordinate_values": clipped_positive + clipped_clean,
    }
    return positive, clean, audit


def predict(two_type: Any, model: Any, rows: Sequence[Mapping[str, Any]],
            batch_size: int, device: Any):
    return two_type.predict_model(model, rows, batch_size, device)


def training_batch(data: Mapping[str, Any], cycles: Mapping[str, Any],
                   rng: random.Random, stage: str, step: int):
    if stage == "finetune":
        rows = cycles["new_positive"].take_distinct(3)
        rows += cycles["replay_positive"].take_distinct(3)
    else:
        rows = cycles["all_positive"].take_distinct(6)
    if step % 12 == 0 and "new_clean" in cycles:
        rows += cycles["new_clean"].take(1)
        rows += cycles["replay_clean"].take_distinct(5)
    else:
        rows += cycles["all_clean"].take_distinct(6)
    if len({str(row["line_id"]) for row in rows}) != len(rows):
        raise Experiment418Error("training batch contains duplicate line IDs")
    rng.shuffle(rows)
    return rows


def train(two_type: Any, type410: Any, data: dict[str, Any], initial: Any,
          args: argparse.Namespace, config: Mapping[str, Any], staging: Path):
    import torch
    from torch.nn import functional

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(SEED)
    random.seed(SEED)
    base_core = data["ablation"].make_unetpp_resnet34(
        data["run_args"].resnet34_weights)
    if config["stage"] == "finetune":
        source_path = args.source_pixel_output / "segmenter_seed42.pt"
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source.get("variant") != "type_dual_topology":
            raise Experiment418Error("fine-tune source is not type_dual_topology")
        model = type410.make_model(
            data["ablation"], data["run_args"].resnet34_weights,
            initial["checkpoint"]["state_dict"], True)
        model.load_state_dict(source["state_dict"], strict=True)
        parameter_count = type410.configure_trainable(model, True)
        source_sha = digest_file(source_path)
        learning_rate = 1e-5
    else:
        model = type410.make_model(
            data["ablation"], data["run_args"].resnet34_weights,
            base_core.state_dict(), True)
        for parameter in model.parameters():
            parameter.requires_grad = True
        parameter_count = {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters()),
        }
        source_sha = None
        learning_rate = 3e-4
    model = model.to(device)
    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=data["run_args"].weight_decay)
    rng = random.Random(SEED)
    partitions = {
        "new_positive": [row for row in data["fit_positive"]
                         if row["typepilot300"]],
        "replay_positive": [row for row in data["fit_positive"]
                            if not row["typepilot300"]],
        "new_clean": [row for row in data["fit_clean"]
                      if row["typepilot300"]],
        "replay_clean": [row for row in data["fit_clean"]
                         if not row["typepilot300"]],
        "all_positive": list(data["fit_positive"]),
        "all_clean": list(data["fit_clean"]),
    }
    if len(partitions["new_positive"]) < 3 or len(partitions["all_clean"]) < 6:
        raise Experiment418Error("type-pilot training partitions are incomplete")
    cycles = {name: data["base"].ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    initial_positive, initial_target = predict(
        two_type, model, data["validation_positive"], 12, device)
    initial_clean, _ = predict(
        two_type, model, data["validation_clean"], 12, device)
    initial_coverage = type410.cleaned_coverage_operating_point(
        data["coverage"], initial_positive, initial_target > 0.5,
        initial_clean)
    best_key = best_state = best_coverage = best_points = None
    best_step = -1
    last_losses = {}
    warmup = int(config["steps"] * (0.25 if config["stage"] == "finetune" else 0.20))
    trace_path = staging / "training_trace.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        trace.write(json.dumps({"coverage": initial_coverage,
                                "event": "initial", "step": 0},
                               sort_keys=True) + "\n")
        for step in range(1, int(config["steps"]) + 1):
            model.train()
            if config["stage"] == "finetune":
                type410.keep_frozen_encoder_eval(model, True)
            rows = training_batch(data, cycles, rng, str(config["stage"]), step)
            images, masks, skeletons, human_types = [], [], [], []
            for row in rows:
                if config["target"] == "human":
                    image, mask, skeleton, typed = two_type.augment_human_types(
                        data["extension"], row, rng)
                    human_types.append(two_type.merge_human_types(typed))
                else:
                    image, mask, skeleton, _ = \
                        data["extension"].paired_affine_with_aux(
                            row["image"], row["mask"], row["line_skeleton"],
                            None, rng)
                images.append(image)
                masks.append(mask)
                skeletons.append(skeleton)
            gray = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            target = torch.from_numpy(
                np.stack(masks).astype(np.float32)[:, None] / 255.0
            ).to(device)
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0
            ).to(device)
            logits, line_logits, dense_logits, gate_logits = \
                model.forward_with_aux(gray)
            standard, skeleton_loss, neighbourhood = \
                data["corrective"].standard_loss_terms(
                    data["extension"], logits, target, skeleton,
                    data["run_args"])
            if config["target"] == "human":
                type_target = torch.from_numpy(
                    np.stack(human_types).astype(np.float32)).to(device)
                line_target = type_target[:, 0:1]
                dense_target = type_target[:, 1:2]
            else:
                line_target, dense_target = type410.type_targets(target, skeleton)
            expert = (type410.balanced_dice_bce(line_logits, line_target)
                      + type410.balanced_dice_bce(dense_logits, dense_target)) / 2.0
            selected = target >= 0.5
            gate = functional.binary_cross_entropy_with_logits(
                gate_logits[selected], line_target[selected]) \
                if bool(selected.any()) else torch.zeros((), device=device)
            scnp = data["extension"].isolated_false_positive_proxy_loss(
                line_logits, line_target, data["run_args"].neighbourhood_kernel)
            loss = standard + 0.35 * skeleton_loss + 0.10 * expert \
                + 0.05 * gate + 0.10 * scnp
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad], 5.0)
            optimiser.step()
            last_losses = {
                "expert": float(expert.detach().cpu()),
                "gate": float(gate.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                "scnp": float(scnp.detach().cpu()),
                "skeleton": float(skeleton_loss.detach().cpu()),
                "standard": float(standard.detach().cpu()),
            }
            if step % 500 and step != int(config["steps"]):
                continue
            positive, validation_target = predict(
                two_type, model, data["validation_positive"], 12, device)
            clean, _ = predict(
                two_type, model, data["validation_clean"], 12, device)
            try:
                coverage = type410.cleaned_coverage_operating_point(
                    data["coverage"], positive, validation_target > 0.5, clean)
            except type410.Experiment410Error as exc:
                if "no cleaned coverage threshold meets precision floor" \
                        not in str(exc):
                    raise
                record = {
                    "best_step": best_step, "event": "validation_ineligible",
                    "reason": "no_threshold_meets_pixel_precision_floor",
                    "step": step, "variant": args.variant,
                }
                trace.write(json.dumps(record, sort_keys=True) + "\n")
                trace.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                continue
            points = data["base"].choose_operating_points(
                positive, validation_target, clean,
                data["run_args"].max_clean_fp,
                data["base"].scaled_min_fire_pixels(
                    data["run_args"].recipe,
                    data["run_args"].min_fire_pixels))
            key = data["coverage"].coverage_key(coverage)
            if step >= warmup and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_coverage, best_points = coverage, points
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            record = {
                "best_step": best_step,
                "covered_ge_10": coverage["covered_ge_10_count"],
                "covered_ge_50": coverage["covered_ge_50_count"],
                "event": "validation",
                "median": coverage["median_gold_component_coverage"],
                "pixel_iou": coverage["pixel_iou"],
                "step": step, "variant": args.variant,
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
    if best_state is None or best_coverage is None or best_points is None:
        raise Experiment418Error("training produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_positive, final_target = predict(
        two_type, model, data["validation_positive"], 12, device)
    final_clean, _ = predict(
        two_type, model, data["validation_clean"], 12, device)
    final_coverage = type410.cleaned_coverage_operating_point(
        data["coverage"], final_positive, final_target > 0.5, final_clean)
    typed = two_type.merge_human_types(np.stack([
        row["human_type_masks"] for row in data["validation_positive"]]))
    per_type = two_type.per_type_component_coverage(
        final_positive, typed, final_coverage["threshold"])
    torch.save({
        "architecture": "type_dual_topology_shallow",
        "best_step": best_step,
        "coverage_operating_point": final_coverage,
        "data_arm": "prior_gold_plus_typepilot300_writer_safe",
        "human_types": ["line", "dense=blot+overwrite"],
        "parameter_count": parameter_count, "seed": SEED,
        "source_checkpoint_sha256": source_sha,
        "state_dict": model.state_dict(), "variant": args.variant,
    }, staging / "segmenter_seed42.pt")
    np.savez_compressed(
        staging / "validation_predictions_seed42.npz",
        clean_probabilities=final_clean.astype(np.float16),
        positive_probabilities=final_positive.astype(np.float16),
        targets=(final_target * 255).astype(np.uint8))
    return {
        "best_step": best_step, "coverage": final_coverage,
        "initial_coverage": initial_coverage, "last_losses": last_losses,
        "operating_points": best_points, "parameter_count": parameter_count,
        "partitions": {name: len(rows) for name, rows in partitions.items()},
        "per_type_coverage": per_type,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS or args.seed != SEED:
        raise Experiment418Error("formal factorial uses a known variant/seed42")
    import torch

    if not torch.cuda.is_available():
        raise Experiment418Error("formal factorial requires CUDA")
    two_type = load_two_type()
    type410, data, source, exclusions, cleaning, type_audit, initial = \
        two_type.prepare_data(args)
    new_positive, new_clean, new_audit = load_new_rows(two_type, data, args)
    data["fit_positive"].extend(new_positive)
    data["fit_clean"].extend(new_clean)
    validation_writers = {str(row["group_uuid"]) for row in
                          data["validation_positive"] + data["validation_clean"]}
    fit_writers = {str(row["group_uuid"]) for row in
                   data["fit_positive"] + data["fit_clean"]}
    if fit_writers & validation_writers:
        raise Experiment418Error("writer isolation failed after new300")
    if args.validate_only:
        return {
            "coverage": None,
            "data": {
                "fit_clean": len(data["fit_clean"]),
                "fit_positive": len(data["fit_positive"]),
                "new_typepilot300": new_audit,
                "validation_clean": len(data["validation_clean"]),
                "validation_positive": len(data["validation_positive"]),
                "writer_overlap_fit_validation": 0,
            },
            "variant": args.variant,
        }
    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment418Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result = train(two_type, type410, data, initial, args,
                       VARIANTS[args.variant], staging)
        original_key = type410.read_json(source["root"] / "validation_key.json")
        clean_key = [row for row in original_key
                     if str(row["line_id"]) not in exclusions]
        write_json(staging / "validation_key_clean.json", clean_key)
        summary = {
            "cleaning": cleaning,
            "coverage": result["coverage"],
            "data": {
                "fit_clean": len(data["fit_clean"]),
                "fit_positive": len(data["fit_positive"]),
                "new_typepilot300": new_audit,
                "validation_clean": len(data["validation_clean"]),
                "validation_positive": len(data["validation_positive"]),
                "writer_overlap_fit_validation": 0,
            },
            "gold_components": type410.EXPECTED_CLEAN_GOLD_COMPONENTS,
            "protocol": {
                **VARIANTS[args.variant],
                "fixed_cleaned_validation": True,
                "frozen_test_touched": False,
                "human_types": ["line", "dense=blot+overwrite"],
                "m6_frozen_test_touched": False,
                "reference_evaluation_materialised": False,
                "seed": SEED, "writer_overlap_fit_validation": 0,
            },
            "result": result, "schema_version": SCHEMA,
            "status": "complete", "type_targets_prior": type_audit,
            "variant": args.variant,
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "run_manifest.json", {
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "frozen_test_touched": False,
            "new_input_sha256": new_audit["input_sha256"],
            "reference_evaluation_materialised": False,
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "seed": SEED, "variant": args.variant,
        })
        write_json(staging / "complete.json", {
            "schema_version": SCHEMA, "status": "complete"})
        data["base"].write_recursive_sha256s(staging)
        staging.rename(output)
        return summary
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source-output", type=Path, required=True)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument("--source-pixel-output", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--active-key", type=Path, required=True)
    parser.add_argument("--active-returned", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--new-key", type=Path, required=True)
    parser.add_argument("--new-returned", type=Path, required=True)
    parser.add_argument("--new-audit", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    try:
        summary = run(build_parser().parse_args())
    except (Experiment418Error, RuntimeError) as exc:
        raise SystemExit(f"experiment 418 refused: {exc}") from exc
    print(json.dumps({"coverage": summary["coverage"],
                      "data": summary.get("data") if summary["coverage"] is None else None,
                      "variant": summary["variant"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fine-tune the locked coverage-first segmenter with the fresh active300 gold.

All arms start from the current standard-neg1 seed-42 checkpoint and reuse its
fixed writer-disjoint development set.  The 300 returned rows are sanitized by
the existing package reader; protected images, the illegible row, and writers
present in validation are excluded.  Human final marks are the only positive
targets.  A hard-negative arm may additionally penalize shown model proposals
that the annotator deleted, after subtracting a dilated final-gold mask.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORER390_PATH = ROOT / "scripts/390_revision_coverage_component_scorer.py"
SCORER390_SHA256 = "8773e3383b96514f14cb0a0741dba6523fae305fd6c2bc1d2000b3d615b08e91"
SCHEMA = "hme-revision-active300-coverage-finetune-seed42-dev-v1"
SOURCE_SCHEMA = "hme-revision-coverage-finetune-seed42-dev-v1"
SOURCE_VARIANT = "standard_neg1"
SOURCE_CHECKPOINT_SHA256 = (
    "29df9cd9e2376e9db6e2feff082e571d801def77386e57e4a274b5a283935b5b"
)
VALIDATION_KEY_SHA256 = (
    "4d9dcf5db1d37cb0725a80cf3d5b4848e0351051c9d4121904cd8a6b5ac15cd0"
)
ACTIVE_KEY_SHA256 = (
    "b6a54aa79fc65cd48150561daf59911f14fc8b033f63802c112d6fbb8182c87b"
)
ACTIVE_RETURNED_SHA256 = (
    "2ce6173a339c2ab8a4b2b29681d24bb7d0fb9bc1e717118c7f6ea74c6c3fb1de"
)
ACTIVE_AUDIT_SHA256 = (
    "3b7ffac70622bd52775432f305c0248ba5035d4784fc90e416a90d9485ee4cb4"
)
ACTIVE_MANIFEST_SHA256 = (
    "f04290ac740c54b63e495ea8e5f6c24d21bda34d6b35d1a73274fab554f60d42"
)
ACTIVE_FINGERPRINT = "e99fe309e2ff7f2b8bca04e3f680cfc1"
ACTIVE_RECORDS = 300
ACTIVE_FULL_QA = {
    "clean_after_sanitation": 12,
    "dropped_zero_length_lines": 0,
    "flags": {"no_revision": 12},
    "mark_types": {"blot": 230, "line": 690, "overwrite": 8},
    "pinned_test_records_not_semantically_read": 0,
    "positive_after_sanitation": 257,
    "protected_image_records_not_semantically_read": 31,
    "raw_marks": 928,
    "safe_records": 269,
    "source_records": 300,
}
LOCKED_VALIDATION_POSITIVE = 127
LOCKED_VALIDATION_CLEAN = 381
LOCKED_GOLD_COMPONENTS = 237
VALIDATION_INTERVAL = 250
LEARNING_RATE = 3e-5
SELECTION_WARMUP = 0.20
HARD_DILATION_PIXELS = 2
VARIANTS = {
    "standard_neg1": {
        "active_positive_per_batch": 3,
        "coverage_variant": "standard_neg1", "hard_weight": 0.0,
        "learning_rate": 3e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
    "tversky_neg1": {
        "active_positive_per_batch": 3,
        "coverage_variant": "tversky_neg1", "hard_weight": 0.0,
        "learning_rate": 3e-5, "loss": "recall_tversky",
        "negative_ratio": 1, "steps": 3000,
    },
    "standard_neg2": {
        "active_positive_per_batch": 2,
        "coverage_variant": "standard_neg2", "hard_weight": 0.0,
        "learning_rate": 3e-5, "loss": "standard",
        "negative_ratio": 2, "steps": 3000,
    },
    "standard_neg1_hard": {
        "active_positive_per_batch": 3,
        "coverage_variant": "standard_neg1", "hard_weight": 0.25,
        "learning_rate": 3e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
    "standard_neg1_natural_lr1": {
        "active_positive_per_batch": 1,
        "coverage_variant": "standard_neg1", "hard_weight": 0.0,
        "learning_rate": 1e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
    "standard_neg1_natural_lr3": {
        "active_positive_per_batch": 1,
        "coverage_variant": "standard_neg1", "hard_weight": 0.0,
        "learning_rate": 3e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
    "standard_neg1_quarter_lr1": {
        "active_positive_per_batch": 2,
        "coverage_variant": "standard_neg1", "hard_weight": 0.0,
        "learning_rate": 1e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
    "standard_neg1_quarter_lr3": {
        "active_positive_per_batch": 2,
        "coverage_variant": "standard_neg1", "hard_weight": 0.0,
        "learning_rate": 3e-5, "loss": "standard",
        "negative_ratio": 1, "steps": 3000,
    },
}


class Experiment402Error(RuntimeError):
    """Raised when an active300, source, or split invariant changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str):
    actual = digest_file(path)
    if actual != expected_sha256:
        raise Experiment402Error(
            f"dependency digest mismatch for {path.name}: {actual}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Experiment402Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment402Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment402Error(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment402Error(f"invalid JSONL: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise Experiment402Error(f"JSONL contains non-object rows: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def verify_checksums(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    sums = Path(root) / "SHA256SUMS"
    if not sums.is_file():
        raise Experiment402Error(f"SHA256SUMS is absent: {root}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or relative in entries:
            raise Experiment402Error(f"malformed SHA256SUMS: {sums}")
        if digest_file(Path(root) / relative) != expected:
            raise Experiment402Error(f"checksum mismatch: {root}/{relative}")
        entries[relative] = expected
    return entries


def verify_source(root: Path) -> dict[str, Any]:
    import torch

    root = Path(root).expanduser().resolve()
    entries = verify_checksums(root)
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    complete = read_json(root / "complete.json")
    checkpoint_path = root / "segmenter_seed42.pt"
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"} \
            or summary.get("schema_version") != SOURCE_SCHEMA \
            or summary.get("variant") != SOURCE_VARIANT \
            or manifest.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("variant") != SOURCE_VARIANT \
            or manifest.get("m6_frozen_test_touched") is not False \
            or manifest.get("reference_evaluation_materialised") is not False:
        raise Experiment402Error("source safety contract changed")
    checkpoint_sha = digest_file(checkpoint_path)
    if checkpoint_sha != SOURCE_CHECKPOINT_SHA256 \
            or entries.get(checkpoint_path.name) != checkpoint_sha \
            or digest_file(root / "validation_key.json") \
            != VALIDATION_KEY_SHA256:
        raise Experiment402Error("source checkpoint or validation key changed")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("seed") != 42 \
            or checkpoint.get("variant") != SOURCE_VARIANT \
            or checkpoint.get("data_arm") \
            != "new1000_plus_all300_plus_active200":
        raise Experiment402Error("source checkpoint metadata changed")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "manifest": manifest,
        "manifest_sha256": digest_file(root / "run_manifest.json"),
        "root": root,
        "summary": summary,
    }


def mark_signature(mark: Mapping[str, Any]) -> str:
    """Canonical geometry/type signature, ignoring proposal display metadata."""
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


def deleted_proposals(original: Sequence[Mapping[str, Any]],
                      returned: Sequence[Mapping[str, Any]]) \
        -> list[dict[str, Any]]:
    retained = collections.Counter(
        mark_signature(mark) for mark in returned if mark.get("proposed"))
    deleted: list[dict[str, Any]] = []
    for mark in original:
        signature = mark_signature(mark)
        if retained[signature]:
            retained[signature] -= 1
        else:
            deleted.append({key: value for key, value in mark.items()
                            if key not in {"weak", "component_score"}})
    if any(retained.values()):
        raise Experiment402Error("returned proposal is absent from original audit")
    return deleted


def load_active300(base: Any, prepared: Mapping[str, Any],
                   args: argparse.Namespace) \
        -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = (
        (Path(args.active_key), ACTIVE_KEY_SHA256, "active key"),
        (Path(args.active_returned), ACTIVE_RETURNED_SHA256, "active returned"),
        (Path(args.active_audit), ACTIVE_AUDIT_SHA256, "active audit"),
        (Path(args.active_manifest), ACTIVE_MANIFEST_SHA256, "active manifest"),
    )
    for path, expected, name in inputs:
        if digest_file(path) != expected:
            raise Experiment402Error(f"{name} digest mismatch")
    manifest = read_json(Path(args.active_manifest))
    returned = read_json(Path(args.active_returned))
    key_rows = sorted(
        read_jsonl(Path(args.active_key)), key=lambda row: int(row["index"]))
    audit_rows = read_jsonl(Path(args.active_audit))
    if manifest.get("page_fingerprint") != ACTIVE_FINGERPRINT \
            or manifest.get("sampled") != ACTIVE_RECORDS \
            or manifest.get("frozen_or_protected_data_touched") is not False \
            or returned.get("page_fingerprint") != ACTIVE_FINGERPRINT \
            or len(returned.get("records") or {}) != ACTIVE_RECORDS \
            or len(key_rows) != ACTIVE_RECORDS \
            or len(audit_rows) != ACTIVE_RECORDS:
        raise Experiment402Error("active300 envelope changed")
    keys = {str(row["id"]): row for row in key_rows}
    audits = {str(row["id"]): row for row in audit_rows}
    if set(keys) != set(audits) or set(keys) != set(returned["records"]):
        raise Experiment402Error("active300 ID coverage changed")

    shown = kept = deleted_count = added = 0
    deleted_by_line: dict[str, list[dict[str, Any]]] = {}
    for record_id, key in keys.items():
        final = returned["records"][record_id].get("marks") or []
        original = audits[record_id].get("final_marks") or [] \
            if int(key.get("proposals", 0)) else []
        if len(original) != int(key.get("proposals", 0)):
            raise Experiment402Error("shown proposal count differs from audit")
        removed = deleted_proposals(original, final)
        line_id = str(key["line_id"])
        deleted_by_line[line_id] = removed
        shown += len(original)
        kept += sum(bool(mark.get("proposed")) for mark in final)
        deleted_count += len(removed)
        added += sum(not bool(mark.get("proposed")) for mark in final)
    if (shown, kept, deleted_count, added) != (473, 292, 181, 761):
        raise Experiment402Error(
            f"active300 correction audit changed: "
            f"{(shown, kept, deleted_count, added)}")

    eval_package = base.load_module(
        base.EVAL_PACKAGE_SCRIPT, "hme_evalpkg325_for402",
        base.EVAL_PACKAGE_SHA256)
    catalogue_rows, _lookup = base.load_catalogue(prepared["segmenter"])
    protected_images = {
        str(entry["image_id"]) for entry in catalogue_rows
        if set(entry.get("data_roles") or [])
        & {"fixed_development_only", "fixed_external_test_only"}
    }
    rows, qa = base.read_package(
        Path(args.active_key), Path(args.active_returned), name="active300",
        expected_key_sha=ACTIVE_KEY_SHA256,
        expected_returned_sha=ACTIVE_RETURNED_SHA256,
        expected_fingerprint=ACTIVE_FINGERPRINT,
        expected_records=ACTIVE_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=eval_package.collect_test_identifiers())
    if qa != ACTIVE_FULL_QA:
        raise Experiment402Error(f"active300 sanitation QA changed: {qa}")
    for row in rows:
        row["hard_candidate_marks"] = deleted_by_line.get(
            str(row["line_id"]), [])
    return rows, {
        "corrections": {
            "added_marks": added,
            "deleted_shown_marks": deleted_count,
            "kept_shown_marks": kept,
            "shown_marks": shown,
        },
        "sanitation": qa,
    }


def prepare_data(scorer390: Any, coverage: Any, source: Mapping[str, Any],
                 args: argparse.Namespace, config: Mapping[str, Any]):
    active = coverage.load_active()
    gold300 = active.load_gold300()
    wavelet = gold300.load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for402", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for402", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for402", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for402", ablation.FULL1000_SHA256)
    command = source["manifest"].get("command")
    if not isinstance(command, list) or len(command) < 3:
        raise Experiment402Error("source command is absent")
    run_args = coverage.build_parser(
        active, gold300, corrective, base, runner, full).parse_args(command[2:])
    run_args.variant = str(config["coverage_variant"])
    run_args.negative_ratio = int(config["negative_ratio"])
    coverage.validate_args(run_args)
    full.configure_base(base)
    coverage.patch_prepare_locked_validation(
        active, gold300, base, run_args)
    prepared = base.prepare(run_args)
    active_rows, active_qa = load_active300(base, prepared, args)
    validation_writers = {
        str(row["group_uuid"]) for row in
        prepared["validation_positive"] + prepared["validation_clean"]
    }
    active_fit = [row for row in active_rows
                  if str(row["group_uuid"]) not in validation_writers]
    active_excluded = [row for row in active_rows
                       if str(row["group_uuid"]) in validation_writers]
    existing_ids = {
        str(row["line_id"]) for row in
        prepared["fit_positive"] + prepared["fit_clean"]
        + prepared["validation_positive"] + prepared["validation_clean"]
    }
    active_ids = {str(row["line_id"]) for row in active_fit}
    if active_ids & existing_ids:
        raise Experiment402Error("active300 overlaps an existing split")
    active_positive = [row for row in active_fit if row["marks"]]
    active_clean = [row for row in active_fit if not row["marks"]]
    if len(active_positive) + len(active_clean) != len(active_fit):
        raise Experiment402Error("active300 fitting partition is incomplete")

    source_positive = list(prepared["fit_positive"])
    gold_clean_count = int(prepared["audit"]["fit_gold_clean"])
    source_gold_clean = list(prepared["fit_clean"][:gold_clean_count])
    total_positive = len(source_positive) + len(active_positive)
    wanted_total_clean = int(config["negative_ratio"]) * total_positive
    wanted_catalogue = wanted_total_clean \
        - len(source_gold_clean) - len(active_clean)
    candidates = [
        row for row in prepared["fit_clean_candidates"]
        if str(row["line_id"]) not in active_ids
    ]
    sampled, hard_selection = base.select_hard_prefix(
        Path(run_args.hard_negative_manifest), candidates, wanted_catalogue,
        validation_seed=run_args.validation_seed,
        validation_fraction=run_args.validation_fraction)
    fit_positive = source_positive + active_positive
    fit_clean = source_gold_clean + active_clean + sampled
    if len(fit_clean) != int(config["negative_ratio"]) * len(fit_positive):
        raise Experiment402Error("active300 clean ratio construction changed")
    fit_writers = {str(row["group_uuid"])
                   for row in fit_positive + fit_clean}
    if fit_writers & validation_writers:
        raise Experiment402Error("writer isolation failed after active300")

    recipe = base.RECIPES[run_args.recipe]
    images_dir = Path(run_args.crop_images_dir)

    def materialise(rows):
        return base.materialise(
            rows, prepared["segmenter"], prepared["base"],
            prepared["catalogue"], images_dir, recipe["crop"],
            prepared["fold_lookup"])

    materialised_positive, clip_positive = materialise(fit_positive)
    materialised_clean, clip_clean = materialise(fit_clean)
    validation_positive, clip_val_positive = materialise(
        prepared["validation_positive"])
    validation_clean, clip_val_clean = materialise(
        prepared["validation_clean"])
    active_lookup = {
        str(row["line_id"]): list(row.get("hard_candidate_marks") or [])
        for row in active_fit
    }
    candidate_rows = [
        {**row, "marks": active_lookup[str(row["line_id"])],
         "source": "active300_deleted_visible_proposals"}
        for row in active_fit if active_lookup.get(str(row["line_id"]))
    ]
    candidate_masks: dict[str, np.ndarray] = {}
    if candidate_rows:
        rendered, _candidate_clipped = materialise(candidate_rows)
        candidate_masks = {
            str(row["line_id"]): np.asarray(row["mask"], dtype=np.uint8)
            for row in rendered
        }
    try:
        import cv2
    except ImportError as exc:
        raise Experiment402Error("OpenCV is required for hard masks") from exc
    kernel = np.ones((5, 5), dtype=np.uint8)
    hard_rows = hard_pixels = 0
    for row in materialised_positive + materialised_clean:
        line_id = str(row["line_id"])
        row["active300"] = line_id in active_ids
        candidate = candidate_masks.get(line_id)
        if candidate is None:
            hard = np.zeros_like(row["mask"], dtype=np.uint8)
        else:
            gold = np.asarray(row["mask"] > 0, dtype=np.uint8)
            protected = cv2.dilate(gold, kernel, iterations=1) > 0
            hard = np.asarray((candidate > 0) & ~protected,
                              dtype=np.uint8) * 255
        row["hard_negative_mask"] = hard
        pixels = int((hard > 0).sum())
        row["hard_negative_pixels"] = pixels
        hard_rows += int(pixels > 0)
        hard_pixels += pixels
    audit = {
        "active300": {
            **active_qa,
            "fitting_clean": len(active_clean),
            "fitting_positive": len(active_positive),
            "fitting_rows": len(active_fit),
            "fitting_writers": len({str(row["group_uuid"])
                                     for row in active_fit}),
            "hard_negative_pixels": hard_pixels,
            "hard_negative_rows": hard_rows,
            "validation_writer_rows_excluded": len(active_excluded),
        },
        "clipped_coordinate_values": clip_positive + clip_clean
                                      + clip_val_positive + clip_val_clean,
        "fit_clean": len(materialised_clean),
        "fit_positive": len(materialised_positive),
        "hard_selection": hard_selection,
        "validation_clean": len(validation_clean),
        "validation_positive": len(validation_positive),
        "writer_overlap_fit_validation": 0,
    }
    return {
        "ablation": ablation,
        "audit": audit,
        "base": base,
        "corrective": corrective,
        "extension": extension,
        "fit_clean": materialised_clean,
        "fit_positive": materialised_positive,
        "run_args": run_args,
        "validation_clean": validation_clean,
        "validation_positive": validation_positive,
    }


def masked_false_positive_loss(logits: Any, hard_negative_mask: Any):
    import torch
    from torch.nn import functional

    selected = hard_negative_mask >= 0.5
    if not bool(selected.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return functional.softplus(logits[selected]).mean()


def training_batch(base: Any, rng: random.Random,
                   config: Mapping[str, Any], cycles: Mapping[str, Any],
                   step: int) -> list[dict[str, Any]]:
    ratio = int(config["negative_ratio"])
    positive_count = 6 if ratio == 1 else 4
    clean_count = 12 - positive_count
    active_count = int(config["active_positive_per_batch"])
    replay_count = positive_count - active_count
    if active_count <= 0 or replay_count <= 0:
        raise Experiment402Error("active/replay positive batch share is invalid")
    rows = cycles["active_positive"].take(active_count) \
        + cycles["replay_positive"].take(replay_count)
    if step % 12 == 0 and "active_clean" in cycles:
        rows += cycles["active_clean"].take(1) \
            + cycles["clean"].take_distinct(clean_count - 1)
    else:
        rows += cycles["clean"].take_distinct(clean_count)
    identities = [str(row["line_id"]) for row in rows]
    if len(rows) != 12 or len(identities) != len(set(identities)):
        raise Experiment402Error("active300 batch size or identity changed")
    rng.shuffle(rows)
    return rows


def train(data: Mapping[str, Any], source: Mapping[str, Any],
          args: argparse.Namespace, config: Mapping[str, Any],
          staging: Path) -> dict[str, Any]:
    import torch

    if args.seed != 42 or not torch.cuda.is_available():
        raise Experiment402Error("formal active300 training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = data["ablation"].make_unetpp_resnet34(
        data["run_args"].resnet34_weights).to(device)
    model.load_state_dict(source["checkpoint"]["state_dict"], strict=True)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=data["run_args"].weight_decay)
    rng = random.Random(args.seed)
    partitions = {
        "active_positive": [row for row in data["fit_positive"]
                            if row["active300"]],
        "replay_positive": [row for row in data["fit_positive"]
                            if not row["active300"]],
        "active_clean": [row for row in data["fit_clean"]
                         if row["active300"]],
        "clean": [row for row in data["fit_clean"]
                  if not row["active300"]],
    }
    if not partitions["active_positive"] \
            or not partitions["replay_positive"] \
            or len(partitions["clean"]) < 8:
        raise Experiment402Error("active300 training pools are incomplete")
    cycles = {name: data["base"].ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    recipe = data["base"].RECIPES[data["run_args"].recipe]
    steps = int(config["steps"])
    start_select = int(round(steps * SELECTION_WARMUP))
    source_positive, source_target = data["ablation"].predict(
        model, data["validation_positive"], 12, device)
    source_clean, _ = data["ablation"].predict(
        model, data["validation_clean"], 12, device)
    source_coverage = data["coverage"].coverage_operating_point(
        source_positive, source_target > 0.5, source_clean)
    best_key = best_state = best_coverage = best_points = None
    best_step = -1
    last_losses: dict[str, float] = {}
    trace_path = staging / "training_trace.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        trace.write(json.dumps({
            "coverage": source_coverage, "event": "source", "step": 0,
        }, sort_keys=True) + "\n")
        for step in range(1, steps + 1):
            model.train()
            rows = training_batch(data["base"], rng, config, cycles, step)
            images, masks, skeletons, hard_masks = [], [], [], []
            for row in rows:
                image, mask, skeleton, hard = \
                    data["extension"].paired_affine_with_aux(
                        row["image"], row["mask"], row["line_skeleton"],
                        np.asarray(row["hard_negative_mask"], dtype=np.float32)
                        / 255.0, rng)
                images.append(image)
                masks.append(mask)
                skeletons.append(skeleton)
                hard_masks.append(hard)
            gray = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            target = torch.from_numpy(
                np.stack(masks).astype(np.float32)[:, None] / 255.0
            ).to(device)
            skeleton = torch.from_numpy(
                np.stack(skeletons).astype(np.float32)[:, None] / 255.0
            ).to(device)
            hard_target = torch.from_numpy(
                np.stack(hard_masks).astype(np.float32)[:, None]).to(device)
            logits = data["ablation"].main_logits(model(gray))
            standard, skeleton_loss, neighbourhood = \
                data["corrective"].standard_loss_terms(
                    data["extension"], logits, target, skeleton,
                    data["run_args"])
            tversky = data["coverage"].recall_tversky_loss(logits, target)
            tversky_weight = data["coverage"].TVERSKY_WEIGHT \
                if config["loss"] == "recall_tversky" else 0.0
            hard_loss = masked_false_positive_loss(logits, hard_target)
            loss = standard + recipe["skeleton_weight"] * skeleton_loss \
                + tversky_weight * tversky \
                + float(config["hard_weight"]) * hard_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            last_losses = {
                "hard_negative": float(hard_loss.detach().cpu()),
                "hard_negative_weight": float(config["hard_weight"]),
                "loss": float(loss.detach().cpu()),
                "neighbourhood_monitor_only": float(neighbourhood.detach().cpu()),
                "skeleton": float(skeleton_loss.detach().cpu()),
                "standard": float(standard.detach().cpu()),
                "tversky": float(tversky.detach().cpu()),
                "tversky_weight": float(tversky_weight),
            }
            if step % VALIDATION_INTERVAL and step != steps:
                continue
            positive_probability, positive_target = data["ablation"].predict(
                model, data["validation_positive"], 12, device)
            clean_probability, _ = data["ablation"].predict(
                model, data["validation_clean"], 12, device)
            coverage = data["coverage"].coverage_operating_point(
                positive_probability, positive_target > 0.5,
                clean_probability)
            points = data["base"].choose_operating_points(
                positive_probability, positive_target, clean_probability,
                data["run_args"].max_clean_fp,
                data["base"].scaled_min_fire_pixels(
                    data["run_args"].recipe,
                    data["run_args"].min_fire_pixels))
            key = data["coverage"].coverage_key(coverage)
            if step >= start_select and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_coverage, best_points = coverage, points
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            record = {
                "best_step": best_step,
                "coverage": coverage,
                "event": "validation",
                "standard_iou": points["localization"]["iou"],
                "step": step,
            }
            trace.write(json.dumps(record, sort_keys=True) + "\n")
            trace.flush()
            print(json.dumps({
                "best_step": best_step,
                "covered_ge_50": coverage["covered_ge_50_count"],
                "median": round(coverage["median_gold_component_coverage"], 6),
                "precision": round(coverage["pixel_precision"], 6),
                "step": step, "variant": args.variant,
            }, sort_keys=True), flush=True)
    if best_state is None or best_coverage is None or best_points is None:
        raise Experiment402Error("active300 training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_positive, final_target = data["ablation"].predict(
        model, data["validation_positive"], 12, device)
    final_clean, _ = data["ablation"].predict(
        model, data["validation_clean"], 12, device)
    final_coverage = data["coverage"].coverage_operating_point(
        final_positive, final_target > 0.5, final_clean)
    if data["coverage"].coverage_key(final_coverage) \
            != data["coverage"].coverage_key(best_coverage):
        raise Experiment402Error("reloaded checkpoint coverage differs")
    torch.save({
        "architecture": {
            "base_variant": "unetpp_resnet34", "input_channels": 1,
            "variant": "active300_coverage_finetune",
        },
        "best_step": best_step,
        "coverage_operating_point": final_coverage,
        "data_arm": "prior1500_plus_active300_writer_safe",
        "negative_ratio": int(config["negative_ratio"]),
        "operating_points": best_points,
        "seed": args.seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "state_dict": model.state_dict(),
        "variant": args.variant,
    }, staging / "segmenter_seed42.pt")
    np.savez_compressed(
        staging / "validation_predictions_seed42.npz",
        clean_probabilities=final_clean.astype(np.float16),
        positive_probabilities=final_positive.astype(np.float16),
        targets=(final_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "coverage": final_coverage,
        "last_losses": last_losses,
        "operating_points": best_points,
        "partitions": {name: len(rows) for name, rows in partitions.items()},
        "source_coverage": source_coverage,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS or args.seed != 42:
        raise Experiment402Error("formal active300 arms use a known variant/seed42")
    scorer390 = load_module(
        SCORER390_PATH, "hme390_for402", SCORER390_SHA256)
    coverage = scorer390.load_module(
        scorer390.COVERAGE_PATH, "hme389_for402", scorer390.COVERAGE_SHA256)
    source = verify_source(args.source_output)
    config = VARIANTS[args.variant]
    data = prepare_data(scorer390, coverage, source, args, config)
    data["coverage"] = coverage
    if data["audit"]["validation_positive"] != LOCKED_VALIDATION_POSITIVE \
            or data["audit"]["validation_clean"] != LOCKED_VALIDATION_CLEAN \
            or data["audit"]["writer_overlap_fit_validation"] != 0:
        raise Experiment402Error("fixed validation contract changed")
    if args.validate_only:
        return {"audit": data["audit"], "variant": args.variant}

    output = args.output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment402Error(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result = train(data, source, args, config, staging)
        shutil.copy2(source["root"] / "validation_key.json",
                     staging / "validation_key.json")
        if digest_file(staging / "validation_key.json") \
                != VALIDATION_KEY_SHA256:
            raise Experiment402Error("copied validation key changed")
        summary = {
            "audit": data["audit"],
            "coverage": result["coverage"],
            "gold_components": LOCKED_GOLD_COMPONENTS,
            "protocol": {
                "active_positive_per_batch":
                    int(config["active_positive_per_batch"]),
                "active300_candidate_pixels_used_as_positive": False,
                "active300_deleted_proposals_hard_weight":
                    float(config["hard_weight"]),
                "fixed_validation_reused": True,
                "gold_masks_modified": False,
                "learning_rate": float(config["learning_rate"]),
                "m6_frozen_test_touched": False,
                "seed_policy": "fixed seed 42",
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "writer_overlap_fit_validation": 0,
            },
            "result": result,
            "schema_version": SCHEMA,
            "status": "complete",
            "variant": args.variant,
        }
        manifest = {
            "active_audit_sha256": digest_file(Path(args.active_audit)),
            "active_key_sha256": digest_file(Path(args.active_key)),
            "active_manifest_sha256": digest_file(Path(args.active_manifest)),
            "active_returned_sha256": digest_file(Path(args.active_returned)),
            "command": [str(value) for value in sys.argv],
            "data_arm": "prior1500_plus_active300_writer_safe",
            "m6_frozen_test_touched": False,
            "reference_evaluation_materialised": False,
            "schema_version": SCHEMA,
            "script_sha256": digest_file(Path(__file__).resolve()),
            "seed": args.seed,
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_manifest_sha256": source["manifest_sha256"],
            "variant": args.variant,
        }
        write_json(staging / "summary.json", summary)
        write_json(staging / "run_manifest.json", manifest)
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
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--active-key", type=Path, required=True)
    parser.add_argument("--active-returned", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    try:
        summary = run(build_parser().parse_args())
    except (Experiment402Error, RuntimeError) as exc:
        raise SystemExit(f"experiment 402 refused: {exc}") from exc
    print(json.dumps({
        "audit": summary["audit"],
        "coverage": summary.get("coverage"),
        "variant": summary["variant"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

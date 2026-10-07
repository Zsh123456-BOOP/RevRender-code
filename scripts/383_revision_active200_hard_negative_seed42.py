#!/usr/bin/env python3
"""Fine-tune the strongest revision segmenter with active-review gold.

Three create-only development arms share seed 42, the same writer-disjoint
validation split, the same strongest source checkpoint, and the same reviewed
gold rows:

``data_only``
    Replay old gold while adding the reviewed active-learning rows.
``hard_sampling``
    Prefer active rows containing model candidate pixels outside final gold.
``hard_loss``
    Use the same sampling and add a masked background loss on those pixels.

The candidate mask never becomes a target.  It is formed only from proposals
that were visibly presented in the high-false-positive review arm, then has a
dilated final-gold mask subtracted.  Thus a deleted-and-redrawn proposal is not
silently labelled negative where it overlaps the corrected gold annotation.
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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOLD300_PATH = ROOT / "scripts/380_revision_gold300_factorial_seed42.py"
GOLD300_SHA256 = "c680ebfe44598043e831f9df621d38091869d0f5fcb4a02ad0650672f58c8686"
SCHEMA = "hme-revision-active200-hard-negative-seed42-dev-v1"
VARIANTS = ("data_only", "hard_sampling", "hard_loss")

ACTIVE_RECORDS = 200
ACTIVE_REVIEWED = 195
ACTIVE_UNTOUCHED = 5
ACTIVE_KEY_SHA256 = "69d09965d0d4c6bc4d0a06ac97ec6893898ee100e2c8872352be01a8d3510b23"
ACTIVE_RETURNED_SHA256 = "a6c8206a2418e776a8061dd2a3aa897e71375f5dc2f73e982ffb84c866afe018"
ACTIVE_AUDIT_SHA256 = "73290dc11d425454fac5d06d2322ac5b03f4d93f57efdfbbe7a1ebc0014d4598"
ACTIVE_FINGERPRINT = "1342533e3090e15fc95e838b8a841e03"
ACTIVE_FULL_QA = {
    "clean_after_sanitation": 15,
    "dropped_zero_length_lines": 2,
    "flags": {"no_revision": 7},
    "mark_types": {"blot": 154, "line": 606, "overwrite": 8},
    "pinned_test_records_not_semantically_read": 0,
    "positive_after_sanitation": 168,
    "protected_image_records_not_semantically_read": 17,
    "raw_marks": 768,
    "safe_records": 183,
    "source_records": 200,
}
ACTIVE_REVIEWED_QA = {
    "clean_after_sanitation": 10,
    "positive_after_sanitation": 168,
    "reviewed_safe_records": 178,
    "untouched_safe_records_excluded": 5,
}
SOURCE_CHECKPOINT_SHA256 = (
    "1b18f3c72ac4b5411ef1289ab6775d8857b21af3d903789cb0366b9e4173a6ec"
)
SOURCE_SCHEMA = "hme-revision-gold300-factorial-seed42-dev-v1"


class Experiment383Error(RuntimeError):
    """Raised when the active-review experiment contract changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment383Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment383Error(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_gold300():
    if digest_file(GOLD300_PATH) != GOLD300_SHA256:
        raise Experiment383Error("experiment-380 dependency digest mismatch")
    specification = importlib.util.spec_from_file_location(
        "hme380_for383", GOLD300_PATH)
    if specification is None or specification.loader is None:
        raise Experiment383Error("cannot import experiment 380")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in
                Path(path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment383Error(f"invalid JSONL: {path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise Experiment383Error(f"non-object JSONL row: {path}")
    return rows


def reviewed_identity_map(key_path: Path, returned_path: Path) \
        -> tuple[dict[str, str], set[str]]:
    key_rows = sorted(read_jsonl(key_path), key=lambda row: int(row["index"]))
    returned = read_json(returned_path)
    records = returned.get("records")
    if len(key_rows) != ACTIVE_RECORDS \
            or [int(row["index"]) for row in key_rows] != list(range(ACTIVE_RECORDS)) \
            or not isinstance(records, dict) or len(records) != ACTIVE_RECORDS:
        raise Experiment383Error("active200 key or returned record count changed")
    id_to_line = {str(row["id"]): str(row["line_id"]) for row in key_rows}
    if len(id_to_line) != ACTIVE_RECORDS or set(records) != set(id_to_line):
        raise Experiment383Error("active200 returned IDs differ from key")
    reviewed = {id_to_line[record_id] for record_id, record in records.items()
                if record.get("t") is not None}
    if len(reviewed) != ACTIVE_REVIEWED:
        raise Experiment383Error(
            f"reviewed active200 rows changed: {len(reviewed)}")
    return id_to_line, reviewed


def proposal_candidates(audit_path: Path, id_to_line: Mapping[str, str]) \
        -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_jsonl(audit_path)
    if len(rows) != ACTIVE_RECORDS or len({str(row.get("id")) for row in rows}) \
            != ACTIVE_RECORDS:
        raise Experiment383Error("active200 proposal audit coverage changed")
    category_counts = Counter(str(row.get("category")) for row in rows)
    if category_counts != Counter({"high_fp": 100, "likely_miss": 50,
                                   "blind": 50}):
        raise Experiment383Error(
            f"active200 proposal categories changed: {category_counts}")
    candidates: dict[str, list[dict[str, Any]]] = {}
    weak = strong = raw = 0
    for row in rows:
        record_id = str(row["id"])
        line_id = id_to_line.get(record_id)
        if line_id is None or str(row.get("line_id")) != line_id:
            raise Experiment383Error("proposal audit identity differs from key")
        marks = row.get("raw_marks") or []
        if row["category"] == "high_fp":
            candidates[line_id] = [dict(mark) for mark in marks]
            raw += len(marks)
            weak += sum(bool(mark.get("weak")) for mark in marks)
            strong += sum(not bool(mark.get("weak")) for mark in marks)
    if (raw, weak, strong) != (535, 314, 221):
        raise Experiment383Error(
            f"active200 proposal counts changed: {(raw, weak, strong)}")
    return candidates, {
        "candidate_marks": raw,
        "high_fp_rows": len(candidates),
        "strong_candidate_marks": strong,
        "weak_candidate_marks": weak,
    }


def load_active_rows(base, prepared: dict[str, Any], args: argparse.Namespace):
    for path, expected, name in (
        (Path(args.active_key), ACTIVE_KEY_SHA256, "active key"),
        (Path(args.active_returned), ACTIVE_RETURNED_SHA256, "active returned"),
        (Path(args.active_audit), ACTIVE_AUDIT_SHA256, "active audit"),
    ):
        if digest_file(path) != expected:
            raise Experiment383Error(f"{name} digest mismatch")
    returned = read_json(Path(args.active_returned))
    if returned.get("schema_version") != base.SCHEMA_ANNOTATION \
            or returned.get("page_fingerprint") != ACTIVE_FINGERPRINT:
        raise Experiment383Error("active200 annotation envelope changed")
    id_to_line, reviewed = reviewed_identity_map(
        Path(args.active_key), Path(args.active_returned))
    candidates, proposal_qa = proposal_candidates(
        Path(args.active_audit), id_to_line)

    eval_package = base.load_module(
        base.EVAL_PACKAGE_SCRIPT, "hme_evalpkg325_for383",
        base.EVAL_PACKAGE_SHA256)
    catalogue_rows, _lookup = base.load_catalogue(prepared["segmenter"])
    protected_images = {
        str(entry["image_id"]) for entry in catalogue_rows
        if set(entry.get("data_roles") or [])
        & {"fixed_development_only", "fixed_external_test_only"}
    }
    rows, full_qa = base.read_package(
        Path(args.active_key), Path(args.active_returned), name="active200",
        expected_key_sha=ACTIVE_KEY_SHA256,
        expected_returned_sha=ACTIVE_RETURNED_SHA256,
        expected_fingerprint=ACTIVE_FINGERPRINT,
        expected_records=ACTIVE_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=eval_package.collect_test_identifiers())
    if full_qa != ACTIVE_FULL_QA:
        raise Experiment383Error(f"active200 QA changed: {full_qa}")
    rows = [row for row in rows if str(row["line_id"]) in reviewed]
    reviewed_qa = {
        "clean_after_sanitation": sum(
            not row["marks"] and row.get("flag") != "illegible" for row in rows),
        "positive_after_sanitation": sum(bool(row["marks"]) for row in rows),
        "reviewed_safe_records": len(rows),
        "untouched_safe_records_excluded":
            int(full_qa["safe_records"]) - len(rows),
    }
    if reviewed_qa != ACTIVE_REVIEWED_QA:
        raise Experiment383Error(
            f"reviewed active200 QA changed: {reviewed_qa}")
    for row in rows:
        row["hard_candidate_marks"] = candidates.get(str(row["line_id"]), [])
    return rows, {"full": full_qa, "proposal": proposal_qa,
                  "reviewed": reviewed_qa}


def verify_source(path: Path) -> dict[str, Any]:
    import torch

    path = Path(path).resolve()
    root = path.parent
    for required in ("complete.json", "summary.json", "run_manifest.json",
                     "SHA256SUMS"):
        if not (root / required).is_file():
            raise Experiment383Error(f"source is missing {required}")
    complete = read_json(root / "complete.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "run_manifest.json")
    if complete != {"schema_version": SOURCE_SCHEMA, "status": "complete"} \
            or summary.get("schema_version") != SOURCE_SCHEMA \
            or manifest.get("schema_version") != SOURCE_SCHEMA \
            or summary.get("variant") != "current_best_all300" \
            or manifest.get("variant") != "current_best_all300" \
            or manifest.get("m6_frozen_test_touched") is not False \
            or manifest.get("reference_evaluation_materialised") is not False \
            or summary.get("data", {}).get("writer_overlap_fit_validation") != 0:
        raise Experiment383Error("strongest source safety contract changed")
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in sums:
            raise Experiment383Error("source SHA256SUMS is malformed")
        sums[name] = digest
    actual = digest_file(path)
    if actual != SOURCE_CHECKPOINT_SHA256 or sums.get(path.name) != actual:
        raise Experiment383Error("strongest source checkpoint digest changed")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    architecture = checkpoint.get("architecture", {})
    if checkpoint.get("seed") != 42 \
            or checkpoint.get("data_arm") != "new1000_plus_all300" \
            or checkpoint.get("correction_set") != "all300" \
            or checkpoint.get("source_family") != "current_best" \
            or architecture.get("base_variant") != "unetpp_resnet34" \
            or architecture.get("input_channels") != 1:
        raise Experiment383Error("strongest source metadata changed")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": actual,
        "manifest_sha256": digest_file(root / "run_manifest.json"),
        "root": root,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.variant not in VARIANTS \
            or args.source_family != "current_best" \
            or args.correction_set != "all300" \
            or args.extra_fingerprint != "b279d1e8e824041a2492498486726559" \
            or args.active_fingerprint != ACTIVE_FINGERPRINT \
            or args.stage != "finetune_replay" \
            or args.mode != "dev" or args.data_arm != "new1000" \
            or args.recipe != "optimized" or args.seeds != "42" \
            or args.input_mode != "image" or args.merge_radius_factor != 0 \
            or not args.gold_confirmed:
        raise Experiment383Error("active200 base protocol changed")
    if args.optimizer_steps != 3000 or args.validation_interval != 100 \
            or not math.isclose(args.selection_warmup_fraction, 0.2) \
            or not math.isclose(args.learning_rate, 3e-5) \
            or args.corrected_positive_per_batch != 1 \
            or args.corrected_clean_interval != 20 \
            or not math.isclose(args.isolated_fp_weight, 0.5):
        raise Experiment383Error("active200 optimisation protocol changed")


def patch_prepare(gold300, base, args: argparse.Namespace) -> None:
    original_prepare = base.prepare

    def prepare(run_args: argparse.Namespace):
        prepared = original_prepare(run_args)
        gold_rows, gold_excluded, gold_qa, gold_selection = \
            gold300.load_selected_rows(base, prepared, run_args)
        active_rows, active_qa = load_active_rows(base, prepared, run_args)
        validation_writers = {
            str(row["group_uuid"]) for row in
            prepared["validation_positive"] + prepared["validation_clean"]
        }
        active_fit = [row for row in active_rows
                      if str(row["group_uuid"]) not in validation_writers]
        active_excluded = [row for row in active_rows
                           if str(row["group_uuid"]) in validation_writers]
        gold_ids = {str(row["line_id"]) for row in gold_rows}
        active_ids = {str(row["line_id"]) for row in active_fit}
        existing_ids = {
            str(row["line_id"]) for row in prepared["fit_positive"]
            + prepared["fit_clean"] + prepared["validation_positive"]
            + prepared["validation_clean"]
        }
        if gold_ids & active_ids or (gold_ids | active_ids) & existing_ids:
            raise Experiment383Error("active/gold300 rows overlap an existing split")
        gold_positive = [row for row in gold_rows if row["marks"]]
        gold_clean = [row for row in gold_rows if not row["marks"]
                      and row.get("flag") != "illegible"]
        active_positive = [row for row in active_fit if row["marks"]]
        active_clean = [row for row in active_fit if not row["marks"]
                        and row.get("flag") != "illegible"]
        if len(gold_positive) != 226 or len(gold_clean) != 4:
            raise Experiment383Error("gold300 fitting partition changed")
        if len(active_positive) + len(active_clean) != len(active_fit):
            raise Experiment383Error("active fitting rows include illegible data")

        original_gold_clean_count = int(prepared["audit"]["fit_gold_clean"])
        original_gold_clean = prepared["fit_clean"][:original_gold_clean_count]
        corrected_ids = gold_ids | active_ids
        candidates = [row for row in prepared["fit_clean_candidates"]
                      if str(row["line_id"]) not in corrected_ids]
        total_positive = len(prepared["fit_positive"]) \
            + len(gold_positive) + len(active_positive)
        total_gold_clean = len(original_gold_clean) \
            + len(gold_clean) + len(active_clean)
        wanted_clean = base.reference_catalogue_clean_count(
            total_positive, total_gold_clean, run_args.negative_ratio)
        sampled, hard_selection = base.select_hard_prefix(
            Path(run_args.hard_negative_manifest), candidates, wanted_clean,
            validation_seed=run_args.validation_seed,
            validation_fraction=run_args.validation_fraction)
        prepared["fit_positive"] = prepared["fit_positive"] \
            + gold_positive + active_positive
        prepared["fit_clean"] = original_gold_clean + gold_clean \
            + active_clean + sampled
        prepared["hard_selection"] = hard_selection
        run_args.gold300_fit_line_ids = tuple(sorted(gold_ids))
        run_args.active_fit_line_ids = tuple(sorted(active_ids))
        run_args.active_candidate_marks = {
            str(row["line_id"]): list(row.get("hard_candidate_marks") or [])
            for row in active_fit if row.get("hard_candidate_marks")
        }
        run_args.expected_partition = {
            "active_clean": len(active_clean),
            "active_positive": len(active_positive),
            "gold300_clean": len(gold_clean),
            "gold300_positive": len(gold_positive),
            "original_clean": len(sampled) + len(original_gold_clean),
            "original_positive": len(prepared["fit_positive"])
                                 - len(gold_positive) - len(active_positive),
        }
        audit = prepared["audit"]
        audit.update({
            "data_arm": "new1000_plus_all300_plus_active200",
            "fit_gold_clean": total_gold_clean,
            "fit_positive": total_positive,
            "hard_selection": hard_selection,
            "sampled_catalogue_clean": len(sampled),
            "total_fit_clean": len(prepared["fit_clean"]),
            "active200": {
                "qa": active_qa,
                "safe_reviewed_before_writer_exclusion": len(active_rows),
                "training_clean": len(active_clean),
                "training_positive": len(active_positive),
                "training_rows": len(active_fit),
                "validation_writer_rows_excluded": len(active_excluded),
            },
            "extra300": {
                **gold_selection,
                "correction_set": "all300",
                "qa_all300": gold_qa,
                "training_clean": len(gold_clean),
                "training_positive": len(gold_positive),
                "training_rows": len(gold_rows),
                "validation_writer_rows_excluded": len(gold_excluded),
            },
        })
        fit_writers = {str(row["group_uuid"]) for row in
                       prepared["fit_positive"] + prepared["fit_clean"]}
        if fit_writers & validation_writers:
            raise Experiment383Error("writer isolation failed after active200")
        audit["writer_overlap_fit_validation"] = 0
        return prepared

    base.prepare = prepare


def patch_materialise(base, args: argparse.Namespace) -> None:
    original_materialise = base.materialise

    def materialise(rows, segmenter, cropper, catalogue, images_dir, crop,
                    fallback=None):
        output, clipped = original_materialise(
            rows, segmenter, cropper, catalogue, images_dir, crop,
            fallback=fallback)
        candidate_lookup = getattr(args, "active_candidate_marks", {})
        candidate_rows = [
            {**row, "marks": candidate_lookup[str(row["line_id"])],
             "source": "active200_visible_candidates"}
            for row in rows if str(row["line_id"]) in candidate_lookup
        ]
        candidate_masks = {}
        if candidate_rows:
            rendered, _ignored = original_materialise(
                candidate_rows, segmenter, cropper, catalogue, images_dir,
                crop, fallback=fallback)
            candidate_masks = {
                str(row["line_id"]): np.asarray(row["mask"], dtype=np.uint8)
                for row in rendered
            }
        try:
            import cv2
        except ImportError as exc:
            raise Experiment383Error("OpenCV is required for gold dilation") from exc
        kernel = np.ones((5, 5), dtype=np.uint8)
        total_pixels = rows_with_pixels = 0
        for row in output:
            candidate = candidate_masks.get(str(row["line_id"]))
            if candidate is None:
                hard = np.zeros_like(row["mask"], dtype=np.uint8)
            else:
                gold = np.asarray(row["mask"] > 0, dtype=np.uint8)
                protected = cv2.dilate(gold, kernel, iterations=1) > 0
                hard = np.asarray((candidate > 0) & ~protected,
                                  dtype=np.uint8) * 255
            pixels = int((hard > 0).sum())
            row["hard_negative_mask"] = hard
            row["hard_negative_pixels"] = pixels
            total_pixels += pixels
            rows_with_pixels += int(pixels > 0)
        args.materialised_hard_negative_pixels = total_pixels
        args.materialised_hard_negative_rows = rows_with_pixels
        return output, clipped

    base.materialise = materialise


def tag_rows(fit_positive: list[dict], fit_clean: list[dict],
             args: argparse.Namespace) -> dict[str, int]:
    active_ids = set(getattr(args, "active_fit_line_ids", ()))
    gold_ids = set(getattr(args, "gold300_fit_line_ids", ()))
    for row in fit_positive + fit_clean:
        line_id = str(row["line_id"])
        row["active200"] = line_id in active_ids
        row["gold300"] = line_id in gold_ids
        if row["active200"] and row["gold300"]:
            raise Experiment383Error("materialised row has two correction tags")
    counts = {
        "active_clean": sum(row["active200"] for row in fit_clean),
        "active_positive": sum(row["active200"] for row in fit_positive),
        "gold300_clean": sum(row["gold300"] for row in fit_clean),
        "gold300_positive": sum(row["gold300"] for row in fit_positive),
        "original_clean": sum(not row["active200"] and not row["gold300"]
                              for row in fit_clean),
        "original_positive": sum(not row["active200"] and not row["gold300"]
                                 for row in fit_positive),
    }
    if counts != getattr(args, "expected_partition", None):
        raise Experiment383Error(f"materialised partition changed: {counts}")
    if len(fit_clean) != 3 * len(fit_positive):
        raise Experiment383Error("negative ratio changed after materialisation")
    return counts


def training_batch(base, rng: random.Random, step: int,
                   args: argparse.Namespace, cycles: dict[str, Any]):
    active_cycle = cycles["active_positive"]
    if args.variant != "data_only" and step % 2 == 1 \
            and "active_hard_positive" in cycles:
        active_cycle = cycles["active_hard_positive"]
    rows = active_cycle.take(1) + cycles["replay_positive"].take(1)
    if step % args.corrected_clean_interval == 0 \
            and "active_clean" in cycles:
        clean_cycle = cycles["active_clean"]
        if args.variant != "data_only" \
                and "active_hard_clean" in cycles:
            clean_cycle = cycles["active_hard_clean"]
        rows += clean_cycle.take(1) + cycles["replay_clean"].take_distinct(5)
    else:
        rows += cycles["replay_clean"].take_distinct(6)
    identities = [str(row["line_id"]) for row in rows]
    if len(rows) != 8 or len(identities) != len(set(identities)):
        raise Experiment383Error("active200 batch size or identity changed")
    rng.shuffle(rows)
    return rows


def masked_false_positive_loss(logits, hard_negative_mask):
    import torch
    from torch.nn import functional as functional

    mask = hard_negative_mask >= 0.5
    if not bool(mask.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return functional.softplus(logits[mask]).mean()


def predict(ablation, model, rows: Sequence[dict[str, Any]], batch_size: int,
            device):
    return ablation.predict(model, rows, batch_size, device)


def train_once(gold300, corrective, ablation, base, extension,
               fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path,
               source: dict[str, Any]):
    import torch

    if seed != 42 or not torch.cuda.is_available():
        raise Experiment383Error("formal active200 training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    counts = tag_rows(fit_positive, fit_clean, args)
    model = ablation.make_unetpp_resnet34(args.resnet34_weights).to(device)
    model.load_state_dict(source["checkpoint"]["state_dict"], strict=True)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(seed)
    partitions = {
        "active_positive": [row for row in fit_positive if row["active200"]],
        "active_hard_positive": [row for row in fit_positive
                                 if row["active200"]
                                 and row.get("hard_negative_pixels", 0) > 0],
        "replay_positive": [row for row in fit_positive
                            if not row["active200"]],
        "active_clean": [row for row in fit_clean if row["active200"]],
        "active_hard_clean": [row for row in fit_clean
                              if row["active200"]
                              and row.get("hard_negative_pixels", 0) > 0],
        "replay_clean": [row for row in fit_clean if not row["active200"]],
    }
    if not partitions["active_positive"] or not partitions["replay_positive"] \
            or len(partitions["replay_clean"]) < 6:
        raise Experiment383Error("active200 training partition is incomplete")
    cycles = {name: base.ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    recipe = base.RECIPES[args.recipe]
    hard_weight = 0.5 if args.variant == "hard_loss" else 0.0
    best_key = best_state = best_points = None
    best_step = -1
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rows = training_batch(base, rng, step, args, cycles)
        images, masks, skeletons, hard_masks = [], [], [], []
        for row in rows:
            image, mask, skeleton, hard_mask = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"],
                np.asarray(row["hard_negative_mask"], dtype=np.float32) / 255.0,
                rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
            hard_masks.append(hard_mask)
        gray = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
        ).to(device)
        target = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
        hard_target = torch.from_numpy(
            np.stack(hard_masks).astype(np.float32)[:, None]).to(device)
        logits = ablation.main_logits(model(gray))
        standard, skeleton_loss, neighbourhood = corrective.standard_loss_terms(
            extension, logits, target, skeleton, args)
        hard_loss = masked_false_positive_loss(logits, hard_target)
        loss = standard + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood + hard_weight * hard_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "hard_negative": float(hard_loss.detach().cpu()),
            "hard_negative_weight": hard_weight,
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard.detach().cpu()),
        }
        if step % args.validation_interval != 0 \
                and step != args.optimizer_steps:
            continue
        positive_probability, positive_target = predict(
            ablation, model, validation_positive, args.batch_size, device)
        clean_probability, _ = predict(
            ablation, model, validation_clean, args.batch_size, device)
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
            "best_step": best_step,
            "event": "validation",
            "seed": seed,
            "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
            "variant": args.variant,
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Experiment383Error("active200 training produced no checkpoint")
    model.load_state_dict(best_state)
    positive_probability, positive_target = predict(
        ablation, model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        ablation, model, validation_clean, args.batch_size, device)
    points = base.choose_operating_points(
        positive_probability, positive_target, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    torch.save({
        "architecture": {
            "base_variant": "unetpp_resnet34",
            "input_channels": 1,
            "variant": "active200_hard_negative_replay",
        },
        "best_step": best_step,
        "data_arm": "new1000_plus_all300_plus_active200",
        "operating_points": points,
        "seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "state_dict": model.state_dict(),
        "variant": args.variant,
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=positive_probability.astype(np.float16),
        targets=(positive_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "last_losses": last_losses,
        "partition": counts,
        "sampling": {
            name: len(rows) for name, rows in partitions.items()
        },
        "seed": seed,
        "validation": points,
        "variant": args.variant,
    }, {"clean": clean_probability, "positive": positive_probability}


def finalise(base, staging: Path, args: argparse.Namespace,
             source: dict[str, Any]) -> dict[str, Any]:
    summary_path = staging / "summary.json"
    manifest_path = staging / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    data = summary.get("data", {})
    if data.get("writer_overlap_fit_validation") != 0 \
            or data.get("active200", {}).get("training_rows", 0) <= 0 \
            or data.get("total_fit_clean") != 3 * data.get("fit_positive", -1):
        raise Experiment383Error("active200 final data contract changed")
    summary["schema_version"] = SCHEMA
    summary["variant"] = args.variant
    summary["protocol"].update({
        "active_untouched_records_excluded": ACTIVE_UNTOUCHED,
        "candidate_pixels_never_used_as_positive_target": True,
        "gold_dilation_before_negative_mask_pixels": 2,
        "hard_negative_loss_weight": 0.5 if args.variant == "hard_loss" else 0.0,
        "seed_policy": "single fixed seed 42",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
        "variant": args.variant,
    })
    manifest.update({
        "active_audit_sha256": digest_file(Path(args.active_audit)),
        "active_key_sha256": digest_file(Path(args.active_key)),
        "active_returned_sha256": digest_file(Path(args.active_returned)),
        "data_arm": "new1000_plus_all300_plus_active200",
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "seed": 42,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "variant": args.variant,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(gold300, corrective, base, runner, full):
    parser = gold300.build_parser(corrective, base, runner, full)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--active-key", required=True)
    parser.add_argument("--active-returned", required=True)
    parser.add_argument("--active-audit", required=True)
    parser.add_argument("--active-fingerprint", required=True)
    return parser


def run(args, gold300, corrective, ablation, base, extension, runner, full):
    validate_args(args)
    source = verify_source(Path(args.source_checkpoint))
    full.configure_base(base)
    patch_prepare(gold300, base, args)
    patch_materialise(base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(gold300, corrective, ablation, b, e, fp, fc, vp, vc, a,
                   seed, staging, source)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment383Error(f"create-only output exists: {output}")
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
    gold300 = load_gold300()
    wavelet = gold300.load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for383", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for383", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for383", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for383", ablation.FULL1000_SHA256)
    args = build_parser(gold300, corrective, base, runner, full).parse_args()
    try:
        summary = run(
            args, gold300, corrective, ablation, base, extension, runner, full)
    except (Experiment383Error, gold300.Experiment380Error,
            wavelet.Experiment376Error, corrective.Experiment374Error,
            ablation.Ablation371Error, base.LocalizerError,
            runner.MechanismError, full.Full1000Error) as exc:
        raise SystemExit(f"experiment 383 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"],
        "ensemble": summary.get("ensemble", {}),
        "variant": args.variant,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

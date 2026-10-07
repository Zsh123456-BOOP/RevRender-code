#!/usr/bin/env python3
"""Run a 2x2 corrective-learning study over source checkpoint and gold set.

The source factor is original-1000 UNet++ versus the existing 1000+safe80
corrective checkpoint.  The supervision factor is the newly reviewed suffix
105:300 versus all 300 reviewed rows.  Every arm uses identical original-data
and hard-clean replay, seed 42, and the unchanged writer-disjoint dev split.
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
WAVELET_PATH = ROOT / "scripts/376_revision_wavelet_ablation_seed42.py"
WAVELET_SHA256 = "d28f3d3441f9fba8564b83e5b5fd5393ca154b52fff4160a4f77a8d5cd9cc21b"
SCHEMA = "hme-revision-gold300-factorial-seed42-dev-v1"
SOURCE_FAMILIES = ("original1000", "current_best")
CORRECTION_SETS = ("new195", "all300")
EXPECTED_RECORDS = 300
OLD_PREFIX_RECORDS = 105
EXPECTED_KEY_SHA256 = (
    "411052032fa831c8f40ce4c6a44dad4b3fcad9d847a2954c22b287da63ac545b"
)
EXPECTED_RETURNED_SHA256 = (
    "f814558c27b7ed15472736ddde4dced6013667d5517e218279cc0f6f57d5f963"
)
EXPECTED_FINGERPRINT = "b279d1e8e824041a2492498486726559"
EXPECTED_QA = {
    "clean_after_sanitation": 5,
    "dropped_zero_length_lines": 0,
    "flags": {"no_revision": 5},
    "mark_types": {"blot": 203, "line": 774, "overwrite": 4},
    "pinned_test_records_not_semantically_read": 0,
    "positive_after_sanitation": 254,
    "protected_image_records_not_semantically_read": 41,
    "raw_marks": 981,
    "safe_records": 259,
    "source_records": 300,
}
EXPECTED_SELECTION = {
    "new195": {
        "requested_records": 195,
        "safe_selected_before_writer_exclusion": 170,
        "training_clean": 1,
        "training_positive": 149,
        "training_rows": 150,
        "validation_writer_rows_excluded": 20,
    },
    "all300": {
        "requested_records": 300,
        "safe_selected_before_writer_exclusion": 259,
        "training_clean": 4,
        "training_positive": 226,
        "training_rows": 230,
        "validation_writer_rows_excluded": 29,
    },
}


class Experiment380Error(RuntimeError):
    """Raised when the locked gold-300 factorial protocol changes."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment380Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment380Error(f"expected JSON object: {path}")
    return value


def load_wavelet():
    if digest_file(WAVELET_PATH) != WAVELET_SHA256:
        raise Experiment380Error("experiment-376 dependency digest mismatch")
    specification = importlib.util.spec_from_file_location(
        "hme376_for380", WAVELET_PATH)
    if specification is None or specification.loader is None:
        raise Experiment380Error("cannot import experiment 376")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_key_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Experiment380Error(f"invalid key JSONL: {path}") from exc
    rows.sort(key=lambda row: int(row["index"]))
    selected = rows[:count]
    if len(selected) != count \
            or [int(row["index"]) for row in selected] != list(range(count)):
        raise Experiment380Error("gold300 key prefix changed")
    if len({str(row["id"]) for row in selected}) != count \
            or len({str(row["line_id"]) for row in selected}) != count:
        raise Experiment380Error("gold300 key prefix identities are not unique")
    return selected


def selected_line_ids(key_rows: Sequence[dict[str, Any]],
                      correction_set: str) -> set[str]:
    if len(key_rows) != EXPECTED_RECORDS:
        raise Experiment380Error("gold300 key count changed")
    start = OLD_PREFIX_RECORDS if correction_set == "new195" else 0
    selected = {str(row["line_id"]) for row in key_rows[start:]}
    expected = EXPECTED_RECORDS - start
    if len(selected) != expected:
        raise Experiment380Error("selected correction identities changed")
    return selected


def validate_args(args: argparse.Namespace) -> None:
    if args.source_family not in SOURCE_FAMILIES \
            or args.correction_set not in CORRECTION_SETS \
            or args.stage != "finetune_replay" \
            or args.mode != "dev" or args.data_arm != "new1000" \
            or args.recipe != "optimized" or args.seeds != "42" \
            or args.input_mode != "image" or args.merge_radius_factor != 0 \
            or not args.gold_confirmed \
            or args.extra_fingerprint != EXPECTED_FINGERPRINT \
            or not math.isclose(args.isolated_fp_weight, 0.5):
        raise Experiment380Error("gold300 factorial base protocol changed")
    if args.optimizer_steps != 3000 or args.validation_interval != 100 \
            or not math.isclose(args.selection_warmup_fraction, 0.2) \
            or not math.isclose(args.learning_rate, 3e-5) \
            or args.corrected_positive_per_batch != 1 \
            or args.corrected_clean_interval != 20:
        raise Experiment380Error("gold300 optimisation or replay protocol changed")


def verify_source(wavelet, corrective, args: argparse.Namespace) \
        -> dict[str, Any]:
    if args.source_family == "original1000":
        source = corrective.verify_source(args.source_checkpoint)
        expected_arm = "new1000"
    else:
        source = wavelet.verify_source(args.source_checkpoint)
        expected_arm = "new1000_plus_safe80"
    if source["checkpoint"].get("data_arm") != expected_arm:
        raise Experiment380Error("source checkpoint family changed")
    source["family"] = args.source_family
    return source


def load_selected_rows(base, prepared: dict[str, Any],
                       args: argparse.Namespace):
    eval_package = base.load_module(
        base.EVAL_PACKAGE_SCRIPT, "hme_evalpkg325_for380",
        base.EVAL_PACKAGE_SHA256)
    catalogue_rows, _lookup = base.load_catalogue(prepared["segmenter"])
    protected_images = {
        str(entry["image_id"]) for entry in catalogue_rows
        if set(entry.get("data_roles") or [])
        & {"fixed_development_only", "fixed_external_test_only"}
    }
    key_rows = load_key_prefix(Path(args.extra_key), EXPECTED_RECORDS)
    returned = read_json(Path(args.extra_returned))
    if digest_file(Path(args.extra_key)) != EXPECTED_KEY_SHA256 \
            or digest_file(Path(args.extra_returned)) != EXPECTED_RETURNED_SHA256:
        raise Experiment380Error("gold300 input digest changed")
    if returned.get("schema_version") != base.SCHEMA_ANNOTATION \
            or returned.get("page_fingerprint") != EXPECTED_FINGERPRINT \
            or len(returned.get("records") or {}) != EXPECTED_RECORDS:
        raise Experiment380Error("gold300 returned-file envelope changed")
    expected_ids = {str(row["id"]) for row in key_rows}
    if set(returned["records"]) != expected_ids:
        raise Experiment380Error("gold300 returned IDs differ from key prefix")
    rows, qa = base.read_package(
        Path(args.extra_key), Path(args.extra_returned), name="extra300",
        expected_key_sha=EXPECTED_KEY_SHA256,
        expected_returned_sha=EXPECTED_RETURNED_SHA256,
        expected_fingerprint=EXPECTED_FINGERPRINT,
        expected_records=EXPECTED_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=eval_package.collect_test_identifiers())
    if qa != EXPECTED_QA:
        raise Experiment380Error(f"gold300 QA changed: {qa}")
    wanted = selected_line_ids(key_rows, args.correction_set)
    selected = [row for row in rows if str(row["line_id"]) in wanted]
    validation_writers = {
        str(row["group_uuid"]) for row in
        prepared["validation_positive"] + prepared["validation_clean"]
    }
    fitting = [row for row in selected
               if str(row["group_uuid"]) not in validation_writers]
    excluded = [row for row in selected
                if str(row["group_uuid"]) in validation_writers]
    return fitting, excluded, qa, {
        "authenticated_prefix_records": EXPECTED_RECORDS,
        "requested_records": len(wanted),
        "safe_selected_before_writer_exclusion": len(selected),
    }


def patch_prepare(base, args: argparse.Namespace) -> None:
    original_prepare = base.prepare

    def prepare(run_args: argparse.Namespace):
        prepared = original_prepare(run_args)
        fitting, excluded, qa, selection = load_selected_rows(
            base, prepared, run_args)
        corrected_ids = {str(row["line_id"]) for row in fitting}
        if not corrected_ids:
            raise Experiment380Error("gold300 arm has no safe fitting rows")
        existing_ids = {
            str(row["line_id"]) for row in prepared["fit_positive"]
            + prepared["fit_clean"] + prepared["validation_positive"]
            + prepared["validation_clean"]
        }
        if corrected_ids & existing_ids:
            raise Experiment380Error("gold300 corrections overlap existing split")
        extra_positive = [row for row in fitting if row["marks"]]
        extra_clean = [row for row in fitting if not row["marks"]
                       and row.get("flag") != "illegible"]
        if len(extra_positive) + len(extra_clean) != len(fitting):
            raise Experiment380Error("gold300 selected rows include illegible data")
        observed_selection = {
            "requested_records": selection["requested_records"],
            "safe_selected_before_writer_exclusion":
                selection["safe_selected_before_writer_exclusion"],
            "training_clean": len(extra_clean),
            "training_positive": len(extra_positive),
            "training_rows": len(fitting),
            "validation_writer_rows_excluded": len(excluded),
        }
        if observed_selection != EXPECTED_SELECTION[run_args.correction_set]:
            raise Experiment380Error(
                f"gold300 selection changed: {observed_selection}")
        existing_gold_clean_count = int(prepared["audit"]["fit_gold_clean"])
        existing_gold_clean = prepared["fit_clean"][:existing_gold_clean_count]
        candidates = [row for row in prepared["fit_clean_candidates"]
                      if str(row["line_id"]) not in corrected_ids]
        total_positive = len(prepared["fit_positive"]) + len(extra_positive)
        total_gold_clean = len(existing_gold_clean) + len(extra_clean)
        wanted_catalogue_clean = base.reference_catalogue_clean_count(
            total_positive, total_gold_clean, run_args.negative_ratio)
        sampled, hard_selection = base.select_hard_prefix(
            Path(run_args.hard_negative_manifest), candidates,
            wanted_catalogue_clean,
            validation_seed=run_args.validation_seed,
            validation_fraction=run_args.validation_fraction)
        prepared["fit_positive"] = prepared["fit_positive"] + extra_positive
        prepared["fit_clean"] = existing_gold_clean + extra_clean + sampled
        prepared["hard_selection"] = hard_selection
        run_args.corrected_fit_line_ids = tuple(sorted(corrected_ids))
        run_args.corrected_expected_positive = len(extra_positive)
        run_args.corrected_expected_clean = len(extra_clean)
        audit = prepared["audit"]
        audit["data_arm"] = f"new1000_plus_{run_args.correction_set}"
        audit["fit_gold_clean"] = total_gold_clean
        audit["fit_positive"] = total_positive
        audit["hard_selection"] = hard_selection
        audit["sampled_catalogue_clean"] = len(sampled)
        audit["total_fit_clean"] = len(prepared["fit_clean"])
        audit["extra300"] = {
            **selection,
            "correction_set": run_args.correction_set,
            "qa_all300": qa,
            "training_clean": len(extra_clean),
            "training_positive": len(extra_positive),
            "training_rows": len(fitting),
            "validation_writer_rows_excluded": len(excluded),
        }
        fit_writers = {str(row["group_uuid"]) for row in
                       prepared["fit_positive"] + prepared["fit_clean"]}
        validation_writers = {str(row["group_uuid"]) for row in
                              prepared["validation_positive"]
                              + prepared["validation_clean"]}
        if fit_writers & validation_writers:
            raise Experiment380Error("writer isolation failed after gold300 addition")
        audit["writer_overlap_fit_validation"] = 0
        return prepared

    base.prepare = prepare


def tag_materialised_rows(fit_positive: list[dict], fit_clean: list[dict],
                          args: argparse.Namespace) -> dict[str, int]:
    corrected_ids = set(getattr(args, "corrected_fit_line_ids", ()))
    if not corrected_ids:
        raise Experiment380Error("materialised correction identity set is absent")
    for row in fit_positive + fit_clean:
        row["corrected300"] = str(row["line_id"]) in corrected_ids
    counts = {
        "corrected_positive": sum(
            bool(row["corrected300"]) for row in fit_positive),
        "original_positive": sum(
            not row["corrected300"] for row in fit_positive),
        "corrected_clean": sum(bool(row["corrected300"]) for row in fit_clean),
        "original_clean": sum(not row["corrected300"] for row in fit_clean),
    }
    if counts["corrected_positive"] != args.corrected_expected_positive \
            or counts["corrected_clean"] != args.corrected_expected_clean \
            or counts["original_positive"] != 718 \
            or counts["original_clean"] + counts["corrected_clean"] \
            != 3 * (counts["original_positive"]
                    + counts["corrected_positive"]):
        raise Experiment380Error(f"materialised partition changed: {counts}")
    return counts


def correction_batch(base, rng: random.Random, step: int,
                     args: argparse.Namespace, cycles: dict[str, Any]):
    positive_count = int(round(args.batch_size / (1.0 + args.negative_ratio)))
    clean_count = args.batch_size - positive_count
    if positive_count != 2 or clean_count != 6:
        raise Experiment380Error("gold300 batch requires 2 positive + 6 clean")
    rows = cycles["corrected_positive"].take(1) \
        + cycles["original_positive"].take(1)
    if args.corrected_expected_clean \
            and step % args.corrected_clean_interval == 0:
        rows += cycles["corrected_clean"].take(1)
        rows += cycles["original_clean"].take_distinct(clean_count - 1)
    else:
        rows += cycles["original_clean"].take_distinct(clean_count)
    identities = [str(row["line_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise Experiment380Error("gold300 batch contains duplicate rows")
    rng.shuffle(rows)
    return rows


def predict(ablation, model, rows: Sequence[dict[str, Any]],
            batch_size: int, device):
    return ablation.predict(model, rows, batch_size, device)


def train_once(corrective, ablation, base, extension,
               fit_positive: list[dict], fit_clean: list[dict],
               validation_positive: list[dict], validation_clean: list[dict],
               args: argparse.Namespace, seed: int, staging: Path,
               source: dict[str, Any]):
    import torch

    if seed != 42 or not torch.cuda.is_available():
        raise Experiment380Error("formal gold300 training requires CUDA seed42")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    counts = tag_materialised_rows(fit_positive, fit_clean, args)
    model = ablation.make_unetpp_resnet34(args.resnet34_weights).to(device)
    model.load_state_dict(source["checkpoint"]["state_dict"], strict=True)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(seed)
    partitions = {
        "corrected_positive": [row for row in fit_positive
                               if row["corrected300"]],
        "original_positive": [row for row in fit_positive
                              if not row["corrected300"]],
        "corrected_clean": [row for row in fit_clean if row["corrected300"]],
        "original_clean": [row for row in fit_clean if not row["corrected300"]],
    }
    if {name: len(rows) for name, rows in partitions.items()} != counts:
        raise Experiment380Error("gold300 training partition changed")
    cycles = {name: base.ShuffledCycle(rows, rng)
              for name, rows in partitions.items() if rows}
    recipe = base.RECIPES[args.recipe]
    best_key = best_state = best_points = None
    best_step = -1
    start_select = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    last_losses: dict[str, float] = {}

    for step in range(1, args.optimizer_steps + 1):
        model.train()
        rows = correction_batch(base, rng, step, args, cycles)
        images, masks, skeletons = [], [], []
        for row in rows:
            image, mask, skeleton, _ = extension.paired_affine_with_aux(
                row["image"], row["mask"], row["line_skeleton"], None, rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        gray = torch.from_numpy(
            np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
        ).to(device)
        target = torch.from_numpy(
            np.stack(masks).astype(np.float32)[:, None] / 255.0).to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32)[:, None] / 255.0).to(device)
        logits = ablation.main_logits(model(gray))
        standard, skeleton_loss, neighbourhood = corrective.standard_loss_terms(
            extension, logits, target, skeleton, args)
        loss = standard + recipe["skeleton_weight"] * skeleton_loss \
            + args.isolated_fp_weight * neighbourhood
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        last_losses = {
            "loss": float(loss.detach().cpu()),
            "neighbourhood": float(neighbourhood.detach().cpu()),
            "skeleton": float(skeleton_loss.detach().cpu()),
            "standard": float(standard.detach().cpu()),
        }
        if step % args.validation_interval != 0 and step != args.optimizer_steps:
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
            "best_step": best_step, "correction_set": args.correction_set,
            "event": "validation", "seed": seed,
            "source_family": args.source_family, "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 6),
        }, sort_keys=True), flush=True)

    if best_state is None or best_points is None:
        raise Experiment380Error("gold300 training produced no checkpoint")
    model.load_state_dict(best_state)
    positive_probability, positive_target = predict(
        ablation, model, validation_positive, args.batch_size, device)
    clean_probability, _ = predict(
        ablation, model, validation_clean, args.batch_size, device)
    points = base.choose_operating_points(
        positive_probability, positive_target, clean_probability,
        args.max_clean_fp,
        base.scaled_min_fire_pixels(args.recipe, args.min_fire_pixels))
    data_arm = f"new1000_plus_{args.correction_set}"
    torch.save({
        "architecture": {
            "base_variant": "unetpp_resnet34", "input_channels": 1,
            "variant": "gold300_corrective_replay"},
        "best_step": best_step,
        "correction_set": args.correction_set,
        "data_arm": data_arm,
        "operating_points": points,
        "seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_family": args.source_family,
        "state_dict": model.state_dict(),
    }, staging / f"segmenter_seed{seed}.pt")
    np.savez_compressed(
        staging / f"validation_predictions_seed{seed}.npz",
        clean_probabilities=clean_probability.astype(np.float16),
        positive_probabilities=positive_probability.astype(np.float16),
        targets=(positive_target * 255).astype(np.uint8))
    return {
        "best_step": best_step,
        "correction_set": args.correction_set,
        "last_losses": last_losses,
        "partition": counts,
        "seed": seed,
        "source_family": args.source_family,
        "validation": points,
        "variant": f"{args.source_family}_{args.correction_set}",
    }, {"clean": clean_probability, "positive": positive_probability}


def finalise(base, staging: Path, args: argparse.Namespace,
             source: dict[str, Any]) -> dict[str, Any]:
    summary_path = staging / "summary.json"
    manifest_path = staging / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    data = summary.get("data", {})
    if data.get("writer_overlap_fit_validation") != 0 \
            or data.get("fit_positive", 0) <= 718 \
            or data.get("total_fit_clean") != 3 * data.get("fit_positive", -1):
        raise Experiment380Error("gold300 final data contract changed")
    variant = f"{args.source_family}_{args.correction_set}"
    summary["schema_version"] = SCHEMA
    summary["variant"] = variant
    summary["protocol"].update({
        "correction_set": args.correction_set,
        "corrected_clean_interval": args.corrected_clean_interval,
        "corrected_positive_per_batch": args.corrected_positive_per_batch,
        "seed_policy": "single fixed seed 42",
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_family": args.source_family,
        "training_runner_sha256": digest_file(Path(__file__).resolve()),
    })
    manifest.update({
        "correction_set": args.correction_set,
        "data_arm": f"new1000_plus_{args.correction_set}",
        "extra_key_sha256": digest_file(Path(args.extra_key)),
        "extra_returned_sha256": digest_file(Path(args.extra_returned)),
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "seed": 42,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_family": args.source_family,
        "source_manifest_sha256": source["manifest_sha256"],
        "variant": variant,
    })
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(staging / "complete.json", {
        "schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(staging)
    return summary


def build_parser(corrective, base, runner, full) -> argparse.ArgumentParser:
    parser = corrective.build_parser(base, runner, full)
    parser.description = __doc__.splitlines()[0]
    parser.add_argument("--source-family", required=True,
                        choices=SOURCE_FAMILIES)
    parser.add_argument("--correction-set", required=True,
                        choices=CORRECTION_SETS)
    parser.add_argument("--extra-fingerprint", required=True)
    return parser


def run(args, wavelet, corrective, ablation, base, extension, runner, full):
    validate_args(args)
    source = verify_source(wavelet, corrective, args)
    full.configure_base(base)
    patch_prepare(base, args)
    runner.train_once = lambda b, e, fp, fc, vp, vc, a, seed, staging: \
        train_once(corrective, ablation, b, e, fp, fc, vp, vc, a, seed,
                   staging, source)
    if args.validate_only:
        return runner.run(args, base, extension)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Experiment380Error(f"create-only output exists: {output}")
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
    wavelet = load_wavelet()
    corrective = wavelet.load_corrective()
    ablation = corrective.load_ablation()
    base = ablation.load_module(
        ablation.BASE_PATH, "hme354_for380", ablation.BASE_SHA256)
    extension = ablation.load_module(
        ablation.EXTENSION_PATH, "hme357_for380", ablation.EXTENSION_SHA256)
    runner = ablation.load_module(
        ablation.RUNNER_PATH, "hme358_for380", ablation.RUNNER_SHA256)
    full = ablation.load_module(
        ablation.FULL1000_PATH, "hme365_for380", ablation.FULL1000_SHA256)
    args = build_parser(corrective, base, runner, full).parse_args()
    try:
        summary = run(
            args, wavelet, corrective, ablation, base, extension, runner, full)
    except (Experiment380Error, wavelet.Experiment376Error,
            corrective.Experiment374Error, ablation.Ablation371Error,
            base.LocalizerError, runner.MechanismError,
            full.Full1000Error) as exc:
        raise SystemExit(f"experiment 380 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"], "ensemble": summary.get("ensemble", {}),
        "variant": f"{args.source_family}_{args.correction_set}",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

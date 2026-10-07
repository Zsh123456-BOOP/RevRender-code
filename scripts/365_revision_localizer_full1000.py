#!/usr/bin/env python3
"""Train the revision localiser on the completed 1000-line gold package.

This development-only runner adapts the pinned experiment-354/358 protocol to
the completed 1000-record assisted annotation package.  It compares two data
arms on one writer-disjoint validation split:

``new1000``
    Safe rows from the completed 1000-record package only.

``new1000_plus_old300``
    The same rows plus safe fitting-writer rows from the earliest 300-record
    package.

The previous three-seed hard-negative ranking was produced without reference,
held or M6 data.  Adding gold rows can only remove candidates from that clean
pool, so this runner validates the entire pinned ranking and takes its exact
intersection with the new safe fitting-writer clean pool.  No score is fitted
or re-ranked on the new validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "354_revision_localizer_optimized.py"
EXTENSION_PATH = ROOT / "scripts" / "357_revision_localizer_extensions.py"
RUNNER_PATH = ROOT / "scripts" / "358_revision_localizer_mechanism.py"

BASE_SHA256 = "73e3582b59186f97899571f832e01f983034fed4bba2269b2feb5d57fa4e2780"
EXTENSION_SHA256 = "10e003b28c4f9c6b1de45a22c27a1051351c95150ecbedd1d25ea8110b5d91ce"
RUNNER_SHA256 = "958b27a5230216c6a20d08f095de3c73b39adf5148dec9bcb562492b24e507d0"

SCHEMA = "hme-revision-localizer-full1000-dev-v1"
NEW_RETURNED_SHA256 = (
    "6f0054fb7c2be585b586a8d4fd0fc13322923c4980e29f2ef359219f8e6a15b3"
)
NEW_RECORDS = 1000
NEW_SAFE_QA = {
    "source_records": 1000,
    "safe_records": 862,
    "pinned_test_records_not_semantically_read": 0,
    "protected_image_records_not_semantically_read": 138,
    "positive_after_sanitation": 845,
    "clean_after_sanitation": 17,
    "raw_marks": 4097,
    "dropped_zero_length_lines": 3,
    "mark_types": {"blot": 532, "line": 3544, "overwrite": 21},
    "flags": {"no_revision": 11},
}
OLD_SAFE_QA = {
    "source_records": 300,
    "safe_records": 298,
    "pinned_test_records_not_semantically_read": 2,
    "protected_image_records_not_semantically_read": 0,
    "positive_after_sanitation": 285,
    "clean_after_sanitation": 12,
    "raw_marks": 620,
    "dropped_zero_length_lines": 0,
    "mark_types": {"blot": 101, "line": 512, "overwrite": 7},
    "flags": {"illegible": 1, "no_revision": 12},
}
EXPECTED_WRITER_OVERLAP = 148

LEGACY_HARD_MANIFEST_SHA256 = (
    "eb9b114017f51ef90020da2b2d82f3e321b35754cb8b1c45fa3edf678f7a7da0"
)
LEGACY_HARD_RANKING_SHA256 = (
    "c37b02696ce9085b664b5061fdcdd12861d45286617c8ecafa7e8a1af48a13b8"
)
LEGACY_HARD_POOL_COUNT = 6437
EXTERNAL_TO_INTERNAL_ARM = {
    "new1000": "new609",
    "new1000_plus_old300": "combined",
}


class Full1000Error(RuntimeError):
    """Raised when provenance or a fail-closed experiment guard is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str, expected_sha256: str):
    actual = digest_file(path)
    if actual != expected_sha256:
        raise Full1000Error(
            f"dependency digest mismatch for {path.name}: {actual}"
        )
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Full1000Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _validate_qa(actual: dict[str, Any], expected: dict[str, Any], name: str) -> None:
    if actual != expected:
        raise Full1000Error(f"{name} counts changed: {actual} != {expected}")


def load_full_gold(base, args: argparse.Namespace, protected_images: set[str],
                   pinned_test_identifiers: set[str]):
    if not args.gold_confirmed:
        raise Full1000Error(
            "returned annotations require explicit --gold-confirmed"
        )
    new_rows, new_qa = base.read_package(
        Path(args.new_key), Path(args.new_returned), name="new1000",
        expected_key_sha=base.NEW_KEY_SHA256,
        expected_returned_sha=NEW_RETURNED_SHA256,
        expected_fingerprint=base.NEW_FINGERPRINT,
        expected_records=NEW_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=pinned_test_identifiers,
    )
    old_rows, old_qa = base.read_package(
        Path(args.old_key), Path(args.old_returned), name="old300",
        expected_key_sha=base.OLD_KEY_SHA256,
        expected_returned_sha=base.OLD_RETURNED_SHA256,
        expected_fingerprint=base.OLD_FINGERPRINT,
        expected_records=base.OLD_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=pinned_test_identifiers,
    )
    _validate_qa(new_qa, NEW_SAFE_QA, "new1000")
    _validate_qa(old_qa, OLD_SAFE_QA, "old300")
    old_usable = [row for row in old_rows if row["flag"] != "illegible"]
    if {row["line_id"] for row in new_rows} & {
            row["line_id"] for row in old_usable}:
        raise Full1000Error("new1000 and old300 overlap by line id")
    writer_overlap = len(
        {row["group_uuid"] for row in new_rows}
        & {row["group_uuid"] for row in old_usable}
    )
    if writer_overlap != EXPECTED_WRITER_OVERLAP:
        raise Full1000Error(
            f"new/old writer overlap changed: {writer_overlap} "
            f"!= {EXPECTED_WRITER_OVERLAP}"
        )
    # ``new609`` is retained only as the private compatibility key expected by
    # experiment 354.  It is relabelled before any output is finalised.
    return new_rows, old_usable, {
        "all_returned_records_confirmed_gold": True,
        "new609": new_qa,
        "old300": old_qa,
        "old_illegible_excluded": 1,
        "writer_overlap": writer_overlap,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full1000Error(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Full1000Error(f"JSON input must be an object: {path}")
    return value


def select_legacy_hard_prefix(base, path: Path, clean_pool: list[dict],
                              count: int, *, validation_seed: int,
                              validation_fraction: float):
    """Take the new safe-pool intersection of the pinned legacy ranking."""
    path = Path(path).resolve()
    if digest_file(path) != LEGACY_HARD_MANIFEST_SHA256:
        raise Full1000Error("legacy hard-negative manifest digest mismatch")
    manifest = _read_json(path)
    expected_safety = {
        "image_level_protected_closure_applied": True,
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "source_data_arm": "new609",
        "source_mode": "dev",
        "source_recipe": "highres",
    }
    if any(manifest.get(key) != value for key, value in expected_safety.items()):
        raise Full1000Error("legacy hard-negative manifest safety contract changed")
    if manifest.get("pool_count") != LEGACY_HARD_POOL_COUNT:
        raise Full1000Error("legacy hard-negative pool count changed")
    if manifest.get("validation_seed") != validation_seed \
            or not math.isclose(
                float(manifest.get("validation_fraction", -1)),
                float(validation_fraction), rel_tol=0.0, abs_tol=0.0):
        raise Full1000Error("legacy hard-negative validation split changed")
    ranking_name = manifest.get("ranking_file")
    if not isinstance(ranking_name, str) or Path(ranking_name).name != ranking_name:
        raise Full1000Error("legacy ranking file is not a local basename")
    ranking_path = path.parent / ranking_name
    if manifest.get("ranking_sha256") != LEGACY_HARD_RANKING_SHA256 \
            or digest_file(ranking_path) != LEGACY_HARD_RANKING_SHA256:
        raise Full1000Error("legacy hard-negative ranking digest mismatch")
    try:
        ranking = [json.loads(line) for line in
                   ranking_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full1000Error("legacy hard-negative ranking is invalid") from exc
    if len(ranking) != LEGACY_HARD_POOL_COUNT:
        raise Full1000Error("legacy hard-negative ranking row count changed")
    previous_key = None
    seen = set()
    for index, row in enumerate(ranking, 1):
        line_id = str(row.get("line_id", ""))
        scores = row.get("score_by_seed")
        score = row.get("ensemble_score")
        if row.get("schema_version") != base.SCHEMA_HARD_RANKING \
                or row.get("record_type") != "ranked_clean" \
                or row.get("rank") != index or not line_id or line_id in seen \
                or not isinstance(scores, dict) or set(scores) != {
                    "354", "1354", "2354"} \
                or not isinstance(score, (int, float)) \
                or not all(isinstance(value, (int, float))
                           and math.isfinite(float(value))
                           for value in scores.values()) \
                or not math.isfinite(float(score)) \
                or not math.isclose(
                    float(score), statistics.fmean(float(v) for v in scores.values()),
                    rel_tol=1e-12, abs_tol=1e-12):
            raise Full1000Error("legacy hard-negative ranking row is invalid")
        key = (-float(score), line_id)
        if previous_key is not None and key < previous_key:
            raise Full1000Error("legacy hard-negative ranking order changed")
        previous_key = key
        seen.add(line_id)

    identities = base.clean_pool_identities(clean_pool)
    pool_lookup = {row["line_id"]: row for row in clean_pool}
    expected_identity = {row["line_id"]: row for row in identities}
    filtered = []
    for row in ranking:
        line_id = str(row["line_id"])
        identity = expected_identity.get(line_id)
        if identity is None:
            continue
        if any(str(row.get(name, "")) != identity[name]
               for name in ("line_id", "group_uuid", "image_id")):
            raise Full1000Error("legacy ranking identity differs in new clean pool")
        filtered.append(row)
    if len(filtered) != len(clean_pool):
        missing = sorted(set(pool_lookup) - {str(row["line_id"]) for row in filtered})
        raise Full1000Error(
            f"new clean pool is not a subset of legacy ranking; missing={missing[:3]}"
        )
    if count < 0 or count > len(filtered):
        raise Full1000Error("filtered hard-negative ranking is too small")
    selected_ids = [str(row["line_id"]) for row in filtered[:count]]
    selected = [pool_lookup[line_id] for line_id in selected_ids]
    filtered_ids_sha256 = hashlib.sha256(
        "\n".join(str(row["line_id"]) for row in filtered).encode("utf-8")
    ).hexdigest()
    return selected, {
        "filter_policy": "exact safe-pool intersection; preserve legacy order",
        "legacy_manifest_sha256": LEGACY_HARD_MANIFEST_SHA256,
        "legacy_pool_count": LEGACY_HARD_POOL_COUNT,
        "legacy_ranking_sha256": LEGACY_HARD_RANKING_SHA256,
        "manifest_path": str(path),
        "manifest_sha256": LEGACY_HARD_MANIFEST_SHA256,
        "pool_sha256": base.clean_pool_sha256(clean_pool),
        "prefix_count": count,
        "prefix_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "ranking_intersection_count": len(filtered),
        "ranking_intersection_ids_sha256": filtered_ids_sha256,
        "ranking_sha256": LEGACY_HARD_RANKING_SHA256,
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "source_seeds": manifest.get("source_seeds"),
        "validation_fraction": float(validation_fraction),
        "validation_seed": validation_seed,
    }


def _relabel_audit(audit: dict[str, Any], external_arm: str) -> None:
    audit["data_arm"] = external_arm
    gold = audit["gold"]
    gold["new1000"] = gold.pop("new609")
    protected = audit["protected_image_closure"]
    protected["new1000_rows_excluded_without_semantic_label_access"] = \
        protected.pop("new609_rows_excluded_without_semantic_label_access")
    pinned = audit["pinned_test_closure"]
    pinned["new1000_records_not_semantically_read"] = \
        pinned.pop("new609_records_not_semantically_read")
    safe = audit["safe_gold_after_protected_image_closure"]
    for suffix in ("clean", "positive", "rows"):
        safe[f"new1000_{suffix}"] = safe.pop(f"new609_{suffix}")
    audit["catalogue_clean_reference_arm"] = "new1000"


def configure_base(base) -> None:
    if getattr(base, "_hme365_configured", False):
        return
    base.NEW_RETURNED_SHA256 = NEW_RETURNED_SHA256
    base.NEW_RECORDS = NEW_RECORDS
    base.NEW_SAFE_RECORDS = NEW_SAFE_QA["safe_records"]
    base.NEW_SAFE_POSITIVES = NEW_SAFE_QA["positive_after_sanitation"]
    base.NEW_SAFE_CLEAN = NEW_SAFE_QA["clean_after_sanitation"]
    base.NEW_SAFE_MARKS_RAW = NEW_SAFE_QA["raw_marks"]
    base.NEW_SAFE_ZERO_LINES = NEW_SAFE_QA["dropped_zero_length_lines"]
    base.NEW_SAFE_MARK_TYPES = dict(NEW_SAFE_QA["mark_types"])
    base.PROTECTED_IMAGE_ROWS_NEW = NEW_SAFE_QA[
        "protected_image_records_not_semantically_read"]
    base.load_gold = lambda args, protected, pinned: load_full_gold(
        base, args, protected, pinned)
    base.select_hard_prefix = lambda path, pool, count, **kwargs: \
        select_legacy_hard_prefix(base, path, pool, count, **kwargs)
    original_prepare = base.prepare

    def prepare(args: argparse.Namespace):
        external_arm = str(args.data_arm)
        if external_arm not in EXTERNAL_TO_INTERNAL_ARM:
            raise Full1000Error(f"unsupported data arm: {external_arm}")
        args.data_arm = EXTERNAL_TO_INTERNAL_ARM[external_arm]
        try:
            prepared = original_prepare(args)
        finally:
            args.data_arm = external_arm
        _relabel_audit(prepared["audit"], external_arm)
        return prepared

    base.prepare = prepare
    base._hme365_configured = True


def build_parser(base, runner) -> argparse.ArgumentParser:
    parser = runner.add_extension_arguments(base.build_parser())
    for action in parser._actions:
        if action.dest == "data_arm":
            action.choices = tuple(EXTERNAL_TO_INTERNAL_ARM)
            break
    return parser


def _finalise_output(base, root: Path, external_arm: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    manifest_path = root / "run_manifest.json"
    complete_path = root / "complete.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    summary["schema_version"] = SCHEMA
    summary["protocol"]["threshold_selected_on"] = (
        "common writer-disjoint new1000 validation only"
    )
    summary["protocol"]["full1000_runner_sha256"] = digest_file(
        Path(__file__).resolve())
    summary["protocol"]["legacy_hard_ranking_reused_only_by_safe_intersection"] = True
    manifest["schema_version"] = SCHEMA
    manifest["script_sha256"] = digest_file(Path(__file__).resolve())
    manifest["dependency_script_sha256"] = {
        "base_354": BASE_SHA256,
        "extension_357": EXTENSION_SHA256,
        "runner_358": RUNNER_SHA256,
    }
    manifest["data_arm"] = external_arm
    manifest["legacy_hard_ranking_reused_only_by_safe_intersection"] = True
    base.write_json(summary_path, summary)
    base.write_json(manifest_path, manifest)
    base.write_json(complete_path, {"schema_version": SCHEMA, "status": "complete"})
    base.write_recursive_sha256s(root)
    return summary


def run(args: argparse.Namespace, base, extension, runner) -> dict[str, Any]:
    configure_base(base)
    if args.validate_only:
        summary = runner.run(args, base, extension)
        summary["schema_version"] = SCHEMA
        summary["data"]["data_arm"] = args.data_arm
        return summary
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise Full1000Error(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.inner-{uuid.uuid4().hex}"
    original_output = args.output_dir
    args.output_dir = temporary
    try:
        runner.run(args, base, extension)
        summary = _finalise_output(base, temporary, str(args.data_arm))
        temporary.rename(output)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    finally:
        args.output_dir = original_output
    return summary


def main() -> int:
    base = load_module(BASE_PATH, "hme354_for365", BASE_SHA256)
    extension = load_module(EXTENSION_PATH, "hme357_for365", EXTENSION_SHA256)
    runner = load_module(RUNNER_PATH, "hme358_for365", RUNNER_SHA256)
    args = build_parser(base, runner).parse_args()
    try:
        summary = run(args, base, extension, runner)
    except (Full1000Error, base.LocalizerError, runner.MechanismError) as exc:
        raise SystemExit(f"experiment 365 refused: {exc}") from exc
    print(json.dumps({
        "data": summary["data"],
        "ensemble": summary["ensemble"],
        "status": summary["mode"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

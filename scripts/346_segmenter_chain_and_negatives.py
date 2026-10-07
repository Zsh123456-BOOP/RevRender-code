#!/usr/bin/env python3
"""Two fixes for the segmenter's false positives, separately and together.

344 reaches held-out IoU 0.368 and, run over 382 pages, recovers 89.5% of the
lines the catalogue calls revised while calling 749 clean lines revised as well:
precision 50.9%. Looking at the pages says what those false positives are. On a
page with no revision at all, the segmenter marks the fraction bars. A fraction
bar is a short horizontal dark run through a formula, which is pixel for pixel
what a short strikethrough is, and nothing in the training data ever said one of
them is not a revision.

Two causes, and each gets its own arm so the comparison says which one mattered.

The first is in the annotation itself. A curved strikethrough is drawn as a
chain of short straight clicks -- in the 339 package 82% of segments have an
endpoint within 3px of another segment's, the median segment is 26px against 36
elsewhere, and one line carries 37 of them. Rasterising each independently
teaches the model that a revision looks like a scatter of short bars, which is
the fraction-bar prior exactly. Chaining endpoint-adjacent segments of the same
type into one polyline turns 1,079 segments into 372 strokes on that package.

The second is that the model has never been shown a clean line. Every training
example contains a revision, so it has no way to learn what a fraction bar is
when it is not one. 7,718 human-backed clean lines survive the exclusion chain
and need no annotation at all: their mask is empty by construction.

    A  chained annotations, no clean lines
    B  clean lines, annotations as drawn
    C  both

All three are scored against the same held-out target -- 325's 109 lines,
rasterised as the annotator drew them, never chained -- so the arms differ in
what they were trained on and in nothing else.

This script deliberately does NOT:
  * chain across mark types, or across a gap wider than the tolerance;
  * alter the held-out ground truth for any arm;
  * train on the evaluation-fold annotations, the blind sets, or any test split;
  * overwrite anything -- the output directory is create-only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SEGMENTER_SCRIPT = ROOT / "scripts" / "344_revision_ink_segmenter_v2.py"
AUTHORITATIVE_SEGMENTER_SHA256 = (
    "b402cf97d0750eb2459890f046b4b707fa927ddcdd63a11b1695636e281bbedd"
)
HUMAN_ORIGINS = ("human_batch_visual_confirmed", "human_individual_review")
FORBIDDEN_ROLES = ("fixed_development_only", "fixed_external_test_only")
EXPECTED_NEGATIVE_POOL = 7_718

CHAIN_TOLERANCE = 6.0
SCHEMA = "hme-segmenter-chain-negatives-v1"


class ArmError(RuntimeError):
    """Raised when a pinned input, guard or gate refuses the run."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_recursive_sha256s(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{digest_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def atomic_staging(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise ArmError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_segmenter():
    if digest_file(SEGMENTER_SCRIPT) != AUTHORITATIVE_SEGMENTER_SHA256:
        raise ArmError("experiment 344 digest mismatch")
    spec = importlib.util.spec_from_file_location("hme_seg344_for346", SEGMENTER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def endpoints(mark):
    return [(float(mark["x1"]), float(mark["y1"])), (float(mark["x2"]), float(mark["y2"]))]


def chain_marks(marks, tolerance: float):
    """Endpoint-adjacent segments of one type become one polyline; boxes untouched."""
    lines = [m for m in marks if m["shape"] == "line"]
    out = [dict(m) for m in marks if m["shape"] != "line"]
    if not lines:
        return out
    parent = list(range(len(lines)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i]["type"] != lines[j]["type"]:
                continue
            close = any(
                math.hypot(p[0] - q[0], p[1] - q[1]) <= tolerance
                for p in endpoints(lines[i]) for q in endpoints(lines[j])
            )
            if close:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    groups = collections.defaultdict(list)
    for i in range(len(lines)):
        groups[find(i)].append(i)
    for members in groups.values():
        segments = [lines[i] for i in members]
        if len(segments) == 1:
            out.append(dict(segments[0]))
            continue
        # Greedy walk: repeatedly attach whichever free endpoint is nearest an end.
        used = [False] * len(segments)
        used[0] = True
        points = list(endpoints(segments[0]))
        for _ in range(len(segments) - 1):
            best = None
            for index, segment in enumerate(segments):
                if used[index]:
                    continue
                for near, far in ((0, 1), (1, 0)):
                    candidate = endpoints(segment)[near]
                    for tail, side in ((points[-1], "tail"), (points[0], "head")):
                        distance = math.hypot(candidate[0] - tail[0], candidate[1] - tail[1])
                        if best is None or distance < best[0]:
                            best = (distance, index, far, side)
            if best is None:
                break
            _distance, index, far, side = best
            used[index] = True
            far_point = endpoints(segments[index])[far]
            if side == "tail":
                points.append(far_point)
            else:
                points.insert(0, far_point)
        merged = dict(segments[0])
        merged["path"] = [[round(x, 2), round(y, 2)] for x, y in points]
        merged["x1"], merged["y1"] = points[0]
        merged["x2"], merged["y2"] = points[-1]
        merged["chained_from"] = len(segments)
        out.append(merged)
    return out


def negative_pool(segmenter, package, catalogue, evaluation_writers):
    """Clean lines need no annotation: their mask is empty by construction."""
    banned = package.collect_test_identifiers()
    forbidden = set(FORBIDDEN_ROLES)
    pool = []
    for entry in catalogue:
        if entry["has_edit"] or entry["label_origin"] not in HUMAN_ORIGINS:
            continue
        if set(entry.get("data_roles") or []) & forbidden:
            continue
        writer = str(entry["group_uuid"])
        if writer in evaluation_writers or writer in banned:
            continue
        if str(entry["image_id"]) in banned:
            continue
        pool.append({"line_id": entry["representative_line_id"], "group_uuid": writer,
                     "image_id": str(entry["image_id"]), "marks": [], "source": "clean"})
    if len(pool) != EXPECTED_NEGATIVE_POOL:
        raise ArmError(f"{len(pool)} clean lines available, expected {EXPECTED_NEGATIVE_POOL}")
    return pool


def run(args: argparse.Namespace) -> dict[str, Any]:
    segmenter = load_segmenter()
    region = segmenter.load_region()
    base = region.load_matched().load_gap().load_base()
    package_spec = importlib.util.spec_from_file_location(
        "hme_evalpkg325_for346",
        ROOT / "scripts" / "325_revision_region_eval_annotation_package.py",
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_spec.name] = package
    package_spec.loader.exec_module(package)

    if digest_file(segmenter.CATALOG) != segmenter.AUTHORITATIVE_CATALOG_SHA256:
        raise ArmError("crop catalogue digest mismatch")
    catalogue_rows, catalogue = [], {}
    for line in segmenter.CATALOG.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        entry = json.loads(line)
        catalogue_rows.append(entry)
        catalogue[entry["representative_line_id"]] = entry
        for alias in entry.get("alias_line_ids") or []:
            catalogue.setdefault(alias, entry)

    fit_rows, held_rows = segmenter.split(segmenter.load_sources())
    if args.chain_tolerance > 0:
        chained = 0
        for row in fit_rows:
            before = len([m for m in row["marks"] if m["shape"] == "line"])
            row["marks"] = chain_marks(row["marks"], args.chain_tolerance)
            after = len([m for m in row["marks"] if m["shape"] == "line"])
            chained += before - after
    else:
        chained = 0

    fold_images = Path(args.fold_images_dir or args.crop_images_dir)
    train_rows, evaluation_rows = base._validate_training_inputs(
        ROOT / "outputs" / "strong_clean_visual_baseline_fold0_input_20260821_v3", fold_images
    )
    evaluation_writers = {str(r["group_uuid"]) for r in evaluation_rows}
    fold_rows = {str(r["line_id"]): r for r in list(train_rows) + list(evaluation_rows)}

    negatives = []
    if args.negatives:
        pool = negative_pool(segmenter, package, catalogue_rows, evaluation_writers)
        rng = random.Random(args.negative_seed)
        rng.shuffle(pool)
        negatives = pool if args.negatives < 0 else pool[: args.negatives]
        fit_writers = {r["group_uuid"] for r in fit_rows}
        held_writers = {r["group_uuid"] for r in held_rows}
        negatives = [n for n in negatives if n["group_uuid"] not in held_writers]
        if not negatives:
            raise ArmError("every sampled clean line belongs to a held-out writer")

    images_dir = Path(args.crop_images_dir)
    fit_data = segmenter.materialise(fit_rows + negatives, base, catalogue, images_dir, fold_rows)
    held_data = segmenter.materialise(held_rows, base, catalogue, images_dir, fold_rows)

    staging = atomic_staging(Path(args.output_dir).resolve())
    results = []
    for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
        outcome = segmenter.train_once(fit_data, held_data, args, seed, staging)
        results.append(outcome)
        print(json.dumps({"arm": args.arm, "seed": seed,
                          "held_out_iou": round(outcome["held_out"]["iou"], 4),
                          "precision": round(outcome["held_out"]["precision"], 4),
                          "recall": round(outcome["held_out"]["recall"], 4),
                          "epochs": outcome["epochs_run"]}, sort_keys=True), flush=True)

    ious = [r["held_out"]["iou"] for r in results]
    summary = {
        "arm": args.arm,
        "across_seeds": {
            "iou_max": max(ious), "iou_mean": statistics.fmean(ious), "iou_min": min(ious),
            "iou_stdev": statistics.stdev(ious) if len(ious) > 1 else 0.0,
            "precision_mean": statistics.fmean([r["held_out"]["precision"] for r in results]),
            "recall_mean": statistics.fmean([r["held_out"]["recall"] for r in results]),
            "seeds": len(ious),
        },
        "data": {
            "annotated_lines": len(fit_rows),
            "chain_tolerance": args.chain_tolerance,
            "clean_lines": len(negatives),
            "held_out_lines": len(held_data),
            "segments_merged_by_chaining": chained,
        },
        "per_seed": results,
        "schema_version": SCHEMA,
        "versus_344": {"iou": 0.3679, "precision": None, "recall": None},
    }
    write_json(staging / "summary.json", summary)
    write_json(staging / "run_manifest.json", {
        "arm": args.arm,
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "evaluation_fold_trained_on": False,
        "external_blind_set_touched": False,
        "git_revision": git_revision(),
        "held_out_target": "325's annotations as drawn, never chained, identical "
                           "across every arm",
        "network_used": False,
        "paid_api_calls": 0,
        "pinned_experiment_344_sha256": AUTHORITATIVE_SEGMENTER_SHA256,
        "platform": platform.platform(),
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "summary": summary["across_seeds"],
    })
    write_json(staging / "complete.json", {"schema_version": SCHEMA, "status": "complete"})
    write_recursive_sha256s(staging)
    staging.rename(Path(args.output_dir).resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crop-images-dir", required=True, type=Path)
    parser.add_argument("--fold-images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("A_chain", "B_negatives", "C_both"))
    parser.add_argument("--chain-tolerance", type=float, default=0.0)
    parser.add_argument("--negatives", type=int, default=0,
                        help="how many clean lines to add; -1 uses the whole pool")
    parser.add_argument("--negative-seed", type=int, default=346)
    parser.add_argument("--seeds", default="344,1344,2344")
    segmenter_defaults = {
        "epochs": 120, "batch_size": 8, "learning_rate": 3e-4, "weight_decay": 1e-4,
        "base_channels": 48, "validation_fraction": 0.15, "patience": 25,
    }
    for name, value in segmenter_defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except ArmError as exc:
        raise SystemExit(f"experiment 346 refused: {exc}") from exc
    print(json.dumps({
        "arm": summary["arm"],
        "across_seeds": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in summary["across_seeds"].items()},
        "data": summary["data"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

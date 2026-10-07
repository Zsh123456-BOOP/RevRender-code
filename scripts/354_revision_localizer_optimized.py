#!/usr/bin/env python3
"""Select and run an improved revision-ink localiser without test leakage.

The human comparison requested for this experiment is:

    new609     the newly returned 609 gold line annotations only
    combined   the same 609 plus the earliest 300 returned annotations

After pinned-test closure, the 300-record package contributes 285 revised
lines and 12 clean lines; its single ``illegible`` record is excluded.  The
safe part of the 609-record package contributes 523 revised and 10 clean
lines.  A hash-stable writer split is shared by every
recipe and both data arms.  Rows from a validation writer never enter fitting,
including old-package rows whose writer also occurs in the new package.
Before splitting, image-level closure removes every row whose source page also
contains a ``fixed_development_only`` or ``fixed_external_test_only`` line.

``--mode dev`` never materialises or scores the established reference sets.  It is
used to select one recipe on the common writer-disjoint development split.
``--mode mine-hard`` is a separate create-only operation: the three completed
``highres`` development checkpoints score every safe fitting-writer catalogue
clean line once, and emit one immutable ensemble ranking.  ``hard`` and
``optimized`` consume prefixes of that same ranking instead of mining online.
Only after that recipe is locked may ``--mode final`` evaluate the 109 mask
lines and the evaluation-writer clean/revised line sets.  Those reference sets
were already used by experiments 348/350 and are not described as untouched or
blind.  The M6 frozen test remains completely outside this program.

The improved recipe changes four things that can be ablated independently:

* 768x96 inputs instead of 512x64, preserving thin strokes;
* fill-based paired affine/photometric augmentation (no wrap-around ``roll``);
* balanced batches with a fixed optimizer-step budget and an optional fixed
  offline hard-clean ranking, so recipes, seeds and data arms are comparable;
* a small centreline-recall term on ``line`` marks only, while BCE and Dice
  continue supervising all three annotation types.

Every output is create-only and records input hashes, seeds, split membership,
training steps, thresholds and environment details.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
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
SEGMENTER_SHA256 = "b402cf97d0750eb2459890f046b4b707fa927ddcdd63a11b1695636e281bbedd"
NEGATIVE_SCRIPT = ROOT / "scripts" / "346_segmenter_chain_and_negatives.py"
NEGATIVE_SHA256 = "a42dfba14e1cbf74b0657bb640455061a4d0a23f2d22250c213d9aec378d84bf"
EVAL_PACKAGE_SCRIPT = ROOT / "scripts" / "325_revision_region_eval_annotation_package.py"
EVAL_PACKAGE_SHA256 = "fed2b83ea9fb09f5ce4d038d06157e9172571c813319b974151a281d50e98d70"

DEFAULT_NEW_KEY = ROOT / "outputs" / "assisted_region_annotation_v1" / "key.jsonl"
NEW_KEY_SHA256 = "6ecc31137ec93c4d7f33d83901cb629df03e933f47b6e6c5f534aaa7563ffd87"
NEW_RETURNED_SHA256 = "9bed9bc41b06b4537a754a596d54f5ae9cc82de34675ac01c56742833ea4a78f"
NEW_FINGERPRINT = "e0d0cb61145ba90b4004a21d28fee026"
NEW_RECORDS = 609
NEW_SAFE_RECORDS = 533
NEW_SAFE_POSITIVES = 523
NEW_SAFE_CLEAN = 10
NEW_SAFE_MARKS_RAW = 2_717
NEW_SAFE_ZERO_LINES = 2
NEW_SAFE_MARK_TYPES = {"blot": 308, "line": 2_394, "overwrite": 15}

DEFAULT_OLD_KEY = ROOT / "outputs" / "revision_region_annotation_v1" / "key.jsonl"
OLD_KEY_SHA256 = "eb7458b240c430b690c5cd43a543f40dbb82a3634c4260b163f9d4eecda69708"
DEFAULT_OLD_RETURNED = (
    ROOT / "work" / "revision_region_annotation_returned_v2"
    / "revision_region_annotation.json"
)
OLD_RETURNED_SHA256 = "88a49f39297c0e665d9f164c0cd6b274099a4c5cd3db6dc50e28ba70ea4a27af"
OLD_FINGERPRINT = "cbdd10c5ed9f3c3f70df6a3d5b928edc"
OLD_RECORDS = 300
OLD_POSITIVES = 285
OLD_CLEAN = 12
OLD_ILLEGIBLE = 1
OLD_MARKS = 620
OLD_SAFE_RECORDS = 298
OLD_PINNED_TEST_ROWS = 2
OLD_SAFE_MARK_TYPES = {"blot": 101, "line": 512, "overwrite": 7}
PROTECTED_IMAGE_ROWS_NEW = 76
PROTECTED_IMAGE_ROWS_OLD = 0

SCHEMA_ANNOTATION = "hme-revision-region-annotation-v1"
SCHEMA_RESULT = "hme-revision-localizer-optimized-v1"
SCHEMA_HARD_MANIFEST = "hme-revision-hard-negative-manifest-v1"
SCHEMA_HARD_RANKING = "hme-revision-hard-negative-ranking-v1"
SCHEMA_RECIPE_LOCK = "hme-revision-localizer-recipe-lock-v1"
ALLOWED_TYPES = {"blot", "line", "overwrite"}
HUMAN_ORIGINS = ("human_batch_visual_confirmed", "human_individual_review")
HELD_MASK_LINES = 109
HELD_CLEAN_LINES = 500
HELD_REVISED_LINES = 200
THRESHOLDS = tuple(round(index * 0.02, 2) for index in range(1, 50))

RECIPES = {
    "baseline": {
        "crop": (512, 64), "augment": "legacy", "hard_fraction": 0.0,
        "skeleton_weight": 0.0,
    },
    "affine512": {
        "crop": (512, 64), "augment": "affine", "hard_fraction": 0.0,
        "skeleton_weight": 0.0,
    },
    "highres": {
        "crop": (768, 96), "augment": "affine", "hard_fraction": 0.0,
        "skeleton_weight": 0.0,
    },
    "hard": {
        "crop": (768, 96), "augment": "affine", "hard_fraction": 1.0,
        "skeleton_weight": 0.0,
    },
    "optimized": {
        "crop": (768, 96), "augment": "affine", "hard_fraction": 1.0,
        "skeleton_weight": 0.15,
    },
}


class LocalizerError(RuntimeError):
    """Raised when an input, split, protocol or create-only guard refuses a run."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_recursive_sha256s(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{digest_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def atomic_staging(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise LocalizerError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_module(path: Path, name: str, expected: str | None = None):
    if expected is not None and digest_file(path) != expected:
        raise LocalizerError(f"{name}: digest mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LocalizerError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pinned_input_dependency_hashes(base, segmenter) -> dict[str, str]:
    """Return every pinned non-checkpoint dependency used to construct a split."""
    fold_hashes = getattr(base, "INPUT_V4_SHA256", None)
    if not isinstance(fold_hashes, dict) or not fold_hashes:
        raise LocalizerError("fold input dependency hashes are unavailable")
    required_fold_files = {"protocol.json", "train_rows.jsonl", "eval_rows.jsonl"}
    if set(fold_hashes) != required_fold_files:
        raise LocalizerError("fold input dependency set changed")
    dependencies = {
        "catalogue": str(segmenter.AUTHORITATIVE_CATALOG_SHA256),
        "eval_package_script": EVAL_PACKAGE_SHA256,
        "fold_eval_rows": str(fold_hashes["eval_rows.jsonl"]),
        "fold_protocol": str(fold_hashes["protocol.json"]),
        "fold_train_rows": str(fold_hashes["train_rows.jsonl"]),
        "negative_script": NEGATIVE_SHA256,
        "new_key": NEW_KEY_SHA256,
        "new_returned": NEW_RETURNED_SHA256,
        "old_key": OLD_KEY_SHA256,
        "old_returned": OLD_RETURNED_SHA256,
        "segmenter_script": SEGMENTER_SHA256,
    }
    if not all(len(value) == 64 and all(character in "0123456789abcdef"
                                        for character in value)
               for value in dependencies.values()):
        raise LocalizerError("input dependency hashes are malformed")
    return dependencies


def validate_mark(mark: dict[str, Any]) -> float | None:
    if mark.get("type") not in ALLOWED_TYPES:
        raise LocalizerError(f"unknown mark type: {mark.get('type')!r}")
    shape = mark.get("shape")
    if shape == "box":
        values = [mark.get(name) for name in ("x", "y", "w", "h")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in values):
            raise LocalizerError("box has non-finite coordinates")
        if float(mark["w"]) <= 0 or float(mark["h"]) <= 0:
            raise LocalizerError("box has non-positive size")
        return None
    if shape != "line":
        raise LocalizerError(f"unknown mark shape: {shape!r}")
    values = [mark.get(name) for name in ("x1", "y1", "x2", "y2")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in values):
        raise LocalizerError("line has non-finite coordinates")
    return math.hypot(float(mark["x2"]) - float(mark["x1"]),
                      float(mark["y2"]) - float(mark["y1"]))


def read_package(key_path: Path, returned_path: Path, *, name: str,
                 expected_key_sha: str, expected_returned_sha: str,
                 expected_fingerprint: str, expected_records: int,
                 protected_images: set[str] | None = None,
                 pinned_test_identifiers: set[str] | None = None) \
        -> tuple[list[dict], dict]:
    if digest_file(key_path) != expected_key_sha:
        raise LocalizerError(f"{name}: key digest mismatch")
    if digest_file(returned_path) != expected_returned_sha:
        raise LocalizerError(f"{name}: returned digest mismatch")
    document = json.loads(returned_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_ANNOTATION:
        raise LocalizerError(f"{name}: schema mismatch")
    if document.get("page_fingerprint") != expected_fingerprint:
        raise LocalizerError(f"{name}: page fingerprint mismatch")
    records = document.get("records")
    if not isinstance(records, dict) or len(records) != expected_records:
        raise LocalizerError(f"{name}: expected {expected_records} records")

    key_rows = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines()
                if line]
    key_rows.sort(key=lambda row: int(row["index"]))
    selected_key = key_rows[:expected_records]
    if [int(row["index"]) for row in selected_key] != list(range(expected_records)):
        raise LocalizerError(f"{name}: key indices are not contiguous")
    if set(records) != {row["id"] for row in selected_key}:
        raise LocalizerError(f"{name}: returned ids do not match the pinned key prefix")

    protected_images = protected_images or set()
    pinned_test_identifiers = pinned_test_identifiers or set()
    rows = []
    type_counts = {mark_type: 0 for mark_type in sorted(ALLOWED_TYPES)}
    raw_marks = dropped_zero_lines = 0
    protected_skipped = pinned_test_skipped = 0
    flags: dict[str, int] = {}
    for key in selected_key:
        if str(key["image_id"]) in protected_images:
            protected_skipped += 1
            continue
        identity = {
            str(key["line_id"]), str(key["group_uuid"]), str(key["image_id"]),
        }
        if identity & pinned_test_identifiers:
            # The whole returned file and ID coverage are authenticated above,
            # but a pinned-test record's label object must never be dereferenced.
            pinned_test_skipped += 1
            continue
        record = records[key["id"]]
        flag = record.get("flag")
        if flag not in (None, "no_revision", "illegible"):
            raise LocalizerError(f"{name}: unknown flag {flag!r}")
        marks = record.get("marks") or []
        if flag is not None and marks:
            raise LocalizerError(f"{name}: record has both marks and flag")
        if flag is not None:
            flags[flag] = flags.get(flag, 0) + 1
        cleaned = []
        for mark in marks:
            length = validate_mark(mark)
            raw_marks += 1
            type_counts[mark["type"]] += 1
            if length == 0.0:
                dropped_zero_lines += 1
                continue
            cleaned.append(dict(mark))
        rows.append({
            "image_id": str(key["image_id"]),
            "line_id": str(key["line_id"]),
            "group_uuid": str(key["group_uuid"]),
            "marks": cleaned,
            "raw_marks": len(marks),
            "flag": flag,
            "source": name,
        })
    if len({row["line_id"] for row in rows}) != len(rows):
        raise LocalizerError(f"{name}: duplicate line ids")
    return rows, {
        "clean_after_sanitation": sum(
            int(not row["marks"] and row["flag"] != "illegible") for row in rows),
        "dropped_zero_length_lines": dropped_zero_lines,
        "flags": flags,
        "mark_types": type_counts,
        "positive_after_sanitation": sum(int(bool(row["marks"])) for row in rows),
        "raw_marks": raw_marks,
        "pinned_test_records_not_semantically_read": pinned_test_skipped,
        "protected_image_records_not_semantically_read": protected_skipped,
        "safe_records": len(rows),
        "source_records": expected_records,
    }


def load_gold(args: argparse.Namespace, protected_images: set[str],
              pinned_test_identifiers: set[str]) \
        -> tuple[list[dict], list[dict], dict]:
    if not args.gold_confirmed:
        raise LocalizerError("returned annotations require explicit --gold-confirmed")
    new_rows, new_qa = read_package(
        Path(args.new_key), Path(args.new_returned), name="new609",
        expected_key_sha=NEW_KEY_SHA256, expected_returned_sha=NEW_RETURNED_SHA256,
        expected_fingerprint=NEW_FINGERPRINT, expected_records=NEW_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=pinned_test_identifiers,
    )
    old_rows, old_qa = read_package(
        Path(args.old_key), Path(args.old_returned), name="old300",
        expected_key_sha=OLD_KEY_SHA256, expected_returned_sha=OLD_RETURNED_SHA256,
        expected_fingerprint=OLD_FINGERPRINT, expected_records=OLD_RECORDS,
        protected_images=protected_images,
        pinned_test_identifiers=pinned_test_identifiers,
    )
    expected_new = {
        "source_records": NEW_RECORDS, "safe_records": NEW_SAFE_RECORDS,
        "pinned_test_records_not_semantically_read": 0,
        "protected_image_records_not_semantically_read": PROTECTED_IMAGE_ROWS_NEW,
        "positive_after_sanitation": NEW_SAFE_POSITIVES,
        "clean_after_sanitation": NEW_SAFE_CLEAN,
        "raw_marks": NEW_SAFE_MARKS_RAW,
        "dropped_zero_length_lines": NEW_SAFE_ZERO_LINES,
        "mark_types": NEW_SAFE_MARK_TYPES, "flags": {"no_revision": 5},
    }
    if new_qa != expected_new:
        raise LocalizerError(f"new609 counts changed: {new_qa} != {expected_new}")
    expected_old = {
        "source_records": OLD_RECORDS, "safe_records": OLD_SAFE_RECORDS,
        "pinned_test_records_not_semantically_read": OLD_PINNED_TEST_ROWS,
        "protected_image_records_not_semantically_read": PROTECTED_IMAGE_ROWS_OLD,
        "positive_after_sanitation": OLD_POSITIVES,
        "clean_after_sanitation": OLD_CLEAN,
        "raw_marks": OLD_MARKS, "dropped_zero_length_lines": 0,
        "mark_types": OLD_SAFE_MARK_TYPES,
        "flags": {"illegible": OLD_ILLEGIBLE, "no_revision": OLD_CLEAN},
    }
    if old_qa != expected_old:
        raise LocalizerError(f"old300 counts changed: {old_qa} != {expected_old}")
    old_usable = [row for row in old_rows if row["flag"] != "illegible"]
    new_ids = {row["line_id"] for row in new_rows}
    if new_ids & {row["line_id"] for row in old_usable}:
        raise LocalizerError("new609 and old300 overlap by line id")
    writer_overlap = len(
        {row["group_uuid"] for row in new_rows}
        & {row["group_uuid"] for row in old_usable}
    )
    if writer_overlap != 121:
        raise LocalizerError(f"new/old writer overlap changed: {writer_overlap} != 121")
    return new_rows, old_usable, {
        "all_returned_records_confirmed_gold": True,
        "new609": new_qa,
        "old300": old_qa,
        "old_illegible_excluded": OLD_ILLEGIBLE,
        "writer_overlap": writer_overlap,
    }


def writer_is_validation(writer: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(
        f"hme354-common-val-{seed}\x1f{writer}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < fraction


def reference_catalogue_clean_count(new_positive: int, new_gold_clean: int,
                                    negative_ratio: float) -> int:
    """Freeze external clean-pool size to the new609 reference arm."""
    return max(0, int(round(new_positive * negative_ratio)) - new_gold_clean)


def clamp_mark(mark: dict[str, Any], width: int, height: int) -> tuple[dict, int]:
    out = dict(mark)
    changed = 0

    def clip(value: float, high: float) -> float:
        nonlocal changed
        clipped = min(max(float(value), 0.0), float(high))
        changed += int(clipped != float(value))
        return clipped

    if out["shape"] == "box":
        x0, y0 = clip(out["x"], width), clip(out["y"], height)
        x1 = clip(float(out["x"]) + float(out["w"]), width)
        y1 = clip(float(out["y"]) + float(out["h"]), height)
        out.update(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        if out["w"] <= 0 or out["h"] <= 0:
            raise LocalizerError("clipping collapsed a box")
    elif out.get("path"):
        out["path"] = [[clip(x, width), clip(y, height)] for x, y in out["path"]]
    else:
        out.update(x1=clip(out["x1"], width), y1=clip(out["y1"], height),
                   x2=clip(out["x2"], width), y2=clip(out["y2"], height))
    return out, changed


def letterbox(image, width: int, height: int, fill: int, *, mask: bool = False):
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized_size = (max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))))
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = image.resize(resized_size, resample)
    canvas = Image.new("L", (width, height), fill)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    resized.close()
    return canvas


def line_skeleton_mask(segmenter, marks: list[dict], width: int, height: int):
    skeleton_marks = []
    for mark in marks:
        if mark["type"] != "line":
            continue
        skeleton = dict(mark)
        skeleton["stroke_width"] = 1
        skeleton_marks.append(skeleton)
    return segmenter.rasterise(skeleton_marks, width, height)


def materialise(rows: list[dict], segmenter, base, catalogue: dict,
                images_dir: Path, crop: tuple[int, int], fallback=None):
    import numpy as np
    from PIL import Image

    width, height = crop
    out, pages, clipped = [], {}, 0
    try:
        for row in sorted(rows, key=lambda value: value["line_id"]):
            entry = catalogue.get(row["line_id"])
            if entry is None and fallback is not None:
                entry = fallback.get(row["line_id"])
            if entry is None:
                raise LocalizerError(f"no crop box for {row['line_id']}")
            image_id = row["image_id"]
            if image_id not in pages:
                for opened in pages.values():
                    opened.close()
                pages = {image_id: Image.open(
                    images_dir / f"{image_id}.jpg").convert("RGB")}
            crop_image = base.crop_line(
                pages[image_id], entry["box_1000"]).convert("L")
            marks = []
            for mark in row["marks"]:
                cleaned, count = clamp_mark(mark, crop_image.width, crop_image.height)
                clipped += count
                marks.append(cleaned)
            mask_image = segmenter.rasterise(marks, crop_image.width, crop_image.height)
            skeleton_image = line_skeleton_mask(
                segmenter, marks, crop_image.width, crop_image.height)
            resized_image = letterbox(crop_image, width, height, 255)
            resized_mask = letterbox(mask_image, width, height, 0, mask=True)
            resized_skeleton = letterbox(skeleton_image, width, height, 0, mask=True)
            out.append({
                "group_uuid": row["group_uuid"],
                "image": np.asarray(resized_image, dtype=np.uint8),
                "image_id": image_id,
                "line_id": row["line_id"],
                "line_skeleton": np.asarray(resized_skeleton, dtype=np.uint8),
                "mask": np.asarray(resized_mask, dtype=np.uint8),
                "source": row["source"],
                "types": sorted({mark["type"] for mark in marks}),
            })
            for opened in (crop_image, mask_image, skeleton_image,
                           resized_image, resized_mask, resized_skeleton):
                opened.close()
    finally:
        for opened in pages.values():
            opened.close()
    return out, clipped


def cut_catalogue(entries: list[dict], base, images_dir: Path,
                  crop: tuple[int, int]):
    import numpy as np
    from PIL import Image

    width, height = crop
    out, pages = [], {}
    try:
        for entry in entries:
            image_id = str(entry["image_id"])
            if image_id not in pages:
                for opened in pages.values():
                    opened.close()
                pages = {image_id: Image.open(
                    images_dir / f"{image_id}.jpg").convert("RGB")}
            line = base.crop_line(pages[image_id], entry["box_1000"]).convert("L")
            resized = letterbox(line, width, height, 255)
            out.append({
                "group_uuid": str(entry["group_uuid"]),
                "image": np.asarray(resized, dtype=np.uint8),
                "image_id": image_id,
                "line_id": str(entry["representative_line_id"]),
            })
            line.close()
            resized.close()
    finally:
        for opened in pages.values():
            opened.close()
    return out


def paired_affine(image, mask, skeleton, rng: random.Random):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    if rng.random() > 0.8:
        return image, mask, skeleton
    height, width = image.shape
    angle = rng.uniform(-3.0, 3.0)
    scale = 1.0 + rng.uniform(-0.06, 0.06)
    dx, dy = rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height
    radians = math.radians(angle)
    cosine, sine = math.cos(radians) / scale, math.sin(radians) / scale
    cx, cy = width / 2.0, height / 2.0
    matrix = (cosine, sine, cx - cosine * cx - sine * cy - dx,
              -sine, cosine, cy + sine * cx - cosine * cy - dy)

    def warp(array, resample, fill):
        return Image.fromarray(array).transform(
            (width, height), Image.Transform.AFFINE, matrix,
            resample=resample, fillcolor=fill,
        )

    picture = warp(image, Image.Resampling.BILINEAR, 255)
    target = warp(mask, Image.Resampling.NEAREST, 0)
    centreline = warp(skeleton, Image.Resampling.NEAREST, 0)
    picture = ImageEnhance.Contrast(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    picture = ImageEnhance.Brightness(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    if rng.random() < 0.20:
        picture = picture.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8)))
    if rng.random() < 0.15:
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=rng.randint(55, 92))
        buffer.seek(0)
        picture = Image.open(buffer).convert("L")
    return (np.asarray(picture, dtype=np.uint8).copy(),
            np.asarray(target, dtype=np.uint8).copy(),
            np.asarray(centreline, dtype=np.uint8).copy())


def pixel_metrics(probabilities, targets, threshold: float) -> dict[str, float]:
    import numpy as np

    predicted, truth = probabilities >= threshold, targets > 0.5
    intersection = (predicted & truth).reshape(len(predicted), -1).sum(axis=1).astype(float)
    union = (predicted | truth).reshape(len(predicted), -1).sum(axis=1).astype(float)
    predicted_sum = predicted.reshape(len(predicted), -1).sum(axis=1).astype(float)
    truth_sum = truth.reshape(len(predicted), -1).sum(axis=1).astype(float)
    micro_intersection = float(intersection.sum())
    micro_union = float(union.sum())
    micro_predicted = float(predicted_sum.sum())
    micro_truth = float(truth_sum.sum())
    per_iou = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
    per_dice = np.divide(2 * intersection, predicted_sum + truth_sum,
                         out=np.ones_like(intersection), where=(predicted_sum + truth_sum) > 0)
    return {
        "dice": 2 * micro_intersection / (micro_predicted + micro_truth)
                if micro_predicted + micro_truth else 1.0,
        "iou": micro_intersection / micro_union if micro_union else 1.0,
        "macro_dice": float(per_dice.mean()),
        "macro_iou": float(per_iou.mean()),
        "precision": micro_intersection / micro_predicted if micro_predicted else 1.0,
        "recall": micro_intersection / micro_truth if micro_truth else 1.0,
    }


def line_fire_rate(probabilities, threshold: float, min_pixels: int) -> float:
    if len(probabilities) == 0:
        return 0.0
    counts = (probabilities >= threshold).reshape(len(probabilities), -1).sum(axis=1)
    return float((counts >= min_pixels).mean())


def line_f1(recall: float, false_alarm: float, positives: int, negatives: int) -> float:
    true_positive = recall * positives
    false_positive = false_alarm * negatives
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def choose_operating_points(positive_probabilities, positive_targets,
                            clean_probabilities, max_clean_fp: float,
                            min_fire_pixels: int,
                            thresholds=THRESHOLDS) -> dict[str, dict]:
    candidates = []
    for threshold in thresholds:
        metrics = pixel_metrics(positive_probabilities, positive_targets, threshold)
        false_alarm = line_fire_rate(clean_probabilities, threshold, min_fire_pixels)
        candidates.append({**metrics, "clean_false_alarm": false_alarm,
                           "threshold": float(threshold)})
    localization = max(candidates, key=lambda item: (
        item["iou"], item["recall"], item["precision"], -item["threshold"]
    ))
    feasible = [item for item in candidates if item["clean_false_alarm"] <= max_clean_fp]
    if feasible:
        compatibility = max(feasible, key=lambda item: (
            item["iou"], item["recall"], item["precision"], -item["threshold"]
        ))
        compatibility = {**compatibility, "constraint_met": True}
    else:
        compatibility = min(candidates, key=lambda item: (
            item["clean_false_alarm"], -item["iou"], -item["recall"]
        ))
        compatibility = {**compatibility, "constraint_met": False}
    return {"localization": localization, "compatibility": compatibility}


def scaled_min_fire_pixels(recipe_name: str, base_min_fire_pixels: int) -> int:
    width, height = RECIPES[recipe_name]["crop"]
    area_scale = (width * height) / float(512 * 64)
    return max(1, int(round(base_min_fire_pixels * area_scale)))


def _binary_erode(mask, radius: int):
    import numpy as np

    padded = np.pad(mask, ((radius, radius), (radius, radius)), constant_values=False)
    output = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return output


def _binary_dilate(mask, radius: int):
    import numpy as np

    padded = np.pad(mask, ((radius, radius), (radius, radius)), constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return output


def boundary_f1(probabilities, targets, threshold: float, tolerance: int = 3) -> float:
    predicted, truth = probabilities >= threshold, targets > 0.5
    matched_predicted = matched_truth = predicted_total = truth_total = 0
    for predicted_mask, truth_mask in zip(predicted, truth):
        predicted_boundary = predicted_mask & ~_binary_erode(predicted_mask, 1)
        truth_boundary = truth_mask & ~_binary_erode(truth_mask, 1)
        predicted_total += int(predicted_boundary.sum())
        truth_total += int(truth_boundary.sum())
        matched_predicted += int((predicted_boundary & _binary_dilate(
            truth_boundary, tolerance)).sum())
        matched_truth += int((truth_boundary & _binary_dilate(
            predicted_boundary, tolerance)).sum())
    precision = matched_predicted / predicted_total if predicted_total else 1.0
    recall = matched_truth / truth_total if truth_total else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predict(model, rows: list[dict], batch_size: int, device):
    import numpy as np
    import torch

    probabilities, targets = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            x = torch.from_numpy(np.stack([
                row["image"] for row in chunk]).astype(np.float32) / 127.5 - 1.0
            )[:, None].to(device)
            probabilities.append(torch.sigmoid(model(x)).cpu().numpy()[:, 0])
            if "mask" in chunk[0]:
                targets.append(np.stack([
                    row["mask"] for row in chunk]).astype(np.float32) / 255.0)
    model.train()
    probability = np.concatenate(probabilities) if probabilities else np.empty((0, 1, 1))
    target = np.concatenate(targets) if targets else None
    return probability, target


class ShuffledCycle:
    def __init__(self, rows: list[dict], rng: random.Random):
        if not rows:
            raise LocalizerError("cannot sample an empty row collection")
        self.rows = rows
        self.rng = rng
        self.order: list[int] = []
        self.position = 0

    def take(self, count: int) -> list[dict]:
        output = []
        while len(output) < count:
            if self.position >= len(self.order):
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
                self.position = 0
            amount = min(count - len(output), len(self.order) - self.position)
            output.extend(self.rows[index] for index in
                          self.order[self.position:self.position + amount])
            self.position += amount
        return output

    def take_distinct(self, count: int) -> list[dict]:
        if count < 0 or count > len(self.rows):
            raise LocalizerError("distinct cycle request exceeds its row pool")
        output = []
        selected_indices = set()
        while len(output) < count:
            if self.position >= len(self.order):
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
                self.position = 0
            index = self.order[self.position]
            self.position += 1
            if index in selected_indices:
                continue
            selected_indices.add(index)
            output.append(self.rows[index])
        return output


def take_balanced_batch(positive_cycle: ShuffledCycle, clean_cycle: ShuffledCycle,
                        batch_size: int, negative_ratio: float) \
        -> tuple[list[dict], int, int]:
    positive_count = int(round(batch_size / (1.0 + negative_ratio)))
    clean_count = batch_size - positive_count
    if positive_count <= 0 or abs(clean_count / positive_count - negative_ratio) > 1e-9:
        raise LocalizerError("batch size must exactly express the requested negative ratio")
    positive_rows = positive_cycle.take_distinct(positive_count)
    clean_rows = clean_cycle.take_distinct(clean_count)
    rows = positive_rows + clean_rows
    line_ids = [str(row["line_id"]) for row in rows]
    if len(line_ids) != len(set(line_ids)):
        raise LocalizerError("balanced batch contains a duplicate line")
    return rows, positive_count, clean_count


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise LocalizerError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise LocalizerError("seeds must be a non-empty unique list")
    return seeds


def clean_pool_identities(rows: list[dict]) -> list[dict[str, str]]:
    identities = []
    for row in rows:
        identity = {name: str(row[name]) for name in
                    ("line_id", "group_uuid", "image_id")}
        if not all(identity.values()):
            raise LocalizerError("clean pool contains an empty identity")
        identities.append(identity)
    identities.sort(key=lambda row: row["line_id"])
    line_ids = [row["line_id"] for row in identities]
    if len(line_ids) != len(set(line_ids)):
        raise LocalizerError("clean pool contains duplicate line ids")
    return identities


def clean_pool_sha256(rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in clean_pool_identities(rows)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def line_ids_sha256(rows: list[dict]) -> str:
    """Hash one stable, duplicate-free line-id set for protocol locking."""
    line_ids = sorted(str(row["line_id"]) for row in rows)
    if any(not line_id for line_id in line_ids) \
            or len(line_ids) != len(set(line_ids)):
        raise LocalizerError("protocol identity contains empty or duplicate line ids")
    return hashlib.sha256("\n".join(line_ids).encode("utf-8")).hexdigest()


def locked_protocol(args: argparse.Namespace, prepared: dict[str, Any],
                    seeds: list[int]) -> dict[str, Any]:
    """Return the complete arm-independent protocol a final lock must pin."""
    hard = prepared["hard_selection"]
    hard_identity = None if hard is None else {
        "manifest_sha256": hard["manifest_sha256"],
        "ranking_sha256": hard["ranking_sha256"],
        "fixed_prefix_count": hard["prefix_count"],
        "fixed_prefix_ids_sha256": hard["prefix_ids_sha256"],
    }
    recipe_config = RECIPES[args.recipe]
    return {
        "checkpoint_rule": args.checkpoint_rule,
        "negative_protocol": {
            "clean_selection": prepared["audit"]["clean_selection"],
            "fixed_catalogue_clean_count":
            prepared["audit"]["sampled_catalogue_clean"],
            "hard_negative_identity": hard_identity,
            "negative_ratio": args.negative_ratio,
            "negative_seed": args.negative_seed,
        },
        "optimizer_steps": args.optimizer_steps,
        "recipe": args.recipe,
        "recipe_config": {
            "augment": recipe_config["augment"],
            "crop": list(recipe_config["crop"]),
            "hard_fraction": recipe_config["hard_fraction"],
            "skeleton_weight": recipe_config["skeleton_weight"],
        },
        "reference_protocol": {
            "held_out_seed": args.held_out_seed,
            "max_clean_fp": args.max_clean_fp,
            "min_fire_pixels": args.min_fire_pixels,
        },
        "seeds": seeds,
        "training_config": {
            "base_channels": args.base_channels,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "selection_warmup_fraction": args.selection_warmup_fraction,
            "weight_decay": args.weight_decay,
        },
        "validation_split": {
            "clean_count": len(prepared["validation_clean"]),
            "clean_line_ids_sha256": line_ids_sha256(
                prepared["validation_clean"]),
            "fraction": args.validation_fraction,
            "interval": args.validation_interval,
            "positive_count": len(prepared["validation_positive"]),
            "positive_line_ids_sha256": line_ids_sha256(
                prepared["validation_positive"]),
            "seed": args.validation_seed,
            "writer_count": prepared["audit"]["validation_writer_count"],
        },
    }


def _first_protocol_difference(expected: Any, observed: Any,
                               path: str = "locked_protocol") -> str | None:
    if type(expected) is not type(observed):
        return f"{path}: expected {expected!r}, observed {observed!r}"
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            return (f"{path}: fields differ; expected={sorted(expected)}, "
                    f"observed={sorted(observed)}")
        for key in sorted(expected):
            difference = _first_protocol_difference(
                expected[key], observed[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(expected, observed)):
            difference = _first_protocol_difference(
                left, right, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return f"{path}: expected {expected!r}, observed {observed!r}"
    return None


def validate_recipe_lock(path: Path, args: argparse.Namespace,
                         prepared: dict[str, Any], seeds: list[int]) \
        -> dict[str, Any]:
    """Fail closed unless a create-only lock exactly matches this final run."""
    path = Path(path).resolve()
    if not path.is_file():
        raise LocalizerError(f"recipe lock missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalizerError("recipe lock is not valid JSON") from exc
    if not isinstance(document, dict) \
            or document.get("schema_version") != SCHEMA_RECIPE_LOCK:
        raise LocalizerError("recipe lock schema mismatch")
    if document.get("m6_frozen_test_touched") is not False \
            or document.get("reference_evaluation_materialised") is not False:
        raise LocalizerError("recipe lock has unsafe evaluation provenance")
    observed = document.get("locked_protocol")
    expected = locked_protocol(args, prepared, seeds)
    difference = _first_protocol_difference(expected, observed)
    if difference is not None:
        raise LocalizerError(f"recipe lock protocol mismatch: {difference}")
    return {
        "path": str(path),
        "sha256": digest_file(path),
        "locked_protocol": expected,
    }


def hardness_scores(probabilities, top_pixels: int = 24):
    import numpy as np

    if probabilities.ndim < 2 or len(probabilities) == 0:
        raise LocalizerError("hard-negative scoring requires non-empty masks")
    if top_pixels <= 0:
        raise LocalizerError("hard-negative top-pixels must be positive")
    flattened = probabilities.reshape(len(probabilities), -1)
    count = min(top_pixels, flattened.shape[1])
    return np.partition(
        flattened, flattened.shape[1] - count, axis=1
    )[:, -count:].mean(1)


def rank_hard_negatives(clean_rows: list[dict], scores_by_seed: dict[int, Any]) -> list[dict]:
    import numpy as np

    identities = clean_pool_identities(clean_rows)
    index_by_id = {str(row["line_id"]): index
                   for index, row in enumerate(clean_rows)}
    if len(scores_by_seed) != 3:
        raise LocalizerError("hard-negative ranking requires exactly three source seeds")
    normalized: dict[int, Any] = {}
    for seed, scores in scores_by_seed.items():
        array = np.asarray(scores, dtype=np.float64)
        if array.shape != (len(clean_rows),) or not np.isfinite(array).all():
            raise LocalizerError(f"invalid hard-negative scores for seed {seed}")
        normalized[int(seed)] = array
    rows = []
    for identity in identities:
        source_index = index_by_id[identity["line_id"]]
        per_seed = {str(seed): float(normalized[seed][source_index])
                    for seed in sorted(normalized)}
        rows.append({
            **identity,
            "ensemble_score": float(statistics.fmean(per_seed.values())),
            "record_type": "ranked_clean",
            "score_by_seed": per_seed,
        })
    rows.sort(key=lambda row: (-row["ensemble_score"], row["line_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["schema_version"] = SCHEMA_HARD_RANKING
    return rows


def load_hard_negative_manifest(path: Path, clean_pool: list[dict], *,
                                validation_seed: int,
                                validation_fraction: float) \
        -> tuple[list[dict], dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise LocalizerError(f"hard-negative manifest missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalizerError("hard-negative manifest is not valid JSON") from exc
    if document.get("schema_version") != SCHEMA_HARD_MANIFEST:
        raise LocalizerError("hard-negative manifest schema mismatch")
    required_safety = {
        "image_level_protected_closure_applied": True,
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": False,
        "source_data_arm": "new609",
        "source_mode": "dev",
        "source_recipe": "highres",
    }
    for field, expected in required_safety.items():
        if document.get(field) != expected:
            raise LocalizerError(
                f"hard-negative manifest has unsafe {field}: {document.get(field)!r}"
            )
    if type(document.get("validation_seed")) is not int \
            or document.get("validation_seed") != validation_seed \
            or isinstance(document.get("validation_fraction"), bool) \
            or not isinstance(document.get("validation_fraction"), (int, float)) \
            or not math.isclose(float(document["validation_fraction"]),
                                float(validation_fraction), rel_tol=0.0, abs_tol=0.0):
        raise LocalizerError("hard-negative manifest validation split mismatch")
    source_seeds = document.get("source_seeds")
    checkpoint_hashes = document.get("source_checkpoint_sha256")
    if not isinstance(source_seeds, list) or len(source_seeds) != 3 \
            or len(set(source_seeds)) != 3 \
            or not all(isinstance(seed, int) for seed in source_seeds) \
            or not isinstance(checkpoint_hashes, dict) \
            or set(checkpoint_hashes) != {str(seed) for seed in source_seeds} \
            or not all(isinstance(value, str) and len(value) == 64
                       and all(character in "0123456789abcdef" for character in value)
                       for value in checkpoint_hashes.values()):
        raise LocalizerError("hard-negative source checkpoint provenance is invalid")
    expected_pool_hash = clean_pool_sha256(clean_pool)
    if document.get("pool_count") != len(clean_pool) \
            or document.get("pool_sha256") != expected_pool_hash:
        raise LocalizerError("hard-negative manifest does not match the full safe clean pool")
    ranking_name = document.get("ranking_file")
    if not isinstance(ranking_name, str) or Path(ranking_name).name != ranking_name:
        raise LocalizerError("hard-negative ranking file must be a local basename")
    ranking_path = path.parent / ranking_name
    if not ranking_path.is_file() \
            or digest_file(ranking_path) != document.get("ranking_sha256"):
        raise LocalizerError("hard-negative ranking digest mismatch")
    try:
        ranking = [json.loads(line) for line in
                   ranking_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalizerError("hard-negative ranking is not valid JSONL") from exc
    if len(ranking) != len(clean_pool):
        raise LocalizerError("hard-negative ranking row count mismatch")
    pool_lookup = {row["line_id"]: row for row in clean_pool_identities(clean_pool)}
    seen = set()
    previous_key = None
    for index, row in enumerate(ranking, 1):
        line_id = str(row.get("line_id", ""))
        if row.get("schema_version") != SCHEMA_HARD_RANKING \
                or row.get("record_type") != "ranked_clean" \
                or row.get("rank") != index:
            raise LocalizerError("hard-negative ranking sequence or schema mismatch")
        if line_id in seen or line_id not in pool_lookup:
            raise LocalizerError("hard-negative ranking contains duplicate or foreign rows")
        expected_identity = pool_lookup[line_id]
        if any(str(row.get(name, "")) != expected_identity[name]
               for name in ("line_id", "group_uuid", "image_id")):
            raise LocalizerError("hard-negative ranking identity mismatch")
        score = row.get("ensemble_score")
        per_seed = row.get("score_by_seed")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)) \
                or not isinstance(per_seed, dict) or len(per_seed) != 3 \
                or set(per_seed) != {str(seed) for seed in source_seeds} \
                or not all(isinstance(value, (int, float)) and math.isfinite(float(value))
                           for value in per_seed.values()):
            raise LocalizerError("hard-negative ranking contains invalid scores")
        if not math.isclose(float(score), statistics.fmean(
                float(value) for value in per_seed.values()),
                rel_tol=1e-12, abs_tol=1e-12):
            raise LocalizerError("hard-negative ensemble score does not match its seeds")
        key = (-float(score), line_id)
        if previous_key is not None and key < previous_key:
            raise LocalizerError("hard-negative ranking is not deterministically sorted")
        previous_key = key
        seen.add(line_id)
    return ranking, {
        "manifest_path": str(path),
        "manifest_sha256": digest_file(path),
        "pool_sha256": expected_pool_hash,
        "ranking_sha256": document["ranking_sha256"],
        "source_checkpoint_sha256": document.get("source_checkpoint_sha256"),
        "source_seeds": document.get("source_seeds"),
        "validation_fraction": float(document["validation_fraction"]),
        "validation_seed": document["validation_seed"],
    }


def select_hard_prefix(path: Path, clean_pool: list[dict], count: int, *,
                       validation_seed: int,
                       validation_fraction: float) \
        -> tuple[list[dict], dict[str, Any]]:
    ranking, audit = load_hard_negative_manifest(
        path, clean_pool, validation_seed=validation_seed,
        validation_fraction=validation_fraction,
    )
    if count < 0 or count > len(ranking):
        raise LocalizerError("hard-negative ranking is smaller than requested prefix")
    lookup = {str(row["line_id"]): row for row in clean_pool}
    selected = [lookup[row["line_id"]] for row in ranking[:count]]
    selected_ids = [str(row["line_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise LocalizerError("hard-negative prefix contains duplicate rows")
    return selected, {
        **audit,
        "prefix_count": count,
        "prefix_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
    }


def train_once(arm: str, recipe_name: str, fit_positive: list[dict],
               fit_clean: list[dict], validation_positive: list[dict],
               validation_clean: list[dict], held: dict[str, list[dict]] | None,
               hard_selection: dict[str, Any] | None, segmenter,
               args: argparse.Namespace, seed: int, staging: Path):
    import numpy as np
    import torch
    from torch import nn

    recipe = RECIPES[recipe_name]
    width, height = recipe["crop"]
    effective_min_fire_pixels = scaled_min_fire_pixels(
        recipe_name, args.min_fire_pixels
    )
    boundary_tolerance = max(1, int(math.ceil(3.0 * height / 64.0)))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") \
            not in (":4096:8", ":16:8"):
        raise LocalizerError("CUDA runs require CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = segmenter.make_unet(args.base_channels).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random(seed)
    positive_cycle = ShuffledCycle(fit_positive, rng)
    clean_cycle = ShuffledCycle(fit_clean, rng)

    best_key, best_step, best_state, best_points = None, -1, None, None
    selection_start_step = int(round(
        args.optimizer_steps * args.selection_warmup_fraction))
    for step in range(1, args.optimizer_steps + 1):
        model.train()
        chunk, _positive_count, _clean_count = take_balanced_batch(
            positive_cycle, clean_cycle, args.batch_size, args.negative_ratio
        )
        rng.shuffle(chunk)
        images, masks, skeletons = [], [], []
        for row in chunk:
            image, mask, skeleton = row["image"], row["mask"], row["line_skeleton"]
            if recipe["augment"] == "legacy":
                image, mask = segmenter.augment(image, mask, rng)
            else:
                image, mask, skeleton = paired_affine(image, mask, skeleton, rng)
            images.append(image)
            masks.append(mask)
            skeletons.append(skeleton)
        x = torch.from_numpy(
            np.stack(images).astype(np.float32) / 127.5 - 1.0)[:, None].to(device)
        y = torch.from_numpy(
            np.stack(masks).astype(np.float32) / 255.0)[:, None].to(device)
        skeleton = torch.from_numpy(
            np.stack(skeletons).astype(np.float32) / 255.0)[:, None].to(device)
        logits = model(x)
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, probability.ndim))
        numerator = 2.0 * (probability * y).sum(dimensions) + 1.0
        denominator = probability.sum(dimensions) + y.sum(dimensions) + 1.0
        positive_target = y.sum(dimensions) > 0
        if not bool(positive_target.any()):
            raise LocalizerError("balanced batch unexpectedly contains no positive mask")
        # Clean rows are already supervised by BCE.  Including six empty masks
        # in a per-example Dice mean (for the fixed 1:3 batch) collapses the
        # network toward all-background; Dice therefore measures positive masks only.
        dice_loss = (1.0 - numerator[positive_target]
                     / denominator[positive_target]).mean()
        loss = bce(logits, y) + dice_loss
        skeleton_sum = skeleton.sum(dimensions)
        valid_skeleton = skeleton_sum > 0
        skeleton_loss = torch.zeros((), device=device)
        if recipe["skeleton_weight"] and bool(valid_skeleton.any()):
            covered = (probability * skeleton).sum(dimensions)
            skeleton_loss = (1.0 - (covered[valid_skeleton] + 1.0)
                             / (skeleton_sum[valid_skeleton] + 1.0)).mean()
            loss = loss + recipe["skeleton_weight"] * skeleton_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if step % args.validation_interval != 0 and step != args.optimizer_steps:
            continue
        validation_probability, validation_target = predict(
            model, validation_positive, args.batch_size, device)
        validation_clean_probability, _ = predict(
            model, validation_clean, args.batch_size, device)
        points = choose_operating_points(
            validation_probability, validation_target, validation_clean_probability,
            args.max_clean_fp, effective_min_fire_pixels)
        primary = points["localization"]
        key = (primary["iou"], primary["macro_iou"], primary["recall"],
               -primary["threshold"])
        eligible = step >= selection_start_step
        if eligible and (best_key is None or key > best_key):
            best_key, best_step, best_points = key, step, points
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(json.dumps({
            "arm": arm, "best_step": best_step, "event": "validation",
            "eligible_for_checkpoint": eligible,
            "loss": round(float(loss.detach().cpu()), 6),
            "recipe": recipe_name, "seed": seed, "step": step,
            "threshold": primary["threshold"],
            "validation_iou": round(primary["iou"], 5),
        }, sort_keys=True), flush=True)

    diagnostic_best_step = best_step
    if args.checkpoint_rule == "last":
        best_step = args.optimizer_steps
        best_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
    if best_state is None or best_points is None:
        raise LocalizerError("training produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_probability, validation_target = predict(
        model, validation_positive, args.batch_size, device)
    validation_clean_probability, _ = predict(
        model, validation_clean, args.batch_size, device)
    points = choose_operating_points(
        validation_probability, validation_target, validation_clean_probability,
        args.max_clean_fp, effective_min_fire_pixels)
    result: dict[str, Any] = {
        "best_step": best_step,
        "checkpoint_rule": args.checkpoint_rule,
        "diagnostic_best_step": diagnostic_best_step,
        "hard_negative_count": 0 if hard_selection is None else
        hard_selection["prefix_count"],
        "hard_negative_ids_sha256": None if hard_selection is None else
        hard_selection["prefix_ids_sha256"],
        "hard_negative_manifest_sha256": None if hard_selection is None else
        hard_selection["manifest_sha256"],
        "recipe": recipe_name,
        "seed": seed,
        "steps_run": args.optimizer_steps,
        "validation": points,
    }
    prediction_bundle = {
        "validation_clean": validation_clean_probability,
        "validation_positive": validation_probability,
    }
    held_probability = held_target = None
    if held is not None:
        held_probability, held_target = predict(
            model, held["mask"], args.batch_size, device)
        held_clean_probability, _ = predict(
            model, held["clean"], args.batch_size, device)
        held_revised_probability, _ = predict(
            model, held["revised"], args.batch_size, device)
        prediction_bundle.update({
            "held_clean": held_clean_probability,
            "held_mask": held_probability,
            "held_revised": held_revised_probability,
        })
        held_results = {}
        for point_name, point in points.items():
            threshold = point["threshold"]
            clean_fp = line_fire_rate(
                held_clean_probability, threshold, effective_min_fire_pixels)
            revised_recall = line_fire_rate(
                held_revised_probability, threshold, effective_min_fire_pixels)
            metrics = pixel_metrics(held_probability, held_target, threshold)
            held_results[point_name] = {
                "boundary_f1": boundary_f1(
                    held_probability, held_target, threshold,
                    tolerance=boundary_tolerance),
                "boundary_tolerance_pixels": boundary_tolerance,
                "clean_false_alarm": clean_fp,
                "line_f1": line_f1(revised_recall, clean_fp,
                                   len(held["revised"]), len(held["clean"])),
                "mask": metrics,
                "revised_line_recall": revised_recall,
                "threshold": threshold,
            }
        result["held_out"] = held_results
        np.savez_compressed(
            staging / f"held_predictions_seed{seed}.npz",
            probabilities=held_probability.astype(np.float16),
            targets=(held_target * 255).astype(np.uint8),
        )

    torch.save({
        "architecture": {"base_channels": args.base_channels,
                         "crop": list(RECIPES[recipe_name]["crop"])},
        "best_step": best_step,
        "effective_min_fire_pixels": effective_min_fire_pixels,
        "data_arm": arm,
        "operating_points": points,
        "recipe": recipe_name,
        "seed": seed,
        "state_dict": model.state_dict(),
    }, staging / f"segmenter_seed{seed}.pt")
    print(json.dumps({
        "arm": arm, "best_step": best_step, "event": "seed_complete",
        "held_iou": None if held is None else
        round(result["held_out"]["localization"]["mask"]["iou"], 5),
        "recipe": recipe_name, "seed": seed,
        "validation_iou": round(points["localization"]["iou"], 5),
    }, sort_keys=True), flush=True)
    return result, prediction_bundle


def load_catalogue(segmenter):
    if digest_file(segmenter.CATALOG) != segmenter.AUTHORITATIVE_CATALOG_SHA256:
        raise LocalizerError("crop catalogue digest mismatch")
    rows, lookup = [], {}
    for line in segmenter.CATALOG.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        entry = json.loads(line)
        rows.append(entry)
        lookup[entry["representative_line_id"]] = entry
        for alias in entry.get("alias_line_ids") or []:
            lookup.setdefault(alias, entry)
    return rows, lookup


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    segmenter = load_module(SEGMENTER_SCRIPT, "hme_seg344_for354", SEGMENTER_SHA256)
    negatives = load_module(NEGATIVE_SCRIPT, "hme_neg346_for354", NEGATIVE_SHA256)
    eval_package = load_module(
        EVAL_PACKAGE_SCRIPT, "hme_evalpkg325_for354", EVAL_PACKAGE_SHA256
    )
    base = segmenter.load_region().load_matched().load_gap().load_base()
    input_dependency_hashes = pinned_input_dependency_hashes(base, segmenter)

    fold_images = Path(args.fold_images_dir or args.crop_images_dir)
    training_rows, evaluation_rows = base._validate_training_inputs(
        Path(args.input_root), fold_images)
    evaluation_writers = {str(row["group_uuid"]) for row in evaluation_rows}
    fold_lookup = {str(row["line_id"]): row
                   for row in list(training_rows) + list(evaluation_rows)}
    catalogue_rows, catalogue = load_catalogue(segmenter)
    pinned_test_identifiers = eval_package.collect_test_identifiers()
    protected_roles = {"fixed_development_only", "fixed_external_test_only"}
    protected_images = {
        str(entry["image_id"]) for entry in catalogue_rows
        if set(entry.get("data_roles") or []) & protected_roles
    }
    new_rows, old_rows, gold_qa = load_gold(
        args, protected_images, pinned_test_identifiers
    )
    all_gold = new_rows + old_rows
    if {row["group_uuid"] for row in all_gold} & evaluation_writers:
        raise LocalizerError("gold training annotations touch an evaluation writer")

    _old_fit, held_mask_rows = segmenter.split(segmenter.load_sources())
    if len(held_mask_rows) != HELD_MASK_LINES:
        raise LocalizerError("held mask count changed")
    held_mask_ids = {row["line_id"] for row in held_mask_rows}
    if held_mask_ids & {row["line_id"] for row in all_gold}:
        raise LocalizerError("gold training annotations overlap held mask rows")
    held_mask_writers = {row["group_uuid"] for row in held_mask_rows}
    held_mask_images = {row["image_id"] for row in held_mask_rows}
    if held_mask_writers & {row["group_uuid"] for row in all_gold}:
        raise LocalizerError("gold training annotations overlap a held mask writer")
    if held_mask_images & {row["image_id"] for row in all_gold}:
        raise LocalizerError("gold training annotations overlap a held mask image")

    clean_pool = negatives.negative_pool(
        segmenter, eval_package, catalogue_rows, evaluation_writers)
    all_gold_ids = {row["line_id"] for row in all_gold}
    clean_pool = [row for row in clean_pool
                  if row["line_id"] not in all_gold_ids
                  and row["group_uuid"] not in held_mask_writers
                  and row["image_id"] not in protected_images]

    union_writers = {row["group_uuid"] for row in all_gold}
    validation_writers = {writer for writer in union_writers
                          if writer_is_validation(writer, args.validation_fraction,
                                                  args.validation_seed)}
    validation_gold = [row for row in new_rows
                       if row["group_uuid"] in validation_writers]
    new_fit = [row for row in new_rows if row["group_uuid"] not in validation_writers]
    old_fit = [row for row in old_rows if row["group_uuid"] not in validation_writers]
    new_fit_positive = [row for row in new_fit if row["marks"]]
    new_fit_gold_clean = [row for row in new_fit if not row["marks"]]
    if args.data_arm == "new609":
        arm_gold = new_fit
    else:
        arm_gold = new_fit + old_fit
    fit_positive = [row for row in arm_gold if row["marks"]]
    fit_gold_clean = [row for row in arm_gold if not row["marks"]]
    validation_positive = [row for row in validation_gold if row["marks"]]

    validation_clean_candidates = [row for row in clean_pool
                                   if row["group_uuid"] in validation_writers]
    validation_clean_candidates.sort(key=lambda row: hashlib.sha256(
        f"hme354-val-clean-{args.validation_seed}\x1f{row['line_id']}".encode()
    ).hexdigest())
    wanted_validation_clean = max(64, int(round(
        len(validation_positive) * args.negative_ratio)))
    validation_clean = validation_clean_candidates[:wanted_validation_clean]
    if not validation_positive or len(validation_clean) < 64:
        raise LocalizerError("common validation lacks positive or clean rows")

    fit_clean_candidates = sorted(
        [row for row in clean_pool if row["group_uuid"] not in validation_writers],
        key=lambda row: row["line_id"],
    )
    full_clean_pool_hash = clean_pool_sha256(fit_clean_candidates)
    # A/B must differ only by the old300 gold package.  Freeze the external
    # catalogue-clean pool to the new609 reference arm; the exact 1:3 training
    # batch ratio is enforced independently by take_balanced_batch().
    wanted_catalogue_clean = reference_catalogue_clean_count(
        len(new_fit_positive), len(new_fit_gold_clean), args.negative_ratio,
    )
    hard_selection = None
    if RECIPES[args.recipe]["hard_fraction"]:
        hard_manifest = getattr(args, "hard_negative_manifest", None)
        if hard_manifest is None:
            raise LocalizerError(
                "hard/optimized recipes require --hard-negative-manifest"
            )
        sampled_clean, hard_selection = select_hard_prefix(
            Path(hard_manifest), fit_clean_candidates, wanted_catalogue_clean,
            validation_seed=args.validation_seed,
            validation_fraction=args.validation_fraction,
        )
        clean_selection = "fixed_hard_ranking_prefix"
    else:
        randomized_clean = list(fit_clean_candidates)
        random.Random(args.negative_seed).shuffle(randomized_clean)
        sampled_clean = randomized_clean[:wanted_catalogue_clean]
        clean_selection = "hash-pinned-random"
    if len(sampled_clean) != wanted_catalogue_clean:
        raise LocalizerError("clean fitting pool is smaller than requested")
    fit_clean = fit_gold_clean + sampled_clean
    fit_clean_ids = [row["line_id"] for row in fit_clean]
    if len(fit_clean_ids) != len(set(fit_clean_ids)):
        raise LocalizerError("fitting clean pool contains duplicate lines")

    banned_identifiers = pinned_test_identifiers
    held_clean_candidates = [
        entry for entry in catalogue_rows
        if not entry["has_edit"] and entry["label_origin"] in HUMAN_ORIGINS
        and str(entry["group_uuid"]) in evaluation_writers
        and str(entry["group_uuid"]) not in banned_identifiers
        and str(entry["image_id"]) not in banned_identifiers
        and str(entry["image_id"]) not in protected_images
    ]
    held_revised_candidates = [
        entry for entry in catalogue_rows
        if entry["has_edit"] and entry["label_origin"] in HUMAN_ORIGINS
        and str(entry["group_uuid"]) in evaluation_writers
        and str(entry["group_uuid"]) not in banned_identifiers
        and str(entry["image_id"]) not in banned_identifiers
        and str(entry["image_id"]) not in protected_images
    ]
    picker = random.Random(args.held_out_seed)
    picker.shuffle(held_clean_candidates)
    picker.shuffle(held_revised_candidates)
    held_clean_candidates = held_clean_candidates[:HELD_CLEAN_LINES]
    held_revised_candidates = held_revised_candidates[:HELD_REVISED_LINES]
    if len(held_clean_candidates) != HELD_CLEAN_LINES \
            or len(held_revised_candidates) != HELD_REVISED_LINES:
        raise LocalizerError("held line pools changed")

    audit = {
        "data_arm": args.data_arm,
        "fit_gold_clean": len(fit_gold_clean),
        "fit_positive": len(fit_positive),
        "gold": gold_qa,
        "m6_frozen_test_touched": False,
        "old_rows_excluded_with_validation_writers": len(old_rows) - len(old_fit),
        "protected_image_closure": {
            "new609_rows_excluded_without_semantic_label_access": PROTECTED_IMAGE_ROWS_NEW,
            "old300_rows_excluded_without_semantic_label_access": PROTECTED_IMAGE_ROWS_OLD,
            "roles": sorted(protected_roles),
        },
        "pinned_test_closure": {
            "identifiers_loaded_before_gold_labels": True,
            "new609_records_not_semantically_read":
            gold_qa["new609"]["pinned_test_records_not_semantically_read"],
            "old300_records_not_semantically_read":
            gold_qa["old300"]["pinned_test_records_not_semantically_read"],
        },
        "safe_gold_after_protected_image_closure": {
            "new609_clean": sum(int(not row["marks"]) for row in new_rows),
            "new609_positive": sum(int(bool(row["marks"])) for row in new_rows),
            "new609_rows": len(new_rows),
            "old300_clean": sum(int(not row["marks"]) for row in old_rows),
            "old300_positive": sum(int(bool(row["marks"])) for row in old_rows),
            "old300_rows": len(old_rows),
        },
        "reference_sets_materialised": args.mode == "final" and not args.validate_only,
        "reference_sets_exclude_pinned_test_and_protected_images": True,
        "clean_selection": clean_selection,
        "catalogue_clean_reference_arm": "new609",
        "catalogue_clean_reference_fit_gold_clean": len(new_fit_gold_clean),
        "catalogue_clean_reference_fit_positive": len(new_fit_positive),
        "full_safe_fit_writer_catalogue_clean": len(fit_clean_candidates),
        "full_safe_fit_writer_catalogue_clean_sha256": full_clean_pool_hash,
        "hard_selection": hard_selection,
        "sampled_catalogue_clean": len(sampled_clean),
        "total_fit_clean": len(fit_clean),
        "validation_clean": len(validation_clean),
        "validation_positive": len(validation_positive),
        "validation_writer_count": len(validation_writers),
        "writer_overlap_fit_validation": 0,
    }
    return {
        "audit": audit,
        "base": base,
        "catalogue": catalogue,
        "fit_clean": fit_clean,
        "fit_clean_candidates": fit_clean_candidates,
        "fit_positive": fit_positive,
        "fold_lookup": fold_lookup,
        "held_clean_candidates": held_clean_candidates,
        "held_mask_rows": held_mask_rows,
        "held_revised_candidates": held_revised_candidates,
        "input_dependency_hashes": input_dependency_hashes,
        "segmenter": segmenter,
        "hard_selection": hard_selection,
        "validation_clean": validation_clean,
        "validation_positive": validation_positive,
    }


def validate_highres_source(source_dir: Path, seeds: list[int], *,
                            expected_pool_sha256: str,
                            validation_seed: int,
                            validation_fraction: float,
                            input_dependency_hashes: dict[str, str],
                            script_sha256: str) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise LocalizerError(f"highres source directory missing: {source_dir}")
    required = {
        name: source_dir / name for name in
        ("complete.json", "summary.json", "run_manifest.json",
         "split_manifest.jsonl")
    }
    if not all(path.is_file() for path in required.values()):
        raise LocalizerError("highres source is incomplete")
    try:
        complete = json.loads(required["complete.json"].read_text(encoding="utf-8"))
        summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))
        run_manifest = json.loads(
            required["run_manifest.json"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalizerError("highres source metadata is invalid") from exc
    if complete.get("status") != "complete":
        raise LocalizerError("highres source is not marked complete")
    split_manifest_sha256 = digest_file(required["split_manifest.jsonl"])
    protocol = summary.get("protocol", {})
    source_data = summary.get("data", {})
    checks = {
        "summary mode": summary.get("mode") == "dev",
        "summary arm": source_data.get("data_arm") == "new609",
        "summary recipe": protocol.get("recipe") == "highres",
        "summary seeds": protocol.get("seeds") == seeds,
        "summary checkpoint rule":
        protocol.get("checkpoint_rule") == "best_after_warmup",
        "summary split manifest":
        protocol.get("split_manifest_sha256") == split_manifest_sha256,
        "summary clean pool":
        protocol.get("full_safe_fit_writer_catalogue_clean_sha256")
        == expected_pool_sha256
        and source_data.get("full_safe_fit_writer_catalogue_clean_sha256")
        == expected_pool_sha256,
        "summary validation seed": protocol.get("validation_seed") == validation_seed,
        "summary validation fraction":
        protocol.get("validation_fraction") == validation_fraction,
        "run mode": run_manifest.get("mode") == "dev",
        "run arm": run_manifest.get("data_arm") == "new609",
        "run recipe": run_manifest.get("recipe") == "highres",
        "run checkpoint rule":
        run_manifest.get("checkpoint_rule") == "best_after_warmup",
        "M6 exclusion": run_manifest.get("m6_frozen_test_touched") is False,
        "reference exclusion":
        run_manifest.get("reference_evaluation_materialised") is False,
        "evaluation-writer exclusion":
        run_manifest.get("evaluation_fold_trained_on") is False,
        "run split manifest":
        run_manifest.get("split_manifest_sha256") == split_manifest_sha256,
        "run clean pool":
        run_manifest.get("full_safe_fit_writer_catalogue_clean_sha256")
        == expected_pool_sha256,
        "run validation seed": run_manifest.get("validation_seed") == validation_seed,
        "run validation fraction":
        run_manifest.get("validation_fraction") == validation_fraction,
        "run input dependencies":
        run_manifest.get("input_dependency_hashes") == input_dependency_hashes,
        "run script digest": run_manifest.get("script_sha256") == script_sha256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise LocalizerError(
            "highres source protocol mismatch: " + ", ".join(failed)
        )
    checkpoint_paths = {}
    checkpoint_hashes = {}
    for seed in seeds:
        checkpoint = source_dir / f"segmenter_seed{seed}.pt"
        if not checkpoint.is_file():
            raise LocalizerError(f"highres checkpoint missing for seed {seed}")
        checkpoint_paths[seed] = checkpoint
        checkpoint_hashes[str(seed)] = digest_file(checkpoint)
    return {
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_hashes,
        "complete_sha256": digest_file(required["complete.json"]),
        "full_safe_fit_writer_catalogue_clean_sha256": expected_pool_sha256,
        "input_dependency_hashes": dict(input_dependency_hashes),
        "run_manifest_sha256": digest_file(required["run_manifest.json"]),
        "script_sha256": script_sha256,
        "source_dir": str(source_dir),
        "split_manifest_sha256": split_manifest_sha256,
        "summary_sha256": digest_file(required["summary.json"]),
        "validation_fraction": validation_fraction,
        "validation_seed": validation_seed,
    }


def load_highres_model(checkpoint_path: Path, seed: int, segmenter,
                       base_channels: int, device):
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_architecture = {
        "base_channels": base_channels,
        "crop": list(RECIPES["highres"]["crop"]),
    }
    if not isinstance(checkpoint, dict) \
            or checkpoint.get("architecture") != expected_architecture \
            or checkpoint.get("data_arm") != "new609" \
            or checkpoint.get("recipe") != "highres" \
            or checkpoint.get("seed") != seed \
            or not isinstance(checkpoint.get("state_dict"), dict):
        raise LocalizerError(f"highres checkpoint metadata mismatch for seed {seed}")
    model = segmenter.make_unet(base_channels)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise LocalizerError(
            f"highres checkpoint weights mismatch for seed {seed}"
        ) from exc
    return model.to(device).eval()


def run_mine_hard(args: argparse.Namespace, prepared: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    seeds = parse_seeds(args.seeds)
    if len(seeds) != 3:
        raise LocalizerError("mine-hard requires exactly three highres checkpoints")
    source = validate_highres_source(
        Path(args.hard_source_dir), seeds,
        expected_pool_sha256=prepared["audit"][
            "full_safe_fit_writer_catalogue_clean_sha256"],
        validation_seed=args.validation_seed,
        validation_fraction=args.validation_fraction,
        input_dependency_hashes=prepared["input_dependency_hashes"],
        script_sha256=digest_file(Path(__file__).resolve()),
    )
    output = Path(args.output_dir).resolve()
    staging = atomic_staging(output)
    crop = RECIPES["highres"]["crop"]
    clean_rows, clipped = materialise(
        prepared["fit_clean_candidates"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], Path(args.crop_images_dir), crop,
        prepared["fold_lookup"],
    )
    if clipped != 0:
        raise LocalizerError("clean hard-negative pool unexpectedly clipped coordinates")
    pool_hash = clean_pool_sha256(clean_rows)
    if pool_hash != prepared["audit"][
            "full_safe_fit_writer_catalogue_clean_sha256"]:
        raise LocalizerError("materialised hard-negative pool identity changed")
    area_scale = (crop[0] * crop[1]) / float(512 * 64)
    top_pixels = max(1, int(round(args.min_fire_pixels * area_scale)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores_by_seed = {}
    for seed in seeds:
        model = load_highres_model(
            source["checkpoint_paths"][seed], seed, prepared["segmenter"],
            args.base_channels, device,
        )
        probability, _ = predict(model, clean_rows, args.batch_size, device)
        scores_by_seed[seed] = hardness_scores(probability, top_pixels)
        del probability, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "event": "hard_pool_scored", "pool_rows": len(clean_rows),
            "seed": seed,
        }, sort_keys=True), flush=True)
    ranking = rank_hard_negatives(clean_rows, scores_by_seed)
    ranking_path = staging / "hard_negative_ranking.jsonl"
    write_jsonl(ranking_path, ranking)
    ranking_hash = digest_file(ranking_path)
    hard_manifest = {
        "image_level_protected_closure_applied": True,
        "m6_frozen_test_touched": False,
        "pool_count": len(clean_rows),
        "pool_sha256": pool_hash,
        "protected_roles": ["fixed_development_only", "fixed_external_test_only"],
        "ranking_file": ranking_path.name,
        "ranking_sha256": ranking_hash,
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA_HARD_MANIFEST,
        "score_definition": {
            "ensemble": "arithmetic mean across source seeds",
            "per_checkpoint": "mean of highest predicted-probability pixels",
            "top_pixels": top_pixels,
        },
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_data_arm": "new609",
        "source_full_safe_fit_writer_catalogue_clean_sha256":
        source["full_safe_fit_writer_catalogue_clean_sha256"],
        "source_input_dependency_hashes": source["input_dependency_hashes"],
        "source_mode": "dev",
        "source_recipe": "highres",
        "source_run_manifest_sha256": source["run_manifest_sha256"],
        "source_script_sha256": source["script_sha256"],
        "source_seeds": seeds,
        "source_split_manifest_sha256": source["split_manifest_sha256"],
        "source_summary_sha256": source["summary_sha256"],
        "tie_break": "line_id ascending",
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
    }
    manifest_path = staging / "hard_negative_manifest.json"
    write_json(manifest_path, hard_manifest)
    summary = {
        "across_seeds": {},
        "data": prepared["audit"],
        "mode": "mine-hard",
        "protocol": {
            "pool_count": len(clean_rows),
            "pool_sha256": pool_hash,
            "ranking_sha256": ranking_hash,
            "score_top_pixels": top_pixels,
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_full_safe_fit_writer_catalogue_clean_sha256":
            source["full_safe_fit_writer_catalogue_clean_sha256"],
            "source_input_dependency_hashes": source["input_dependency_hashes"],
            "source_script_sha256": source["script_sha256"],
            "source_seeds": seeds,
            "source_split_manifest_sha256": source["split_manifest_sha256"],
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
        },
        "schema_version": SCHEMA_RESULT,
    }
    write_json(staging / "summary.json", summary)
    write_json(staging / "run_manifest.json", {
        "command": [sys.executable, *sys.argv],
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_arm": "new609",
        "evaluation_fold_trained_on": False,
        "git_revision": git_revision(),
        "gold_confirmation": bool(args.gold_confirmed),
        "hard_negative_manifest_sha256": digest_file(manifest_path),
        "image_level_protected_closure_applied": True,
        "input_dependency_hashes": prepared["input_dependency_hashes"],
        "input_hashes": {
            "new_key": NEW_KEY_SHA256, "new_returned": NEW_RETURNED_SHA256,
            "old_key": OLD_KEY_SHA256, "old_returned": OLD_RETURNED_SHA256,
            "source_checkpoints": source["checkpoint_sha256"],
        },
        "m6_frozen_test_touched": False,
        "mode": "mine-hard",
        "network_used": False,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "recipe": "highres",
        "reference_evaluation_materialised": False,
        "schema_version": SCHEMA_RESULT,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "source_full_safe_fit_writer_catalogue_clean_sha256":
        source["full_safe_fit_writer_catalogue_clean_sha256"],
        "source_script_sha256": source["script_sha256"],
        "source_split_manifest_sha256": source["split_manifest_sha256"],
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
    })
    write_json(staging / "complete.json", {
        "schema_version": SCHEMA_RESULT, "status": "complete",
    })
    write_recursive_sha256s(staging)
    staging.rename(output)
    return summary


def aggregate(results: list[dict], path: tuple[str, ...]) -> dict[str, float]:
    values = []
    for result in results:
        current: Any = result
        for key in path:
            current = current[key]
        values.append(float(current))
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def accumulate_probability_bundle(totals: dict[str, Any], bundle: dict[str, Any],
                                  required_keys: tuple[str, ...]) -> None:
    import numpy as np

    required = set(required_keys)
    if set(bundle) != required:
        missing = sorted(required - set(bundle))
        unexpected = sorted(set(bundle) - required)
        raise LocalizerError(
            "prediction bundle keys changed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in required_keys:
        probability = np.asarray(bundle[name])
        if probability.ndim != 3 or not np.issubdtype(
                probability.dtype, np.floating):
            raise LocalizerError(
                f"prediction bundle {name} must be a floating N x H x W array"
            )
        if not np.isfinite(probability).all() \
                or bool((probability < 0).any()) \
                or bool((probability > 1).any()):
            raise LocalizerError(
                f"prediction bundle {name} contains invalid probabilities"
            )
        if name not in totals:
            totals[name] = probability.astype(np.float64, copy=True)
        elif totals[name].shape != probability.shape:
            raise LocalizerError(
                f"prediction bundle {name} shape changed: "
                f"{probability.shape} != {totals[name].shape}"
            )
        else:
            totals[name] += probability


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    if args.mode in ("dev", "mine-hard") and args.data_arm != "new609":
        raise LocalizerError("development and mining must use the new609 arm")
    recipe_lock_path = getattr(args, "recipe_lock", None)
    if args.mode == "final" and recipe_lock_path is None:
        raise LocalizerError("final evaluation requires --recipe-lock FILE")
    if args.mode != "final" and recipe_lock_path is not None:
        raise LocalizerError("--recipe-lock is only valid for final evaluation")
    if args.mode == "mine-hard":
        if args.recipe != "highres":
            raise LocalizerError("mine-hard must use the completed highres recipe")
        if getattr(args, "hard_source_dir", None) is None:
            raise LocalizerError("mine-hard requires --hard-source-dir")
    elif getattr(args, "hard_source_dir", None) is not None:
        raise LocalizerError("--hard-source-dir is only valid for mine-hard")
    if RECIPES[getattr(args, "recipe", "baseline")]["hard_fraction"] \
            and getattr(args, "hard_negative_manifest", None) is None:
        raise LocalizerError("hard/optimized require --hard-negative-manifest")
    if not RECIPES[getattr(args, "recipe", "baseline")]["hard_fraction"] \
            and getattr(args, "hard_negative_manifest", None) is not None:
        raise LocalizerError("only hard/optimized may consume a hard-negative manifest")
    if args.optimizer_steps <= 0 or args.validation_interval <= 0:
        raise LocalizerError("optimizer steps and validation interval must be positive")
    if not 0.0 < args.selection_warmup_fraction <= 1.0:
        raise LocalizerError("selection warmup fraction must be in (0, 1]")
    seeds = parse_seeds(args.seeds)
    prepared = prepare(args)
    recipe_lock = None
    if args.mode == "final":
        recipe_lock = validate_recipe_lock(
            Path(recipe_lock_path), args, prepared, seeds
        )
    if args.validate_only:
        if args.mode == "mine-hard":
            if len(seeds) != 3:
                raise LocalizerError(
                    "mine-hard requires exactly three highres checkpoints"
                )
            validate_highres_source(
                Path(args.hard_source_dir), seeds,
                expected_pool_sha256=prepared["audit"][
                    "full_safe_fit_writer_catalogue_clean_sha256"],
                validation_seed=args.validation_seed,
                validation_fraction=args.validation_fraction,
                input_dependency_hashes=prepared["input_dependency_hashes"],
                script_sha256=digest_file(Path(__file__).resolve()),
            )
        print(json.dumps(prepared["audit"], ensure_ascii=False, sort_keys=True))
        return {"schema_version": SCHEMA_RESULT, "status": "validated",
                **prepared["audit"]}
    if args.mode == "mine-hard":
        return run_mine_hard(args, prepared)
    output = Path(args.output_dir).resolve()
    staging = atomic_staging(output)
    recipe = RECIPES[args.recipe]
    images_dir = Path(args.crop_images_dir)
    fit_positive, clipped_positive = materialise(
        prepared["fit_positive"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    fit_clean, clipped_clean = materialise(
        prepared["fit_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    validation_positive, clipped_validation_positive = materialise(
        prepared["validation_positive"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
    validation_clean, clipped_validation_clean = materialise(
        prepared["validation_clean"], prepared["segmenter"], prepared["base"],
        prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])

    held = None
    if args.mode == "final":
        held_mask, clipped_held = materialise(
            prepared["held_mask_rows"], prepared["segmenter"], prepared["base"],
            prepared["catalogue"], images_dir, recipe["crop"], prepared["fold_lookup"])
        held = {
            "mask": held_mask,
            "clean": cut_catalogue(
                prepared["held_clean_candidates"], prepared["base"], images_dir,
                recipe["crop"]),
            "revised": cut_catalogue(
                prepared["held_revised_candidates"], prepared["base"], images_dir,
                recipe["crop"]),
        }
    else:
        clipped_held = 0

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
    split_manifest_path = staging / "split_manifest.jsonl"
    write_jsonl(split_manifest_path, sorted(
        split_rows, key=lambda row: (row["split"], row["line_id"])))
    split_manifest_sha256 = digest_file(split_manifest_path)

    required_prediction_keys = (
        "validation_positive", "validation_clean",
    )
    if args.mode == "final":
        required_prediction_keys += (
            "held_mask", "held_clean", "held_revised",
        )
    results: list[dict[str, Any]] = []
    probability_totals: dict[str, Any] = {}
    for seed in seeds:
        result, prediction_bundle = train_once(
            args.data_arm, args.recipe, fit_positive, fit_clean,
            validation_positive, validation_clean, held, prepared["hard_selection"],
            prepared["segmenter"], args, seed, staging)
        results.append(result)
        accumulate_probability_bundle(
            probability_totals, prediction_bundle, required_prediction_keys
        )
        del prediction_bundle

    mean_probabilities = {
        name: total / len(seeds) for name, total in probability_totals.items()
    }
    validation_target = np.stack(
        [row["mask"] for row in validation_positive]
    ).astype(np.float32) / 255.0
    effective_min_fire_pixels = scaled_min_fire_pixels(
        args.recipe, args.min_fire_pixels
    )
    ensemble_operating_points = choose_operating_points(
        mean_probabilities["validation_positive"], validation_target,
        mean_probabilities["validation_clean"], args.max_clean_fp,
        effective_min_fire_pixels,
    )

    summary: dict[str, Any] = {
        "across_seeds": {
            "validation_localization_iou": aggregate(
                results, ("validation", "localization", "iou")),
        },
        "data": prepared["audit"],
        "ensemble_operating_points": ensemble_operating_points,
        "mode": args.mode,
        "per_seed": results,
        "protocol": {
            "batch_size": args.batch_size,
            "fixed_optimizer_steps": args.optimizer_steps,
            "negative_ratio": args.negative_ratio,
            "clean_selection": prepared["audit"]["clean_selection"],
            "hard_selection": prepared["hard_selection"],
            "recipe": args.recipe,
            "recipe_config": recipe,
            "seeds": seeds,
            "validation_interval": args.validation_interval,
            "checkpoint_selection_starts_at_step": int(round(
                args.optimizer_steps * args.selection_warmup_fraction)),
            "checkpoint_rule": args.checkpoint_rule,
            "recipe_lock_sha256": None if recipe_lock is None
            else recipe_lock["sha256"],
            "full_safe_fit_writer_catalogue_clean_sha256":
            prepared["audit"]["full_safe_fit_writer_catalogue_clean_sha256"],
            "ensemble_definition":
            "arithmetic mean probabilities across selected seed checkpoints",
            "ensemble_threshold_chosen_on":
            "common writer-disjoint validation mean probabilities",
            "split_manifest_sha256": split_manifest_sha256,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
        },
        "schema_version": SCHEMA_RESULT,
    }
    if args.mode == "final":
        for point in ("localization", "compatibility"):
            summary["across_seeds"][f"held_{point}_iou"] = aggregate(
                results, ("held_out", point, "mask", "iou"))
            summary["across_seeds"][f"held_{point}_precision"] = aggregate(
                results, ("held_out", point, "mask", "precision"))
            summary["across_seeds"][f"held_{point}_recall"] = aggregate(
                results, ("held_out", point, "mask", "recall"))
            summary["across_seeds"][f"held_{point}_clean_fp"] = aggregate(
                results, ("held_out", point, "clean_false_alarm"))
            summary["across_seeds"][f"held_{point}_line_recall"] = aggregate(
                results, ("held_out", point, "revised_line_recall"))
        if held is None:
            raise LocalizerError("final mode lost its held reference materialisation")
        mean_probability = mean_probabilities["held_mask"]
        held_target = np.stack(
            [row["mask"] for row in held["mask"]]
        ).astype(np.float32) / 255.0
        boundary_tolerance = max(
            1, int(math.ceil(3.0 * recipe["crop"][1] / 64.0))
        )
        ensemble_held_out = {}
        for point_name, point in ensemble_operating_points.items():
            threshold = point["threshold"]
            clean_fp = line_fire_rate(
                mean_probabilities["held_clean"], threshold,
                effective_min_fire_pixels,
            )
            revised_recall = line_fire_rate(
                mean_probabilities["held_revised"], threshold,
                effective_min_fire_pixels,
            )
            ensemble_held_out[point_name] = {
                "boundary_f1": boundary_f1(
                    mean_probability, held_target, threshold,
                    tolerance=boundary_tolerance,
                ),
                "boundary_tolerance_pixels": boundary_tolerance,
                "clean_false_alarm": clean_fp,
                "line_f1": line_f1(
                    revised_recall, clean_fp,
                    len(held["revised"]), len(held["clean"]),
                ),
                "mask": pixel_metrics(
                    mean_probability, held_target, threshold
                ),
                "revised_line_recall": revised_recall,
                "threshold": threshold,
            }
        summary["ensemble_held_out"] = ensemble_held_out
        np.savez_compressed(
            staging / "held_predictions_mean.npz",
            images=np.stack([row["image"] for row in held["mask"]]).astype(np.uint8),
            probabilities=mean_probability.astype(np.float16),
            targets=np.stack([row["mask"] for row in held["mask"]]).astype(np.uint8),
        )
        write_json(staging / "held_review_key.json", [{
            "image_id": row["image_id"], "line_id": row["line_id"],
            "source": row["source"], "types": row["types"],
        } for row in held["mask"]])
    write_json(staging / "summary.json", summary)
    write_json(staging / "run_manifest.json", {
        "clipped_coordinate_values": clipped_positive + clipped_clean
        + clipped_validation_positive + clipped_validation_clean + clipped_held,
        "command": [sys.executable, *sys.argv],
        "checkpoint_rule": args.checkpoint_rule,
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_arm": args.data_arm,
        "evaluation_fold_trained_on": False,
        "m6_frozen_test_touched": False,
        "reference_evaluation_materialised": args.mode == "final",
        "reference_evaluation_previously_used_by_experiments_348_350": True,
        "git_revision": git_revision(),
        "gold_confirmation": bool(args.gold_confirmed),
        "full_safe_fit_writer_catalogue_clean_sha256":
        prepared["audit"]["full_safe_fit_writer_catalogue_clean_sha256"],
        "input_dependency_hashes": prepared["input_dependency_hashes"],
        "input_hashes": {
            "new_key": NEW_KEY_SHA256, "new_returned": NEW_RETURNED_SHA256,
            "old_key": OLD_KEY_SHA256, "old_returned": OLD_RETURNED_SHA256,
            "hard_negative_manifest": None if prepared["hard_selection"] is None
            else prepared["hard_selection"]["manifest_sha256"],
            "hard_negative_ranking": None if prepared["hard_selection"] is None
            else prepared["hard_selection"]["ranking_sha256"],
            "recipe_lock": None if recipe_lock is None else recipe_lock["sha256"],
        },
        "mode": args.mode,
        "network_used": False,
        "optimizer_steps_equal_across_arms": True,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "recipe": args.recipe,
        "recipe_lock_path": None if recipe_lock is None else recipe_lock["path"],
        "recipe_lock_sha256": None if recipe_lock is None else recipe_lock["sha256"],
        "recipe_locked_before_reference_evaluation": recipe_lock is not None,
        "schema_version": SCHEMA_RESULT,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "split_manifest_sha256": split_manifest_sha256,
        "threshold_chosen_on":
        "common writer-disjoint validation mean probabilities for ensemble; "
        "per-seed validation probabilities for each checkpoint",
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
    })
    write_json(staging / "complete.json", {
        "schema_version": SCHEMA_RESULT, "status": "complete"})
    write_recursive_sha256s(staging)
    staging.rename(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--crop-images-dir", required=True, type=Path)
    parser.add_argument("--fold-images-dir", type=Path, default=None)
    parser.add_argument("--new-key", type=Path, default=DEFAULT_NEW_KEY)
    parser.add_argument("--new-returned", required=True, type=Path)
    parser.add_argument("--old-key", type=Path, default=DEFAULT_OLD_KEY)
    parser.add_argument("--old-returned", type=Path, default=DEFAULT_OLD_RETURNED)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("dev", "mine-hard", "final"),
                        required=True)
    parser.add_argument("--data-arm", choices=("new609", "combined"), required=True)
    parser.add_argument("--recipe", choices=tuple(RECIPES), required=True)
    parser.add_argument("--hard-source-dir", type=Path, default=None)
    parser.add_argument("--hard-negative-manifest", type=Path, default=None)
    parser.add_argument("--recipe-lock", type=Path, default=None)
    parser.add_argument("--gold-confirmed", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--negative-seed", type=int, default=354)
    parser.add_argument("--held-out-seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-seed", type=int, default=354)
    parser.add_argument("--max-clean-fp", type=float, default=0.15)
    parser.add_argument("--min-fire-pixels", type=int, default=24)
    parser.add_argument("--optimizer-steps", type=int, default=18_000)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--selection-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--checkpoint-rule", choices=("last", "best_after_warmup"),
                        default="last")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--seeds", default="354,1354,2354")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except LocalizerError as exc:
        raise SystemExit(f"experiment 354 refused: {exc}") from exc
    if not args.validate_only:
        print(json.dumps({
            "across_seeds": summary["across_seeds"],
            "data": summary["data"],
            "mode": summary["mode"],
            "protocol": summary["protocol"],
        }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

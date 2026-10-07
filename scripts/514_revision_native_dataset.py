#!/usr/bin/env python3
"""Export the reviewed 445 fit/dev rows for native instance baselines.

The exporter calls the existing preparation code under --legacy-code-root,
applies experiment 500's correction once to the uncorrected positive masks,
and writes a create-only, hashed bundle. No training or frozen test access.
The importable load_bundle(path) API needs only NumPy and SciPy, not legacy
assets or torch. Images are uint8 RGB CHW (grayscale copied to three channels).
Masks are bool, and instances are 8-connected components of corrected masks.

Pass ordinary 445 arguments after this script's --legacy-code-root and
--output-dir options. --arm defaults to instance for the preparation call.
Alternatively --legacy-args-json accepts a JSON array of those argument words.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hme-revision-native-dataset-v1"
IMAGE_SHAPE = (3, 96, 768)
MASK_SHAPE = (96, 768)
COUNTS = {"fit_positive": 1515, "fit_clean": 1319,
          "validation_positive": 126, "validation_clean": 379}
DEV_COMPONENTS = 221
SOURCE_PINS = {
    "491_revision_query_memory_resolution_seed42.py": "2187508895edda0855c11fbf459f896711ad025e75b019c8234f9952a1ee6c96",
    "445_revision_joint_head_probe.py": "a5f140681abe1fae6d92b029f1cc9971ce018b40ff42348450766ef91277e986",
    "422_revision_paper_baselines.py": "a566110f4a629ba038c97b895ee0d442f9d445cadb7fa961f048175cddb2a8b9",
    "418_revision_typepilot300_factorial_seed42.py": "ad88c2542a11a45c2770ddd7f589dab2eb9d8b8fe0a987cca445d272b189275f",
    "500_revision_gold_pen_width_reeval.py": "626feedbe59c8835986aa54c86fb64fb7ce4129a3bed212ffa7e4e8dfc7f5534",
}


class DatasetError(RuntimeError):
    """The declared fit/dev bundle contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(path: Path) -> Path:
    expanded = path.expanduser()
    for candidate in (expanded.absolute(), expanded.resolve()):
        for part in candidate.parts:
            if re.search(r"frozen|holdout|冻结|封存|(?:^|[_-])m6(?:$|[_-])", part, re.I):
                raise DatasetError(f"frozen/M6 path refused: {path}")
    return expanded.resolve()


def instances_from_mask(mask: Any) -> Any:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return np.zeros((0, *mask.shape), dtype=bool)
    return np.stack([labels == index for index in range(1, count + 1)])


def normalize_row(row: dict[str, Any], positive: bool,
                  correct_mask: Callable[[Any], Any]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    image = np.asarray(row["image"])
    mask = np.asarray(row["mask"])
    require(image.dtype == np.uint8 and image.shape == MASK_SHAPE,
            "legacy image must already be uint8 grayscale 96x768; exporter never resizes")
    require(mask.dtype == np.uint8 and mask.shape == MASK_SHAPE
            and np.isin(mask, [0, 255]).all(), "legacy mask must be binary uint8 96x768")
    require(bool(mask.any()) == positive, "positive/clean membership disagrees with legacy mask")
    require(not row.get("native_514_corrected", False), "refusing to correct a mask twice")
    corrected = np.asarray(correct_mask(mask.copy()), dtype=bool) if positive else np.zeros(MASK_SHAPE, dtype=bool)
    require(corrected.shape == MASK_SHAPE and bool(corrected.any()) == positive,
            "mask corrector changed shape or foreground membership")
    group_uuid, image_id, line_id = (str(row[name]) for name in ("group_uuid", "image_id", "line_id"))
    require(all(value and value != "None" for value in (group_uuid, image_id, line_id)),
            "missing UUID/line identifiers")
    # Existing UUIDs are retained, with no raw identity mapping copied.
    sample_id = f"{image_id}:{line_id}"
    metadata = {"sample_id": sample_id, "group_uuid": group_uuid,
                "image_id": image_id, "line_id": line_id,
                "is_positive": positive, "native_514_corrected": True}
    return np.repeat(image[None], 3, axis=0), corrected, metadata


def write_bundle(data: dict[str, Any], output_dir: Path,
                 correct_mask: Callable[[Any], Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Export uncorrected legacy rows; source arrays/dictionaries stay unchanged."""
    import numpy as np

    output = safe_path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), f"create-only output exists: {output_dir}")
    require(not any(key in data for key in ("corrected_gold_rows", "corrected_gold_components")),
            "prepared data already carries gold-correction markers")
    require(all(len(data.get(name, [])) == count for name, count in COUNTS.items()),
            "fit/dev row counts differ from the locked 1515/1319/126/379 split")
    ordered = [(name, row) for name in COUNTS for row in data[name]]
    groups = {split: {str(row["group_uuid"]) for name, row in ordered
                      if (name.startswith("fit_") if split == "train" else name.startswith("validation_"))}
              for split in ("train", "dev")}
    require(groups["train"] and groups["dev"] and not groups["train"] & groups["dev"],
            "writer overlap or empty train/dev split")
    sample_ids = [f"{row['image_id']}:{row['line_id']}" for _, row in ordered]
    require(len(sample_ids) == len(set(sample_ids)), "duplicate sample identifiers")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        images = np.lib.format.open_memmap(staging / "images.npy", mode="w+", dtype=np.uint8,
                                          shape=(len(ordered), *IMAGE_SHAPE))
        masks = np.lib.format.open_memmap(staging / "masks.npy", mode="w+", dtype=np.bool_,
                                         shape=(len(ordered), *MASK_SHAPE))
        records, dev_components, train_components = [], 0, 0
        train_sum = np.zeros(3, dtype=np.float64)
        train_square_sum = np.zeros(3, dtype=np.float64)
        train_pixels = 0
        for index, (name, row) in enumerate(ordered):
            positive = name.endswith("positive")
            image, mask, metadata = normalize_row(row, positive, correct_mask)
            count = len(instances_from_mask(mask))
            split = "train" if name.startswith("fit_") else "dev"
            images[index], masks[index] = image, mask
            records.append({**metadata, "index": index, "split": split,
                            "legacy_split": name, "instance_count": count})
            if split == "dev":
                dev_components += count
            else:
                train_components += count
                train_sum += image.sum(axis=(1, 2), dtype=np.float64)
                train_square_sum += np.square(image, dtype=np.float64).sum(axis=(1, 2))
                train_pixels += image.shape[1] * image.shape[2]
        require(dev_components == DEV_COMPONENTS,
                f"corrected development CC count {dev_components} != {DEV_COMPONENTS}; no second correction allowed")
        images.flush()
        masks.flush()
        del images, masks
        mean = train_sum / train_pixels
        std = np.sqrt(np.maximum(0, train_square_sum / train_pixels - mean ** 2))
        (staging / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "image_layout": "NCHW", "image_dtype": "uint8", "image_shape": list(IMAGE_SHAPE),
            "mask_dtype": "bool", "mask_shape": list(MASK_SHAPE),
            "counts": dict(COUNTS), "writer_overlap_train_dev": 0,
            "corrected_dev_components": dev_components,
            "gold_correction": {"source": "500.correct_mask", "applications_per_positive": 1,
                                "connectivity": 8, "input": "uncorrected legacy masks"},
            "train_statistics": {"instances": train_components, "pixels_per_channel": train_pixels,
                                 "rgb_mean_0_255": mean.tolist(), "rgb_std_0_255": std.tolist()},
            "training_performed": False, "frozen_test_touched": False,
            "create_only": True, "provenance": provenance,
            "files": {name: {"sha256": digest(staging / name), "bytes": (staging / name).stat().st_size}
                      for name in ("images.npy", "masks.npy", "records.json")},
        }
        (staging / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        require(not output.exists() and not output.is_symlink(), "output appeared during export")
        staging.rename(output)
        return manifest
    except BaseException:
        # Only the directory created by this call is removed; original evidence stays intact.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_bundle(path: Path | str) -> dict[str, Any]:
    """Return train/dev lists of record dictionaries with mmap image/mask views.

    Required record keys: sample_id, group_uuid, image, mask, instances,
    is_positive. Arrays are read-only. Callers must copy before augmentation.
    Instance masks are reconstructed on CPU; no model/data-loader is imported.
    """
    import numpy as np

    root = safe_path(Path(path))
    manifest_path = root / "run_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "bundle manifest absent/symlinked")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == SCHEMA, "unsupported bundle schema")
    require(manifest.get("frozen_test_touched") is False
            and manifest.get("writer_overlap_train_dev") == 0, "unsafe bundle provenance")
    require(manifest.get("counts") == COUNTS and manifest.get("corrected_dev_components") == DEV_COMPONENTS,
            "bundle count contract changed")
    require(manifest.get("gold_correction", {}).get("applications_per_positive") == 1
            and manifest.get("gold_correction", {}).get("connectivity") == 8,
            "gold correction contract changed")
    for name in ("images.npy", "masks.npy", "records.json"):
        item = root / name
        require(item.is_file() and not item.is_symlink() and safe_path(item).is_relative_to(root),
                f"missing/symlinked bundle file: {name}")
        expected = manifest["files"][name]
        require(item.stat().st_size == expected["bytes"] and digest(item) == expected["sha256"],
                f"bundle integrity mismatch: {name}")
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    images = np.load(root / "images.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)
    count = sum(COUNTS.values())
    require(images.shape == (count, *IMAGE_SHAPE) and images.dtype == np.uint8,
            "unexpected bundle image array")
    require(masks.shape == (count, *MASK_SHAPE) and masks.dtype == np.bool_,
            "unexpected bundle mask array")
    require(isinstance(records, list) and len(records) == count, "invalid bundle record count")
    result: dict[str, Any] = {"schema_version": SCHEMA, "train": [], "dev": [], "manifest": manifest}
    seen, groups = set(), {"train": set(), "dev": set()}
    observed_counts = {key: 0 for key in COUNTS}
    dev_components = 0
    for index, record in enumerate(records):
        require(type(record.get("index")) is int and record["index"] == index, "bundle index/order mismatch")
        name = record["legacy_split"]
        require(name in COUNTS, "undeclared/frozen split")
        split = "train" if name.startswith("fit_") else "dev"
        require(record["split"] == split and type(record["is_positive"]) is bool
                and record["is_positive"] == name.endswith("positive"), "bundle split/label mismatch")
        require(record["sample_id"] not in seen, "duplicate bundle sample ID")
        seen.add(record["sample_id"])
        groups[split].add(record["group_uuid"])
        observed_counts[name] += 1
        mask = masks[index]
        require(bool(mask.any()) == record["is_positive"], "bundle foreground/label mismatch")
        instances = instances_from_mask(mask)
        require(len(instances) == record["instance_count"], "instance count differs from archived mask")
        if split == "dev":
            dev_components += len(instances)
        instances.flags.writeable = False
        result[split].append({**record, "image": images[index], "mask": mask, "instances": instances})
    require(observed_counts == COUNTS and dev_components == DEV_COMPONENTS, "loaded bundle counts changed")
    require(not groups["train"] & groups["dev"], "loaded bundle writer overlap")
    return result


def load_legacy(code_root: Path) -> tuple[Any, Any, Any, Any]:
    code = safe_path(code_root)
    for filename, expected in SOURCE_PINS.items():
        path = safe_path(code / "scripts" / filename)
        require(path.is_file() and digest(path) == expected, f"legacy source drift: {filename}")
    path = code / "scripts/491_revision_query_memory_resolution_seed42.py"
    spec = importlib.util.spec_from_file_location("hme491_for514", path)
    require(spec is not None and spec.loader is not None, "cannot import legacy 491")
    memory = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = memory
    spec.loader.exec_module(memory)
    joint = memory.load_module(memory.JOINT_PATH, "hme445_for514")
    baseline = memory.load_module(memory.BASELINE_PATH, "hme422_for514")
    typepilot = memory.load_module(memory.TYPEPILOT_PATH, "hme418_for514")
    gold = memory.load_module(code / "scripts/500_revision_gold_pen_width_reeval.py", "hme500_for514")
    return joint, baseline, typepilot, gold


def export_legacy(code_root: Path, output_dir: Path, legacy_arguments: Sequence[str]) -> dict[str, Any]:
    code, output = safe_path(code_root), safe_path(output_dir)
    require(not output.exists() and not output_dir.is_symlink(), "create-only output already exists")
    require(not output.is_relative_to(code), "runtime bundle must live outside legacy source tree")
    # Reject explicit forbidden paths before importing the legacy preparation chain.
    for argument in legacy_arguments:
        if argument.startswith("/") or argument.startswith("~/"):
            safe_path(Path(argument))
    joint, baseline, typepilot, gold = load_legacy(code)
    parser = joint.build_parser(baseline, typepilot)
    words = list(legacy_arguments)
    require(not any(value == "--output-dir" or value.startswith("--output-dir=") for value in words),
            "use exporter --output-dir; legacy arguments must not replace it")
    if not any(value == "--arm" or value.startswith("--arm=") for value in words):
        words += ["--arm", "instance"]
    args = parser.parse_args([*words, "--output-dir", str(output)])
    for value in vars(args).values():
        if isinstance(value, Path):
            safe_path(value)
    require(args.seed == 42, "data export is seed-independent; keep preparation seed42")
    data = joint.prepare(baseline, typepilot, args)
    # collect only modules actually loaded under the declared legacy checkout.
    source_files = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(code):
                source_files[str(path)] = digest(path)
    input_files = {}
    for key, value in vars(args).items():
        if not isinstance(value, Path) or key.endswith("weights") or key == "output_dir":
            continue
        path = safe_path(value)
        candidates = [path] if path.is_file() else [path / name for name in ("run_manifest.json", "summary.json")]
        for candidate in candidates:
            if candidate.is_file():
                input_files[str(candidate)] = digest(candidate)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    provenance = {"exporter_source": str(Path(__file__).resolve()),
                  "exporter_sha256": digest(Path(__file__).resolve()), "exporter_git_revision": commit.stdout.strip() or None,
                  "legacy_code_root": str(code), "legacy_source_sha256": source_files,
                  "explicit_input_metadata_sha256": input_files,
                  "legacy_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "environment": {"python": platform.python_version(), "platform": platform.platform()},
                  "image_transform": "existing legacy letterbox 96x768 unchanged; grayscale replicated to RGB; no resize",
                  "normalization_statistics_scope": "train pixels only; no dev mean/std computed"}
    return write_bundle(data, output, gold.correct_mask, provenance)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--legacy-code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-args-json", type=Path,
                        help="optional JSON array of legacy argv words; alternative to trailing arguments")
    args, trailing = parser.parse_known_args(argv)
    try:
        if trailing[:1] == ["--"]:
            trailing = trailing[1:]
        if args.legacy_args_json:
            require(not trailing, "choose either legacy-args-json or trailing legacy arguments")
            path = safe_path(args.legacy_args_json)
            trailing = json.loads(path.read_text(encoding="utf-8"))
            require(isinstance(trailing, list) and all(isinstance(word, str) for word in trailing),
                    "legacy-args-json must contain a JSON array of strings")
        manifest = export_legacy(args.legacy_code_root, args.output_dir, trailing)
        print(json.dumps({"output_dir": str(args.output_dir), "schema_version": SCHEMA,
                          "counts": manifest["counts"], "corrected_dev_components": manifest["corrected_dev_components"]}, ensure_ascii=False))
        return 0
    except (DatasetError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"514 dataset refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

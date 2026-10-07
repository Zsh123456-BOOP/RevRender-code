#!/usr/bin/env python3
"""Build the region-marking page for the *frozen evaluation fold*.

Experiment 324 annotated 300 revised lines drawn from the fold-0 **train**
split, and the returned work settled what real revision ink looks like: the
median struck segment spans 14% of the line, only 8.8% of segments run the full
width, and a line carries 2.0 marks on average. Every synthetic source built so
far draws one full-width, near-horizontal rule through the middle of the line --
a mode that accounts for 8.8% of real segments. That is why 313/315, 314 and 316
all sit near chance and why the learned generator of 318 stalls at 0.618.

Those 300 lines are fitting material. They cannot measure whether a fix works,
because they are train rows: scoring a fold-0-trained model on them is scoring
on its own training data. A localisation claim needs ground truth on the frozen
evaluation fold, and that is what this package collects.

324's docstring says it deliberately avoids the frozen evaluation fold. This
script reverses that for the opposite purpose, and the reversal is the point:
drawing *fitting* material from the evaluation fold would contaminate it, while
drawing *ground truth* is what makes the fold scoreable at all. The protections
that keep the reversal honest are enforced below, not merely asserted:

  * every writer that appears as ``split == test`` in any pinned split table is
    removed before sampling, so no FTV test row can reach the annotator;
  * the page never shows an existing label, so the annotator cannot copy one;
  * the returned marks are ground truth for scoring only. They must never be
    fed to a generator fit, a hyper-parameter choice, or a gate threshold --
    that is what experiment 324's output is for, and mixing the two would turn
    the evaluation fold into training data by the back door.

The page, the three tools and the two escapes are taken unchanged from 324 by
import, so the annotator meets an identical interface and the returned JSON
parses with the same reader.

This script deliberately does NOT:
  * run a model, use a GPU, reach the network, or call a paid API;
  * draw a policy-assigned label -- only human-backed positives;
  * touch the FTV test split or the external blind set;
  * show the annotator any existing label;
  * sample -- it takes every eligible row, so the fold is covered exactly once;
  * overwrite anything -- the output directory is create-only.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REGION_SCRIPT = ROOT / "scripts" / "324_revision_region_annotation_package.py"
AUTHORITATIVE_REGION_SHA256 = (
    "da5fd32a040a834f64cda3fdb1106329a1c3c4e5c12c724c34d4c022b58dc442"
)

# Every split table that names an FTV/line-edit test row. A writer listed as
# test in any one of them is removed, whichever column carries the identifier.
PINNED_SPLIT_TABLES = {
    "outputs/ftv_v1/m0/split_assignments.csv":
        "fb2a1e793227f44123ca1576a62256600af83eb9fcf15fba8f3dbebbb9e22440",
    "outputs/ftv_v1/m0_v2/split_assignments.csv":
        "fb2a1e793227f44123ca1576a62256600af83eb9fcf15fba8f3dbebbb9e22440",
    "outputs/line_edit_v1/m0/split_assignments.csv":
        "d0151afdc962e2a09e8af9d857cd510f0409bbc7f4e79c8a6489a89bfd350c85",
    "outputs/stage1_3_server_artifacts_v1/g0/split_assignments.csv":
        "10379a463f1959e1b82e268634d1b27ae7273f8de0e5e65d70578e6b2b31b737",
    "outputs/stage3_g0_full755_v2/split_assignments.csv":
        "352cb91ada3a758469e2a3555730e9360c9c57e7783ab63f3928b0f38889d0ac",
}

# Fail-closed arithmetic. If the pinned fold or the pinned split tables ever
# move, these stop the run rather than quietly shipping a different package.
EXPECTED_EVAL_ROWS = 329
EXPECTED_EVAL_POSITIVES = 118
EXPECTED_EXCLUDED_WRITERS = 1
EXPECTED_ITEMS = 117

SAMPLE_NAMESPACE = "hme-region-eval-v1"
SCHEMA_PACKAGE = "hme-revision-region-annotation-v1"


class EvalRegionPackageError(RuntimeError):
    """Raised when a pinned input or a leakage guard refuses the run."""


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
        raise EvalRegionPackageError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_region_package():
    """Import 324 so the page, the tools and the escapes are shared, not copied."""
    if not REGION_SCRIPT.is_file():
        raise EvalRegionPackageError(f"missing pinned experiment 324: {REGION_SCRIPT}")
    if digest_file(REGION_SCRIPT) != AUTHORITATIVE_REGION_SHA256:
        raise EvalRegionPackageError("experiment 324 digest mismatch")
    spec = importlib.util.spec_from_file_location("hme_region324_for325", REGION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_test_identifiers() -> set[str]:
    """Every identifier a pinned split table marks as test, in any column."""
    banned: set[str] = set()
    for relative, expected in sorted(PINNED_SPLIT_TABLES.items()):
        path = ROOT / relative
        if not path.is_file():
            raise EvalRegionPackageError(f"missing pinned split table: {relative}")
        if digest_file(path) != expected:
            raise EvalRegionPackageError(f"split table digest mismatch: {relative}")
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "split" not in rows[0]:
            raise EvalRegionPackageError(f"split table carries no split column: {relative}")
        for row in rows:
            if row.get("split") != "test":
                continue
            for column, value in row.items():
                if column == "split" or not value:
                    continue
                banned.add(value)
                # image ids are "<writer-uuid>_<page>_<n>"; ban the writer too.
                banned.add(value.split("_")[0])
    if not banned:
        raise EvalRegionPackageError("no test identifiers found -- guard would be vacuous")
    return banned


def item_key(line_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_NAMESPACE}\x1f{line_id}".encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    output_dir = Path(args.output_dir).resolve()
    region = load_region_package()
    base = region.load_matched().load_gap().load_base()
    images_dir = Path(args.images_dir)

    try:
        _train_rows, eval_rows = base._validate_training_inputs(
            Path(args.input_root), images_dir
        )
    except base.StrongVisual250Error as exc:
        raise EvalRegionPackageError(f"pinned fold validation failed: {exc}") from exc

    if len(eval_rows) != EXPECTED_EVAL_ROWS:
        raise EvalRegionPackageError(
            f"evaluation fold holds {len(eval_rows)} rows, expected {EXPECTED_EVAL_ROWS}"
        )
    positives = [row for row in eval_rows if row["has_edit"]]
    if len(positives) != EXPECTED_EVAL_POSITIVES:
        raise EvalRegionPackageError(
            f"evaluation fold holds {len(positives)} revised rows, "
            f"expected {EXPECTED_EVAL_POSITIVES}"
        )

    banned = collect_test_identifiers()
    kept, dropped_writers = [], set()
    for row in positives:
        writer, image_id = str(row["group_uuid"]), str(row["image_id"])
        if writer in banned or image_id in banned:
            dropped_writers.add(writer)
            continue
        kept.append(row)
    if len(dropped_writers) != EXPECTED_EXCLUDED_WRITERS:
        raise EvalRegionPackageError(
            f"test guard removed {len(dropped_writers)} writers, "
            f"expected {EXPECTED_EXCLUDED_WRITERS}"
        )
    if len(kept) != EXPECTED_ITEMS:
        raise EvalRegionPackageError(
            f"{len(kept)} rows survive the test guard, expected {EXPECTED_ITEMS}"
        )

    picked = sorted(kept, key=lambda row: item_key(str(row["line_id"])))

    pages: dict[str, Any] = {}
    items, key_rows = [], []
    for index, row in enumerate(picked):
        image_id = str(row["image_id"])
        if image_id not in pages:
            pages.clear()
            pages[image_id] = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
        crop = base.crop_line(pages[image_id], row["box_1000"])
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        item_id = item_key(str(row["line_id"]))[:16]
        items.append({"id": item_id, "png": f"data:image/png;base64,{encoded}",
                      "w": crop.width, "h": crop.height})
        key_rows.append({
            "group_uuid": str(row["group_uuid"]),
            "id": item_id,
            "image_id": image_id,
            "index": index,
            "line_id": str(row["line_id"]),
            "question_global_id": str(row["question_global_id"]),
        })
    for opened in pages.values():
        opened.close()

    if len({item["id"] for item in items}) != len(items):
        raise EvalRegionPackageError("item ids collide")

    fingerprint = hashlib.sha256(
        "\x1f".join(item["id"] for item in items).encode("utf-8")
    ).hexdigest()[:32]
    page = region.build_page(items, fingerprint)

    staging = atomic_staging(output_dir)
    (staging / "标注页.html").write_text(page, encoding="utf-8")
    write_jsonl(staging / "key.jsonl", key_rows)
    manifest = {
        "create_only": True,
        "escapes": ["no_revision", "illegible"],
        "evaluation_fold_annotated": True,
        "excluded_test_writers": sorted(dropped_writers),
        "external_blind_set_touched": False,
        "ftv_test_rows_reachable": 0,
        "git_commit": git_revision(),
        "intended_use": "ground truth for scoring localisation on the frozen "
                        "evaluation fold; never for fitting, tuning or gating",
        "labels_shown_to_annotator": False,
        "network_used": False,
        "page_fingerprint": fingerprint,
        "page_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
        "paid_api_calls": 0,
        "pinned_experiment_324_sha256": AUTHORITATIVE_REGION_SHA256,
        "pinned_split_tables": dict(sorted(PINNED_SPLIT_TABLES.items())),
        "pool_size": len(positives),
        "sample_namespace": SAMPLE_NAMESPACE,
        "sampled": len(items),
        "schema_version": SCHEMA_PACKAGE,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "source": "human-backed revised lines of the pinned fold-0 evaluation fold, "
                  "less every writer named as test by a pinned split table",
        "tools": list(region.TOOLS),
    }
    write_json(staging / "package_manifest.json", manifest)
    write_json(staging / "complete.json", {"schema_version": SCHEMA_PACKAGE, "status": "complete"})
    write_recursive_sha256s(staging)
    staging.rename(output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = run(args)
    except EvalRegionPackageError as exc:
        raise SystemExit(f"experiment 325 refused: {exc}") from exc
    print(json.dumps({
        "excluded_test_writers": manifest["excluded_test_writers"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "page_fingerprint": manifest["page_fingerprint"],
        "pool_size": manifest["pool_size"],
        "sampled": manifest["sampled"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

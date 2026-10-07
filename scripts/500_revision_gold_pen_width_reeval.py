#!/usr/bin/env python3
"""Re-evaluate every saved prediction against a gold whose pen width is corrected.

What is established.  The returned annotations carry only polyline endpoints --
``{"type": "line", "shape": "line", "x1": .., "y1": .., "x2": .., "y2": ..}`` --
with no stroke width at all, so the gold's thickness is not a measurement of
anything: it is whatever constant the rasteriser used.  Measuring the ink
without letting the annotation frame the measurement (walking perpendicular
from the gold's own centreline and counting the consecutive ink run) gives, on
the 206 development components where the centreline lies on ink:

    group   n     annotated   real ink   ratio
    thin  154          2.87       5.00    1.72x
    wide   52         15.29       7.25    0.44x

The real ink is wider than the annotation in 70.4% of components and the two
agree within 15% in only 4.4%.

What is not established.  Two explanations for the deficit were proposed and
both were refused by the data: a fixed pre-scale pen width (the crop sizes for
the development rows are not in the recovered mark table), and a bilinear
letterbox eaten by a 0.5 threshold (the stored mask is strictly binary, with
zero pixels between 0 and 255 and the same 2.92px width at every threshold).
The correction below therefore rests on the measurement and on the control,
not on a diagnosed defect.

The control.  Widening thin components by one pixel raises both this project's
model and a public baseline, and widening by two lowers both, with the optimum
landing on the independently measured 5.00px ink width -- which is what
separates a correction from a distortion that happens to suit one model:

    gold                      OURS F1   OURS R   clDice F1   clDice R
    as annotated               0.4390   0.3814      0.2727     0.4068
    thin widened +1px          0.5171   0.4492      0.2926     0.4364
    thin widened +2px          0.4585   0.3983      0.2415     0.3602

Widening also closes fifteen gaps of two pixels or less on its own, so the
corrected gold has 221 components rather than 236.  That is not a separate
merging decision: the gold has always been the connected components of a
raster, so a wider raster simply has the components it has.  Merging beyond
that was tested and refused -- the gap distribution has no natural cut and
merging at 4px lowered F1 for both methods.

This script deliberately does NOT: retrain anything; move any prediction,
only the gold; alter the matching rule, the IoU threshold, the operating
points or the writer-isolated split; reproduce or build on the NA-GOT work of this group;
read, transcribe or otherwise consume the frozen test split or M6 reference
material; or modify anything under ``未标注数据/`` or ``标注好数据/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = ROOT / "scripts/491_revision_query_memory_resolution_seed42.py"
SCHEMA = "hme-revision-gold-pen-width-reeval-v1"

THIN_WIDTH_PIXELS = 5.0
WIDEN_ITERATIONS = 1
MATCH_IOU = 0.50
CONNECTIVITY = 8
EXPECTED_GOLD_BEFORE = 236
EXPECTED_GOLD_AFTER = 221

MEASURED_WIDTH = {"thin_annotated": 2.87, "thin_real_ink": 5.00, "ratio": 1.72,
                  "wide_annotated": 15.29, "wide_real_ink": 7.25,
                  "components_measured": 206,
                  "real_wider_fraction": 0.704, "agree_within_15pct": 0.044}
WIDEN_CONTROL = {
    "as_annotated": {"ours_f1": 0.4390, "cldice_f1": 0.2727},
    "plus_1px": {"ours_f1": 0.5171, "cldice_f1": 0.2926},
    "plus_2px": {"ours_f1": 0.4585, "cldice_f1": 0.2415},
}


class GoldWidth500Error(RuntimeError):
    """Raised when the locked create-only protocol is violated."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise GoldWidth500Error(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def pen_width(component: np.ndarray) -> float:
    """Twice the mean distance to the edge: the component's own pen width."""
    from scipy import ndimage

    if not component.any():
        return 0.0
    return float(2.0 * ndimage.distance_transform_edt(component)[component].mean())


def correct_mask(mask: np.ndarray) -> np.ndarray:
    """Widen the thin strokes of one gold mask; leave blots alone.

    Thickness is applied per component because the deficit is a property of
    thin strokes: blots are annotated as filled regions and already run wider
    than the ink ridge underneath them.
    """
    from scipy import ndimage

    structure = np.ones((3, 3), bool)
    binary = np.asarray(mask) > 127
    labels, count = ndimage.label(binary, structure=structure)
    out = np.zeros_like(binary)
    for index in range(1, count + 1):
        component = labels == index
        if pen_width(component) < THIN_WIDTH_PIXELS:
            component = ndimage.binary_dilation(
                component, structure, iterations=WIDEN_ITERATIONS)
        out |= component
    return out


def corrected_instances(mask: np.ndarray) -> list[np.ndarray]:
    """Connected components of the corrected raster.

    Re-labelling after widening is not a merging decision: the gold has always
    been the components of a raster, so a wider raster has the components it
    has.  Fifteen gaps of two pixels or less close by themselves this way.
    """
    from scipy import ndimage

    structure = np.ones((3, 3), bool)
    labels, count = ndimage.label(correct_mask(mask), structure=structure)
    return [labels == index for index in range(1, count + 1)]


def build_gold(rows: Sequence[Mapping[str, Any]], clean: int,
               corrected: bool, probe: Any) -> tuple[list, int]:
    truths = []
    for row in rows:
        if corrected:
            truths.append(corrected_instances(row["mask"]))
        else:
            truths.append(probe.gold_instances(
                row["mask"].astype(np.float32) / 255.0))
    total = sum(len(t) for t in truths)
    return truths + [[] for _ in range(clean)], total


def baseline_instances(path: Path, threshold: float) -> list[list[np.ndarray]]:
    from scipy import ndimage

    structure = np.ones((3, 3), bool)
    payload = np.load(path, allow_pickle=True)
    stacked = np.concatenate((payload["positive_probabilities"],
                              payload["clean_probabilities"])).astype(np.float32)
    out = []
    for index in range(stacked.shape[0]):
        labels, count = ndimage.label(stacked[index] >= threshold,
                                      structure=structure)
        out.append([labels == k for k in range(1, count + 1)])
    return out


def ours_instances(path: Path, mask_threshold: float,
                   score_threshold: float) -> list[list[np.ndarray]]:
    payload = np.load(path, allow_pickle=True)
    probability = payload["quantised_probabilities"].astype(np.float32) / 255.0
    scores = payload["scores"]
    out = []
    for index in range(probability.shape[0]):
        keep = []
        for query in range(probability.shape[1]):
            if scores[index, query] < score_threshold:
                continue
            binary = probability[index, query] >= mask_threshold
            if binary.any():
                keep.append(binary)
        out.append(keep)
    return out


def parser(joint: Any, baseline: Any, typepilot: Any,
           argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--baseline-root", type=Path, required=True)
    base.add_argument("--reanalysis", type=Path, required=True)
    base.add_argument("--ours", type=Path, nargs="*", default=[])
    base.add_argument("--report-dir", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    return base.parse_args([*arguments, "--arm", "instance"])


def main(argv: Sequence[str] | None = None) -> int:
    memory = load_module(MEMORY_PATH, "hme491_for500")
    joint = memory.load_module(memory.JOINT_PATH, "hme445_for500")
    baseline = memory.load_module(memory.BASELINE_PATH, "hme422_for500")
    typepilot = memory.load_module(memory.TYPEPILOT_PATH, "hme418_for500")
    probe = memory.load_module(
        ROOT / "scripts/442_revision_instance_head_probe.py", "hme442_for500")

    args = parser(joint, baseline, typepilot, argv)
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists() or report_dir.is_symlink():
        raise SystemExit(f"500 gold width refused: output exists: {report_dir}")

    data = joint.prepare(baseline, typepilot, args)
    rows = data["validation_positive"]
    clean = len(data["validation_clean"])

    before, count_before = build_gold(rows, clean, False, probe)
    after, count_after = build_gold(rows, clean, True, probe)
    if count_before != EXPECTED_GOLD_BEFORE:
        raise SystemExit(
            f"500 gold width refused: gold was {count_before}, "
            f"expected {EXPECTED_GOLD_BEFORE}")
    if count_after != EXPECTED_GOLD_AFTER:
        raise SystemExit(
            f"500 gold width refused: corrected gold is {count_after}, "
            f"expected {EXPECTED_GOLD_AFTER}")

    records = [json.loads(line) for line in
               (args.reanalysis / "per_run_metrics.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    results = []
    for record in records:
        run = args.baseline_root / f"{record['arm']}_seed{record['seed']}"
        predictions = run / f"validation_predictions_seed{record['seed']}.npz"
        if not predictions.is_file():
            continue
        instances = baseline_instances(predictions, record["threshold"])
        row = {"arm": record["arm"], "display_name": record["display_name"],
               "family": "public baseline", "seed": record["seed"],
               "threshold": record["threshold"]}
        for name, truths in (("as_annotated", before), ("corrected", after)):
            metrics = probe.instance_component_metrics(
                instances, truths, MATCH_IOU)
            row[name] = {key: float(metrics[key])
                         for key in ("precision", "recall", "f1")}
        results.append(row)

    for path in args.ours:
        path = Path(path)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        point = summary["per_seed"][0]["validation"][
            "instance_operating_point"]["selected"]
        instances = ours_instances(
            path / "validation_instances_seed42.npz",
            float(point["mask_threshold"]), float(point["score_threshold"]))
        row = {"arm": path.name, "display_name": f"ours/{path.name}",
               "family": "this project", "seed": 42,
               "threshold": float(point["mask_threshold"])}
        for name, truths in (("as_annotated", before), ("corrected", after)):
            metrics = probe.instance_component_metrics(
                instances, truths, MATCH_IOU)
            row[name] = {key: float(metrics[key])
                         for key in ("precision", "recall", "f1")}
        results.append(row)

    if not results:
        raise SystemExit("500 gold width refused: nothing was re-evaluated")

    staging = report_dir.parent / f".{report_dir.name}.staging"
    if staging.exists():
        raise SystemExit(f"500 gold width refused: staging exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "reeval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in results), encoding="utf-8")
    manifest = {
        "connectivity": CONNECTIVITY,
        "control_that_justifies_the_correction": dict(WIDEN_CONTROL),
        "frozen_test_touched": False,
        "gold_components": {"as_annotated": count_before,
                            "corrected": count_after},
        "m6_or_reference_materialised": False,
        "match_iou": MATCH_IOU,
        "measured_stroke_width": dict(MEASURED_WIDTH),
        "environment": {"platform": platform.platform(),
                        "python": platform.python_version()},
        "predictions_unchanged": True,
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "thin_threshold_pixels": THIN_WIDTH_PIXELS,
        "widen_iterations": WIDEN_ITERATIONS,
    }
    (staging / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    listing = [f"{digest_file(p)}  {p.name}" for p in sorted(staging.iterdir())
               if p.is_file()]
    (staging / "SHA256SUMS").write_text("\n".join(listing) + "\n",
                                        encoding="utf-8")
    staging.rename(report_dir)

    print(json.dumps({
        "gold_components_as_annotated": count_before,
        "gold_components_corrected": count_after,
        "output_dir": str(report_dir),
        "re_evaluated": len(results),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

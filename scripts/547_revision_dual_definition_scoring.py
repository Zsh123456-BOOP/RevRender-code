#!/usr/bin/env python3
"""Score every model against both instance definitions, and report both.

The instance definition in use is the 8-connected components of the corrected
gold raster.  That was never a decision -- the annotations are polyline
endpoints, so they are per stroke, and "instance = connected component" fell
out of rasterisation.  It is also unstable: dilating thin strokes by one pixel
took the gold from 236 components to 221 in experiment 500.

Experiment 544 measured that 73.8 percent of pairs on two-revision rows overlap
in their column ranges; scripts 545 and 546 rendered them; the rule in
``gold/涂改事件口径.md`` was fixed and committed before any judgement; and
``gold/涂改事件分组_v1.jsonl`` records the grouping of all 61 multi-component
rows -- 58 decided, 3 left unsure.  Two or three parallel lines through one
expression, the two strokes of an X-shaped cross-out and one long rule broken
in two are single acts of revising; separate characters struck individually are
not, and the rule separates them.

The direction this moves the numbers was known before the judgement was made:
merging gold components should raise the score of a model that draws a
strike-out as one region and lower the coverage of one that proposes each
stroke separately.  That is exactly why both definitions are reported here
rather than one replacing the other, and why the grouping was judged from the
scans alone with no model output in view.

Component F1 is reported at each model's own selected operating point.
Coverage is reported twice: once over the candidates that operating point keeps,
and once over the whole proposal pool with no score threshold at all, which is
the ceiling any read-out of that pool could reach.  The two differ a lot for
models that propose many low-scoring candidates, so quoting one for the other
misstates them.

This script deliberately does NOT: train or fine-tune anything; re-derive the
gold, which is experiment 500's corrected definition unchanged; re-judge or
alter the grouping, which is read from the committed file; dilate any mask;
pick a winner, since the point is that the two definitions disagree; read,
transcribe or otherwise consume the frozen test split or M6 reference material;
contact the network; or modify anything under ``未标注数据/`` or ``标注好数据/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "scripts/500_revision_gold_pen_width_reeval.py"
RULE_PATH = ROOT / "gold/涂改事件口径.md"
GROUPING_PATH = ROOT / "gold/涂改事件分组_v1.jsonl"
SCHEMA = "hme-revision-dual-definition-scoring-v1"

MATCH_IOU = 0.50
EXPECTED_POSITIVE_ROWS = 126
EXPECTED_STROKE_GOLD = 221
EXPECTED_EVENT_GOLD = 169
EXPECTED_MULTI_ROWS = 61
EXPECTED_UNSURE_ROWS = 3
POINTREND_SCORE_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


class Dual547Error(RuntimeError):
    """Raised when the locked create-only protocol is violated."""


def digest_file(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Dual547Error(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def read_grouping(path: Path) -> dict[str, Any]:
    """The committed judgement, keyed by line id."""
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_MULTI_ROWS:
        raise Dual547Error(
            f"{len(records)} grouping records, expected {EXPECTED_MULTI_ROWS}")
    unsure = sum(1 for r in records if r["decision"] == "unsure")
    if unsure != EXPECTED_UNSURE_ROWS:
        raise Dual547Error(
            f"{unsure} unsure rows, expected {EXPECTED_UNSURE_ROWS}")
    return {record["line_id"]: record for record in records}


def event_truths(components: Sequence[np.ndarray],
                 record: dict[str, Any] | None) -> list[np.ndarray]:
    """Components merged into revision events, per the committed grouping.

    A row with no record is single-component; a row judged unsure keeps every
    component, which is the conservative direction -- it leaves the gold larger
    and so cannot manufacture the effect being measured.
    """
    if record is None or record["grouping"] is None:
        return list(components)
    if record["components"] != len(components):
        raise Dual547Error(
            f"{record['line_id']} was judged with {record['components']} "
            f"components but carries {len(components)}")
    merged = []
    for group in record["grouping"]:
        union = np.zeros_like(components[group[0]])
        for index in group:
            union = np.logical_or(union, components[index])
        merged.append(union)
    return merged


def coverage(instances: Sequence[Sequence[np.ndarray]],
             truths: Sequence[Sequence[np.ndarray]]) -> int:
    """Gold objects reached by some candidate at the matching threshold.

    No score threshold and no one-to-one constraint: a ceiling on what any
    read-out of this proposal pool could recover, not a score.
    """
    hit = 0
    for predicted, truth in zip(instances, truths):
        for target in truth:
            area = int(target.sum())
            for candidate in predicted:
                intersection = int(np.logical_and(candidate, target).sum())
                if not intersection:
                    continue
                union = int(candidate.sum()) + area - intersection
                if union and intersection / union >= MATCH_IOU:
                    hit += 1
                    break
    return hit


def pointrend_index(directory: Path) -> dict[str, Path]:
    """Archive path per line id.

    The stored sample id is ``<image id>:<line id>`` and the line id itself
    begins with the image id, so the key is everything after the first colon;
    matching on a substring instead collides across lines of one page.
    """
    archives: dict[str, Path] = {}
    for path in sorted(directory.glob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        line = str(payload["sample_id"]).split(":", 1)[1]
        if line in archives:
            raise Dual547Error(f"two PointRend archives for {line}")
        archives[line] = path
    return archives


def pointrend_instances(archives: dict[str, Path], order: Sequence[str],
                        score_threshold: float) -> list[list[np.ndarray]]:
    """Candidates per validation row, in the dataset's own row order."""
    out = []
    for key in order:
        if key not in archives:
            raise Dual547Error(f"no PointRend archive for {key}")
        payload = np.load(archives[key], allow_pickle=True)
        shape = tuple(int(v) for v in payload["shape"])
        masks = np.unpackbits(payload["masks_packed"], axis=1)
        masks = masks[:, :shape[1] * shape[2]].reshape(
            shape[0], shape[1], shape[2]).astype(bool)
        scores = payload["scores"]
        out.append([masks[i] for i in range(shape[0])
                    if scores[i] >= score_threshold and masks[i].any()])
    return out


def score(instances: Sequence[Sequence[np.ndarray]],
          truths: Sequence[Sequence[np.ndarray]], probe: Any,
          pool: Sequence[Sequence[np.ndarray]] | None = None) -> dict[str, Any]:
    """Matched metrics on the kept candidates, coverage on both sets."""
    metrics = probe.instance_component_metrics(instances, truths, MATCH_IOU)
    result = {"f1": float(metrics["f1"]),
              "precision": float(metrics["precision"]),
              "recall": float(metrics["recall"]),
              "true_positive": int(metrics["true_positive"]),
              "false_positive": int(metrics["false_positive"]),
              "false_negative": int(metrics["false_negative"]),
              "covered_at_operating_point": coverage(instances, truths)}
    result["covered_whole_pool"] = result["covered_at_operating_point"] \
        if pool is None else coverage(pool, truths)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    gold = load_module(GOLD_PATH, "hme500_for547")
    memory = load_module(gold.MEMORY_PATH, "hme491_for547")
    joint = memory.load_module(memory.JOINT_PATH, "hme445_for547")
    baseline = memory.load_module(memory.BASELINE_PATH, "hme422_for547")
    typepilot = memory.load_module(memory.TYPEPILOT_PATH, "hme418_for547")
    probe = memory.load_module(
        ROOT / "scripts/442_revision_instance_head_probe.py", "hme442_for547")

    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--ours", type=Path, action="append", required=True,
                      help="a completed run directory of ours; repeatable, so "
                           "several of our runs share one pass over the "
                           "baselines instead of recomputing them each time")
    base.add_argument("--baseline-root", type=Path, required=True)
    base.add_argument("--baseline-thresholds", type=Path, required=True,
                      help="experiment 512's comparison.json")
    base.add_argument("--pointrend", type=Path, default=None)
    base.add_argument("--report-dir", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(value == "--output-dir" or value.startswith("--output-dir=")
               for value in arguments):
        index = arguments.index("--report-dir") + 1
        arguments = [*arguments, "--output-dir",
                     str(Path(arguments[index]) / "unused")]
    args = base.parse_args([*arguments, "--arm", "instance"])

    try:
        report_dir = args.report_dir.expanduser().resolve()
        if report_dir.exists() or report_dir.is_symlink():
            raise Dual547Error(f"output exists: {report_dir}")
        grouping = read_grouping(GROUPING_PATH)

        data = joint.prepare(baseline, typepilot, args)
        rows = data["validation_positive"]
        clean = len(data["validation_clean"])
        if len(rows) != EXPECTED_POSITIVE_ROWS:
            raise Dual547Error(
                f"{len(rows)} positive rows, expected "
                f"{EXPECTED_POSITIVE_ROWS}")

        stroke, event = [], []
        for row in rows:
            components = gold.corrected_instances(row["mask"])
            stroke.append(components)
            event.append(event_truths(
                components, grouping.get(str(row["line_id"]))))
        stroke_total = sum(len(t) for t in stroke)
        event_total = sum(len(t) for t in event)
        if stroke_total != EXPECTED_STROKE_GOLD:
            raise Dual547Error(
                f"stroke gold is {stroke_total}, expected "
                f"{EXPECTED_STROKE_GOLD}")
        if event_total != EXPECTED_EVENT_GOLD:
            raise Dual547Error(
                f"event gold is {event_total}, expected "
                f"{EXPECTED_EVENT_GOLD}")
        stroke += [[] for _ in range(clean)]
        event += [[] for _ in range(clean)]

        results: list[dict[str, Any]] = []

        for directory in args.ours:
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8"))
            point = summary["per_seed"][0]["validation"][
                "instance_operating_point"]["selected"]
            archives = sorted(directory.glob("validation_instances_seed*.npz"))
            if len(archives) != 1:
                raise Dual547Error(
                    f"{len(archives)} instance archives in {directory}")
            ours = gold.ours_instances(
                archives[0], float(point["mask_threshold"]),
                float(point["score_threshold"]))
            ours_pool = gold.ours_instances(
                archives[0], float(point["mask_threshold"]), -1.0)
            results.append({
                "model": f"ours:{directory.name}", "family": "this work",
                "operating_point": {"mask": float(point["mask_threshold"]),
                                    "score": float(point["score_threshold"])},
                "stroke_level": score(ours, stroke, probe, ours_pool),
                "event_level": score(ours, event, probe, ours_pool)})

        recalibrated = {record["arm"]: record for record in json.loads(
            args.baseline_thresholds.read_text(encoding="utf-8"))}
        for arm, record in sorted(recalibrated.items()):
            threshold = float(record["recalibrated_grid"]["threshold"])
            path = args.baseline_root / f"{arm}_seed42" \
                / "validation_predictions_seed42.npz"
            if not path.is_file():
                path = args.baseline_root / arm \
                    / "validation_predictions_seed42.npz"
            instances = gold.baseline_instances(path, threshold)
            results.append({
                "model": arm, "family": "retrained baseline",
                "operating_point": {"mask": threshold, "score": None},
                "stroke_level": score(instances, stroke, probe),
                "event_level": score(instances, event, probe)})

        if args.pointrend is not None:
            order = [str(row["line_id"]) for row in rows]
            order += [str(row["line_id"]) for row in data["validation_clean"]]
            archives = pointrend_index(args.pointrend)
            pool = pointrend_instances(archives, order, -1.0)
            best = None
            for threshold in POINTREND_SCORE_GRID:
                instances = pointrend_instances(archives, order, threshold)
                stroke_score = score(instances, stroke, probe, pool)
                event_score = score(instances, event, probe, pool)
                candidate = {"model": "pointrend",
                             "family": "public instance baseline",
                             "operating_point": {"mask": 0.5,
                                                 "score": threshold},
                             "stroke_level": stroke_score,
                             "event_level": event_score}
                if best is None or stroke_score["f1"] > \
                        best["stroke_level"]["f1"]:
                    best = candidate
            best["note"] = ("score threshold chosen to maximise its own "
                            "stroke-level F1 over " +
                            repr(list(POINTREND_SCORE_GRID)) +
                            "; generous to it and stated as such")
            results.append(best)

        report = {
            "schema_version": SCHEMA,
            "match_iou": MATCH_IOU,
            "gold": {"stroke_level": stroke_total, "event_level": event_total,
                     "multi_component_rows": EXPECTED_MULTI_ROWS,
                     "rows_left_unsure": EXPECTED_UNSURE_ROWS,
                     "unsure_rows_keep_every_component": True},
            "results": results,
        }
        report_dir.mkdir(parents=True)
        (report_dir / "dual_scoring.json").write_text(
            json.dumps(report, indent=1, sort_keys=True,
                       ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "run_manifest.json").write_text(json.dumps({
            "script_sha256": digest_file(Path(__file__).resolve()),
            "gold_script_sha256": digest_file(GOLD_PATH),
            "rule_sha256": digest_file(RULE_PATH),
            "grouping_sha256": digest_file(GROUPING_PATH),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "create_only": True,
            "training_performed": False,
            "frozen_test_touched": False,
        }, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
    except Dual547Error as exc:
        raise SystemExit("547 dual scoring refused: " + str(exc))

    print(json.dumps({
        "event_gold": event_total,
        "models": len(results),
        "report_dir": str(report_dir),
        "stroke_gold": stroke_total,
        "training_performed": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

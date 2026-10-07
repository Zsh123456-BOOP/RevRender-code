#!/usr/bin/env python3
"""Run a strictly isolated, sample-matched repair of experiment 312.

Experiment 312 had two coupled confounds: its real arm was 2.7 times larger
than its synthetic arm, and 204 of the 1,699 synthetic positives were guard
rows that shared either an evaluation writer fold or an evaluation question
fold.  This follow-up removes both problems before making a source-domain
claim.

Two arms are trained consecutively in the same process.  Each contains 1,495
positives and the exact same 1,495 real negatives in the same positions:

    synthetic_strict  all generated positives whose writer AND question folds
                      are strict-train for evaluation fold zero
    real_matched      a hash-ranked sample of strict-train real positives

The paired arms share label order, negative line IDs, architecture, schedule,
model seed, evaluation rows and student-clustered bootstrap.  Only the positive
images differ.  The script imports the SHA-pinned experiment 312 implementation
so preprocessing cannot silently drift, but repairs its role filtering here.

This script deliberately does NOT:
  * modify experiment 312 or reuse its confounded synthetic score as primary;
  * reach the network, call a paid API, or touch an external blind/test split;
  * use HWG or any other public dataset;
  * overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GAP_SCRIPT = ROOT / "scripts" / "312_synthetic_to_real_gap.py"
AUTHORITATIVE_GAP_SHA256 = (
    "7ba7cb2a24887fc754dd36464b50300d307dab4c7be4e1044f144a3a652c607b"
)
AUTHORITATIVE_TRAIN_SHA256 = (
    "cc6aa56165e240d3df27c5fcebefdb759766cb1be79033a5221ad00b41bebeb2"
)
AUTHORITATIVE_EVAL_SHA256 = (
    "775b564c2800d26518ba70fefa2d06a6b01d29292d8f6ef8d43933b28fe6e8c4"
)
AUTHORITATIVE_SYNTHETIC_SHA256 = (
    "e6628aa0fa37e4ecca1b5f5d007a3ff38f00ad99f2790c9061b56978741b6154"
)
AUTHORITATIVE_SYNTHETIC_SUMS_SHA256 = (
    "4f15120e4f80d9346c5d16339537763ade61fc00533676f6cb9492ce56b74cba"
)
AUTHORITATIVE_SYNTHETIC_MANIFEST_SHA256 = (
    "2bf77e58255547b83af8104e90634b234864ae48c300d0b9126b6ebf825eab37"
)
AUTHORITATIVE_TRAIN_ROWS = 9_212
AUTHORITATIVE_EVAL_ROWS = 329
AUTHORITATIVE_SYNTHETIC_ROWS = 2_100
EXPECTED_STRICT_SYNTHETIC = 1_495
EXPECTED_GUARD_SYNTHETIC = 540
EXPECTED_EVAL_SYNTHETIC = 65
MATCHED_POSITIVES = EXPECTED_STRICT_SYNTHETIC
MATCHED_NEGATIVES = EXPECTED_STRICT_SYNTHETIC

SCHEMA_VERSION = "hme-real-matched-synthetic-gap-control-v1"
EVALUATION_FOLD = 0
MODEL_SEED = 312
BOOTSTRAP = 2_000
ARMS = ("synthetic_strict", "real_matched")
FORMAL = {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}
POSITIVE_NAMESPACE = "hme-real-matched-positive-v1"
NEGATIVE_NAMESPACE = "hme-paired-negative-v1"
SYNTHETIC_ORDER_NAMESPACE = "hme-strict-synthetic-order-v1"


class MatchedControlError(RuntimeError):
    """Refuse to publish a paired result when an authority or role drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gap():
    if not GAP_SCRIPT.is_file():
        raise MatchedControlError(f"missing pinned experiment 312: {GAP_SCRIPT}")
    actual = sha256_file(GAP_SCRIPT)
    if actual != AUTHORITATIVE_GAP_SHA256:
        raise MatchedControlError(
            f"experiment 312 digest mismatch: expected {AUTHORITATIVE_GAP_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("hme_gap312", GAP_SCRIPT)
    if spec is None or spec.loader is None:
        raise MatchedControlError("cannot import pinned experiment 312")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("bootstrap_delta", "load_base", "load_jsonl", "run_arm"):
        if not hasattr(module, name):
            raise MatchedControlError(f"experiment 312 lacks required function {name}")
    if module.SEED != MODEL_SEED or module.FORMAL != FORMAL:
        raise MatchedControlError("experiment 312 seed or frozen schedule drifted")
    return module


def strict_role(student_fold: int, question_fold: int) -> str:
    if student_fold == EVALUATION_FOLD and question_fold == EVALUATION_FOLD:
        return "eval"
    if student_fold != EVALUATION_FOLD and question_fold != EVALUATION_FOLD:
        return "train"
    return "guard"


def hash_ranked_sample(rows: Iterable[dict], count: int, namespace: str) -> list[dict]:
    """Return an order-independent deterministic sample without silent shrinkage."""
    materialized = list(rows)
    if len(materialized) < count:
        raise MatchedControlError(
            f"cannot sample {count} rows from a pool of {len(materialized)}"
        )
    identifiers = [str(row["line_id"]) for row in materialized]
    if len(set(identifiers)) != len(identifiers):
        raise MatchedControlError("candidate line_id values are not unique")

    def rank(row: dict) -> tuple[str, str]:
        line_id = str(row["line_id"])
        digest = hashlib.sha256(f"{namespace}\0{line_id}".encode("utf-8")).hexdigest()
        return digest, line_id

    return sorted(materialized, key=rank)[:count]


def line_ids_sha256(rows: Iterable[dict]) -> str:
    payload = "".join(f"{row['line_id']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_synthetic_checksums(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise MatchedControlError("synthetic root must be a regular directory")
    sums = root / "SHA256SUMS"
    manifest = root / "input_manifest.json"
    if sha256_file(sums) != AUTHORITATIVE_SYNTHETIC_SUMS_SHA256:
        raise MatchedControlError("synthetic SHA256SUMS authority drifted")
    if sha256_file(manifest) != AUTHORITATIVE_SYNTHETIC_MANIFEST_SHA256:
        raise MatchedControlError("synthetic input_manifest authority drifted")
    resolved_root = root.resolve()
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        expected, separator, relative_text = line.partition("  ")
        relative = Path(relative_text)
        if (
            not separator
            or len(expected) != 64
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise MatchedControlError(f"invalid synthetic SHA256SUMS row {number}")
        candidate = root / relative
        artifact = candidate.resolve()
        if (
            not artifact.is_relative_to(resolved_root)
            or candidate.is_symlink()
            or not artifact.is_file()
            or sha256_file(artifact) != expected
        ):
            raise MatchedControlError(f"synthetic checksum failure: {relative}")


def _real_item(row: dict) -> dict:
    return {
        "kind": "real",
        "path": None,
        "image_id": row["image_id"],
        "box_1000": row["box_1000"],
        "line_id": row["line_id"],
        "group_uuid": row["group_uuid"],
        "y": int(bool(row["has_edit"])),
    }


def _synthetic_item(row: dict, synthetic_root: Path) -> dict:
    relative = Path(str(row["crop_relpath"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise MatchedControlError(f"unsafe synthetic crop path: {relative}")
    resolved_root = synthetic_root.resolve()
    candidate = synthetic_root / relative
    if candidate.is_symlink():
        raise MatchedControlError(f"synthetic crop is a symlink: {relative}")
    resolved_path = candidate.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise MatchedControlError(f"synthetic crop escaped its root: {relative}")
    return {
        "kind": "synthetic",
        "path": resolved_path,
        "image_id": None,
        "box_1000": None,
        "line_id": row["synthetic_id"],
        "group_uuid": row["group_uuid"],
        "question_global_id": row["question_global_id"],
        "source_line_id": row["source_line_id"],
        "source_image_id": row["source_image_id"],
        "y": 1,
        "style": row["style"],
    }


def build_paired_arms(
    train_rows: list[dict],
    eval_rows: list[dict],
    synthetic_rows: list[dict],
    synthetic_root: Path,
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    eval_writers = {str(row["group_uuid"]) for row in eval_rows}
    eval_questions = {str(row["question_global_id"]) for row in eval_rows}
    eval_images = {str(row["image_id"]) for row in eval_rows}
    if any(
        strict_role(int(row["student_fold"]), int(row["question_fold"])) != "train"
        for row in train_rows
    ):
        raise MatchedControlError("authoritative train rows contain a non-train role")
    if any(
        strict_role(int(row["student_fold"]), int(row["question_fold"])) != "eval"
        for row in eval_rows
    ):
        raise MatchedControlError("authoritative eval rows contain a non-eval role")

    real = [_real_item(row) for row in train_rows]
    train_line_ids = {str(row["line_id"]) for row in train_rows}
    real_positives = [row for row in real if row["y"] == 1]
    real_negatives = [row for row in real if row["y"] == 0]
    selected_real_positives = hash_ranked_sample(
        real_positives, MATCHED_POSITIVES, POSITIVE_NAMESPACE
    )
    shared_negatives = hash_ranked_sample(
        real_negatives, MATCHED_NEGATIVES, NEGATIVE_NAMESPACE
    )

    role_counts = Counter(
        strict_role(int(row["student_fold"]), int(row["question_fold"]))
        for row in synthetic_rows
    )
    expected_roles = Counter(
        train=EXPECTED_STRICT_SYNTHETIC,
        guard=EXPECTED_GUARD_SYNTHETIC,
        eval=EXPECTED_EVAL_SYNTHETIC,
    )
    if role_counts != expected_roles:
        raise MatchedControlError(
            f"synthetic role counts drifted: expected {dict(expected_roles)}, got {dict(role_counts)}"
        )
    synthetic_ids = [str(row.get("synthetic_id")) for row in synthetic_rows]
    source_ids = [str(row.get("source_line_id")) for row in synthetic_rows]
    if len(set(synthetic_ids)) != len(synthetic_ids):
        raise MatchedControlError("synthetic_id values are not unique")
    if len(set(source_ids)) != len(source_ids):
        raise MatchedControlError("synthetic source_line_id values are not unique")
    strict_source = []
    for row in synthetic_rows:
        if strict_role(int(row["student_fold"]), int(row["question_fold"])) != "train":
            continue
        if row.get("source_has_edit") is not False:
            raise MatchedControlError("synthetic positive was drawn on an edited source line")
        if row.get("synthetic_label") is not True:
            raise MatchedControlError("synthetic row lacks a true synthetic_label")
        if str(row["source_line_id"]) not in train_line_ids:
            raise MatchedControlError("strict synthetic source_line_id is outside train authority")
        if str(row["group_uuid"]) in eval_writers:
            raise MatchedControlError("strict synthetic row shares an evaluation writer")
        if str(row["question_global_id"]) in eval_questions:
            raise MatchedControlError("strict synthetic row shares an evaluation question")
        if str(row["source_image_id"]) in eval_images:
            raise MatchedControlError("strict synthetic row shares an evaluation image")
        strict_source.append(_synthetic_item(row, synthetic_root))
    strict_synthetic = hash_ranked_sample(
        strict_source, MATCHED_POSITIVES, SYNTHETIC_ORDER_NAMESPACE
    )

    arms = {
        "synthetic_strict": strict_synthetic + shared_negatives,
        "real_matched": selected_real_positives + shared_negatives,
    }
    expected_labels = [1] * MATCHED_POSITIVES + [0] * MATCHED_NEGATIVES
    for name, rows in arms.items():
        if [int(row["y"]) for row in rows] != expected_labels:
            raise MatchedControlError(f"arm {name} label schedule drifted")
        writers = {str(row["group_uuid"]) for row in rows}
        if writers & eval_writers:
            raise MatchedControlError(f"arm {name} shares an evaluation writer")
    synthetic_questions = {
        str(row["question_global_id"]) for row in strict_synthetic
    }
    if synthetic_questions & eval_questions:
        raise MatchedControlError("synthetic_strict shares an evaluation question")
    if [row["line_id"] for row in arms["synthetic_strict"][MATCHED_POSITIVES:]] != [
        row["line_id"] for row in arms["real_matched"][MATCHED_POSITIVES:]
    ]:
        raise MatchedControlError("paired arms do not share identical negative order")

    audit = {
        "rows_per_arm": MATCHED_POSITIVES + MATCHED_NEGATIVES,
        "positives_per_arm": MATCHED_POSITIVES,
        "negatives_per_arm": MATCHED_NEGATIVES,
        "synthetic_role_counts_all_2100": dict(sorted(role_counts.items())),
        "strict_synthetic_style_counts": dict(
            sorted(Counter(row["style"] for row in strict_synthetic).items())
        ),
        "strict_synthetic_writers": len(
            {row["group_uuid"] for row in strict_synthetic}
        ),
        "strict_synthetic_questions": len(synthetic_questions),
        "real_matched_positive_writers": len(
            {row["group_uuid"] for row in selected_real_positives}
        ),
        "shared_negative_writers": len(
            {row["group_uuid"] for row in shared_negatives}
        ),
        "synthetic_positive_line_ids_sha256": line_ids_sha256(strict_synthetic),
        "real_positive_line_ids_sha256": line_ids_sha256(selected_real_positives),
        "shared_negative_line_ids_sha256": line_ids_sha256(shared_negatives),
        "positive_selection_namespace": POSITIVE_NAMESPACE,
        "negative_selection_namespace": NEGATIVE_NAMESPACE,
        "synthetic_order_namespace": SYNTHETIC_ORDER_NAMESPACE,
        "writer_and_question_disjoint_from_eval": True,
        "image_disjoint_from_eval": True,
        "identical_negative_ids_and_order": True,
        "identical_label_schedule": True,
    }
    return arms, audit


def fixed_metrics(labels: list[bool], scores: list[float]) -> dict[str, float]:
    if len(labels) != len(scores):
        raise MatchedControlError("metric inputs have different lengths")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise MatchedControlError("fixed metrics require both classes")
    predicted = [float(score) >= 0.5 for score in scores]
    tp = sum(bool(label) and pred for label, pred in zip(labels, predicted))
    tn = sum((not bool(label)) and (not pred) for label, pred in zip(labels, predicted))
    tpr = tp / positives
    tnr = tn / negatives
    return {
        "balanced_accuracy_at_0_5": (tpr + tnr) / 2,
        "tpr_at_0_5": tpr,
        "tnr_at_0_5": tnr,
    }


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise MatchedControlError(
            f"create-only output directory already exists: {output_dir}"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise MatchedControlError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch"
        )
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible_devices) != 1 or visible_devices[0] == "-1":
        raise MatchedControlError(
            "paired training requires exactly one CUDA_VISIBLE_DEVICES entry"
        )

    gap = load_gap()
    base = gap.load_base()
    input_root = Path(args.input_root)
    train_path = input_root / "train_rows.jsonl"
    eval_path = input_root / "eval_rows.jsonl"
    synthetic_root = Path(args.synthetic_root)
    synthetic_path = synthetic_root / "synthetic_rows.jsonl"
    authorities = {
        train_path: AUTHORITATIVE_TRAIN_SHA256,
        eval_path: AUTHORITATIVE_EVAL_SHA256,
        synthetic_path: AUTHORITATIVE_SYNTHETIC_SHA256,
    }
    for path, expected in authorities.items():
        actual = sha256_file(path)
        if actual != expected:
            raise MatchedControlError(
                f"authority digest mismatch for {path.name}: expected {expected}, got {actual}"
            )
    try:
        train_rows, eval_rows = base._validate_training_inputs(
            input_root, Path(args.images_dir)
        )
    except base.StrongVisual250Error as exc:
        raise MatchedControlError(f"pinned fold-0 input validation failed: {exc}") from exc
    if len(train_rows) != AUTHORITATIVE_TRAIN_ROWS or len(eval_rows) != AUTHORITATIVE_EVAL_ROWS:
        raise MatchedControlError("pinned fold-0 row counts drifted")
    validate_synthetic_checksums(synthetic_root)
    synthetic_rows = gap.load_jsonl(
        synthetic_path, AUTHORITATIVE_SYNTHETIC_ROWS
    )
    arms, selection_audit = build_paired_arms(
        train_rows, eval_rows, synthetic_rows, synthetic_root
    )
    for row in arms["synthetic_strict"][:MATCHED_POSITIVES]:
        path = Path(row["path"])
        if path.is_symlink() or not path.is_file():
            raise MatchedControlError(f"missing regular synthetic crop: {path}")

    weights = Path(args.resnet_weights)
    try:
        model_authority = base._model_source_metadata(
            base.MODEL_IDS[0], weights, None
        )
    except base.StrongVisual250Error as exc:
        raise MatchedControlError(f"pinned ResNet weight validation failed: {exc}") from exc
    weight_sha256 = str(model_authority["weight_sha256"])

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        results[name] = gap.run_arm(
            base,
            arms[name],
            eval_rows,
            Path(args.images_dir),
            weights,
            args,
        )
        print(
            json.dumps(
                {
                    "arm": name,
                    "average_precision": results[name]["average_precision"],
                    "train_rows": results[name]["train_rows"],
                    "train_seconds": results[name]["train_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    labels = [bool(row["has_edit"]) for row in eval_rows]
    synthetic_scores = [float(value) for value in results["synthetic_strict"]["scores"]]
    real_scores = [float(value) for value in results["real_matched"]["scores"]]
    delta = float(results["real_matched"]["average_precision"]) - float(
        results["synthetic_strict"]["average_precision"]
    )
    bootstrap = gap.bootstrap_delta(
        base, eval_rows, synthetic_scores, real_scores, args.bootstrap, MODEL_SEED
    )
    arm_metrics = {}
    for name, scores in (
        ("synthetic_strict", synthetic_scores),
        ("real_matched", real_scores),
    ):
        arm_metrics[name] = {
            key: value for key, value in results[name].items() if key != "scores"
        } | fixed_metrics(labels, scores) | {"auroc": base.auroc(labels, scores)}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "arms": arm_metrics,
        "real_matched_minus_synthetic_strict": {
            "point": delta,
            "bootstrap_students": bootstrap,
        },
        "gate": {
            "strict_sample_matched_gap_confirmed": bool(
                delta > 0 and bootstrap.get("excludes_zero")
            ),
            "reading": "a positive student-bootstrap interval shows that neither "
            "sample size nor writer/question guard leakage explains the source gap",
        },
        "selection_audit": selection_audit,
        "eval_fold": {
            "rows": len(eval_rows),
            "students": len({row["group_uuid"] for row in eval_rows}),
            "questions": len({row["question_global_id"] for row in eval_rows}),
            "writer_and_question_disjoint_from_every_arm": True,
        },
        "limitations": {
            "single_fold": True,
            "single_paired_training_seed": True,
            "bitwise_reproducible": False,
            "reason": "the pinned experiment 312 ResNet backward keeps "
            "torch deterministic algorithms in warn-only mode",
        },
    }

    score_rows = [
        {
            "line_id": row["line_id"],
            "group_uuid": row["group_uuid"],
            "question_global_id": row["question_global_id"],
            "has_edit": bool(row["has_edit"]),
            "synthetic_strict": synthetic_scores[index],
            "real_matched": real_scores[index],
        }
        for index, row in enumerate(eval_rows)
    ]
    selection_rows = []
    for name in ARMS:
        for slot, row in enumerate(arms[name]):
            selection_rows.append(
                {
                    "arm": name,
                    "slot": slot,
                    "class": "positive" if row["y"] else "negative",
                    "kind": row["kind"],
                    "line_id": row["line_id"],
                    "group_uuid": row["group_uuid"],
                    "source_line_id": row.get("source_line_id"),
                    "style": row.get("style"),
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "create_only": True,
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_used": True,
        "network_used": False,
        "paid_api_calls": 0,
        "external_blind_set_touched": False,
        "model_seed": MODEL_SEED,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "pinned_experiment_312_sha256": AUTHORITATIVE_GAP_SHA256,
        "input_sha256": {
            "train_rows.jsonl": AUTHORITATIVE_TRAIN_SHA256,
            "eval_rows.jsonl": AUTHORITATIVE_EVAL_SHA256,
            "synthetic_rows.jsonl": AUTHORITATIVE_SYNTHETIC_SHA256,
            "synthetic_SHA256SUMS": AUTHORITATIVE_SYNTHETIC_SUMS_SHA256,
            "synthetic_input_manifest": AUTHORITATIVE_SYNTHETIC_MANIFEST_SHA256,
            "resnet_weights": weight_sha256,
        },
        "frozen_hyperparameters": FORMAL,
        "paired_arm_order": list(ARMS),
        "cuda_visible_devices_count": len(visible_devices),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_allowed": False,
        "bitwise_reproducible": False,
        "known_nondeterministic_cuda_backward": True,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "bootstrap": {"rounds": args.bootstrap, "seed": MODEL_SEED, "unit": "student"},
        "only_varied": "strict-train positive images; both arms contain 1495 "
        "positives followed by the identical 1495 real negatives",
        "summary": summary,
    }
    staging = base.atomic_staging(output_dir)
    try:
        base.write_json(staging / "summary.json", summary)
        base.write_jsonl(staging / "scores.jsonl", score_rows)
        base.write_jsonl(staging / "selection.jsonl", selection_rows)
        base.write_json(staging / "run_manifest.json", manifest)
        base.write_json(
            staging / "complete.json",
            {"schema_version": SCHEMA_VERSION, "status": "complete"},
        )
        base.write_sha256s(staging)
        base.finalize_staging(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "synthetic_strict_ap": results["synthetic_strict"]["average_precision"],
        "real_matched_ap": results["real_matched"]["average_precision"],
        "delta": delta,
        "strict_sample_matched_gap_confirmed": summary["gate"][
            "strict_sample_matched_gap_confirmed"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--synthetic-root", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--resnet-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=FORMAL["epochs"])
    parser.add_argument("--batch-size", type=int, default=FORMAL["batch_size"])
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=FORMAL["gradient_accumulation"],
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=FORMAL["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL["weight_decay"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for key, expected in FORMAL.items():
        if getattr(args, key) != expected:
            raise SystemExit(f"matched control refused: {key} is frozen at {expected}")
    try:
        result = build(args)
    except (MatchedControlError, OSError, ValueError) as exc:
        raise SystemExit("matched control refused: " + str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

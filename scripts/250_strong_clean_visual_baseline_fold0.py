#!/usr/bin/env python3
"""Pre-register and run experiment 250's clean fold-0 visual baselines.

The formal protocol uses only the 12,740 internal, individually reviewed line
labels from experiment 239.  Fold 0 is strictly unseen-student AND
unseen-question: train rows have neither fold id equal to zero, evaluation rows
have both fold ids equal to zero, and all remaining rows are an unused guard.

Two end-to-end arms receive exactly the same five image views, deterministic
augmentations, epochs, optimizer, learning rate, and effective line batch:

* ImageNet-pretrained ResNet50 loaded from a pinned local weight file.
* ``facebook/dinov2-small`` loaded from a pinned local snapshot.

No experiment-233 feature tensor or learned head is accepted by this script.
Evaluation happens once, after the fixed final epoch; no threshold or
hyperparameter is selected from fold-0 outcomes.  All outputs are create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_INPUT = "hme-strong-clean-visual-fold0-input-v1"
SCHEMA_RESULT = "hme-strong-clean-visual-fold0-result-v1"
SCHEMA_COMPARE = "hme-strong-clean-visual-fold0-comparison-v1"
SEED = 250
EVAL_FOLD = 0
FOLD_COUNT = 5
EXPECTED = {
    "human_rows": 12_740,
    "positive": 3_929,
    "negative": 8_811,
    "images_available": 3_976,
    "train_rows": 9_212,
    "train_positive": 2_822,
    "guard_rows": 3_199,
    "eval_rows": 329,
    "eval_positive": 118,
    "hard_rows": 22,
    "hard_positive": 7,
}
HUMAN_INDIVIDUAL_SOURCES = frozenset(
    {
        "historical_human",
        "assisted_full_human_review",
        "assisted_round5_human",
        "human_review_dual_model_disagreement",
        "human_review_dual_model_both_negative_audit",
    }
)
HARD_SOURCE = "human_review_dual_model_disagreement"
MASTER_LINES_SHA256 = "48bcd6921a124ff25247fec0af882cc2e9bf0abf34ee7f5cc2b3ab22df6053b6"
CLEAN_AUTHORITY_SHA256 = "d0d4d8f6b291dd4874669f83c31a67b23eaf9a38622688479405c5f85876d647"
DINO_MODEL_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
RESNET_WEIGHT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
RESNET_WEIGHT_BYTES = 102_540_417
DINO_FILE_SHA256 = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}
FORMAL_TRAINING_SCRIPT_SHA256 = "79245833e1e307d07938506b13d501df7434b478086bac1fcf0acc97f0694fd0"
INPUT_V4_SHA256 = {
    "protocol.json": "f23b44d691f649143137c7e81c63bf9a2fc0dd68b29d8642b525298976fb4753",
    "train_rows.jsonl": "cc6aa56165e240d3df27c5fcebefdb759766cb1be79033a5221ad00b41bebeb2",
    "eval_rows.jsonl": "775b564c2800d26518ba70fefa2d06a6b01d29292d8f6ef8d43933b28fe6e8c4",
}
CLEAN_FROZEN_PREDICTIONS_SHA256 = "40f09860c114debb359877ecc67fef35f2eb2af3a7c257feafa64dfac5029835"
MODEL_IDS = ("resnet50_imagenet1k_v2", "facebook_dinov2_small")
INPUT_SIZE = (224, 224)
VIEW_COUNT = 5
LOCAL_SLICE_COUNT = 4
LOCAL_OVERLAP = 0.25
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
FORMAL = {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "effective_line_batch": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "workers": 2,
}


class StrongVisual250Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StrongVisual250Error(f"missing regular {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrongVisual250Error(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise StrongVisual250Error(f"{label} must be a JSON object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise StrongVisual250Error(f"missing regular {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise StrongVisual250Error(f"{label} row {number} is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrongVisual250Error(f"invalid {label}: {path}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(path, 0o600)


def write_sha256s(root: Path) -> None:
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    os.chmod(root / "SHA256SUMS", 0o600)


def validate_sha256s(root: Path) -> None:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise StrongVisual250Error(f"missing SHA256SUMS: {root}")
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or Path(relative).name != relative:
            raise StrongVisual250Error(f"invalid SHA256SUMS row {number}: {root}")
        artifact = root / relative
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != expected:
            raise StrongVisual250Error(f"checksum failure: {artifact}")


def atomic_staging(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise StrongVisual250Error(f"output exists; create-only refusal: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def finalize_staging(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise StrongVisual250Error(f"output appeared during run: {output}")
    staging.rename(output)


def strict_role(student_fold: int, question_fold: int, eval_fold: int = EVAL_FOLD) -> str:
    values = (student_fold, question_fold, eval_fold)
    if any(type(value) is not int or not 0 <= value < FOLD_COUNT for value in values):
        raise StrongVisual250Error("fold ids must be integers in [0,4]")
    if student_fold == eval_fold and question_fold == eval_fold:
        return "eval"
    if student_fold != eval_fold and question_fold != eval_fold:
        return "train"
    return "guard"


def crop_line(image: Any, box_1000: Sequence[float]) -> Any:
    if len(box_1000) != 4:
        raise StrongVisual250Error("box_1000 must have four values")
    width, height = image.size
    x0, y0, x1, y1 = (float(value) for value in box_1000)
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise StrongVisual250Error("box_1000 must be ordered within [0,1000]")
    left = max(0, min(width - 1, math.floor(x0 * width / 1000)))
    top = max(0, min(height - 1, math.floor(y0 * height / 1000)))
    right = max(left + 1, min(width, math.ceil(x1 * width / 1000)))
    bottom = max(top + 1, min(height, math.ceil(y1 * height / 1000)))
    return image.crop((left, top, right, bottom))


def letterbox(image: Any, input_size: Sequence[int] = INPUT_SIZE) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise StrongVisual250Error("Pillow is required") from exc
    height, width = (int(input_size[0]), int(input_size[1]))
    if height < 1 or width < 1:
        raise StrongVisual250Error("input size must be positive")
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def horizontal_slice_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise StrongVisual250Error("slice dimensions must be positive integers")
    coverage_units = 1.0 + (LOCAL_SLICE_COUNT - 1) * (1.0 - LOCAL_OVERLAP)
    window = max(1, min(width, math.ceil(width / coverage_units)))
    travel = max(0, width - window)
    starts = [round(index * travel / (LOCAL_SLICE_COUNT - 1)) for index in range(LOCAL_SLICE_COUNT)]
    starts[0] = 0
    starts[-1] = travel
    return [(start, 0, min(width, start + window), height) for start in starts]


def make_five_views(crop: Any) -> list[Any]:
    width, height = crop.size
    views = [letterbox(crop)]
    views.extend(letterbox(crop.crop(box)) for box in horizontal_slice_boxes(width, height))
    if len(views) != VIEW_COUNT:
        raise StrongVisual250Error("five-view construction failed")
    return views


def augment_view(image: Any, line_id: str, epoch: int, view_index: int) -> Any:
    """Apply the same fixed, mild handwriting-safe augmentation in both arms."""

    try:
        from PIL import Image, ImageEnhance
    except ImportError as exc:
        raise StrongVisual250Error("Pillow is required") from exc
    rng = random.Random(stable_seed("augment", line_id, epoch, view_index))
    dx = rng.randint(-4, 4)
    dy = rng.randint(-3, 3)
    resampling = getattr(Image, "Resampling", Image)
    transform_kinds = getattr(Image, "Transform", Image)
    shifted = image.transform(
        image.size,
        transform_kinds.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=resampling.BILINEAR,
        fillcolor="white",
    )
    shifted = ImageEnhance.Brightness(shifted).enhance(rng.uniform(0.92, 1.08))
    return ImageEnhance.Contrast(shifted).enhance(rng.uniform(0.90, 1.10))


def rank_percentile(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        rank = (0.5 * (start + end - 1) + 0.5) / len(order)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def average_precision(labels: Sequence[bool | int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise StrongVisual250Error("AP requires nonempty aligned inputs")
    positives = sum(bool(value) for value in labels)
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    tp = fp = 0
    previous_recall = result = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[start]]):
            end += 1
        tp += sum(bool(labels[order[position]]) for position in range(start, end))
        fp += sum(not bool(labels[order[position]]) for position in range(start, end))
        recall = tp / positives
        precision = tp / max(tp + fp, 1)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def auroc(labels: Sequence[bool | int], scores: Sequence[float]) -> float | None:
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (float(scores[index]), index))
    negative_below = 0
    concordant = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(scores[order[end]]) == float(scores[order[start]]):
            end += 1
        group_positive = sum(bool(labels[order[position]]) for position in range(start, end))
        group_negative = end - start - group_positive
        concordant += group_positive * negative_below + 0.5 * group_positive * group_negative
        negative_below += group_negative
        start = end
    return float(concordant / (positives * negatives))


def fixed_metrics(labels: Sequence[bool | int], scores: Sequence[float]) -> dict[str, Any]:
    if len(labels) != len(scores) or not labels or any(not math.isfinite(float(score)) for score in scores):
        raise StrongVisual250Error("metrics require nonempty aligned finite scores")
    positives = sum(bool(value) for value in labels)
    negatives = len(labels) - positives
    tp = sum(bool(label) and float(score) >= 0.5 for label, score in zip(labels, scores))
    tn = sum(not bool(label) and float(score) < 0.5 for label, score in zip(labels, scores))
    sensitivity = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    return {
        "rows": len(labels),
        "positive": positives,
        "negative": negatives,
        "average_precision": average_precision(labels, scores),
        "auroc": auroc(labels, scores),
        "balanced_accuracy_at_fixed_0_5": 0.5 * (sensitivity + specificity),
        "threshold_tuned": False,
    }


def student_cluster_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    score_a_key: str,
    score_b_key: str,
    replicates: int,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise StrongVisual250Error("comparison requires numpy") from exc
    by_student: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_student.setdefault(str(row["group_uuid"]), []).append(index)
    students = sorted(by_student)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(replicates):
        sampled = rng.choice(students, size=len(students), replace=True)
        indexes = [index for student in sampled for index in by_student[str(student)]]
        labels = [bool(rows[index]["has_edit"]) for index in indexes]
        if len(set(labels)) < 2:
            continue
        deltas.append(
            average_precision(labels, [float(rows[index][score_a_key]) for index in indexes])
            - average_precision(labels, [float(rows[index][score_b_key]) for index in indexes])
        )
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "method": "student_uuid_cluster_bootstrap_with_replacement",
        "seed": SEED,
        "clusters": len(students),
        "requested_replicates": replicates,
        "valid_replicates": len(deltas),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
    }


def _validate_clean_authority(authority_root: Path) -> list[dict[str, Any]]:
    validate_sha256s(authority_root)
    path = authority_root / "human_internal_authority.jsonl"
    if sha256_file(path) != CLEAN_AUTHORITY_SHA256:
        raise StrongVisual250Error("clean experiment-239 authority drifted")
    rows = load_jsonl(path, "clean human authority")
    if (
        len(rows) != EXPECTED["human_rows"]
        or sum(bool(row.get("has_edit")) for row in rows) != EXPECTED["positive"]
        or {str(row.get("label_source")) for row in rows} - HUMAN_INDIVIDUAL_SOURCES
    ):
        raise StrongVisual250Error("clean human authority counts or sources drifted")
    return rows


def prepare(args: argparse.Namespace) -> int:
    authority_root = Path(args.authority_root).expanduser().resolve()
    master_path = Path(args.master_lines).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if master_path.is_symlink() or not master_path.is_file() or sha256_file(master_path) != MASTER_LINES_SHA256:
        raise StrongVisual250Error("Round-9 master-line authority drifted")
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise StrongVisual250Error("images-dir must be a regular directory")
    image_paths = sorted(path for path in images_dir.glob("*.jpg") if path.is_file() and not path.is_symlink())
    if len(image_paths) != EXPECTED["images_available"]:
        raise StrongVisual250Error("images-dir is not the internal-3976 authority")
    available_images = {path.stem for path in image_paths}
    authority = _validate_clean_authority(authority_root)
    master = load_jsonl(master_path, "Round-9 master lines")
    master_by_line = {str(row.get("line_id")): row for row in master}
    if len(master_by_line) != len(master):
        raise StrongVisual250Error("duplicate Round-9 line id")
    prepared = []
    for row in authority:
        line_id = str(row["line_id"])
        source = master_by_line.get(line_id)
        if source is None:
            raise StrongVisual250Error(f"clean line missing from master authority: {line_id}")
        image_id = str(source.get("image_id"))
        box = source.get("box_1000")
        if image_id not in available_images or not isinstance(box, list) or len(box) != 4:
            raise StrongVisual250Error(f"invalid internal image or line box: {line_id}")
        for key in ("has_edit", "group_uuid", "label_source"):
            if source.get(key) != row.get(key):
                raise StrongVisual250Error(f"authority mismatch for {line_id}: {key}")
        role = strict_role(int(row["student_fold"]), int(row["question_fold"]))
        prepared.append(
            {
                **row,
                "image_id": image_id,
                "box_1000": [float(value) for value in box],
                "role": role,
            }
        )
    counts = Counter(row["role"] for row in prepared)
    positive_by_role = Counter(row["role"] for row in prepared if bool(row["has_edit"]))
    hard_eval = [row for row in prepared if row["role"] == "eval" and row["label_source"] == HARD_SOURCE]
    if (
        counts != Counter(train=EXPECTED["train_rows"], guard=EXPECTED["guard_rows"], eval=EXPECTED["eval_rows"])
        or positive_by_role["train"] != EXPECTED["train_positive"]
        or positive_by_role["eval"] != EXPECTED["eval_positive"]
        or len(hard_eval) != EXPECTED["hard_rows"]
        or sum(bool(row["has_edit"]) for row in hard_eval) != EXPECTED["hard_positive"]
    ):
        raise StrongVisual250Error("fold-0 counts drifted")
    train_rows = sorted((row for row in prepared if row["role"] == "train"), key=lambda row: str(row["line_id"]))
    eval_rows = sorted((row for row in prepared if row["role"] == "eval"), key=lambda row: str(row["line_id"]))
    staging = atomic_staging(output)
    try:
        write_jsonl(staging / "train_rows.jsonl", train_rows)
        write_jsonl(staging / "eval_rows.jsonl", eval_rows)
        write_json(
            staging / "protocol.json",
            {
                "schema_version": SCHEMA_INPUT,
                "status": "complete",
                "seed": SEED,
                "eval_fold": EVAL_FOLD,
                "strict_unseen_student_and_question": True,
                "counts": {**EXPECTED, "eval_students": len({row["group_uuid"] for row in eval_rows})},
                "training_sources": sorted(HUMAN_INDIVIDUAL_SOURCES),
                "excluded_sources": ["human_batch", "policy", "strategy", "external1026", "frozen_test"],
                "models": [
                    {"id": MODEL_IDS[0], "source": "pinned local torchvision ResNet50 ImageNet-1K V2 weight"},
                    {"id": MODEL_IDS[1], "source": DINO_MODEL_ID, "revision": DINO_REVISION},
                ],
                "fixed_training": FORMAL,
                "views": "exact line crop; whole-line plus four horizontal 25%-overlap slices; no contextual padding",
                "augmentation": "deterministic +/-4px x, +/-3px y, brightness .92-1.08, contrast .90-1.10",
                "cuda_determinism": "seeds+cudnn deterministic+CUBLAS workspace; deterministic algorithms warn_only because both arms have nondeterministic CUDA backward kernels",
                "evaluation_timing": "once after fixed final epoch; no early stopping or outcome-based model selection",
                "hard_source_policy": "n=22 is descriptive only and cannot tune the method",
                "primary_confirmatory_arm": MODEL_IDS[1],
                "secondary_architecture_control": MODEL_IDS[0],
                "five_fold_gate": "predeclared DINOv2-small minus clean frozen ResNet AP >=.02, student-bootstrap lower CI>0, AUC delta>=-.01",
                "experiment233_features_or_head_used": False,
                "external_or_frozen_data_used": False,
                "labels_modified": False,
            },
        )
        write_json(
            staging / "run_manifest.json",
            {
                "schema_version": SCHEMA_INPUT,
                "created_at": utc_now(),
                "command": " ".join(sys.argv),
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "clean_authority_sha256": sha256_file(authority_root / "human_internal_authority.jsonl"),
                "master_lines_sha256": sha256_file(master_path),
                "images_dir": str(images_dir),
                "images_available": len(image_paths),
                "used_image_ids": len({row["image_id"] for row in train_rows + eval_rows}),
                "guard_rows_exported_or_used": False,
                "create_only": True,
            },
        )
        write_json(staging / "complete.json", {"schema_version": SCHEMA_INPUT, "status": "complete"})
        write_sha256s(staging)
        finalize_staging(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


def _validate_training_inputs(input_root: Path, images_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_sha256s(input_root)
    for name, expected in INPUT_V4_SHA256.items():
        if sha256_file(input_root / name) != expected:
            raise StrongVisual250Error(f"formal run requires pinned v4 input: {name}")
    protocol = load_json(input_root / "protocol.json", "experiment-250 protocol")
    if (
        protocol.get("schema_version") != SCHEMA_INPUT
        or protocol.get("status") != "complete"
        or protocol.get("strict_unseen_student_and_question") is not True
        or protocol.get("experiment233_features_or_head_used") is not False
        or protocol.get("external_or_frozen_data_used") is not False
        or protocol.get("fixed_training") != FORMAL
        or protocol.get("eval_fold") != EVAL_FOLD
        or protocol.get("primary_confirmatory_arm") != MODEL_IDS[1]
        or "warn_only" not in str(protocol.get("cuda_determinism"))
    ):
        raise StrongVisual250Error("experiment-250 input protocol drifted")
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise StrongVisual250Error("images-dir must be a regular directory")
    image_paths = [path for path in images_dir.glob("*.jpg") if path.is_file() and not path.is_symlink()]
    if len(image_paths) != EXPECTED["images_available"]:
        raise StrongVisual250Error("formal run requires internal-3976 image directory")
    train_rows = load_jsonl(input_root / "train_rows.jsonl", "fold-0 train rows")
    eval_rows = load_jsonl(input_root / "eval_rows.jsonl", "fold-0 eval rows")
    if (
        len(train_rows) != EXPECTED["train_rows"]
        or sum(bool(row["has_edit"]) for row in train_rows) != EXPECTED["train_positive"]
        or len(eval_rows) != EXPECTED["eval_rows"]
        or sum(bool(row["has_edit"]) for row in eval_rows) != EXPECTED["eval_positive"]
        or any(row.get("role") != "train" for row in train_rows)
        or any(row.get("role") != "eval" for row in eval_rows)
        or {str(row.get("label_source")) for row in train_rows + eval_rows} - HUMAN_INDIVIDUAL_SOURCES
    ):
        raise StrongVisual250Error("formal train/eval authority drifted")
    train_students = {str(row["group_uuid"]) for row in train_rows}
    eval_students = {str(row["group_uuid"]) for row in eval_rows}
    train_questions = {str(row["question_global_id"]) for row in train_rows}
    eval_questions = {str(row["question_global_id"]) for row in eval_rows}
    train_images = {str(row["image_id"]) for row in train_rows}
    eval_images = {str(row["image_id"]) for row in eval_rows}
    if train_students & eval_students or train_questions & eval_questions or train_images & eval_images:
        raise StrongVisual250Error("student, question, or image leakage in fold 0")
    return train_rows, eval_rows


def _model_source_metadata(model_id: str, resnet_weights: Path | None, dino_dir: Path | None) -> dict[str, Any]:
    if model_id == MODEL_IDS[0]:
        if resnet_weights is None or resnet_weights.is_symlink() or not resnet_weights.is_file():
            raise StrongVisual250Error("ResNet arm requires a regular --resnet-weights file")
        weight_sha256 = sha256_file(resnet_weights)
        if weight_sha256 != RESNET_WEIGHT_SHA256 or resnet_weights.stat().st_size != RESNET_WEIGHT_BYTES:
            raise StrongVisual250Error("pinned ResNet50 ImageNet-V2 weight drifted")
        return {
            "model_id": model_id,
            "source": "torchvision ResNet50 ImageNet-1K V2 official weight",
            "weight_file": str(resnet_weights),
            "weight_sha256": weight_sha256,
            "weight_bytes": resnet_weights.stat().st_size,
        }
    if model_id == MODEL_IDS[1]:
        if dino_dir is None or dino_dir.is_symlink() or not dino_dir.is_dir():
            raise StrongVisual250Error("DINO arm requires a regular --dino-model-dir")
        required = ("config.json", "model.safetensors")
        files = {}
        for name in required:
            path = dino_dir / name
            if path.is_symlink() or not path.is_file():
                raise StrongVisual250Error(f"missing pinned DINO file: {path}")
            digest = sha256_file(path)
            if digest != DINO_FILE_SHA256[name]:
                raise StrongVisual250Error(f"pinned DINO file drifted: {name}")
            files[name] = {"sha256": digest, "bytes": path.stat().st_size}
        return {
            "model_id": model_id,
            "source": DINO_MODEL_ID,
            "revision": DINO_REVISION,
            "snapshot_dir": str(dino_dir),
            "files": files,
        }
    raise StrongVisual250Error(f"unsupported model id: {model_id}")


def _build_model(model_id: str, source: Mapping[str, Any], torch: Any) -> Any:
    class MultiViewClassifier(torch.nn.Module):
        def __init__(self, backbone: Any, feature_dim: int, kind: str):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(feature_dim, 1)
            self.kind = kind

        def forward(self, views: Any) -> Any:
            batch, count = views.shape[:2]
            flat = views.reshape(batch * count, *views.shape[2:])
            if self.kind == "dino":
                representation = self.backbone(pixel_values=flat).last_hidden_state[:, 0]
            else:
                representation = self.backbone(flat)
            view_logits = self.head(representation).reshape(batch, count)
            log_no_edit = torch.nn.functional.logsigmoid(-view_logits).sum(dim=1)
            probability = (-torch.expm1(log_no_edit)).clamp(1e-6, 1 - 1e-6)
            return torch.logit(probability)

    if model_id == MODEL_IDS[0]:
        from torchvision.models import resnet50

        try:
            backbone = resnet50(weights=None)
        except TypeError:  # compatibility with the preflight-only legacy package
            backbone = resnet50(pretrained=False)
        state = torch.load(str(source["weight_file"]), map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = torch.nn.Identity()
        return MultiViewClassifier(backbone, feature_dim, "resnet")
    from transformers import Dinov2Model

    backbone = Dinov2Model.from_pretrained(
        str(source["snapshot_dir"]),
        local_files_only=True,
        use_safetensors=True,
    )
    return MultiViewClassifier(backbone, int(backbone.config.hidden_size), "dino")


def _run_training(args: argparse.Namespace, smoke: bool) -> int:
    if args.device != "cuda" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise StrongVisual250Error("GPU run requires --device cuda and CUDA_VISIBLE_DEVICES pinned before launch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise StrongVisual250Error("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch")
    try:
        import numpy as np
        import torch
        import torchvision
        import transformers
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise StrongVisual250Error("training requires torch, Pillow, and numpy") from exc
    if not torch.cuda.is_available():
        raise StrongVisual250Error("CUDA is unavailable")
    if torchvision.__version__ != "0.19.1+cu118":
        raise StrongVisual250Error("formal runtime requires isolated torchvision 0.19.1+cu118 overlay")
    input_root = Path(args.input_root).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise StrongVisual250Error(f"output exists; create-only refusal before GPU work: {output}")
    train_rows, eval_rows = _validate_training_inputs(input_root, images_dir)
    source = _model_source_metadata(
        args.model_id,
        Path(args.resnet_weights).expanduser().resolve() if args.resnet_weights else None,
        Path(args.dino_model_dir).expanduser().resolve() if args.dino_model_dir else None,
    )
    if smoke:
        positive = next(row for row in train_rows if bool(row["has_edit"]))
        negative = next(row for row in train_rows if not bool(row["has_edit"]))
        third = next(row for row in train_rows if row["line_id"] not in {negative["line_id"], positive["line_id"]})
        train_rows = [negative, positive, third]
        eval_rows = []
        epochs, batch_size, accumulation, workers = 1, 3, 1, 0
    else:
        for key, expected in FORMAL.items():
            if key == "effective_line_batch":
                continue
            argument = "gradient_accumulation" if key == "gradient_accumulation" else key
            if getattr(args, argument) != expected:
                raise StrongVisual250Error(f"formal {key} is frozen at {expected}")
        epochs = args.epochs
        batch_size = args.batch_size
        accumulation = args.gradient_accumulation
        workers = args.workers
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # ResNet adaptive pooling and DINO positional interpolation have no fully
    # deterministic CUDA backward on this stack.  Keep every available
    # deterministic control, but warn instead of making the formal run crash.
    torch.use_deterministic_algorithms(True, warn_only=True)
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1)

    class LineDataset(Dataset):
        def __init__(self, rows: Sequence[Mapping[str, Any]], training: bool):
            self.rows = list(rows)
            self.training = training
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, Any, int]:
            row = self.rows[index]
            path = images_dir / f"{row['image_id']}.jpg"
            if path.is_symlink() or not path.is_file():
                raise StrongVisual250Error(f"missing regular internal image: {path}")
            with Image.open(path) as opened:
                crop = crop_line(opened.convert("RGB"), row["box_1000"])
                views = make_five_views(crop)
            tensors = []
            for view_index, view in enumerate(views):
                if self.training:
                    view = augment_view(view, str(row["line_id"]), self.epoch, view_index)
                array = np.asarray(view, dtype=np.float32).copy() / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                tensors.append((tensor - mean) / std)
            return torch.stack(tensors), torch.tensor(float(row["has_edit"])), index

    train_dataset = LineDataset(train_rows, training=True)
    eval_dataset = LineDataset(eval_rows, training=False)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    ) if eval_rows else None
    model = _build_model(args.model_id, source, torch).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positive = sum(bool(row["has_edit"]) for row in train_rows)
    pos_weight = torch.tensor([(len(train_rows) - positive) / max(positive, 1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    started = time.monotonic()
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        optimizer_steps = 0
        for step, (views, labels, _indices) in enumerate(train_loader):
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(views)
                raw_loss = loss_fn(logits, labels)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            total_loss += float(raw_loss.detach()) * len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": total_loss / len(train_rows),
                "optimizer_steps": optimizer_steps,
            }
        )
        print(json.dumps({"event": "epoch_complete", **history[-1]}, sort_keys=True), flush=True)
    predictions: list[dict[str, Any]] = []
    if eval_loader is not None:
        model.eval()
        with torch.inference_mode():
            for views, _labels, indices in eval_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = torch.sigmoid(model(views.to(device, non_blocking=True))).float().cpu().tolist()
                for index, score in zip(indices.tolist(), scores):
                    row = eval_rows[index]
                    predictions.append(
                        {
                            "fold": EVAL_FOLD,
                            "line_id": row["line_id"],
                            "group_uuid": row["group_uuid"],
                            "question_global_id": row["question_global_id"],
                            "label_source": row["label_source"],
                            "has_edit": bool(row["has_edit"]),
                            "score": float(score),
                        }
                    )
    elapsed = time.monotonic() - started
    labels = [bool(row["has_edit"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    overall = fixed_metrics(labels, scores) if predictions else None
    hard = [row for row in predictions if row["label_source"] == HARD_SOURCE]
    source_metrics = {}
    for source_name in sorted({str(row["label_source"]) for row in predictions}):
        stratum = [row for row in predictions if row["label_source"] == source_name]
        source_metrics[source_name] = fixed_metrics(
            [bool(row["has_edit"]) for row in stratum], [float(row["score"]) for row in stratum]
        )
    result = {
        "schema_version": SCHEMA_RESULT,
        "status": "complete",
        "run_kind": "three_line_smoke" if smoke else "formal_fold0_final_epoch",
        "model": source,
        "scope": {
            "internal3976_only": True,
            "human_individual_rows_only": True,
            "strict_unseen_student_and_question": not smoke,
            "experiment233_features_or_head_used": False,
            "external1026_or_frozen_test_used": False,
            "guard_rows_used": False,
            "eval_used_during_training_or_selection": False,
        },
        "training": {
            "seed": SEED,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "effective_line_batch": batch_size * accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "precision": "fp16 autocast with GradScaler",
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "history": history,
            "elapsed_seconds": elapsed,
        },
        "overall": overall,
        "hard_disagreement_descriptive_only": (
            {**fixed_metrics([bool(row["has_edit"]) for row in hard], [float(row["score"]) for row in hard]),
             "small_n_warning": "n=22; no tuning or inferential claim"}
            if hard else None
        ),
        "source_strata": source_metrics,
    }
    staging = atomic_staging(output)
    try:
        write_json(staging / "metrics.json", result)
        write_jsonl(staging / "fold0_predictions.jsonl", sorted(predictions, key=lambda row: str(row["line_id"])))
        with (staging / "final_model.pt").open("xb") as handle:
            torch.save(
                {
                    "schema_version": SCHEMA_RESULT,
                    "model_id": args.model_id,
                    "state_dict": model.state_dict(),
                    "seed": SEED,
                    "formal_protocol": not smoke,
                },
                handle,
            )
        write_json(
            staging / "run_manifest.json",
            {
                "schema_version": SCHEMA_RESULT,
                "created_at": utc_now(),
                "command": " ".join(sys.argv),
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "input_root": str(input_root),
                "input_sha256s_validated": True,
                "images_dir": str(images_dir),
                "model_source": source,
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "torchvision_file": str(Path(torchvision.__file__).resolve()),
                    "transformers": transformers.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                },
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "deterministic_algorithms": "enabled_warn_only",
                "create_only": True,
                "labels_modified": False,
            },
        )
        write_json(staging / "complete.json", {"schema_version": SCHEMA_RESULT, "status": "complete"})
        write_sha256s(staging)
        finalize_staging(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


def train(args: argparse.Namespace) -> int:
    return _run_training(args, smoke=False)


def smoke(args: argparse.Namespace) -> int:
    return _run_training(args, smoke=True)


def _load_arm(root: Path, expected_model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_sha256s(root)
    complete = load_json(root / "complete.json", "arm completion")
    metrics = load_json(root / "metrics.json", "arm metrics")
    manifest = load_json(root / "run_manifest.json", "arm run manifest")
    rows = load_jsonl(root / "fold0_predictions.jsonl", "arm predictions")
    scope = metrics.get("scope", {})
    model = metrics.get("model", {})
    environment = manifest.get("environment", {})
    if (
        complete.get("schema_version") != SCHEMA_RESULT
        or complete.get("status") != "complete"
        or metrics.get("run_kind") != "formal_fold0_final_epoch"
        or model.get("model_id") != expected_model
        or len(rows) != EXPECTED["eval_rows"]
        or len({str(row.get("line_id")) for row in rows}) != EXPECTED["eval_rows"]
        or any(not math.isfinite(float(row.get("score", float("nan")))) for row in rows)
        or manifest.get("script_sha256") != FORMAL_TRAINING_SCRIPT_SHA256
        or manifest.get("cuda_visible_devices") != "1"
        or manifest.get("cublas_workspace_config") != ":4096:8"
        or manifest.get("deterministic_algorithms") != "enabled_warn_only"
        or environment.get("torchvision") != "0.19.1+cu118"
        or scope.get("internal3976_only") is not True
        or scope.get("human_individual_rows_only") is not True
        or scope.get("strict_unseen_student_and_question") is not True
        or scope.get("experiment233_features_or_head_used") is not False
        or scope.get("external1026_or_frozen_test_used") is not False
        or scope.get("guard_rows_used") is not False
        or scope.get("eval_used_during_training_or_selection") is not False
    ):
        raise StrongVisual250Error(f"invalid formal arm: {root}")
    if expected_model == MODEL_IDS[0] and (
        model.get("weight_sha256") != RESNET_WEIGHT_SHA256
        or model.get("weight_bytes") != RESNET_WEIGHT_BYTES
    ):
        raise StrongVisual250Error("formal ResNet source hash drifted")
    if expected_model == MODEL_IDS[1] and (
        model.get("revision") != DINO_REVISION
        or {name: record.get("sha256") for name, record in model.get("files", {}).items()} != DINO_FILE_SHA256
    ):
        raise StrongVisual250Error("formal DINO source hash drifted")
    return metrics, rows


def compare(args: argparse.Namespace) -> int:
    resnet_root = Path(args.resnet_root).expanduser().resolve()
    dino_root = Path(args.dino_root).expanduser().resolve()
    frozen_root = Path(args.clean_frozen_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    resnet_metrics, resnet_rows = _load_arm(resnet_root, MODEL_IDS[0])
    dino_metrics, dino_rows = _load_arm(dino_root, MODEL_IDS[1])
    validate_sha256s(frozen_root)
    if sha256_file(frozen_root / "strict_ood_predictions.jsonl") != CLEAN_FROZEN_PREDICTIONS_SHA256:
        raise StrongVisual250Error("clean frozen ResNet baseline prediction authority drifted")
    frozen_all = load_jsonl(frozen_root / "strict_ood_predictions.jsonl", "clean frozen predictions")
    frozen_rows = [row for row in frozen_all if int(row["fold"]) == EVAL_FOLD]
    if len(frozen_rows) != EXPECTED["eval_rows"]:
        raise StrongVisual250Error("clean frozen fold-0 count drifted")
    resnet_by_line = {str(row["line_id"]): row for row in resnet_rows}
    dino_by_line = {str(row["line_id"]): row for row in dino_rows}
    frozen_by_line = {str(row["line_id"]): row for row in frozen_rows}
    if set(resnet_by_line) != set(dino_by_line) or set(resnet_by_line) != set(frozen_by_line):
        raise StrongVisual250Error("comparison line sets differ")
    joined = []
    for line_id in sorted(resnet_by_line):
        resnet_row = resnet_by_line[line_id]
        dino_row = dino_by_line[line_id]
        frozen_row = frozen_by_line[line_id]
        if (
            bool(resnet_row["has_edit"]) != bool(dino_row["has_edit"])
            or bool(resnet_row["has_edit"]) != bool(frozen_row["has_edit"])
            or str(resnet_row["group_uuid"]) != str(dino_row["group_uuid"])
            or str(resnet_row["group_uuid"]) != str(frozen_row["group_uuid"])
        ):
            raise StrongVisual250Error(f"comparison metadata mismatch: {line_id}")
        joined.append(
            {
                **resnet_row,
                "resnet50_e2e_score": float(resnet_row["score"]),
                "dinov2_small_e2e_score": float(dino_row["score"]),
                "resnet50_frozen_clean_score": float(frozen_row["visual_score"]),
            }
        )
    labels = [bool(row["has_edit"]) for row in joined]
    metric_by_arm = {
        "resnet50_e2e": fixed_metrics(labels, [float(row["resnet50_e2e_score"]) for row in joined]),
        "dinov2_small_e2e": fixed_metrics(labels, [float(row["dinov2_small_e2e_score"]) for row in joined]),
        "resnet50_frozen_clean": fixed_metrics(labels, [float(row["resnet50_frozen_clean_score"]) for row in joined]),
    }
    descriptive_best = max(
        ("resnet50_e2e", "dinov2_small_e2e"),
        key=lambda arm: metric_by_arm[arm]["average_precision"],
    )
    bootstrap_primary = student_cluster_bootstrap_delta(
        joined, "dinov2_small_e2e_score", "resnet50_frozen_clean_score", args.bootstrap_replicates
    )
    bootstrap_resnet_control = student_cluster_bootstrap_delta(
        joined, "resnet50_e2e_score", "resnet50_frozen_clean_score", args.bootstrap_replicates
    )
    dino_vs_resnet = student_cluster_bootstrap_delta(
        joined, "dinov2_small_e2e_score", "resnet50_e2e_score", args.bootstrap_replicates
    )
    delta_ap = metric_by_arm["dinov2_small_e2e"]["average_precision"] - metric_by_arm["resnet50_frozen_clean"]["average_precision"]
    primary_auc = metric_by_arm["dinov2_small_e2e"]["auroc"]
    frozen_auc = metric_by_arm["resnet50_frozen_clean"]["auroc"]
    delta_auc = float(primary_auc) - float(frozen_auc)
    gate = delta_ap >= 0.02 and bootstrap_primary["ci95"][0] > 0.0 and delta_auc >= -0.01
    hard = [row for row in joined if row["label_source"] == HARD_SOURCE]
    hard_metrics = {
        arm: fixed_metrics(
            [bool(row["has_edit"]) for row in hard],
            [float(row[key]) for row in hard],
        )
        for arm, key in (
            ("resnet50_e2e", "resnet50_e2e_score"),
            ("dinov2_small_e2e", "dinov2_small_e2e_score"),
            ("resnet50_frozen_clean", "resnet50_frozen_clean_score"),
        )
    }
    result = {
        "schema_version": SCHEMA_COMPARE,
        "status": "complete",
        "scope": {
            "fold": EVAL_FOLD,
            "strict_unseen_student_and_question": True,
            "rows": len(joined),
            "students": len({row["group_uuid"] for row in joined}),
            "hard_rows": len(hard),
            "hard_inference": "descriptive only because n=22",
            "outcome_used_for_training_or_hyperparameter_selection": False,
            "primary_arm_predeclared_before_outcome_read": "dinov2_small_e2e",
            "resnet50_e2e_role": "secondary architecture control; cannot trigger five-fold continuation",
        },
        "overall": metric_by_arm,
        "descriptive_best_point_estimate_no_inferential_use": descriptive_best,
        "primary_dinov2_minus_frozen": {
            "delta_ap": delta_ap,
            "delta_auc": delta_auc,
            "student_cluster_bootstrap_ap": bootstrap_primary,
        },
        "secondary_resnet50_minus_frozen_ap_bootstrap": bootstrap_resnet_control,
        "dinov2_minus_resnet50_e2e_ap_bootstrap": dino_vs_resnet,
        "hard_disagreement_descriptive_only": hard_metrics,
        "predeclared_five_fold_gate": {
            "definition": "predeclared DINOv2-small AP - clean frozen AP >=.02; student-bootstrap AP lower CI>0; AUC delta>=-.01",
            "pass": gate,
            "decision": "recommend_five_fold_dinov2_small" if gate else "stop_after_fold0_screen",
        },
        "training_runtime_seconds": {
            "resnet50_e2e": resnet_metrics["training"]["elapsed_seconds"],
            "dinov2_small_e2e": dino_metrics["training"]["elapsed_seconds"],
        },
    }
    staging = atomic_staging(output)
    try:
        write_json(staging / "metrics.json", result)
        write_jsonl(staging / "fold0_joined_predictions.jsonl", joined)
        (staging / "report.md").write_text(
            f"""# Experiment 250 clean strong visual fold-0 screen

- Primary confirmatory arm: `dinov2_small_e2e`
- Descriptive best point estimate (no inferential use): `{descriptive_best}`
- E2E ResNet50 AP/AUC: {metric_by_arm['resnet50_e2e']['average_precision']:.6f} / {metric_by_arm['resnet50_e2e']['auroc']:.6f}
- E2E DINOv2-small AP/AUC: {metric_by_arm['dinov2_small_e2e']['average_precision']:.6f} / {metric_by_arm['dinov2_small_e2e']['auroc']:.6f}
- Clean frozen ResNet AP/AUC: {metric_by_arm['resnet50_frozen_clean']['average_precision']:.6f} / {metric_by_arm['resnet50_frozen_clean']['auroc']:.6f}
- Primary DINOv2 minus frozen AP: {delta_ap:+.6f}
- Primary student-bootstrap 95% CI: [{bootstrap_primary['ci95'][0]:.6f}, {bootstrap_primary['ci95'][1]:.6f}]
- Five-fold gate: `{gate}`
- Hard disagreement: 22 rows only; descriptive, not inferential.

Only the internal 3,976-image pool and 12,740 individually reviewed labels
were eligible.  Student and question identities are both disjoint between
fold-0 train and evaluation.  No experiment-233 learned head, external1026,
frozen test, batch label, policy label, or strategy label was used.
""",
            encoding="utf-8",
        )
        os.chmod(staging / "report.md", 0o600)
        write_json(
            staging / "run_manifest.json",
            {
                "schema_version": SCHEMA_COMPARE,
                "created_at": utc_now(),
                "command": " ".join(sys.argv),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "resnet_root": str(resnet_root),
                "dino_root": str(dino_root),
                "clean_frozen_root": str(frozen_root),
                "all_input_sha256s_validated": True,
                "create_only": True,
            },
        )
        write_json(staging / "complete.json", {"schema_version": SCHEMA_COMPARE, "status": "complete"})
        write_sha256s(staging)
        finalize_staging(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True, choices=MODEL_IDS)
    parser.add_argument("--resnet-weights", type=Path)
    parser.add_argument("--dino-model-dir", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--epochs", type=int, default=FORMAL["epochs"])
    parser.add_argument("--batch-size", type=int, default=FORMAL["batch_size"])
    parser.add_argument("--gradient-accumulation", type=int, default=FORMAL["gradient_accumulation"])
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=FORMAL["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL["weight_decay"])
    parser.add_argument("--workers", type=int, default=FORMAL["workers"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--authority-root", required=True, type=Path)
    prepare_parser.add_argument("--master-lines", required=True, type=Path)
    prepare_parser.add_argument("--images-dir", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.set_defaults(func=prepare)
    smoke_parser = commands.add_parser("smoke")
    _add_model_args(smoke_parser)
    smoke_parser.set_defaults(func=smoke)
    train_parser = commands.add_parser("train")
    _add_model_args(train_parser)
    train_parser.set_defaults(func=train)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--resnet-root", required=True, type=Path)
    compare_parser.add_argument("--dino-root", required=True, type=Path)
    compare_parser.add_argument("--clean-frozen-root", required=True, type=Path)
    compare_parser.add_argument("--output-dir", required=True, type=Path)
    compare_parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    compare_parser.set_defaults(func=compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("epochs", "batch_size", "gradient_accumulation", "eval_batch_size", "workers"):
        if hasattr(args, name) and (type(getattr(args, name)) is not int or getattr(args, name) < (0 if name == "workers" else 1)):
            raise StrongVisual250Error(f"invalid {name}")
    if hasattr(args, "learning_rate") and (args.learning_rate <= 0 or args.weight_decay < 0):
        raise StrongVisual250Error("invalid optimizer hyperparameters")
    if hasattr(args, "bootstrap_replicates") and args.bootstrap_replicates < 100:
        raise StrongVisual250Error("bootstrap-replicates must be at least 100")
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StrongVisual250Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

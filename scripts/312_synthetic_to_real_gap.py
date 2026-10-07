#!/usr/bin/env python3
"""Do synthetic strikethroughs teach a detector what a real one looks like?

Public strikethrough resources are dominated by strokes drawn onto clean words
by a generator; the taxonomy this project's synthetic set follows (single,
double, diagonal, cross, wave, zig-zag, scratch) comes from that literature. An
earlier experiment showed that *adding* such synthetic lines to real training
data changes nothing (AP -0.0024, interval spanning zero), but adding data to
an already-sufficient real set is a weak test: a null there is equally
consistent with "the synthetic strokes are realistic but redundant".

The decisive question is what happens with no real positives at all. Three arms
share one evaluation fold, one architecture, one schedule and one seed:

    real       real revised lines as positives
    synthetic  synthetic strokes as positives, real clean lines as negatives
    mixed      both

Only the positive class differs between `real` and `synthetic`; the negatives
and the backgrounds are real handwriting in every arm, so a gap cannot be
blamed on the paper texture or the crop pipeline. It isolates one thing: does a
drawn stroke look, to a detector, like a stroke a student actually made.

The synthetic arm is deliberately *not* called synthetic-only in the outputs.
Its negatives are real, and saying otherwise would overstate how synthetic the
arm is.

Writer isolation is enforced twice: the evaluation fold's writers are removed
from every training arm including the synthetic one, and the script refuses to
run if any writer appears on both sides.

This script deliberately does NOT:
  * reach the network, download anything, or call a paid API;
  * touch the external blind set, the FTV test split, or a policy-assigned
    label;
  * tune anything per arm -- the schedule is frozen and identical;
  * overwrite anything: the output directory is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "250_strong_clean_visual_baseline_fold0.py"
AUTHORITATIVE_BASE_SHA256 = (
    "ec3ef8823fa65c2a59928c55c6b16fdd57306467e74cc3f4420dce26cc0a99e3"
)
AUTHORITATIVE_SYNTHETIC_ROWS = 2100
AUTHORITATIVE_TRAIN_ROWS = 9212
AUTHORITATIVE_EVAL_ROWS = 329

SCHEMA_VERSION = "hme-synthetic-to-real-gap-v1"
SEED = 312
INPUT_SIZE = (224, 224)
ARMS = ("real", "synthetic", "mixed")
BOOTSTRAP = 2000
FORMAL = {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}


class GapError(RuntimeError):
    """Base error for refusing to publish a confounded domain-gap claim."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base():
    actual = sha256_file(BASE_SCRIPT)
    if actual != AUTHORITATIVE_BASE_SHA256:
        raise GapError(f"pinned baseline digest mismatch: got {actual}")
    spec = importlib.util.spec_from_file_location("hme_baseline_250", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def five_views_at(base, crop, size):
    width, height = crop.size
    views = [base.letterbox(crop, size)]
    views.extend(base.letterbox(crop.crop(box), size)
                 for box in base.horizontal_slice_boxes(width, height))
    if len(views) != 5:
        raise GapError("five-view construction failed")
    return views


def load_jsonl(path: Path, expected: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if expected is not None and len(rows) != expected:
        raise GapError(f"{path.name}: expected {expected} rows, got {len(rows)}")
    return rows


def build_arms(train_rows, eval_rows, synthetic_rows, synthetic_root) -> dict[str, list[dict]]:
    eval_students = {row["group_uuid"] for row in eval_rows}
    real = [
        {"kind": "real", "path": None, "image_id": row["image_id"], "box_1000": row["box_1000"],
         "line_id": row["line_id"], "group_uuid": row["group_uuid"], "y": int(bool(row["has_edit"]))}
        for row in train_rows
    ]
    negatives = [row for row in real if row["y"] == 0]
    synthetic_positive = []
    for row in synthetic_rows:
        if row["group_uuid"] in eval_students:
            continue
        if str(row["source_has_edit"]) != "False":
            raise GapError("a synthetic row was drawn on a line that already had a revision")
        synthetic_positive.append(
            {"kind": "synthetic", "path": synthetic_root / row["crop_relpath"],
             "image_id": None, "box_1000": None, "line_id": row["synthetic_id"],
             "group_uuid": row["group_uuid"], "y": 1, "style": row["style"]}
        )
    if not synthetic_positive:
        raise GapError("no synthetic positive survived writer isolation")
    rng = random.Random(f"{SEED}:negatives")
    sampled_negatives = rng.sample(negatives, min(len(negatives), len(synthetic_positive)))
    arms = {
        "real": real,
        "synthetic": synthetic_positive + sampled_negatives,
        "mixed": real + synthetic_positive,
    }
    for name, rows in arms.items():
        writers = {row["group_uuid"] for row in rows}
        if writers & eval_students:
            raise GapError(f"arm {name} shares a writer with the evaluation fold")
        if not any(row["y"] == 1 for row in rows) or not any(row["y"] == 0 for row in rows):
            raise GapError(f"arm {name} is single class")
    return arms


def run_arm(base, rows, eval_rows, images_dir, weights, args) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import resnet50

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    class Lines(Dataset):
        def __init__(self, items, training):
            self.items, self.training, self.epoch = list(items), training, 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            if item["kind"] == "synthetic":
                with Image.open(item["path"]) as opened:
                    crop = opened.convert("RGB")
            else:
                path = images_dir / f"{item['image_id']}.jpg"
                if path.is_symlink() or not path.is_file():
                    raise GapError(f"missing regular image: {path}")
                with Image.open(path) as opened:
                    crop = base.crop_line(opened.convert("RGB"), item["box_1000"])
            views = five_views_at(base, crop, INPUT_SIZE)
            tensors = []
            for view_index, view in enumerate(views):
                if self.training:
                    view = base.augment_view(view, str(item["line_id"]), self.epoch, view_index)
                array = np.asarray(view, dtype=np.float32).copy() / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                tensors.append((tensor - mean) / std)
            return torch.stack(tensors), torch.tensor(float(item["y"])), index

    model = resnet50()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    torch.nn.init.zeros_(model.fc.bias)
    torch.nn.init.normal_(model.fc.weight, std=0.01)
    model = model.to(device)

    def noisy_or_logit(view_logits):
        probability = torch.sigmoid(view_logits)
        keep = torch.clamp(1.0 - probability, min=1e-6)
        page = torch.clamp(1.0 - torch.exp(torch.log(keep).sum(dim=1)), min=1e-6, max=1 - 1e-6)
        return torch.log(page / (1 - page))

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = Lines(rows, True)
    generator = torch.Generator().manual_seed(SEED)
    started = time.time()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, generator=generator)
        model.train()
        optimiser.zero_grad(set_to_none=True)
        for step, (views, labels, _) in enumerate(loader):
            batch, count = views.shape[0], views.shape[1]
            flat = views.view(batch * count, *views.shape[2:]).to(device, non_blocking=True)
            logits = model(flat).view(batch, count)
            loss = loss_fn(noisy_or_logit(logits), labels.to(device)) / args.gradient_accumulation
            loss.backward()
            if (step + 1) % args.gradient_accumulation == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

    evaluation = [
        {"kind": "real", "path": None, "image_id": row["image_id"], "box_1000": row["box_1000"],
         "line_id": row["line_id"], "group_uuid": row["group_uuid"], "y": int(bool(row["has_edit"]))}
        for row in eval_rows
    ]
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        loader = DataLoader(Lines(evaluation, False), batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=args.workers)
        for views, _, _ in loader:
            batch, count = views.shape[0], views.shape[1]
            flat = views.view(batch * count, *views.shape[2:]).to(device)
            logits = model(flat).view(batch, count)
            scores.extend(torch.sigmoid(noisy_or_logit(logits)).cpu().tolist())
    labels = [bool(row["has_edit"]) for row in eval_rows]
    return {
        "train_rows": len(rows),
        "train_positives": sum(1 for row in rows if row["y"] == 1),
        "synthetic_positives": sum(1 for row in rows if row["kind"] == "synthetic"),
        "average_precision": base.average_precision(labels, scores),
        "scores": scores,
        "train_seconds": round(time.time() - started, 1),
    }


def bootstrap_delta(base, eval_rows, left, right, rounds, seed) -> dict[str, Any]:
    by_student: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eval_rows):
        by_student[row["group_uuid"]].append(index)
    students = sorted(by_student)
    rng = random.Random(seed)
    labels = [bool(row["has_edit"]) for row in eval_rows]
    deltas: list[float] = []
    for _ in range(rounds):
        picked: list[int] = []
        for _ in range(len(students)):
            picked.extend(by_student[students[rng.randrange(len(students))]])
        drawn = [labels[i] for i in picked]
        if not any(drawn) or all(drawn):
            continue
        deltas.append(base.average_precision(drawn, [right[i] for i in picked])
                      - base.average_precision(drawn, [left[i] for i in picked]))
    if len(deltas) < 100:
        return {"rounds": len(deltas), "ci95": None}
    deltas.sort()
    low, high = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]
    return {"rounds": len(deltas), "mean": statistics.fmean(deltas), "ci95": [low, high],
            "excludes_zero": bool(low > 0 or high < 0)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise GapError(f"create-only output directory already exists: {output_dir}")
    base = load_base()
    input_root = Path(args.input_root)
    train_rows = load_jsonl(input_root / "train_rows.jsonl", AUTHORITATIVE_TRAIN_ROWS)
    eval_rows = load_jsonl(input_root / "eval_rows.jsonl", AUTHORITATIVE_EVAL_ROWS)
    synthetic_root = Path(args.synthetic_root)
    synthetic_rows = load_jsonl(synthetic_root / "synthetic_rows.jsonl", AUTHORITATIVE_SYNTHETIC_ROWS)
    arms = build_arms(train_rows, eval_rows, synthetic_rows, synthetic_root)

    results: dict[str, Any] = {}
    for name in ARMS:
        results[name] = run_arm(base, arms[name], eval_rows, Path(args.images_dir),
                                Path(args.resnet_weights), args)
        print(json.dumps({"arm": name, "ap": results[name]["average_precision"],
                          "train_rows": results[name]["train_rows"],
                          "seconds": results[name]["train_seconds"]}, sort_keys=True), flush=True)

    gap = bootstrap_delta(base, eval_rows, results["synthetic"]["scores"],
                          results["real"]["scores"], args.bootstrap, SEED)
    mixed_gain = bootstrap_delta(base, eval_rows, results["real"]["scores"],
                                 results["mixed"]["scores"], args.bootstrap, SEED)

    output_dir.mkdir(parents=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in results.items()},
        "note": "the synthetic arm's negatives and backgrounds are real handwriting; "
                "only its positive strokes are drawn, so it is not a synthetic-only arm",
        "domain_gap_real_minus_synthetic": {
            "point": results["real"]["average_precision"] - results["synthetic"]["average_precision"],
            "bootstrap_students": gap,
        },
        "mixed_minus_real": {
            "point": results["mixed"]["average_precision"] - results["real"]["average_precision"],
            "bootstrap_students": mixed_gain,
        },
        "gate": {
            "gap_is_real": bool(gap.get("excludes_zero")
                                and results["real"]["average_precision"]
                                > results["synthetic"]["average_precision"]),
            "reading": "a confirmed gap motivates diagnosing which stroke properties the "
                       "generator gets wrong; no gap means synthetic strokes already look "
                       "real enough and this direction is closed",
        },
        "eval_fold": {"rows": len(eval_rows),
                      "students": len({row["group_uuid"] for row in eval_rows}),
                      "writer_disjoint_from_every_arm": True},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "scores.jsonl").write_text(
        "".join(json.dumps({"line_id": row["line_id"], "group_uuid": row["group_uuid"],
                            "has_edit": bool(row["has_edit"]),
                            **{k: results[k]["scores"][i] for k in ARMS}},
                           ensure_ascii=False, sort_keys=True) + "\n"
                for i, row in enumerate(eval_rows)), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "create_only": True,
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip() or "unknown",
        "python_version": platform.python_version(), "platform": platform.platform(),
        "gpu_used": True, "network_used": False, "paid_api_calls": 0, "seed": SEED,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "pinned_baseline_sha256": AUTHORITATIVE_BASE_SHA256,
        "frozen_hyperparameters": FORMAL, "input_size": list(INPUT_SIZE),
        "only_varied": "which lines supply the positive class",
        "synthetic_style_taxonomy": "single/double/diagonal/cross/wave/zigzag/scratch",
        "external_blind_set_touched": False,
        "bootstrap": {"rounds": args.bootstrap, "seed": SEED, "unit": "student"},
        "summary": summary,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir),
            "ap": {k: results[k]["average_precision"] for k in ARMS},
            "gap": summary["domain_gap_real_minus_synthetic"]["point"],
            "gap_confirmed": summary["gate"]["gap_is_real"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--synthetic-root", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--resnet-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=FORMAL["epochs"])
    parser.add_argument("--batch-size", type=int, default=FORMAL["batch_size"])
    parser.add_argument("--gradient-accumulation", type=int, default=FORMAL["gradient_accumulation"])
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=FORMAL["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL["weight_decay"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    for key, expected in FORMAL.items():
        if getattr(args, key) != expected:
            raise SystemExit(f"gap experiment refused: {key} is frozen at {expected}")
    try:
        result = build(args)
    except GapError as exc:
        raise SystemExit("gap experiment refused: " + str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

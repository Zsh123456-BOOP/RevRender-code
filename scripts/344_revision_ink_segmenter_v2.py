#!/usr/bin/env python3
"""Say which pixels are revision ink, from every annotation the project holds.

The method the first paper needs is not erasure. Deleting a struck span turns
"a-struck c" into "ac", which is worse than leaving it, and the probe measured
exactly that: a model recovers the struck text from the untouched crop 48.0% of
the time and from a cleaned one 36.6%. What the model cannot do is decide which
half of a revision counts -- it returns the corrected version 62.4% of the time
and the struck version 16.5%, and nothing in the image tells it which it should
have chosen. So the intervention is to say where the revision is, and leave the
pixels alone.

That needs a mask on pages, not on the 592 lines a person has marked. It does
not need a page-level model: the catalogue carries a box for all 46,902 lines of
all 4,990 pages, and 1,023 of the 1,026 error-detection images are among them,
so a line-level segmenter composes back to a page through boxes that already
exist. Training and inference then share a granularity, which is the one the
annotator worked at.

Experiment 331 trained this on 287 lines and reached IoU 0.279 with a
non-monotonic size curve -- 200 lines scored below 100 -- which says its variance
exceeded its data effect. It was one seed, sixty epochs, no augmentation, no
pretrained stem, and it was set aside as too weak rather than tuned. This run
has 455 training lines instead of 287, augmentation, and several seeds, and
reports the spread rather than one draw.

The split is by provenance and by writer. 325 collected its 117 evaluation-fold
lines for scoring and its manifest permits nothing else, so they are the held-out
set and are never trained on. The other two packages -- 324's 300 and 339's 175,
overlapping in seven lines -- form the training pool, split again by writer into
fit and validation so no writer is scored on a model that saw them.

This script deliberately does NOT:
  * train on the evaluation-fold annotations;
  * let a writer appear on both sides of any split;
  * use a pretrained backbone -- the input is a one-channel line crop, and an
    ImageNet stem would make the size and seed curves unreadable;
  * overwrite anything -- the output directory is create-only.
"""

from __future__ import annotations

import argparse
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

REGION_SCRIPT = ROOT / "scripts" / "324_revision_region_annotation_package.py"
AUTHORITATIVE_REGION_SHA256 = (
    "da5fd32a040a834f64cda3fdb1106329a1c3c4e5c12c724c34d4c022b58dc442"
)
CATALOG = (ROOT / "gold" / "line_edit_all5002_policy_decisions_20260814_v1"
           / "canonical" / "exact_crops_46902.jsonl")
AUTHORITATIVE_CATALOG_SHA256 = (
    "eaf3aaeb2d811c2e7c64a548e3131f0379938d6e660ab420eddf76562040d424"
)

SOURCES = {
    "324_train": {
        "key": "outputs/revision_region_annotation_v1/key.jsonl",
        "key_sha256": "eb7458b240c430b690c5cd43a543f40dbb82a3634c4260b163f9d4eecda69708",
        "returned": "work/revision_region_annotation_returned_v2/revision_region_annotation.json",
        "returned_sha256": "88a49f39297c0e665d9f164c0cd6b274099a4c5cd3db6dc50e28ba70ea4a27af",
        "role": "fit", "usable": 287,
    },
    "339_pair": {
        "key": "outputs/pair_region_annotation_v2/key.jsonl",
        "key_sha256": "faa3dbf384f8dc06f0a39203ca66ffddae7dd42a88400a9a4fd6ffdfe2f5ca99",
        "returned": "work/pair_region_returned_v1/revision_region_annotation.json",
        "returned_sha256": "a60978ab578f1827a519617d23bed8285c99a2dd7af7ff2f3d8c2fd14b7551be",
        "role": "fit", "usable": 175,
    },
    "325_eval": {
        "key": "outputs/revision_region_eval_annotation_v1/key.jsonl",
        "key_sha256": "844e96cea8d1d0dcc98bb0bd0b4e5cc3cc09b465f02f4aaa201a69e4f74ff8bf",
        "returned": "work/revision_region_eval_returned_v2/revision_region_annotation.json",
        "returned_sha256": "11320440eaea164ed8ab408c3372a5f8d62df4b582f7c60cd1af6c144f49a8b6",
        "role": "held_out", "usable": 109,
    },
}
EXPECTED_FIT_POOL = 455        # 287 + 175 less the seven lines both packages cover
EXPECTED_HELD_OUT = 109
EXPECTED_OVERLAP = 7

CROP_WIDTH = 512
CROP_HEIGHT = 64
# 326 measured a 3px median dark run under a real revision stroke.
MARK_STROKE_PX = 3

FORMAL = {
    "epochs": 120,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "base_channels": 48,
    "validation_fraction": 0.15,
    "patience": 25,
}
SEEDS = (344, 1344, 2344)
THRESHOLD_SWEEP = tuple(round(0.05 * i, 2) for i in range(1, 20))
SCHEMA_RESULT = "hme-revision-ink-segmenter-v2"


class SegmenterError(RuntimeError):
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
        raise SegmenterError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    return staging


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def load_region():
    if digest_file(REGION_SCRIPT) != AUTHORITATIVE_REGION_SHA256:
        raise SegmenterError("experiment 324 digest mismatch")
    spec = importlib.util.spec_from_file_location("hme_region324_for344", REGION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_sources() -> dict[str, list[dict]]:
    """Every annotated line, with its marks, grouped by which package it came from."""
    out: dict[str, list[dict]] = {}
    for name, spec in SOURCES.items():
        key_path, returned_path = ROOT / spec["key"], ROOT / spec["returned"]
        if digest_file(key_path) != spec["key_sha256"]:
            raise SegmenterError(f"{name}: key digest mismatch")
        if digest_file(returned_path) != spec["returned_sha256"]:
            raise SegmenterError(f"{name}: returned annotation digest mismatch")
        key = {r["id"]: r for r in (json.loads(l) for l in
                                    key_path.read_text(encoding="utf-8").splitlines() if l)}
        records = json.loads(returned_path.read_text(encoding="utf-8"))["records"]
        if set(records) != set(key):
            raise SegmenterError(f"{name}: returned annotation does not cover its key")
        rows = [
            {"line_id": key[i]["line_id"], "group_uuid": str(key[i]["group_uuid"]),
             "image_id": str(key[i]["image_id"]), "marks": record["marks"], "source": name}
            for i, record in records.items()
            if not record.get("flag") and record.get("marks")
        ]
        if len(rows) != spec["usable"]:
            raise SegmenterError(f"{name}: {len(rows)} usable rows, expected {spec['usable']}")
        out[name] = rows
    return out


def split(sources) -> tuple[list[dict], list[dict]]:
    """325 is held out because its manifest says so; the rest is the fitting pool."""
    held_out = sources["325_eval"]
    pool: dict[str, dict] = {}
    for name in ("324_train", "339_pair"):
        for row in sources[name]:
            # A line both packages cover keeps the later, more finely marked pass.
            pool[row["line_id"]] = row
    overlap = len(sources["324_train"]) + len(sources["339_pair"]) - len(pool)
    if overlap != EXPECTED_OVERLAP:
        raise SegmenterError(f"{overlap} lines overlap between packages, expected {EXPECTED_OVERLAP}")
    fit = list(pool.values())
    if len(fit) != EXPECTED_FIT_POOL:
        raise SegmenterError(f"fitting pool holds {len(fit)}, expected {EXPECTED_FIT_POOL}")
    if len(held_out) != EXPECTED_HELD_OUT:
        raise SegmenterError(f"held-out holds {len(held_out)}, expected {EXPECTED_HELD_OUT}")
    fit_writers = {r["group_uuid"] for r in fit}
    held_writers = {r["group_uuid"] for r in held_out}
    if fit_writers & held_writers:
        raise SegmenterError("a writer appears in both the fitting pool and the held-out set")
    return fit, held_out


def rasterise(marks, width: int, height: int):
    from PIL import Image, ImageDraw

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for mark in marks:
        if mark["shape"] == "box":
            x0, y0 = float(mark["x"]), float(mark["y"])
            draw.rectangle([x0, y0, x0 + float(mark["w"]), y0 + float(mark["h"])], fill=255)
        elif mark.get("path") and len(mark["path"]) > 1:
            draw.line([tuple(p) for p in mark["path"]], fill=255,
                      width=int(mark.get("stroke_width") or MARK_STROKE_PX), joint="curve")
        else:
            draw.line([(float(mark["x1"]), float(mark["y1"])),
                       (float(mark["x2"]), float(mark["y2"]))],
                      fill=255, width=int(mark.get("stroke_width") or MARK_STROKE_PX))
    return mask


def letterbox(image, width: int, height: int, fill: int):
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.BILINEAR,
    )
    canvas = Image.new("L", (width, height), fill)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def materialise(rows, base, catalogue, images_dir: Path, fallback=None):
    import numpy as np
    from PIL import Image

    out, pages = [], {}
    try:
        for row in sorted(rows, key=lambda r: r["line_id"]):
            entry = catalogue.get(row["line_id"])
            if entry is None and fallback is not None:
                # A handful of annotated lines predate the catalogue's dedup and
                # survive only in the pinned fold, which carries the same box.
                entry = fallback.get(row["line_id"])
            if entry is None:
                raise SegmenterError(f"no crop box for {row['line_id']}")
            if row["image_id"] not in pages:
                for opened in pages.values():
                    opened.close()
                pages = {row["image_id"]:
                         Image.open(images_dir / f"{row['image_id']}.jpg").convert("RGB")}
            crop = base.crop_line(pages[row["image_id"]], entry["box_1000"]).convert("L")
            mask = rasterise(row["marks"], crop.width, crop.height)
            out.append({
                **{k: row[k] for k in ("line_id", "group_uuid", "source")},
                "image": np.asarray(letterbox(crop, CROP_WIDTH, CROP_HEIGHT, 255), dtype=np.uint8),
                "mask": np.asarray(letterbox(mask, CROP_WIDTH, CROP_HEIGHT, 0), dtype=np.uint8),
            })
    finally:
        for opened in pages.values():
            opened.close()
    return out


def augment(image, mask, rng):
    """Photometric and small geometric jitter; never a flip -- writing has a direction."""
    import numpy as np

    picture = image.astype(np.float32)
    picture = picture * rng.uniform(0.85, 1.15) + rng.uniform(-18.0, 18.0)
    if rng.random() < 0.3:
        picture = picture + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0.0, 4.0, picture.shape
        )
    shift = rng.randint(-3, 3)
    if shift:
        picture = np.roll(picture, shift, axis=0)
        mask = np.roll(mask, shift, axis=0)
    across = rng.randint(-8, 8)
    if across:
        picture = np.roll(picture, across, axis=1)
        mask = np.roll(mask, across, axis=1)
    return np.clip(picture, 0, 255).astype(np.uint8), mask


def make_unet(base_channels: int):
    import torch
    from torch import nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.d1, self.d2, self.d3 = block(1, c), block(c, c * 2), block(c * 2, c * 4)
            self.bottom = block(c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.u1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.c3, self.c2, self.c1 = block(c * 8, c * 4), block(c * 4, c * 2), block(c * 2, c)
            self.head = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            s1 = self.d1(x)
            s2 = self.d2(self.pool(s1))
            s3 = self.d3(self.pool(s2))
            b = self.bottom(self.pool(s3))
            x = self.c3(torch.cat([self.u3(b), s3], 1))
            x = self.c2(torch.cat([self.u2(x), s2], 1))
            x = self.c1(torch.cat([self.u1(x), s1], 1))
            return self.head(x)

    return UNet(base_channels)


def metrics(probabilities, targets, threshold: float) -> dict[str, float]:
    predicted, truth = probabilities >= threshold, targets > 0.5
    intersection = float((predicted & truth).sum())
    union = float((predicted | truth).sum())
    predicted_sum, truth_sum = float(predicted.sum()), float(truth.sum())
    return {
        "dice": 2 * intersection / (predicted_sum + truth_sum) if (predicted_sum + truth_sum) else 0.0,
        "iou": intersection / union if union else 0.0,
        "precision": intersection / predicted_sum if predicted_sum else 0.0,
        "recall": intersection / truth_sum if truth_sum else 0.0,
    }


def train_once(fit_rows, held_out, args, seed: int, staging: Path):
    import numpy as np
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 32))
    writers = sorted({r["group_uuid"] for r in fit_rows})
    rng.shuffle(writers)
    cut = max(1, int(round(len(writers) * args.validation_fraction)))
    validation_writers = set(writers[:cut])
    train = [r for r in fit_rows if r["group_uuid"] not in validation_writers]
    validation = [r for r in fit_rows if r["group_uuid"] in validation_writers]
    if not train or not validation:
        raise SegmenterError("the writer split left a side empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_unet(args.base_channels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    def batches(rows, size, shuffle, jitter):
        order = list(range(len(rows)))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), size):
            chunk = [rows[i] for i in order[start:start + size]]
            pictures, masks = [], []
            for entry in chunk:
                picture, mask = entry["image"], entry["mask"]
                if jitter:
                    picture, mask = augment(picture, mask, rng)
                pictures.append(picture)
                masks.append(mask)
            x = torch.from_numpy(np.stack(pictures).astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
            y = torch.from_numpy(np.stack(masks).astype(np.float32) / 255.0).unsqueeze(1)
            yield x.to(device), y.to(device)

    def score(rows):
        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for x, y in batches(rows, args.batch_size, False, False):
                probabilities.append(torch.sigmoid(model(x)).cpu().numpy())
                targets.append(y.cpu().numpy())
        model.train()
        return np.concatenate(probabilities), np.concatenate(targets)

    best, best_epoch, best_state = -1.0, -1, None
    for epoch in range(args.epochs):
        for x, y in batches(train, args.batch_size, True, True):
            logits = model(x)
            probability = torch.sigmoid(logits)
            numerator = 2.0 * (probability * y).sum() + 1.0
            denominator = probability.sum() + y.sum() + 1.0
            loss = bce(logits, y) + (1.0 - numerator / denominator)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        probabilities, targets = score(validation)
        current = max(metrics(probabilities, targets, t)["iou"] for t in THRESHOLD_SWEEP)
        if current > best + 1e-5:
            best, best_epoch = current, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_probabilities, validation_targets = score(validation)
    threshold = max(THRESHOLD_SWEEP,
                    key=lambda t: metrics(validation_probabilities, validation_targets, t)["iou"])
    held_probabilities, held_targets = score(held_out)
    torch.save({"architecture": {"base_channels": args.base_channels,
                                 "crop": [CROP_WIDTH, CROP_HEIGHT]},
                "operating_threshold": float(threshold),
                "seed": seed,
                "state_dict": model.state_dict()}, staging / f"segmenter_seed{seed}.pt")
    return {
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "held_out": metrics(held_probabilities, held_targets, threshold),
        "held_out_best": max(
            (metrics(held_probabilities, held_targets, t) for t in THRESHOLD_SWEEP),
            key=lambda m: m["iou"]),
        "seed": seed,
        "threshold": float(threshold),
        "validation_iou": best,
        "validation_writers": len(validation_writers),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    region = load_region()
    base = region.load_matched().load_gap().load_base()
    if digest_file(CATALOG) != AUTHORITATIVE_CATALOG_SHA256:
        raise SegmenterError("crop catalogue digest mismatch")
    # A crop is reached by its representative line id or by any alias: the
    # annotation packages were drawn from different splits, and 324's lines
    # carry the fold's ids rather than the catalogue's representatives.
    catalogue = {}
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        entry = json.loads(line)
        catalogue[entry["representative_line_id"]] = entry
        for alias in entry.get("alias_line_ids") or []:
            catalogue.setdefault(alias, entry)

    fold_rows = {}
    if args.input_root is not None:
        train_rows, eval_rows = base._validate_training_inputs(
            Path(args.input_root), Path(args.fold_images_dir or args.crop_images_dir)
        )
        fold_rows = {str(r["line_id"]): r for r in list(train_rows) + list(eval_rows)}

    fit_rows, held_rows = split(load_sources())
    images_dir = Path(args.crop_images_dir)
    fit_data = materialise(fit_rows, base, catalogue, images_dir, fold_rows)
    held_data = materialise(held_rows, base, catalogue, images_dir, fold_rows)

    staging = atomic_staging(Path(args.output_dir).resolve())
    results = []
    for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
        outcome = train_once(fit_data, held_data, args, seed, staging)
        results.append(outcome)
        print(json.dumps({"seed": seed, "held_out_iou": round(outcome["held_out"]["iou"], 4),
                          "threshold": outcome["threshold"],
                          "epochs": outcome["epochs_run"]}, sort_keys=True), flush=True)

    ious = [r["held_out"]["iou"] for r in results]
    summary = {
        "across_seeds": {
            "iou_max": max(ious), "iou_mean": statistics.fmean(ious), "iou_min": min(ious),
            "iou_stdev": statistics.stdev(ious) if len(ious) > 1 else 0.0,
            "seeds": len(ious),
        },
        "data": {
            "fit_lines": len(fit_data), "fit_writers": len({r["group_uuid"] for r in fit_rows}),
            "held_out_lines": len(held_data),
            "held_out_writers": len({r["group_uuid"] for r in held_rows}),
            "sources": {n: s["usable"] for n, s in SOURCES.items()},
        },
        "per_seed": results,
        "schema_version": SCHEMA_RESULT,
        "versus_331": {"iou": 0.2792, "fit_lines": 287, "seeds": 1},
    }
    write_json(staging / "summary.json", summary)
    write_json(staging / "run_manifest.json", {
        "augmentation": "brightness, contrast, gaussian noise, small vertical and "
                        "horizontal roll; never a flip",
        "create_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "evaluation_fold_trained_on": False,
        "external_blind_set_touched": False,
        "frozen_hyperparameters": dict(FORMAL),
        "git_revision": git_revision(),
        "network_used": False,
        "paid_api_calls": 0,
        "pinned_catalogue_sha256": AUTHORITATIVE_CATALOG_SHA256,
        "pinned_experiment_324_sha256": AUTHORITATIVE_REGION_SHA256,
        "pinned_sources": {n: {k: s[k] for k in ("key_sha256", "returned_sha256", "role")}
                           for n, s in SOURCES.items()},
        "platform": platform.platform(),
        "pretrained_backbone_used": False,
        "python_version": platform.python_version(),
        "schema_version": SCHEMA_RESULT,
        "script_sha256": digest_file(Path(__file__).resolve()),
        "summary": summary["across_seeds"],
    })
    write_json(staging / "complete.json", {"schema_version": SCHEMA_RESULT, "status": "complete"})
    write_recursive_sha256s(staging)
    staging.rename(Path(args.output_dir).resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crop-images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--fold-images-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    for name, value in FORMAL.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except SegmenterError as exc:
        raise SystemExit(f"experiment 344 refused: {exc}") from exc
    print(json.dumps({
        "across_seeds": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in summary["across_seeds"].items()},
        "fit_lines": summary["data"]["fit_lines"],
        "held_out_lines": summary["data"]["held_out_lines"],
        "versus_331_iou": summary["versus_331"]["iou"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

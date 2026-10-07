#!/usr/bin/env python3
"""Bit-exact faster training loop for revision instance experiments.

This module deliberately leaves experiment 445 untouched.  Downstream
experiments can load it and replace ``joint.train_once`` with a closure around
``fast_train_once``.  The runnable entry point applies the loop to experiment
480's ``fullres_dup_off`` configuration and can also execute a 300-step
legacy-versus-fast bit-exact equivalence check.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import platform
import queue
import random
import shutil
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOINT_PATH = ROOT / "scripts/445_revision_joint_head_probe.py"
BASELINE_PATH = ROOT / "scripts/422_revision_paper_baselines.py"
TYPEPILOT_PATH = ROOT / "scripts/418_revision_typepilot300_factorial_seed42.py"
FUSION_PATH = ROOT / "scripts/478_revision_instance_preserving_fusion_seed42.py"
FULLRES_PATH = ROOT / "scripts/480_revision_fullres_instance_seed42.py"
SCHEMA = "hme-revision-fast-train-loop-v1"

SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236
LOCKED_JOINT_SHA256 = \
    "a5f140681abe1fae6d92b029f1cc9971ce018b40ff42348450766ef91277e986"
PREFETCH_DEPTH = 1
EARLY_VALIDATION_MULTIPLIER = 4
EQUIVALENCE_STEPS = 300
EQUIVALENCE_VALIDATION_INTERVAL = 100
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"


class FastTrain483Error(RuntimeError):
    """Raised when the bit-exact fast-loop protocol is violated."""


STRICT_DETERMINISM = False


def set_strict_determinism(enabled: bool) -> None:
    """Strict CUDA determinism is a *verification* setting, not a run setting.

    Experiment 446 already measured it at 2.2x (58 -> 129 minutes) and settled
    on cudnn mode as its default; experiments 480 and 481 never call
    ``use_deterministic_algorithms`` at all.  Enabling it for a production arm
    therefore costs roughly double *and* puts that arm in a different numerical
    regime from the 480/481 baselines it is compared against.  The 300-step
    equivalence check turns it on for itself; nothing else should.
    """
    global STRICT_DETERMINISM
    STRICT_DETERMINISM = bool(enabled)


def enable_strict_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != \
            DETERMINISTIC_CUBLAS_WORKSPACE:
        raise FastTrain483Error(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 is required for bit-exact runs")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_determinism() -> str:
    """Reproduce experiment 445's own regime unless bit-exactness was asked for."""
    import torch

    if STRICT_DETERMINISM:
        enable_strict_determinism()
        return "strict"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return "cudnn"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise FastTrain483Error(f"cannot import dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def validation_is_due(step: int, optimizer_steps: int,
                      validation_interval: int, start_select: int) -> bool:
    if step == optimizer_steps:
        return True
    multiplier = EARLY_VALIDATION_MULTIPLIER if step < start_select else 1
    return step % (multiplier * validation_interval) == 0


def validation_steps(optimizer_steps: int, validation_interval: int) -> list[int]:
    start_select = int(round(optimizer_steps * 0.5))
    return [step for step in range(1, optimizer_steps + 1)
            if validation_is_due(step, optimizer_steps, validation_interval,
                                 start_select)]


class OneDeepBatchPrefetcher:
    """Single-producer, depth-one augmentation prefetcher.

    The producer is the sole owner of ``rng`` and the shuffled cycles.  Its
    random-call order is consequently identical to the original serial loop.
    """

    def __init__(self, base: Any, extension: Any, data: Mapping[str, Any],
                 args: argparse.Namespace, rng: random.Random):
        self.base = base
        self.extension = extension
        self.args = args
        self.rng = rng
        self.positive_cycle = base.ShuffledCycle(data["fit_positive"], rng)
        self.clean_cycle = base.ShuffledCycle(data["fit_clean"], rng)
        self.items: queue.Queue[tuple[str, Any]] = queue.Queue(
            maxsize=PREFETCH_DEPTH)
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._produce, name="revision-batch-prefetch", daemon=True)

    def _put(self, value: tuple[str, Any]) -> bool:
        while not self.stop.is_set():
            try:
                self.items.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for _step in range(1, self.args.optimizer_steps + 1):
                if self.stop.is_set():
                    return
                rows, positive_in_batch, _clean = \
                    self.base.take_balanced_batch(
                        self.positive_cycle, self.clean_cycle,
                        self.args.batch_size, self.args.negative_ratio)
                labels = np.concatenate((
                    np.ones(positive_in_batch, dtype=np.float32),
                    np.zeros(len(rows) - positive_in_batch,
                             dtype=np.float32)))
                order = list(range(len(rows)))
                self.rng.shuffle(order)
                rows = [rows[index] for index in order]
                labels = labels[order]

                images, masks = [], []
                for row in rows:
                    image, mask, _skeleton, _aux = \
                        self.extension.paired_affine_with_aux(
                            row["image"], row["mask"], row["line_skeleton"],
                            None, self.rng)
                    images.append(image)
                    masks.append(mask)
                if not self._put(("batch", (images, masks, labels))):
                    return
        except BaseException as exc:
            self._put(("error", exc))

    def start(self) -> None:
        self.thread.start()

    def next(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        kind, value = self.items.get()
        if kind == "error":
            raise FastTrain483Error("batch prefetch failed") from value
        return value

    def close(self) -> None:
        self.stop.set()
        try:
            while True:
                self.items.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise FastTrain483Error("batch prefetch thread did not stop")


def fast_train_once(joint: Any, baseline: Any, probe: Any, extensions: Any,
                    data: Mapping[str, Any], args: argparse.Namespace,
                    staging: Path) -> dict[str, Any]:
    """Drop-in replacement for 445.train_once with bit-exact optimisations."""
    import torch

    configure_determinism()
    seed = args.seed
    if seed not in joint.SEEDS or not torch.cuda.is_available():
        raise joint.JointProbeError(
            "the probe requires CUDA and a declared seed")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base = data["base"]
    model = joint.build_model(baseline, data["ablation"], args).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(seed)
    prefetcher = OneDeepBatchPrefetcher(
        base, data["extension"], data, args, rng)

    validation_rows = data["validation_positive"] + data["validation_clean"]
    positive_count = len(data["validation_positive"])
    gold_masks = [row["mask"].astype(np.float32) / 255.0
                  for row in data["validation_positive"]]

    curve, best_key, best_state, best_step = [], None, None, -1
    start_select = int(round(args.optimizer_steps * 0.5))
    last_losses: dict[str, float] = {}
    prefetcher.start()
    try:
        for step in range(1, args.optimizer_steps + 1):
            model.train()
            rate = joint.learning_rate_at(step, args)
            for group in optimiser.param_groups:
                group["lr"] = rate

            images, masks, labels = prefetcher.next()
            value = torch.from_numpy(
                np.stack(images).astype(np.float32)[:, None] / 127.5 - 1.0
            ).to(device)
            stacked = np.stack(masks).astype(np.float32) / 255.0
            target = torch.from_numpy(stacked[:, None]).to(device)
            row_labels = torch.from_numpy(labels).to(device)

            mask_logits, objectness, row_logits = model(value)
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, size=value.shape[-2:], mode="bilinear",
                align_corners=False)
            batch_targets = []
            for index in range(len(images)):
                pieces = probe.gold_instances(stacked[index])
                if not pieces:
                    batch_targets.append([])
                    continue
                block = torch.from_numpy(
                    np.stack(pieces).astype(np.float32)).to(device)
                batch_targets.append(list(block))
            mask_part, object_part = joint.instance_loss(
                probe, mask_logits, objectness, batch_targets)
            loss = mask_part + args.objectness_weight * object_part
            supervised = agreement = torch.zeros((), device=device)
            if args.arm == "joint":
                supervised, agreement = joint.row_losses(
                    row_logits, objectness, row_labels)
                loss = loss + args.row_weight * supervised \
                    + joint.agreement_weight(step, args) * agreement
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            if not validation_is_due(
                    step, args.optimizer_steps, args.validation_interval,
                    start_select):
                continue
            last_losses = {
                "loss": float(loss.detach()),
                "mask": float(mask_part.detach()),
                "objectness": float(object_part.detach()),
                "row": float(supervised.detach()),
                "agreement": float(agreement.detach()),
            }
            union, masks_out, scores_out, rows_out = joint.predict(
                model, validation_rows, args.batch_size, device, args.arm)
            reading = joint.evaluate(
                probe, extensions, union, masks_out, scores_out, rows_out,
                positive_count, gold_masks, args)
            entry = {
                "step": step,
                "learning_rate": rate,
                "pixel_iou": reading["pixel_iou"],
                "instance_f1_050": reading["instance"]["0.50"]["f1"],
                "instance_recall_050": reading["instance"]["0.50"]["recall"],
                "row_bacc": reading["row"]["balanced_accuracy_head"],
                "loss": last_losses["loss"],
            }
            curve.append(entry)
            key = (reading["instance"]["0.50"]["f1"], reading["pixel_iou"])
            if step >= start_select and (best_key is None or key > best_key):
                best_key, best_step = key, step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            print(json.dumps({
                "arm": args.arm,
                "best_step": best_step,
                "event": "validation",
                "instance_f1_050": round(entry["instance_f1_050"], 6),
                "instance_recall_050": round(
                    entry["instance_recall_050"], 6),
                "seed": seed,
                "step": step,
            }, sort_keys=True), flush=True)
    finally:
        prefetcher.close()

    if best_state is None:
        raise joint.JointProbeError("training produced no eligible checkpoint")
    model.load_state_dict(best_state)
    union, masks_out, scores_out, rows_out = joint.predict(
        model, validation_rows, args.batch_size, device, args.arm)
    final = joint.evaluate(
        probe, extensions, union, masks_out, scores_out, rows_out,
        positive_count, gold_masks, args)
    binary = masks_out >= args.mask_threshold
    np.savez_compressed(
        staging / f"validation_instances_seed{seed}.npz",
        packed_masks=np.packbits(binary, axis=None),
        mask_shape=np.asarray(binary.shape, dtype=np.int64),
        scores=scores_out.astype(np.float32),
        row_probabilities=rows_out.astype(np.float32),
        mask_threshold=np.asarray(args.mask_threshold, dtype=np.float32),
        positive_count=np.asarray(positive_count, dtype=np.int64),
        quantised_probabilities=np.clip(
            np.rint(masks_out * 255.0), 0, 255).astype(np.uint8))
    with (staging / f"curve_seed{seed}.jsonl").open(
            "w", encoding="utf-8") as handle:
        for entry in curve:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return {
        "seed": seed,
        "best_step": best_step,
        "last_losses": last_losses,
        "validation": final,
        "curve_length": len(curve),
    }


def configure_fullres(joint: Any, baseline: Any, fusion: Any, fullres: Any,
                      args: argparse.Namespace, holder: dict[str, Any],
                      training_loop: str,
                      trace: list[dict[str, float]] | None = None,
                      augmentation_trace: list[str] | None = None) -> None:
    def build(_baseline: Any, _ablation: Any,
              run_args: argparse.Namespace) -> Any:
        model = fullres.build_model(baseline, fusion, run_args, holder)
        if trace is not None:
            digest = hashlib.sha256()
            for name, value in model.state_dict().items():
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().numpy().tobytes())
            holder["initial_state_sha256"] = digest.hexdigest()
        return model

    joint.build_model = build
    base_loss = fullres.make_loss_adapter(holder, args, fusion)

    if trace is None:
        joint.instance_loss = base_loss
    else:
        def traced_loss(probe: Any, mask_logits: Any, objectness: Any,
                        batch_targets: Sequence[Sequence[Any]]) -> tuple[Any, Any]:
            mask_part, object_part = base_loss(
                probe, mask_logits, objectness, batch_targets)
            total = mask_part + args.objectness_weight * object_part
            trace.append({
                "loss": float(total.detach()),
                "mask": float(mask_part.detach()),
                "objectness": float(object_part.detach()),
            })
            return mask_part, object_part

        joint.instance_loss = traced_loss
    joint.predict = fullres.predict
    joint.evaluate = lambda probe, extensions, union, masks, scores, rows, \
            positive_count, gold_masks, run_args: fullres.evaluate(
                probe, extensions, union, masks, scores, rows,
                positive_count, gold_masks, run_args, joint)
    if training_loop == "fast":
        joint.train_once = lambda run_baseline, probe, extensions, data, \
                run_args, staging: fast_train_once(
                    joint, run_baseline, probe, extensions, data, run_args,
                    staging)
    elif training_loop != "legacy":
        raise FastTrain483Error(f"unknown training loop: {training_loop}")
    if augmentation_trace is not None:
        untraced_train_once = joint.train_once

        def traced_train_once(run_baseline: Any, probe: Any, extensions: Any,
                              data: Mapping[str, Any], run_args: argparse.Namespace,
                              staging: Path) -> dict[str, Any]:
            extension = data["extension"]
            original_affine = extension.paired_affine_with_aux

            def traced_affine(*values: Any, **keywords: Any) -> Any:
                result = original_affine(*values, **keywords)
                digest = hashlib.sha256()
                for array in result[:2]:
                    materialised = np.asarray(array)
                    digest.update(str(materialised.shape).encode("ascii"))
                    digest.update(str(materialised.dtype).encode("ascii"))
                    digest.update(materialised.tobytes())
                augmentation_trace.append(digest.hexdigest())
                return result

            extension.paired_affine_with_aux = traced_affine
            try:
                return untraced_train_once(
                    run_baseline, probe, extensions, data, run_args, staging)
            finally:
                extension.paired_affine_with_aux = original_affine

        joint.train_once = traced_train_once
    args.arm = "instance"


def snapshot_after_last_step(joint: Any, holder: Mapping[str, Any],
                             args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    """Patch AdamW.step only to observe state after the final update."""
    import torch

    captured: dict[str, Any] = {}
    counter = 0
    original = torch.optim.AdamW.step

    def observed(optimiser: Any, *values: Any, **keywords: Any) -> Any:
        nonlocal counter
        result = original(optimiser, *values, **keywords)
        counter += 1
        if counter == args.optimizer_steps:
            captured.update({
                name: value.detach().cpu().clone()
                for name, value in holder["model"].state_dict().items()
            })
        return result

    torch.optim.AdamW.step = observed
    return captured, original


def restore_adamw_step(original: Any) -> None:
    import torch

    torch.optim.AdamW.step = original


def run_engine(args: argparse.Namespace, training_loop: str,
               *, trace_enabled: bool = False) -> tuple[
                   Any, list[Any], list[str], dict[str, Any], float,
                   dict[str, Any], list[Any]]:
    suffix = uuid.uuid4().hex
    joint = load_module(JOINT_PATH, f"hme445_for483_{training_loop}_{suffix}")
    baseline = load_module(
        BASELINE_PATH, f"hme422_for483_{training_loop}_{suffix}")
    typepilot = load_module(
        TYPEPILOT_PATH, f"hme418_for483_{training_loop}_{suffix}")
    fusion = load_module(
        FUSION_PATH, f"hme478_for483_{training_loop}_{suffix}")
    fullres = load_module(
        FULLRES_PATH, f"hme480_for483_{training_loop}_{suffix}")
    holder: dict[str, Any] = {}
    trace: list[dict[str, float]] = []
    augmentation_trace: list[str] = []
    configure_fullres(
        joint, baseline, fusion, fullres, args, holder, training_loop,
        trace if trace_enabled else None,
        augmentation_trace if trace_enabled else None)
    captured: dict[str, Any] = {}
    original_step = None
    configure_determinism()
    if trace_enabled:
        captured, original_step = snapshot_after_last_step(joint, holder, args)
    started = time.perf_counter()
    try:
        result = joint.run(args, baseline, typepilot)
        elapsed = time.perf_counter() - started
    finally:
        if original_step is not None:
            restore_adamw_step(original_step)
    modules = [joint, baseline, typepilot, fusion, fullres]
    return (result, trace, augmentation_trace, captured, elapsed,
            holder, modules)


def float_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def compare_traces(legacy: Sequence[Mapping[str, float]],
                   fast: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    first_mismatch = None
    if len(legacy) == len(fast):
        for index, (left, right) in enumerate(zip(legacy, fast), start=1):
            for name in ("loss", "mask", "objectness"):
                if float_bits(left[name]) != float_bits(right[name]):
                    first_mismatch = {
                        "step": index,
                        "field": name,
                        "legacy": left[name],
                        "fast": right[name],
                    }
                    break
            if first_mismatch is not None:
                break
    else:
        first_mismatch = {
            "field": "trace_length",
            "legacy": len(legacy),
            "fast": len(fast),
        }
    return {
        "bit_exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "steps": len(legacy),
    }


def compare_states(legacy: Mapping[str, Any],
                   fast: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    mismatches = []
    keys_match = tuple(legacy) == tuple(fast)
    if keys_match:
        for name in legacy:
            if not torch.equal(legacy[name], fast[name]):
                mismatches.append(name)
    return {
        "bit_exact": keys_match and not mismatches,
        "keys_match": keys_match,
        "mismatched_tensors": mismatches,
        "tensor_count": len(legacy),
    }


def compare_string_traces(legacy: Sequence[str],
                          fast: Sequence[str]) -> dict[str, Any]:
    mismatch = None
    if len(legacy) == len(fast):
        for index, (left, right) in enumerate(zip(legacy, fast), start=1):
            if left != right:
                mismatch = {"augmentation_call": index,
                            "legacy": left, "fast": right}
                break
    else:
        mismatch = {"field": "trace_length", "legacy": len(legacy),
                    "fast": len(fast)}
    return {"bit_exact": mismatch is None,
            "first_mismatch": mismatch, "calls": len(legacy)}


def release_run(values: Sequence[Any]) -> None:
    import torch

    for module in values:
        sys.modules.pop(module.__name__, None)
    gc.collect()
    torch.cuda.empty_cache()


def run_equivalence(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FastTrain483Error(f"create-only output exists: {output}")
    output.mkdir(parents=True)
    # Bit-exactness is only decidable under strict determinism, and this is the
    # one place that needs it: the production arms run in 445's cudnn regime.
    set_strict_determinism(True)
    legacy_args = copy.deepcopy(args)
    legacy_args.optimizer_steps = EQUIVALENCE_STEPS
    legacy_args.validation_interval = EQUIVALENCE_VALIDATION_INTERVAL
    legacy_args.output_dir = output / "legacy"
    legacy = run_engine(legacy_args, "legacy", trace_enabled=True)
    (_legacy_result, legacy_trace, legacy_augmentation, legacy_state,
     legacy_time, legacy_holder, legacy_modules) = legacy
    legacy_summary = json.loads(
        (legacy_args.output_dir / "summary.json").read_text(encoding="utf-8"))
    legacy_reading = legacy_summary["per_seed"][0]
    legacy_initial = legacy_holder.get("initial_state_sha256")
    legacy_holder.clear()
    release_run(legacy_modules)
    del legacy, legacy_modules

    fast_args = copy.deepcopy(args)
    fast_args.optimizer_steps = EQUIVALENCE_STEPS
    fast_args.validation_interval = EQUIVALENCE_VALIDATION_INTERVAL
    fast_args.output_dir = output / "fast"
    fast = run_engine(fast_args, "fast", trace_enabled=True)
    (_fast_result, fast_trace, fast_augmentation, fast_state,
     fast_time, fast_holder, fast_modules) = fast
    fast_summary = json.loads(
        (fast_args.output_dir / "summary.json").read_text(encoding="utf-8"))
    fast_reading = fast_summary["per_seed"][0]
    fast_initial = fast_holder.get("initial_state_sha256")

    trace_check = compare_traces(legacy_trace, fast_trace)
    state_check = compare_states(legacy_state, fast_state)
    augmentation_check = compare_string_traces(
        legacy_augmentation, fast_augmentation)
    report = {
        "augmentation_trace": augmentation_check,
        "determinism": "strict",
        "equivalence_steps": EQUIVALENCE_STEPS,
        "fast_curve_length": fast_reading["curve_length"],
        "fast_wall_seconds": fast_time,
        "frozen_test_touched": False,
        "initial_state": {
            "bit_exact": legacy_initial == fast_initial,
            "fast_sha256": fast_initial,
            "legacy_sha256": legacy_initial,
        },
        "legacy_curve_length": legacy_reading["curve_length"],
        "legacy_wall_seconds": legacy_time,
        "loss_trace": trace_check,
        "schema_version": "hme-revision-fast-loop-equivalence-v1",
        "seed": args.seed,
        "state_dict": state_check,
    }
    (output / "equivalence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(
                f"{digest_file(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8")
    fast_holder.clear()
    release_run(fast_modules)
    if (not augmentation_check["bit_exact"]
            or legacy_initial != fast_initial
            or not trace_check["bit_exact"]
            or not state_check["bit_exact"]):
        raise FastTrain483Error(
            "300-step legacy/fast equivalence check failed")
    return report


def recompute_checksums(directory: Path) -> None:
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (directory / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")


def finalise_run(engine_output: Path, final_output: Path,
                 args: argparse.Namespace, training_loop: str,
                 elapsed: float, holder: Mapping[str, Any],
                 modules: Sequence[Any]) -> None:
    fullres = modules[-1]
    intermediate = final_output.parent / (
        f".{final_output.name}.finalised-{uuid.uuid4().hex}")
    fullres.finalise(engine_output, intermediate, args, holder, {
        "baseline": modules[1],
        "fusion": modules[3],
        "joint": modules[0],
        "typepilot": modules[2],
    })
    schedule = {
        "early_interval": args.validation_interval \
            * (EARLY_VALIDATION_MULTIPLIER if training_loop == "fast" else 1),
        "early_rule": "step < round(optimizer_steps * 0.5)",
        "late_interval": args.validation_interval,
        "training_loop": training_loop,
        "determinism": "strict" if STRICT_DETERMINISM else "cudnn",
        "cublas_workspace_config":
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if STRICT_DETERMINISM
            else None,
    }
    summary_path = intermediate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = SCHEMA
    summary["wall_time_seconds"] = elapsed
    summary["protocol"]["validation_schedule"] = schedule
    summary["protocol"]["training_loop_optimisations"] = (
        ["numpy_target_components", "batched_component_upload",
         "validation_only_loss_materialisation", "depth_one_prefetch",
         "sparse_early_validation"] if training_loop == "fast" else [])
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest_path = intermediate / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = SCHEMA
    manifest["script_sha256"] = digest_file(Path(__file__).resolve())
    manifest["dependencies"]["fullres"] = fullres.__source_sha256__
    manifest["protocol"] = summary["protocol"]
    manifest["wall_time_seconds"] = elapsed
    manifest_path.write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    complete_path = intermediate / "complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["training_loop"] = training_loop
    complete_path.write_text(
        json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8")
    recompute_checksums(intermediate)
    intermediate.rename(final_output)


def peel_control_arguments(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    control = argparse.ArgumentParser(add_help=False)
    control.add_argument(
        "--training-loop", choices=("legacy", "fast"), default="fast")
    control.add_argument("--equivalence-check", action="store_true")
    # Off by default: 480 and 481 never enabled it, and an arm that does is in a
    # different numerical regime from the baseline it is compared against.
    control.add_argument("--strict-determinism", action="store_true")
    known, remaining = control.parse_known_args(list(argv))
    return known, remaining


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    control, remaining = peel_control_arguments(arguments)
    set_strict_determinism(control.strict_determinism)
    if digest_file(JOINT_PATH) != LOCKED_JOINT_SHA256:
        raise SystemExit("483 fast loop refused: experiment 445 changed")

    joint = load_module(JOINT_PATH, "hme445_for483_parser")
    baseline = load_module(BASELINE_PATH, "hme422_for483_parser")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for483_parser")
    fullres = load_module(FULLRES_PATH, "hme480_for483_parser")
    args = fullres.parser(joint, baseline, typepilot, remaining)
    if args.seed != SEED:
        raise SystemExit("483 fast loop refused: only seed 42 is declared")
    if args.pixel_decoder != "fullres" or args.duplicate_mode != "off" \
            or args.mask_loss != "dice":
        raise SystemExit(
            "483 fast loop requires 480 fullres_dup_off configuration")
    final_output = args.output_dir.expanduser().resolve()

    if control.equivalence_check:
        report = run_equivalence(args, final_output)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"483 fast loop refused: output already exists: {final_output}")

    engine_output = final_output.parent / (
        f".{final_output.name}.engine-{uuid.uuid4().hex}")
    run_args = copy.deepcopy(args)
    run_args.output_dir = engine_output
    result, _trace, _augmentation, _state, elapsed, holder, modules = run_engine(
        run_args, control.training_loop)
    if tuple(result.get("counts", EXPECTED_COUNTS)) != EXPECTED_COUNTS:
        raise SystemExit("483 fast loop refused: sample counts changed")
    if args.validate_only:
        print(json.dumps({
            "counts": list(EXPECTED_COUNTS),
            "frozen_test_touched": False,
            "training_loop": control.training_loop,
            "validated": True,
        }, sort_keys=True))
        return 0
    finalise_run(
        engine_output, final_output, run_args, control.training_loop,
        elapsed, holder, modules)
    validation = json.loads((final_output / "summary.json").read_text(
        encoding="utf-8"))["per_seed"][0]["validation"]
    print(json.dumps({
        "curve_length": result["curve_length"],
        "instance_f1_050_selected":
            validation["instance_selected"]["0.50"]["f1"],
        "output_dir": str(final_output),
        "pixel_iou": validation["pixel_iou"],
        "training_loop": control.training_loop,
        "wall_time_seconds": elapsed,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

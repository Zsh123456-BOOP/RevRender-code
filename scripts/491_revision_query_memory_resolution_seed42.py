#!/usr/bin/env python3
"""Query-memory resolution sweep on experiment 480's ``fullres_dup_off`` model.

Motivation, measured before this script was written.  Experiment 480 pools the
stride-4 ``quarter`` map by ``ATTENTION_STRIDE=8`` to build the cross-attention
memory the instance queries read, which leaves **72 tokens of 32x32 input
pixels each** for a whole 96x768 line.  On the fit and development positive
rows the median nearest-sibling gap between two gold revision components is
**12.0 px** (p25 6.0, p75 30.3), so the majority of co-occurring revisions fall
inside a single memory token.  That is an information bound on the query path,
and it predicts exactly the failure this project has measured repeatedly:
oracle recall 0.750 on single-gold lines against 0.284 on multi-gold lines,
with 176 of 236 development gold components living on multi-gold lines.

Each arm changes ``ATTENTION_STRIDE`` and nothing else.  Stride 8 reproduces
experiment 480 ``fullres_dup_off`` and is the same-batch reference, which
matters because seed 42 is the maximum of that arm's five-seed distribution
(0.4319 against a 0.4159 mean) and single-seed challengers were previously
judged against that inflated draw.

This script deliberately does NOT: change the encoder, the pixel decoder, the
refiner, the quality head, the fusion gate, the losses, the query count, the
query dimension or depth; add positional encoding, relative coordinates,
dynamic convolutions or query self-attention (experiment 482 tested those at
the unchanged stride and none of them can position what the memory never
resolved); read, transcribe or otherwise consume the frozen test split or M6
reference material; modify anything under ``未标注数据/`` or ``标注好数据/``;
retrain or re-evaluate any published baseline; or enable strict CUDA
determinism, which experiment 446 measured at 2.2x and which would place these
arms in a different numerical regime from the 480/481 readings they answer to.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
JOINT_PATH = ROOT / "scripts/445_revision_joint_head_probe.py"
BASELINE_PATH = ROOT / "scripts/422_revision_paper_baselines.py"
TYPEPILOT_PATH = ROOT / "scripts/418_revision_typepilot300_factorial_seed42.py"
FUSION_PATH = ROOT / "scripts/478_revision_instance_preserving_fusion_seed42.py"
FULLRES_PATH = ROOT / "scripts/480_revision_fullres_instance_seed42.py"
SIZE_PATH = ROOT / "scripts/481_revision_size_separation_seed42.py"
SPATIAL_PATH = ROOT / "scripts/482_revision_spatial_query_seed42.py"
FAST_LOOP_PATH = ROOT / "scripts/483_revision_fast_train_loop.py"
SCHEMA = "hme-revision-query-memory-resolution-seed42-v1"

SEED = 42
EXPECTED_COUNTS = (1515, 1319, 126, 379)
EXPECTED_GOLD_INSTANCES = 236

# Stride 8 is experiment 480 unchanged and serves as the same-batch reference.
ATTENTION_STRIDES = (1, 2, 4, 8)
REFERENCE_STRIDE = 8

# The measurement that motivates the sweep, recorded so the manifest carries it.
QUARTER_STRIDE = 4
LINE_CANVAS = (96, 768)
MEASURED_SIBLING_GAP_PIXELS = {"p25": 6.0, "median": 12.0, "p75": 30.3,
                               "pairs": 2089, "multi_gold_rows": 757,
                               "positive_rows": 1641}


class QueryMemory491Error(RuntimeError):
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
        raise QueryMemory491Error(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    module.__source_sha256__ = digest_file(path)
    return module


def memory_geometry(attention_stride: int) -> dict[str, Any]:
    """Token count and input footprint of the cross-attention memory."""
    if attention_stride not in ATTENTION_STRIDES:
        raise QueryMemory491Error(
            f"undeclared attention stride: {attention_stride}")
    height, width = LINE_CANVAS
    rows = height // (QUARTER_STRIDE * attention_stride)
    columns = width // (QUARTER_STRIDE * attention_stride)
    if rows < 1 or columns < 1:
        raise QueryMemory491Error(
            f"attention stride {attention_stride} leaves no memory tokens")
    footprint = QUARTER_STRIDE * attention_stride
    return {
        "attention_stride": attention_stride,
        "effective_input_stride": footprint,
        "memory_columns": columns,
        "memory_rows": rows,
        "memory_tokens": rows * columns,
        "token_footprint_pixels": [footprint, footprint],
        "sibling_gap_median_in_tokens":
            MEASURED_SIBLING_GAP_PIXELS["median"] / footprint,
    }


def apply_attention_stride(fullres: Any, attention_stride: int) -> dict[str, Any]:
    """Point experiment 480's module-level stride at this arm's value.

    ``ATTENTION_STRIDE`` is read from module globals inside the forward pass,
    so rebinding it before ``build_model`` is what changes the memory, and the
    change is visible in the returned geometry rather than implied.
    """
    if getattr(fullres, "ATTENTION_STRIDE", None) is None:
        raise QueryMemory491Error("experiment 480 no longer exposes "
                                  "ATTENTION_STRIDE")
    if fullres.ATTENTION_STRIDE != REFERENCE_STRIDE:
        raise QueryMemory491Error(
            "experiment 480 no longer ships the reference stride "
            f"{REFERENCE_STRIDE}; found {fullres.ATTENTION_STRIDE}")
    fullres.ATTENTION_STRIDE = attention_stride
    return memory_geometry(attention_stride)


def verify_built_memory(fullres: Any, model: Any,
                        geometry: Mapping[str, Any]) -> None:
    """Run one forward pass and refuse if the memory is not the declared size."""
    import torch
    from torch.nn import functional

    height, width = LINE_CANVAS
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        probe = torch.zeros(1, 1, height, width, device=device)
        features = model.encoder(probe)
        quarter = model.shared_projection(model.dense_decoder(features))
        memory = functional.avg_pool2d(
            quarter, fullres.ATTENTION_STRIDE, fullres.ATTENTION_STRIDE)
    if was_training:
        model.train()
    observed = int(memory.shape[-2]) * int(memory.shape[-1])
    if observed != int(geometry["memory_tokens"]):
        raise QueryMemory491Error(
            f"memory token count is {observed}, declared "
            f"{geometry['memory_tokens']}")


def parser(joint: Any, baseline: Any, typepilot: Any,
           argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = joint.build_parser(baseline, typepilot)
    base.description = __doc__.splitlines()[0]
    base.add_argument("--attention-stride", type=int,
                      choices=ATTENTION_STRIDES, required=True)
    base.add_argument("--dense-loss-weight", type=float, default=1.0)
    base.add_argument("--fused-loss-weight", type=float, default=1.0)
    base.add_argument("--consistency-weight", type=float, default=0.15)
    base.add_argument("--quality-weight", type=float, default=0.5)
    # Off by default; experiments 480 and 481 never enabled it, so an arm that
    # does cannot be compared with their readings without a loose variable.
    base.add_argument("--strict-determinism", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    forbidden = ("--arm", "--pixel-decoder", "--duplicate-mode", "--mask-loss")
    if any(any(value == flag or value.startswith(flag + "=")
               for flag in forbidden) for value in arguments):
        raise QueryMemory491Error(
            "experiment 491 locks the experiment-480 architecture")
    args = base.parse_args([*arguments, "--arm", "instance"])
    args.pixel_decoder = "fullres"
    args.duplicate_mode = "off"
    args.mask_loss = "dice"
    args.cldice_weight = 0.20
    args.duplicate_weight = 0.0
    return args


def finalise(engine_output: Path, final_output: Path,
             args: argparse.Namespace, holder: Mapping[str, Any],
             geometry: Mapping[str, Any],
             dependencies: Mapping[str, Any]) -> None:
    if final_output.exists() or final_output.is_symlink():
        raise QueryMemory491Error(f"create-only output exists: {final_output}")
    for name in ("SHA256SUMS", "complete.json"):
        path = engine_output / name
        if path.exists():
            path.unlink()
    summary_path = engine_output / "summary.json"
    manifest_path = engine_output / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = summary["per_seed"][0]["validation"]
    if int(validation["gold_instances"]) != EXPECTED_GOLD_INSTANCES:
        raise QueryMemory491Error("development gold-instance count changed")
    if "instance_diagnostics" not in validation:
        raise QueryMemory491Error("required instance diagnostics are absent")

    configuration = dict(holder["configuration"])
    configuration["query_memory"] = dict(geometry)
    protocol = {
        "attention_stride": args.attention_stride,
        "cublas_workspace_config":
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if args.strict_determinism else None,
        "determinism": "strict" if args.strict_determinism else "cudnn",
        "factory_instance_workpoint": {
            "mask_threshold": args.mask_threshold,
            "score_threshold": args.score_threshold,
        },
        "frozen_test_touched": False,
        "gold_area_bins_pixels": [300, 900],
        "gold_count_strata": ["one_gold", "multiple_gold"],
        "m6_or_reference_materialised": False,
        "measured_sibling_gap_pixels": dict(MEASURED_SIBLING_GAP_PIXELS),
        "primary_metric": "instance_f1_match_iou_0.50",
        "query_memory": dict(geometry),
        "reference_arm": {
            "attention_stride": REFERENCE_STRIDE,
            "equals_experiment": "480 fullres_dup_off",
        },
        "selected_instance_workpoint": validation[
            "instance_operating_point"]["selected"],
        "training_loop": "fast_bit_exact_483",
        "validation_schedule": {
            "early_interval": args.validation_interval * 4,
            "early_rule": "step < round(optimizer_steps * 0.5)",
            "late_interval": args.validation_interval,
        },
        "writer_overlap_fit_validation": 0,
    }
    summary.update({
        "architecture": configuration,
        "method": "query_memory_resolution_sweep",
        "schema_version": SCHEMA,
    })
    summary["protocol"].update(protocol)
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "architecture": configuration,
        "dependencies": {name: module.__source_sha256__
                         for name, module in dependencies.items()},
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "schema_version": SCHEMA,
        "script_sha256": digest_file(Path(__file__).resolve()),
    })
    manifest["protocol"] = summary["protocol"]
    manifest_path.write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    import torch

    torch.save({
        "architecture": configuration,
        "schema_version": "hme-revision-query-memory-checkpoint-v1",
        "seed": args.seed,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in holder["model"].state_dict().items()
        },
    }, engine_output / f"query_memory_checkpoint_seed{args.seed}.pt")
    listing = [
        f"{digest_file(path)}  {path.name}"
        for path in sorted(engine_output.iterdir())
        if path.is_file() and path.name not in {"SHA256SUMS", "complete.json"}
    ]
    (engine_output / "SHA256SUMS").write_text(
        "\n".join(listing) + "\n", encoding="utf-8")
    (engine_output / "complete.json").write_text(json.dumps({
        "attention_stride": args.attention_stride,
        "seed": args.seed,
        "status": "complete",
    }, sort_keys=True) + "\n", encoding="utf-8")
    engine_output.rename(final_output)


def main(argv: Sequence[str] | None = None) -> int:
    joint = load_module(JOINT_PATH, "hme445_for491_memory")
    baseline = load_module(BASELINE_PATH, "hme422_for491_memory")
    typepilot = load_module(TYPEPILOT_PATH, "hme418_for491_memory")
    fusion = load_module(FUSION_PATH, "hme478_for491_memory")
    fullres = load_module(FULLRES_PATH, "hme480_for491_memory")
    sizeprobe = load_module(SIZE_PATH, "hme481_for491_memory")
    spatial = load_module(SPATIAL_PATH, "hme482_for491_memory")
    fastloop = load_module(FAST_LOOP_PATH, "hme483_for491_memory")

    args = parser(joint, baseline, typepilot, argv)
    fastloop.set_strict_determinism(args.strict_determinism)
    if args.seed != SEED:
        raise SystemExit("491 query memory refused: only seed 42 is declared")
    final_output = args.output_dir.expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        raise SystemExit(
            f"491 query memory refused: output already exists: {final_output}")

    try:
        geometry = apply_attention_stride(fullres, args.attention_stride)
    except QueryMemory491Error as exc:
        raise SystemExit("491 query memory refused: " + str(exc))

    holder: dict[str, Any] = {}

    def build(_baseline: Any, _ablation: Any, run_args: argparse.Namespace):
        model = fullres.build_model(baseline, fusion, run_args, holder)
        verify_built_memory(fullres, model, geometry)
        return model

    joint.build_model = build
    joint.instance_loss = fullres.make_loss_adapter(holder, args, fusion)
    joint.predict = fullres.predict
    joint.evaluate = lambda probe, extensions, union, masks, scores, rows, \
            positive_count, gold_masks, run_args: \
        spatial.evaluate_with_diagnostics(
            probe, extensions, union, masks, scores, rows, positive_count,
            gold_masks, run_args, joint, fullres, sizeprobe)
    joint.train_once = lambda run_baseline, probe, extensions, data, \
            run_args, staging: fastloop.fast_train_once(
                joint, run_baseline, probe, extensions, data, run_args,
                staging)
    args.arm = "instance"

    if args.validate_only:
        result = joint.run(args, baseline, typepilot)
        if tuple(result["counts"]) != EXPECTED_COUNTS:
            raise SystemExit("491 query memory refused: sample counts changed")
        print(json.dumps({
            "attention_stride": args.attention_stride,
            "counts": result["counts"],
            "frozen_test_touched": False,
            "query_memory": geometry,
            "seed": args.seed,
            "validated": True,
        }, sort_keys=True))
        return 0

    final_output.parent.mkdir(parents=True, exist_ok=True)
    engine_output = final_output.parent / (
        f".{final_output.name}.engine-{uuid.uuid4().hex}")
    args.output_dir = engine_output
    try:
        joint.run(args, baseline, typepilot)
        finalise(engine_output, final_output, args, holder, geometry, {
            "baseline": baseline,
            "fastloop": fastloop,
            "fullres": fullres,
            "fusion": fusion,
            "joint": joint,
            "sizeprobe": sizeprobe,
            "spatial": spatial,
            "typepilot": typepilot,
        })
    except QueryMemory491Error as exc:
        if engine_output.is_dir():
            shutil.rmtree(engine_output)
        raise SystemExit("491 query memory refused: " + str(exc))
    except BaseException:
        if engine_output.is_dir():
            shutil.rmtree(engine_output)
        raise

    validation = json.loads((final_output / "summary.json").read_text(
        encoding="utf-8"))["per_seed"][0]["validation"]
    diagnostics = validation["instance_diagnostics"]
    print(json.dumps({
        "attention_stride": args.attention_stride,
        "gold_count_stratified_oracle_recall_050":
            diagnostics["gold_count_stratified_oracle_recall_050"],
        "gold_count_stratified_recall_050":
            diagnostics["gold_count_stratified_recall_050"],
        "instance_f1_050_factory":
            validation["instance_factory"]["0.50"]["f1"],
        "instance_f1_050_selected":
            validation["instance_selected"]["0.50"]["f1"],
        "memory_tokens": geometry["memory_tokens"],
        "oracle_recall_050": diagnostics["oracle_recall_050"]["recall"],
        "output_dir": str(final_output),
        "pixel_iou": validation["pixel_iou"],
        "seed": args.seed,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

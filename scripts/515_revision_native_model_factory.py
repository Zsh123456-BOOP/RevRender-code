#!/usr/bin/env python3
"""Official instance-segmentation models behind one RGB line-image interface.

No proxy heads and no training loop live here. Each method uses its pinned
official R50 instance configuration, including its native matching and losses.
The caller supplies RGB float32 CHW images in [0,255], already letterboxed to
96x768, and Detectron2 Instances with BitMasks/Boxes/class zero for training.
Inference returns ordinary Detectron2 Instances with boolean full-image masks.
Upstream data mappers are not called, so COCO's 800/1024px resizing is not used.

Use one method per Python process: the independent official research packages
register overlapping backbone names. Compatibility patches to those checkouts
are allowed, but their tracked diffs and hashes are included in model metadata.
The smoke flag does not reduce queries, decoder layers, point budgets or losses.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import importlib
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "hme-revision-native-model-factory-v1"
CANVAS = (96, 768)
METHODS = ("mask2former", "maskdino", "pointrend", "condinst", "maskrcnn", "solov2", "dimaskdino")
UPSTREAM_PINS = {
    "detectron2": "a2f4a8771ab77e8411c26b27f24f9489a28a2453",
    "Mask2Former": "9b0651c6c1d5b3af2e6da0589b719c514ec0d69a",
    "MaskDINO": "3831d8514a3728535ace8d4ecc7d28044c42dd14",
    "AdelaiDet": "5e19cb172b8363820b409ed1a2754fb19ad3acb8",
    "DI-MaskDINO": "6a321a511b25e2cbcca9785c3c7891fb845a0ef1",
}
SPECIFICATIONS = {
    "maskrcnn": {
        "repository": "detectron2", "package": "detectron2",
        "url": "https://github.com/facebookresearch/detectron2",
        "config": "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "meta_architecture": "GeneralizedRCNN", "pretraining_kind": "msra",
        "native_input_format": "BGR", "queries": None,
        "native_components": {"roi_heads.mask_head": "MaskRCNNConvUpsampleHead"},
    },
    "solov2": {
        "repository": "AdelaiDet", "package": "adet",
        "url": "https://github.com/aim-uofa/AdelaiDet",
        "config": "configs/SOLOv2/R50_3x.yaml",
        "meta_architecture": "SOLOv2", "pretraining_kind": "msra",
        "native_input_format": "BGR", "queries": None,
        "native_components": {"ins_head": "SOLOv2InsHead", "mask_head": "SOLOv2MaskHead"},
    },
    "dimaskdino": {
        "repository": "DI-MaskDINO", "package": "dimaskdino",
        "url": "https://github.com/CQU-ADHRI-Lab/DI-MaskDINO",
        "config": "configs/dimaskdino_r50_4scale_bs16_50ep.yaml",
        "meta_architecture": "MaskDINO", "pretraining_kind": "torchvision",
        "native_input_format": "RGB", "queries": 300, "license": "Apache-2.0",
        "native_components": {"sem_seg_head.pixel_decoder": "MaskDINOEncoder",
                              "sem_seg_head.predictor": "DIMaskDINODecoder"},
    },
    "mask2former": {
        "repository": "Mask2Former", "package": "mask2former",
        "url": "https://github.com/facebookresearch/Mask2Former",
        "config": "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        "meta_architecture": "MaskFormer", "pretraining_kind": "torchvision",
        "native_input_format": "RGB", "queries": 100,
        "native_components": {"sem_seg_head.pixel_decoder": "MSDeformAttnPixelDecoder",
                              "sem_seg_head.predictor": "MultiScaleMaskedTransformerDecoder"},
    },
    "maskdino": {
        "repository": "MaskDINO", "package": "maskdino",
        "url": "https://github.com/IDEA-Research/MaskDINO",
        "config": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        "meta_architecture": "MaskDINO", "pretraining_kind": "torchvision",
        "native_input_format": "RGB", "queries": 300,
        "native_components": {"sem_seg_head.pixel_decoder": "MaskDINOEncoder",
                              "sem_seg_head.predictor": "MaskDINODecoder"},
    },
    "pointrend": {
        "repository": "detectron2", "package": "detectron2.projects.point_rend",
        "url": "https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend",
        "config": "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml",
        "meta_architecture": "GeneralizedRCNN", "pretraining_kind": "msra",
        "native_input_format": "BGR", "queries": None,
        "native_components": {"roi_heads.mask_head": "PointRendMaskHead"},
    },
    "condinst": {
        "repository": "AdelaiDet", "package": "adet",
        "url": "https://github.com/aim-uofa/AdelaiDet",
        "config": "configs/CondInst/MS_R_50_1x.yaml",
        "meta_architecture": "CondInst", "pretraining_kind": "msra",
        "native_input_format": "BGR", "queries": None,
        "native_components": {"proposal_generator": "FCOS", "mask_head": "DynamicMaskHead"},
    },
}
_ACTIVE_METHOD: str | None = None


class NativeFactory515Error(RuntimeError):
    """The official-source, input or pretrained-backbone contract failed."""


def describe_model(method: str) -> dict[str, Any]:
    if method not in SPECIFICATIONS:
        raise NativeFactory515Error(f"unknown native method: {method}")
    spec = copy.deepcopy(SPECIFICATIONS[method])
    spec.update({
        "schema": SCHEMA, "method": method, "num_classes": 1,
        "backbone": "ResNet-50 ImageNet", "input_format": "RGB",
        "input_dtype": "float32", "input_range": [0, 255],
        "canvas_hw": list(CANVAS), "native_padding_allowed": True,
        "native_dataset_mapper_used": False, "mask_threshold": 0.5,
        "detection_limit_per_image": 100,
        "pretraining_url": "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/"
            + ("torchvision" if spec["pretraining_kind"] == "torchvision" else "MSRA") + "/R-50.pkl",
        "expected_upstream_commits": {name: UPSTREAM_PINS[name] for name in
                                      {"detectron2", spec["repository"]}},
        "scope": "native architecture under the project's controlled data/canvas/budget; not the upstream COCO training recipe",
        "smoke_changes_architecture": False})
    return spec


def get_pretraining_kind(method: str) -> str:
    return describe_model(method)["pretraining_kind"]


def validate_pretraining_author(kind: str, author: str | None) -> None:
    allowed = {"torchvision": {"torchvision"}, "msra": {"Caffe2", "MSRA"}}
    if author not in allowed.get(kind, set()):
        raise NativeFactory515Error(
            f"checkpoint author {author!r} does not match {kind} normalization/stride convention")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(path), *arguments],
                            check=False, capture_output=True)
    if result.returncode:
        raise NativeFactory515Error(f"cannot inspect upstream Git checkout: {path}")
    return result.stdout


def inspect_upstream(method: str, upstream_root: str | Path) -> dict[str, Any]:
    spec = describe_model(method)
    root = Path(upstream_root).expanduser().resolve()
    result = {}
    for name, expected in spec["expected_upstream_commits"].items():
        path = root / name
        actual = _git(path, "rev-parse", "HEAD").decode().strip()
        if actual != expected:
            raise NativeFactory515Error(f"{name} commit {actual} differs from pinned {expected}")
        changed = _git(path, "diff", "HEAD", "--name-only").decode().splitlines()
        patch = _git(path, "diff", "HEAD", "--binary")
        result[name] = {
            "path": str(path), "commit": actual,
            "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
            "tracked_patch_present": bool(patch),
            "modified_tracked_files": {filename: digest_file(path / filename)
                                       if (path / filename).is_file() else "deleted"
                                       for filename in changed}}
    config = root / spec["repository"] / spec["config"]
    if not config.is_file():
        raise NativeFactory515Error(f"official config is missing: {config}")
    result["selected_config"] = {"path": str(config), "sha256": digest_file(config)}
    return result


def _activate_source(package: str, repository: Path) -> None:
    repository = repository.resolve()
    if package in sys.modules:
        loaded = getattr(sys.modules[package], "__file__", None)
        if loaded is None or not Path(loaded).resolve().is_relative_to(repository):
            raise NativeFactory515Error(f"{package} is already imported from another checkout")
    value = str(repository)
    if value not in sys.path:
        sys.path.insert(0, value)


def _register_and_configure(method: str, root: Path, device: str) -> Any:
    global _ACTIVE_METHOD
    if _ACTIVE_METHOD is not None and _ACTIVE_METHOD != method:
        raise NativeFactory515Error("use a separate Python process per official method to avoid registry collisions")
    spec = SPECIFICATIONS[method]
    _activate_source("detectron2", root / "detectron2")
    _activate_source(spec["package"].split(".")[0], root / spec["repository"])
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog

    if method in ("mask2former", "maskdino", "dimaskdino"):
        # Each pinned project builds its identically named CUDA extension in
        # place. Select that project's binary, never a shared global install.
        ops = root / spec["repository"] / spec["package"] / "modeling/pixel_decoder/ops"
        _activate_source("MultiScaleDeformableAttention", ops)
        from detectron2.projects.deeplab import add_deeplab_config
        package = importlib.import_module(spec["package"])
        cfg = get_cfg()
        add_deeplab_config(cfg)
        (package.add_maskformer2_config if method == "mask2former"
         else package.add_dimaskdino_config if method == "dimaskdino"
         else package.add_maskdino_config)(cfg)
    elif method == "pointrend":
        from detectron2.projects.point_rend import add_pointrend_config
        cfg = get_cfg()
        add_pointrend_config(cfg)
    elif method == "maskrcnn":
        cfg = get_cfg()
    else:
        import adet  # noqa: F401 -- official registration
        from adet.config import get_cfg as get_adet_cfg
        cfg = get_adet_cfg()
        if method == "solov2":
            importlib.import_module("adet.modeling.solov2")
    cfg.merge_from_file(str(root / spec["repository"] / spec["config"]))
    _ACTIVE_METHOD = method
    configure_task(cfg, method, device)
    dataset_name = "hme_native515_" + method
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["revision"], thing_dataset_id_to_contiguous_id={1: 0})
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    return cfg


def configure_task(cfg: Any, method: str, device: str) -> None:
    """Explicit task adaptations; never replace native heads or loss functions."""
    spec = describe_model(method)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = ""  # only the supplied local checkpoint may be loaded
    cfg.INPUT.FORMAT = spec["native_input_format"]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = (CANVAS[0],)
    cfg.INPUT.MAX_SIZE_TRAIN = CANVAS[1]
    cfg.INPUT.MIN_SIZE_TEST = CANVAS[0]
    cfg.INPUT.MAX_SIZE_TEST = CANVAS[1]
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    if method in ("mask2former", "maskdino", "dimaskdino"):
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        head = cfg.MODEL.MASK_FORMER if method == "mask2former" else cfg.MODEL.MaskDINO
        head.TEST.SEMANTIC_ON = False
        head.TEST.PANOPTIC_ON = False
        head.TEST.INSTANCE_ON = True
        head.SIZE_DIVISIBILITY = 32
    elif method in ("pointrend", "maskrcnn"):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        if method == "pointrend":
            cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    elif method == "solov2":
        cfg.MODEL.SOLOV2.NUM_CLASSES = 1
        cfg.MODEL.SOLOV2.SCORE_THR = 0.0
        cfg.MODEL.SOLOV2.UPDATE_THR = 0.0
        cfg.MODEL.SOLOV2.MAX_PER_IMG = 100
    else:
        cfg.MODEL.FCOS.NUM_CLASSES = 1
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.0
        cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
        cfg.MODEL.BOXINST.ENABLED = False


def verify_native_components(model: Any, method: str) -> dict[str, Any]:
    spec = describe_model(method)
    objects = {"meta_architecture": model}
    if type(model).__name__ != spec["meta_architecture"]:
        raise NativeFactory515Error("model is not the required official meta architecture")
    for dotted, expected in spec["native_components"].items():
        component = model
        for name in dotted.split("."):
            component = getattr(component, name)
        if type(component).__name__ != expected:
            raise NativeFactory515Error(f"native component {dotted} must be {expected}")
        objects[dotted] = component
    return {name: {"class": type(value).__name__, "module": type(value).__module__,
                   "source": inspect.getfile(type(value))}
            for name, value in objects.items()}


def native_training_instances(method: str, instances: Any) -> Any:
    """Match the upstream GT container without mutating the caller's Instances.

    The two MaskFormer-derived dataset mappers store an NxHxW tensor in
    gt_masks; PointRend and CondInst consume Detectron2 BitMasks instead.
    """
    if method not in ("mask2former", "maskdino", "dimaskdino"):
        return instances
    fields = dict(instances.get_fields())
    fields["gt_masks"] = fields["gt_masks"].tensor
    return type(instances)(instances.image_size, **fields)


def native_execution_context(method: str, torch: Any) -> Any:
    """Keep official JIT losses, disabling only M2F's failing optimization path.

    On torch 2.5.1+cu121 the real batch_dice_loss_jit fails with "Global alloc
    not supported yet" after target counts change across calls. Unoptimized
    execution preserves the ScriptFunction and equations, including gradients.
    The context restores the previous JIT setting after the native forward.
    """
    return torch.jit.optimized_execution(False) if method == "mask2former" else nullcontext()


def _load_pretrained_backbone(model: Any, path: Path, kind: str, torch: Any) -> dict[str, Any]:
    from detectron2.checkpoint import DetectionCheckpointer

    class AuditedCheckpointer(DetectionCheckpointer):
        def _load_file(self, filename):
            checkpoint = super()._load_file(filename)
            self.source_author = checkpoint.get("__author__")
            validate_pretraining_author(kind, self.source_author)
            keys = list(checkpoint["model"])
            if any(any(token in key for token in ("roi_heads.", "sem_seg_head.", "proposal_generator.", "mask_head."))
                   for key in keys):
                raise NativeFactory515Error("supplied checkpoint contains task heads; ImageNet-only initialization required")
            return checkpoint

        def _load_model(self, checkpoint):
            result = super()._load_model(checkpoint)
            self.incompatibility = result
            return result

    # FPN is deliberately randomly initialized; only the ResNet bottom-up must
    # be completely supplied by ImageNet pretraining.
    prefix = "backbone.bottom_up." if hasattr(model.backbone, "bottom_up") else "backbone."
    trunk_keys = {name for name in model.state_dict() if name.startswith(prefix)
                  and not name.endswith("num_batches_tracked")}
    stem_key = next((name for name in trunk_keys if name.endswith("stem.conv1.weight")), None)
    if stem_key is None:
        raise NativeFactory515Error("official ResNet stem not found")
    stem_before = model.state_dict()[stem_key].detach().clone()
    checkpointer = AuditedCheckpointer(model)
    checkpointer.load(str(path))
    incompatible = checkpointer.incompatibility
    missing = sorted(trunk_keys.intersection(incompatible.missing_keys))
    wrong_shapes = [str(item) for item in incompatible.incorrect_shapes if item[0] in trunk_keys]
    if missing or wrong_shapes:
        raise NativeFactory515Error(f"ImageNet ResNet was not completely loaded: missing={missing}, shapes={wrong_shapes}")
    if torch.equal(stem_before, model.state_dict()[stem_key]):
        raise NativeFactory515Error("pretrained stem did not change from random initialization")
    return {"path": str(path), "sha256": digest_file(path), "kind": kind,
            "source_author": checkpointer.source_author, "resnet_state_keys": len(trunk_keys),
            "missing_resnet_keys": missing, "incorrect_resnet_shapes": wrong_shapes,
            "stem_changed_from_initialization": True,
            "unexpected_keys": list(incompatible.unexpected_keys),
            "missing_new_head_keys": list(incompatible.missing_keys)}


def make_input_adapter(native: Any, method: str, cfg: Any, manifest: dict[str, Any], torch: Any) -> Any:
    """Normalize the external contract while retaining the native forward/loss."""
    from detectron2.structures import BitMasks

    class NativeLineInstanceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native_model = native
            self.native_manifest = manifest
            self.native_cfg = cfg
            self.baseline_metadata = manifest
            self.native_config_yaml = cfg.dump() if hasattr(cfg, "dump") else None

        def forward(self, batch):
            if not isinstance(batch, list) or not batch:
                raise NativeFactory515Error("expected a nonempty Detectron2 input list")
            converted = []
            for row in batch:
                image = row["image"]
                if image.dtype != torch.float32 or tuple(image.shape) != (3, *CANVAS):
                    raise NativeFactory515Error("input must be RGB float32 CHW 3x96x768 in [0,255]")
                if row.get("height", CANVAS[0]) != CANVAS[0] or row.get("width", CANVAS[1]) != CANVAS[1]:
                    raise NativeFactory515Error("input height/width metadata must match the 96x768 canvas")
                item = dict(row)
                if cfg.INPUT.FORMAT == "BGR":
                    item["image"] = image[[2, 1, 0], :, :].contiguous()
                item["height"], item["width"] = CANVAS
                if self.training:
                    if "instances" not in item or not item["instances"].has("gt_masks") \
                            or not isinstance(item["instances"].gt_masks, BitMasks):
                        raise NativeFactory515Error("training requires Instances with BitMasks, including empty rows")
                    item["instances"] = native_training_instances(method, item["instances"])
                else:
                    # In particular, CondInst otherwise computes GT-dependent
                    # auxiliary data during eval when an instances field exists.
                    item.pop("instances", None)
                converted.append(item)
            with native_execution_context(method, torch):
                result = self.native_model(converted)
            if self.training:
                if not isinstance(result, dict) or not result:
                    raise NativeFactory515Error("native training did not return a loss dictionary")
                return result
            if not isinstance(result, list) or len(result) != len(batch):
                raise NativeFactory515Error("native inference returned an invalid batch")
            for output in result:
                instances = output["instances"]
                if not instances.has("pred_masks"):
                    if len(instances):
                        raise NativeFactory515Error("nonempty native instances have no masks")
                    instances.pred_masks = torch.zeros((0, *CANVAS), dtype=torch.bool,
                                                       device=instances.scores.device)
                masks = instances.pred_masks
                if method == "solov2" and len(instances) == 0:
                    masks = torch.zeros((0, *CANVAS), dtype=torch.bool, device=masks.device)
                if masks.ndim != 3 or tuple(masks.shape[1:]) != CANVAS:
                    raise NativeFactory515Error("native inference masks are not Nx96x768")
                instances.pred_masks = masks > 0.5
                if method == "mask2former":
                    # Official Mask2Former emits zero boxes by default. Compute
                    # output boxes from the already-final masks; no mask changes.
                    instances.pred_boxes = BitMasks(instances.pred_masks).get_bounding_boxes()
            return result

    return NativeLineInstanceModel()


def build_model(method: str, upstream_root: str | Path, weights_path: str | Path,
                device: str = "cuda", seed: int = 42, smoke: bool = False) -> Any:
    spec = describe_model(method)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise NativeFactory515Error("seed must be an integer in [0, 2**32)")
    path = Path(weights_path).expanduser().resolve()
    if path.suffix != ".pkl" or not path.is_file():
        raise NativeFactory515Error("supply the official local ImageNet R-50.pkl, not an arbitrary torchvision .pth")
    upstream = inspect_upstream(method, upstream_root)
    _activate_source("detectron2", Path(upstream_root).expanduser().resolve() / "detectron2")
    try:
        import numpy as np
        import torch
        from detectron2.utils.logger import setup_logger
    except (ImportError, OSError) as error:
        raise NativeFactory515Error(f"native environment is missing dependencies: {error}") from error
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise NativeFactory515Error("CUDA device requested but unavailable")
        torch.cuda.manual_seed_all(seed)
    setup_logger(name="detectron2")
    cfg = _register_and_configure(method, Path(upstream_root).expanduser().resolve(), device)
    cfg.freeze()
    from detectron2.modeling import build_model as build_detectron2_model
    native = build_detectron2_model(cfg)
    components = verify_native_components(native, method)
    weights = _load_pretrained_backbone(native, path, spec["pretraining_kind"], torch)
    manifest = {
        **spec, "seed": seed, "device": device, "smoke": bool(smoke),
        "upstream": upstream, "native_components_observed": components,
        "pretrained_load": weights, "resolved_config_yaml": cfg.dump(),
        "factory_sha256": digest_file(Path(__file__)),
        "inference_score_floor": 0.0 if method in ("pointrend", "condinst", "maskrcnn", "solov2") else "native top-k",
        "inference_native_nms_preserved": True,
        "input_channel_conversion": "RGB to BGR" if cfg.INPUT.FORMAT == "BGR" else "none",
        "native_training_gt_mask_type": "Tensor" if method in ("mask2former", "maskdino", "dimaskdino") else "BitMasks",
        "runtime_compatibility": {
            "torchscript_optimized_execution_during_native_forward": False,
            "scope": "Mask2Former forward only; previous global setting restored on exit",
            "reason": "torch 2.5.1+cu121 native batch_dice_loss_jit fails with Global alloc not supported yet for varying target counts",
            "native_script_functions_and_loss_equations_preserved": True,
            "verification": "official Dice/CE cost functions at 0/1/2 targets match eager values; scripted Dice backward matches eager gradients",
        } if method == "mask2former" else {},
        "mask2former_boxes_derived_from_final_masks": method == "mask2former",
        "external_trainer_controls_optimizer_schedule_and_augmentation": True}
    manifest["native_optimizer_defaults"] = {
        "optimizer": getattr(cfg.SOLVER, "OPTIMIZER", "SGD"),
        "base_lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        "backbone_multiplier": getattr(cfg.SOLVER, "BACKBONE_MULTIPLIER", 1.0),
        "backbone_freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
        "gradient_clip_enabled": cfg.SOLVER.CLIP_GRADIENTS.ENABLED,
        "gradient_clip_type": cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE,
        "gradient_clip_value": cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
        "note": "configuration provenance only; external trainer must record its actual optimizer"}
    if method in ("mask2former", "maskdino", "dimaskdino"):
        extension = Path(sys.modules["MultiScaleDeformableAttention"].__file__).resolve()
        manifest["deformable_attention_extension"] = {"path": str(extension), "sha256": digest_file(extension)}
    return make_input_adapter(native, method, cfg, manifest, torch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--describe", action="store_true",
                        help="print the static model contract without importing ML libraries")
    args = parser.parse_args(argv)
    print(json.dumps(describe_model(args.method), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

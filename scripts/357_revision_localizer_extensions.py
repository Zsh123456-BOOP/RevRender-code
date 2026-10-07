#!/usr/bin/env python3
"""Pure mechanisms used by the development-only revision-localizer ablation."""

from __future__ import annotations

import html
import io
import json
import math
import random
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


def make_unet(in_channels: int, base_channels: int):
    """Return the existing shallow U-Net with a parameterised input width.

    The output remains exactly one segmentation channel and one prediction head.
    """
    import torch
    from torch import nn

    if in_channels not in (1, 2, 3, 4):
        raise ValueError("in_channels must be in [1,4]")
    if base_channels <= 0:
        raise ValueError("base_channels must be positive")

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class SingleHeadUNet(nn.Module):
        def __init__(self, channels: int, width: int):
            super().__init__()
            self.input_channels = channels
            self.output_channels = 1
            self.head_count = 1
            self.d1 = block(channels, width)
            self.d2 = block(width, width * 2)
            self.d3 = block(width * 2, width * 4)
            self.bottom = block(width * 4, width * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
            self.u2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
            self.u1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
            self.c3 = block(width * 8, width * 4)
            self.c2 = block(width * 4, width * 2)
            self.c1 = block(width * 2, width)
            self.head = nn.Conv2d(width, 1, 1)

        def forward(self, x):
            s1 = self.d1(x)
            s2 = self.d2(self.pool(s1))
            s3 = self.d3(self.pool(s2))
            bottom = self.bottom(self.pool(s3))
            value = self.c3(torch.cat([self.u3(bottom), s3], dim=1))
            value = self.c2(torch.cat([self.u2(value), s2], dim=1))
            value = self.c1(torch.cat([self.u1(value), s1], dim=1))
            output = self.head(value)
            if output.ndim != 4 or output.shape[1] != 1:
                raise RuntimeError("single-head U-Net output contract changed")
            return output

    return SingleHeadUNet(in_channels, base_channels)


def _line_endpoints(mark: Mapping[str, Any]):
    if mark.get("type") != "line":
        return None
    path = mark.get("path")
    if isinstance(path, list) and len(path) >= 2:
        return ((float(path[0][0]), float(path[0][1])),
                (float(path[-1][0]), float(path[-1][1])))
    if all(name in mark for name in ("x1", "y1", "x2", "y2")):
        return ((float(mark["x1"]), float(mark["y1"])),
                (float(mark["x2"]), float(mark["y2"])))
    return None


def _canonical_line_key(mark: Mapping[str, Any]):
    endpoints = _line_endpoints(mark)
    if endpoints is None:
        raise ValueError("line mark lacks endpoints")
    ordered = tuple(sorted(endpoints))
    return (*ordered[0], *ordered[1], float(mark.get("stroke_width") or 3.0),
            json.dumps(dict(mark), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")))


def bridge_near_line_endpoints(
    marks: Sequence[Mapping[str, Any]], *, radius_factor: float,
    default_stroke_width: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every source mark and add a deterministic minimum bridge forest.

    Direction is intentionally ignored.  Candidate endpoints are eligible when
    their Euclidean distance is at most ``radius_factor`` times the pair's mean
    stroke width.  Kruskal's algorithm prevents redundant bridges and preserves
    branches instead of collapsing them into one polyline.
    """
    if radius_factor < 0 or default_stroke_width <= 0:
        raise ValueError("invalid endpoint-merge configuration")
    output = deepcopy(list(marks))
    nodes = []
    for mark in marks:
        endpoints = _line_endpoints(mark)
        if endpoints is None:
            continue
        width = float(mark.get("stroke_width") or default_stroke_width)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("line stroke width must be finite and positive")
        nodes.append({
            "endpoints": endpoints,
            "key": _canonical_line_key(mark),
            "width": width,
        })
    nodes.sort(key=lambda value: value["key"])
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        return True

    candidates = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            radius = radius_factor * (nodes[left]["width"] + nodes[right]["width"]) / 2.0
            for left_index, left_point in enumerate(nodes[left]["endpoints"]):
                for right_index, right_point in enumerate(nodes[right]["endpoints"]):
                    distance = math.dist(left_point, right_point)
                    if distance <= radius + 1e-12:
                        canonical_points = tuple(sorted((left_point, right_point)))
                        candidates.append((
                            distance, canonical_points, left, right,
                            left_index, right_index, radius,
                        ))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3],
                                      item[4], item[5]))
    bridges, lengths = [], []
    for distance, _, left, right, left_index, right_index, radius in candidates:
        if not union(left, right):
            continue
        if distance <= 1e-12:
            continue
        left_point = nodes[left]["endpoints"][left_index]
        right_point = nodes[right]["endpoints"][right_index]
        width = (nodes[left]["width"] + nodes[right]["width"]) / 2.0
        bridge = {
            "derived": "endpoint_bridge_v1",
            "shape": "line",
            "stroke_width": max(1, int(round(width))),
            "type": "line",
            "x1": left_point[0], "y1": left_point[1],
            "x2": right_point[0], "y2": right_point[1],
        }
        if distance > radius + 1e-9:
            raise AssertionError("bridge exceeded its eligible radius")
        bridges.append(bridge)
        lengths.append(distance)
    bridges.sort(key=_canonical_line_key)
    output.extend(bridges)
    return output, {
        "bridge_length": float(sum(lengths)),
        "bridges_added": len(bridges),
        "candidate_pairs": len(candidates),
        "max_bridge_length": max(lengths, default=0.0),
    }


COMMAND_MAP = {
    r"\therefore": "∴", r"\because": "∵", r"\times": "×",
    r"\cdot": "·", r"\div": "÷", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\parallel": "∥", r"\perp": "⊥",
    r"\angle": "∠", r"\triangle": "△", r"\Delta": "△",
    r"\pm": "±", r"\infty": "∞", r"\circ": "°", r"\sqrt": "√",
}


def canonicalize_ocr_text(raw: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(raw)))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"```(?:markdown|latex|tex|json)?", "", value, flags=re.I)
    value = value.replace("$$", "").replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "").replace("$", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for command, symbol in COMMAND_MAP.items():
        value = value.replace(command, symbol)
    value = re.sub(
        r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}",
        r"\1", value)
    value = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*", "", value)
    value = re.sub(r"\\(?:quad|qquad)\b|\\[,;:!]", "", value)
    value = value.replace("\\\\", "")
    value = value.translate(str.maketrans({"：": ":", "，": ",", "。": ".", "；": ";"}))
    value = re.sub(r"[{}]", "", value)
    return re.sub(r"\s+", "", value)


def _semantic_display(raw: str) -> str:
    value = canonicalize_ocr_text(raw).replace(r"\frac", "⁄")
    return re.sub(r"\\[A-Za-z]+", "□", value) or "□"


def build_qwen_text_lookup(source_path: Path, prediction_path: Path,
                           retry_path: Path | None = None):
    """Load label-blind full-line OCR and expand its canonical aliases."""
    sources, predictions = [], []
    with Path(source_path).open("r", encoding="utf-8") as handle:
        sources = [json.loads(line) for line in handle if line.strip()]
    with Path(prediction_path).open("r", encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle if line.strip()]
    if len(sources) != len(predictions):
        raise ValueError("Qwen source/prediction row count mismatch")
    retries = {}
    if retry_path is not None:
        with Path(retry_path).open("r", encoding="utf-8") as handle:
            retries = {str(row["crop_id"]): row for row in
                       (json.loads(line) for line in handle if line.strip())}
    lookup, valid_count = {}, 0
    for source, prediction in zip(sources, predictions):
        crop_id = str(source["crop_id"])
        if str(prediction.get("crop_id")) != crop_id:
            raise ValueError("Qwen row identity mismatch")
        chosen = retries.get(crop_id, prediction)
        text = str(chosen.get("model_text", "")).strip()
        valid = bool(text) and not bool(chosen.get("empty_output")) \
            and not bool(chosen.get("truncated_at_max_new_tokens")) \
            and bool(canonicalize_ocr_text(text))
        valid_count += int(valid)
        record = {"crop_id": crop_id, "text": text, "valid": valid}
        aliases = list(source.get("alias_line_ids") or [])
        representative = source.get("representative_line_id")
        if representative:
            aliases.append(representative)
        for alias in aliases:
            key = str(alias)
            if key in lookup and lookup[key]["crop_id"] != crop_id:
                raise ValueError(f"duplicate Qwen line alias: {key}")
            lookup[key] = record
    return lookup, {
        "canonical_crops": len(sources),
        "logical_line_ids": len(lookup),
        "valid_crops": valid_count,
    }


def render_ocr_channel(text: str, font_path: Path, width: int, height: int, *,
                       valid: bool = True):
    """Render a deliberately soft, whole-line glyph prior without character boxes."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    if width <= 0 or height <= 0:
        raise ValueError("OCR channel dimensions must be positive")
    if not valid:
        return np.zeros((height, width), dtype=np.float32)
    display = _semantic_display(text)
    font = ImageFont.truetype(str(font_path), size=max(12, int(round(height * 0.50))))
    probe = Image.new("L", (16, 16), 0)
    box = ImageDraw.Draw(probe).textbbox((0, 0), display, font=font)
    glyph_width, glyph_height = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    glyph = Image.new("L", (glyph_width + 12, glyph_height + 12), 0)
    ImageDraw.Draw(glyph).text((6 - box[0], 6 - box[1]), display, fill=255, font=font)
    margin = max(2, int(round(height * 0.08)))
    scale = min((width - 2 * margin) / glyph.width,
                (height - 2 * margin) / glyph.height)
    resized = glyph.resize((max(1, int(round(glyph.width * scale))),
                            max(1, int(round(glyph.height * scale)))),
                           Image.Resampling.BILINEAR)
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(resized, ((width - resized.width) // 2,
                           (height - resized.height) // 2))
    # Low-frequency on purpose: no character coordinates imply no pixel-accurate
    # alignment.  The channel is a soft normal-writing prior, not a target mask.
    small = canvas.resize((max(1, width // 4), max(1, height // 4)),
                          Image.Resampling.BILINEAR)
    soft = small.resize((width, height), Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(radius=max(0.5, height / 96.0)))
    return np.asarray(soft, dtype=np.float32) / 255.0


def paired_affine_with_aux(image, mask, skeleton, aux, rng: random.Random):
    """Apply one geometry to image/target/skeleton/aux; photometry only to image."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    if rng.random() > 0.8:
        return image, mask, skeleton, aux
    height, width = image.shape
    angle = rng.uniform(-3.0, 3.0)
    scale = 1.0 + rng.uniform(-0.06, 0.06)
    dx, dy = rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height
    radians = math.radians(angle)
    cosine, sine = math.cos(radians) / scale, math.sin(radians) / scale
    cx, cy = width / 2.0, height / 2.0
    matrix = (cosine, sine, cx - cosine * cx - sine * cy - dx,
              -sine, cosine, cy + sine * cx - cosine * cy - dy)

    def warp(array, resample, fill):
        return Image.fromarray(array).transform(
            (width, height), Image.Transform.AFFINE, matrix,
            resample=resample, fillcolor=fill)

    picture = warp(image, Image.Resampling.BILINEAR, 255)
    target = warp(mask, Image.Resampling.NEAREST, 0)
    centreline = warp(skeleton, Image.Resampling.NEAREST, 0)
    auxiliary = None
    if aux is not None:
        aux_array = np.asarray(aux)
        if aux_array.ndim == 2:
            aux_array = aux_array[None]
            squeeze_aux = True
        elif aux_array.ndim == 3:
            squeeze_aux = False
        else:
            raise ValueError("auxiliary channel must be H x W or C x H x W")
        warped_aux = []
        for channel in aux_array:
            aux_u8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
            warped_aux.append(np.asarray(
                warp(aux_u8, Image.Resampling.BILINEAR, 0),
                dtype=np.float32).copy() / 255.0)
        auxiliary = np.stack(warped_aux)
        if squeeze_aux:
            auxiliary = auxiliary[0]
    picture = ImageEnhance.Contrast(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    picture = ImageEnhance.Brightness(picture).enhance(1.0 + rng.uniform(-0.12, 0.12))
    if rng.random() < 0.20:
        picture = picture.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8)))
    if rng.random() < 0.15:
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=rng.randint(55, 92))
        buffer.seek(0)
        picture = Image.open(buffer).convert("L")
    return (
        np.asarray(picture, dtype=np.uint8).copy(),
        np.asarray(target, dtype=np.uint8).copy(),
        np.asarray(centreline, dtype=np.uint8).copy(),
        auxiliary,
    )


def scnp_logits_binary(logits, target, kernel_size: int = 3):
    """Worst same-class neighbourhood logits from SCNP Eq. 1 (binary case)."""
    import torch
    from torch.nn import functional as functional

    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    truth = target >= 0.5
    foreground = torch.where(truth, logits, torch.full_like(logits, torch.inf))
    foreground_min = -functional.max_pool2d(
        -foreground, kernel_size, stride=1, padding=kernel_size // 2)
    background = torch.where(~truth, logits, torch.full_like(logits, -torch.inf))
    background_max = functional.max_pool2d(
        background, kernel_size, stride=1, padding=kernel_size // 2)
    output = torch.where(truth, foreground_min, background_max)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("SCNP neighbourhood logits became non-finite")
    return output


def _positive_ce_dice(logits, target):
    import torch
    from torch.nn import functional as functional

    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    target_sum = target.sum(dimensions)
    positive = target_sum > 0
    if not bool(positive.any()):
        dice = torch.zeros((), device=logits.device, dtype=logits.dtype)
    else:
        numerator = 2.0 * (probability * target).sum(dimensions) + 1.0
        denominator = probability.sum(dimensions) + target_sum + 1.0
        dice = (1.0 - numerator[positive] / denominator[positive]).mean()
    return functional.binary_cross_entropy_with_logits(logits, target) + dice


def isolated_false_positive_proxy_loss(logits, target, kernel_size: int = 3):
    """SCNP-inspired extra CE+positive-Dice term used beside the base loss.

    This is named a proxy because the experiment applies a configurable weight;
    it must not be reported as an exact reproduction of another paper's recipe.
    """
    return _positive_ce_dice(scnp_logits_binary(logits, target, kernel_size), target)


def _label_components(mask, connectivity: int):
    import numpy as np

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    binary = np.asarray(mask, dtype=bool)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        count, labels = cv2.connectedComponents(
            binary.astype(np.uint8), connectivity=connectivity)
        areas = np.bincount(labels.ravel(), minlength=count)[1:].astype(int).tolist()
        return labels.astype(np.int32, copy=False), areas
    labels = np.zeros(binary.shape, dtype=np.int32)
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbours += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    component_areas = []
    next_label = 0
    height, width = binary.shape
    for y, x in zip(*np.nonzero(binary)):
        if labels[y, x]:
            continue
        next_label += 1
        stack, pixels = [(int(y), int(x))], []
        labels[y, x] = next_label
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy, dx in neighbours:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width \
                        and binary[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = next_label
                    stack.append((ny, nx))
        component_areas.append(len(pixels))
    return labels, component_areas


def component_metrics(probabilities, targets, threshold: float, *,
                      connectivity: int = 8, match_iou: float = 0.10):
    """Measure component detection without any size filter or deletion rule."""
    import numpy as np

    probability = np.asarray(probabilities)
    truth = np.asarray(targets)
    if probability.shape != truth.shape or probability.ndim != 3:
        raise ValueError("component inputs must be equal N x H x W arrays")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must be in (0,1]")
    true_positive = false_positive = false_negative = 0
    matched_ious, beta_errors = [], []
    total_predicted = total_truth = 0
    for predicted_mask, truth_mask in zip(probability >= threshold, truth > 0.5):
        predicted_labels, predicted_areas = _label_components(
            predicted_mask, connectivity)
        truth_labels, truth_areas = _label_components(truth_mask, connectivity)
        predicted_count, truth_count = len(predicted_areas), len(truth_areas)
        total_predicted += predicted_count
        total_truth += truth_count
        beta_errors.append(abs(predicted_count - truth_count))
        adjacency = [[] for _ in range(predicted_count)]
        iou_lookup = {}
        if predicted_count and truth_count:
            joint = predicted_labels.astype(np.int64) * (truth_count + 1) \
                + truth_labels.astype(np.int64)
            overlap = np.bincount(
                joint.ravel(), minlength=(predicted_count + 1) * (truth_count + 1)
            ).reshape(predicted_count + 1, truth_count + 1)
            predicted_indices, truth_indices = np.nonzero(overlap[1:, 1:])
            for predicted_index, truth_index in zip(
                    predicted_indices.tolist(), truth_indices.tolist()):
                intersection = int(overlap[predicted_index + 1, truth_index + 1])
                union = (predicted_areas[predicted_index]
                         + truth_areas[truth_index] - intersection)
                iou = intersection / union
                if iou >= match_iou:
                    adjacency[predicted_index].append(truth_index)
                    iou_lookup[(predicted_index, truth_index)] = iou
        for predicted_index, candidates in enumerate(adjacency):
            candidates.sort(key=lambda index: (
                -iou_lookup[(predicted_index, index)], index))
        matched_truth = {}

        def augment(predicted_index: int, visited: set[int]) -> bool:
            for truth_index in adjacency[predicted_index]:
                if truth_index in visited:
                    continue
                visited.add(truth_index)
                previous = matched_truth.get(truth_index)
                if previous is None or augment(previous, visited):
                    matched_truth[truth_index] = predicted_index
                    return True
            return False

        for predicted_index in range(predicted_count):
            augment(predicted_index, set())
        matches = [(predicted_index, truth_index)
                   for truth_index, predicted_index in matched_truth.items()]
        true_positive += len(matches)
        false_positive += predicted_count - len(matches)
        false_negative += truth_count - len(matches)
        matched_ious.extend(iou_lookup[pair] for pair in matches)
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "abs_beta0_error_mean": float(np.mean(beta_errors)) if beta_errors else 0.0,
        "connectivity": connectivity,
        "f1": f1,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "fp_components_per_line": false_positive / len(probability)
            if len(probability) else 0.0,
        "match_iou": match_iou,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "min_area_filter": None,
        "precision": precision,
        "predicted_components": total_predicted,
        "recall": recall,
        "true_components": total_truth,
        "true_positive": true_positive,
    }

# RevRender

Code for pixel-level instance segmentation of revisions in handwritten exam answers. RevRender combines a query-based detector, a full-resolution detail pathway, and an in-box point renderer with 32 context tokens.

This is the **code-only public repository**. It contains the model, the paper training recipe, validation threshold selection, mask AP evaluation, an inference entry point, and synthetic unit tests. It does **not** contain student answer sheets, annotations, identity mappings, checkpoints, manuscript files, or the private research Git history.

## Setup

Use Python 3.11 and a Linux CUDA environment for the original GPU recipe. The recorded RevRender run used PyTorch 2.9.1 with CUDA 12.8 and one RTX 4090 (24 GB). Install the compatible PyTorch/torchvision build from [PyTorch](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -r requirements.txt
python revrender.py --help
python -m pytest -q
```

The release was checked on CPU for syntax, source completeness, command-line interfaces, data validation, and metric tests. Model tests skip if PyTorch is absent. This release has not been retrained or GPU-benchmarked independently; the requirements file is a dependency specification, not a certified environment lock.

## Data and weights

Training uses a prepared train/validation bundle and a separately corrected training bundle. Both must be supplied with their manifest SHA-256 values. No test images enter this training or selection path. See [data formats](docs/data_formats.md).

ExamRev contains student work and is not distributed here. The data application contact and agreement are being finalized. There is currently no public download link or guarantee of access. Do not submit student images or personal information to GitHub issues. Trained RevRender weights are not included in this code-only release.

The backbone is the Detectron2-format `R-50-torchvision.pkl` used by the original protocol, with SHA-256 `16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6`. Obtain it from the [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md), following its license. A different backbone file is intentionally rejected.

## Training

The wrapper selects the paper's `control` configuration: seed 42, learning rate 3e-4, weight decay 1e-4, and early-stop patience 6. Set the visible GPU before running. The full recipe is sequential: `smoke_a`, an exact replay `smoke_b`, a learning gate, and a formal run. Each stage writes a new output directory; existing directories are refused.

```sh
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
python revrender.py train --help
```

Run the following with your actual input paths and manifest digests (replace the capitalized placeholders):

```sh
python revrender.py train --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --corrected-bundle work/corrected --corrected-sha CORRECTED_SHA256 --backbone weights/R-50-torchvision.pkl --stage smoke --output-dir work/run/smoke_a
python revrender.py train --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --corrected-bundle work/corrected --corrected-sha CORRECTED_SHA256 --backbone weights/R-50-torchvision.pkl --stage smoke --output-dir work/run/smoke_b --replay-of work/run/smoke_a
python revrender.py train --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --corrected-bundle work/corrected --corrected-sha CORRECTED_SHA256 --backbone weights/R-50-torchvision.pkl --stage learning --output-dir work/run/learning
python revrender.py train --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --corrected-bundle work/corrected --corrected-sha CORRECTED_SHA256 --backbone weights/R-50-torchvision.pkl --stage formal --output-dir work/run/formal --gate-root work/run
```

The original baseline or historical annotation CLIs in the dependency directory are not the public entry point. They may require separate packages/assets. The release retains their shared functions because the model and metrics import them; it does not advertise all historical experiments as standalone reproductions.

## Freeze thresholds and evaluate

Use only the validation bundle for selection:

```sh
python revrender.py freeze --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --run work/run/formal --output work/run/freeze.json
python revrender.py evaluate --bundle work/train_dev --bundle-sha BUNDLE_SHA256 --freeze RevRender=work/run/freeze.json --output work/run/mask_ap.json
```

Freeze selects the mask threshold, mask NMS, and confidence cutoff using validation component F1 at IoU ≥ 0.50. Evaluation reports mask AP over IoU 0.50:0.05:0.95 with 101 recall samples. Do not use the evaluation command to tune on a held-out test split.

## Inference

Input is a single, explicitly prepared 96×768 grayscale `uint8` line (`.npy` or grayscale image). The wrapper never silently resizes it. It loads the paper `control` architecture and applies the thresholds in its validation freeze record:

```sh
python revrender.py predict --image work/line.npy --checkpoint work/run/formal/best.pt --freeze work/run/freeze.json --backbone weights/R-50-torchvision.pkl --output work/prediction.npz --device cuda
```

Output contains boolean instance masks, scores, and original query indices, plus a JSON provenance record. Input data and outputs are ignored by Git. The checkpoint hash must match the freeze record.

## Code map

| Entry | Purpose |
|---|---|
| `revrender.py` | Portable public command line |
| `scripts/816_revision_alignment_refinement.py` | Point features and in-box cross-attention |
| `scripts/674_revision_region_point_model.py` | Detector and ROI point renderer |
| `scripts/750_revision_resnet50_base.py` | Shared ResNet-50 backbone |
| `scripts/880_revision_tuning.py` | Paper control model and declared ablations |
| `scripts/881_revision_tuning_train.py` | Corrected-data training recipe |
| `scripts/867_revision_mask_threshold_freeze.py` | Validation operating point |
| `scripts/883_revision_mask_ap.py` | COCO-style mask AP |
| `source_manifest.json` | Source provenance and SHA-256 fingerprints |

Numbered dependency filenames are retained to preserve import and checkpoint provenance. Workstation-specific filesystem prefixes have been made relative, and affected source-file fingerprints updated. Dataset/weight fingerprints and numerical algorithms have not been changed. No private Git history is included.

## License and attribution

Original project code is released under [MIT](LICENSE). Imported libraries and separately obtained baseline implementations, pretrained weights, and data retain their own licenses; MIT does not grant access to or relicense ExamRev or third-party assets. See [third-party dependencies](docs/third_party.md).

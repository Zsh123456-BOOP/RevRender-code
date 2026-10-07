# Third-party dependencies

This repository imports, rather than vendors, external libraries. Respect their licenses independently of this repository's MIT license.

- PyTorch / torchvision: https://github.com/pytorch/pytorch and https://github.com/pytorch/vision
- segmentation-models-pytorch (SegFormer decoder, ResNet backbone): https://github.com/qubvel-org/segmentation_models.pytorch
- timm: https://github.com/huggingface/pytorch-image-models
- NumPy, SciPy, Pillow, pytest: their official distributions and licenses.
- Detectron2-format pretrained ResNet file: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

The inherited dependency tree also contains factory code for comparison models. Running those optional historical factories requires obtaining the corresponding original packages under their own terms. Detectron2, AdelaiDet, Mask2Former, MaskDINO, and DI-MaskDINO code or weights are not bundled here. Main RevRender training does not require running their factories.

The point-rendering approach builds on PointRend (Kirillov et al., CVPR 2020), and the transformer detector uses ideas from DETR/Deformable DETR. The residual in-box context implementation and the experiment protocol are in this repository. References to prior research do not imply ownership of those projects.

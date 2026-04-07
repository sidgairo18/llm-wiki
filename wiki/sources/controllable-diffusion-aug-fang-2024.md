---
title: "Data Augmentation for Object Detection via Controllable Diffusion Models"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, controlnet, object-detection, data-augmentation, clip-filter]
sources: [raw/ControllableDiffusionAug_Fang_WACV2024.pdf]
arxiv: "2308.10685"
---

## Summary

Fang et al. (WACV 2024) propose a controllable diffusion-based augmentation pipeline for object detection that uses **ControlNet with HED edge priors** to faithfully preserve the geometric layout (and therefore the bounding boxes) of real training images while resynthesizing appearance under diverse text prompts. A **category-calibrated CLIP filter** then removes generated samples whose semantics drift away from the original class distribution. The approach turns one real annotated image into many appearance-diverse but layout-faithful copies, improving detection in low-data and few-shot settings.

## Key Contributions

- Uses ControlNet with HED soft-edge maps as a structural prior so that generated images keep the same object outlines (and hence the original boxes are still valid annotations).
- Category-calibrated CLIP filtering: per-class similarity thresholds that adapt to each class's natural CLIP score distribution, avoiding biases from a single global threshold.
- Demonstrates large gains in few-shot detection on PASCAL VOC and COCO compared to standard photometric augmentation, Cut-Paste, and prior diffusion-based augmentation.

## Technical Details

**Layout-preserving generation.** For real image $\mathbf{x}$ with boxes $\{b_k\}$, extract HED edge map $E(\mathbf{x})$. ControlNet-conditioned Stable Diffusion generates new image
$$
\tilde{\mathbf{x}} \sim p_\theta(\,\cdot\, \mid c, E(\mathbf{x}))
$$
with text prompt $c$ derived from class names plus randomized scene/style descriptors. Because HED edges constrain object silhouettes, the original boxes $\{b_k\}$ remain valid annotations on $\tilde{\mathbf{x}}$.

**Category-calibrated CLIP filter.** For each class $y$, compute CLIP image–text similarity statistics over real training crops:
$$
\mu_y, \sigma_y = \text{mean}, \text{std}\bigl(\{\cos(\text{CLIP}(\text{crop}_i), \text{CLIP}(\text{name}_y))\}\bigr).
$$
Accept a generated crop iff its similarity is within $[\mu_y - \alpha\sigma_y,\, \mu_y + \beta\sigma_y]$, where $\alpha,\beta$ are global hyper-parameters. This is more robust than a single fixed threshold because some classes have intrinsically lower CLIP scores.

**Prompt construction.** Templates of the form `"a {style} photo of a {class} in {context}"` randomized over a curated style/context vocabulary, increasing appearance diversity without changing geometry.

**Training.** Generated images are mixed with real ones at a controlled ratio; standard detectors (Faster R-CNN, YOLOv5) are trained from scratch or fine-tuned.

## Results

- **PASCAL VOC few-shot (1-, 2-, 3-, 5-shot)**: +3 to +7 mAP gains over no-augmentation and over baseline ControlNet-aug without CLIP filter.
- **COCO low-data (1%, 5%, 10%)**: consistent gains, larger when real data is scarce.
- Ablations: HED control beats Canny and depth control for box preservation; category-calibrated filter beats global-threshold filter; prompt diversity matters as much as image count.

## Limitations

- HED edges leak fine appearance information, so "diversity" is limited to texture/style — geometry is unchanged, which caps generalization to novel poses/viewpoints.
- CLIP filter calibration needs enough real samples per class, which is itself problematic in few-shot regimes.
- Pipeline is slow: ControlNet inference per image plus CLIP filtering.

## Connections

- Closely related to [[sources/diffusionengine-zhang-2023]] (complementary: DE generates from scratch, this work edits real images).
- Uses [[concepts/controlnet]], [[entities/clip]].
- Compare with mask-faithful generation in [[sources/freemask-yang-2024]] for segmentation.

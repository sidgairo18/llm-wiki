---
title: "ForAug: Recombining Foregrounds and Backgrounds to Improve Vision Transformer Training"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [data-augmentation, vision-transformers, inpainting, grounded-sam, bias, classification]
sources: [raw/ForAug_Nauen_2025.pdf]
arxiv: "2503.09399"
---

## Summary

ForAug (Nauen et al., 2025; arXiv:2503.09399, DFKI) is a simple but effective image-classification augmentation that explicitly **decomposes training images into foreground and background layers and recombines them across examples**. Foregrounds are extracted with [[entities/grounded-sam|Grounded SAM]]; the background hole is filled with [[sources/lama-suvorov-2021|LaMa]] or Attentive Eraser inpainting; during training a foreground and a background from different images are recomposed. On ImageNet-1k, ForAug improves [[entities/vit|ViT]] top-1 accuracy by **+4.5 percentage points** and measurably **reduces center bias and size bias** in the resulting classifier, as diagnosed by shift/scale robustness probes.

## Key Contributions

- A clean FG/BG decomposition augmentation that does not require any generative model at train time — all inpainting is precomputed offline.
- **+4.5pp ImageNet-1k top-1** on ViT-S/16 and similar gains on other ViT scales.
- Quantitative evidence that ForAug **reduces center bias** (accuracy drop when objects are shifted) and **size bias** (accuracy drop when objects are rescaled), two known failure modes of ViTs trained on ImageNet.
- Systematic ablations over grounding method, inpainter, recombination policy, and mixing probability.

## Technical Details

### Offline FG/BG bank

For each training image $\mathbf{x}_i$ with class label $y_i$:

1. **Ground**: Grounded SAM with the class name as prompt produces an instance mask $M_i$.
2. **Inpaint**: [[sources/lama-suvorov-2021|LaMa]] (or Attentive Eraser) fills $\mathbf{x}_i \odot (1 - M_i)$ to produce a clean background $\mathbf{b}_i$.
3. **Store** the foreground $(\mathbf{x}_i \odot M_i, M_i, y_i)$ and background $\mathbf{b}_i$ in two banks.

### Train-time recombination

With probability $p$, replace a training sample by a recombined image:

$$
\tilde{\mathbf{x}} = \mathbf{x}_i \odot M_i + \mathbf{b}_j \odot (1 - M_i), \quad y = y_i,
$$

where $j \neq i$ is drawn at random (optionally from a different class). The label is the **foreground class**, so the model is pushed to rely on FG features rather than background cues. Standard augmentations (RandAugment, Mixup, etc.) are applied on top.

### Bias probes

- **Center bias**: translate the object within the frame and measure accuracy drop.
- **Size bias**: rescale the object and measure accuracy drop.

ForAug-trained ViTs degrade significantly less than baseline ViTs under both probes, indicating genuine reduction of spatial/scale shortcuts.

## Results

- **ImageNet-1k top-1**: +4.5pp on ViT-S/16; consistent gains on ViT-B, ViT-L.
- Reduces center and size biases substantially vs. RandAugment-only baselines.
- Compatible with and additive to Mixup, CutMix, RandAugment.
- Ablations: Grounded SAM > simpler salient-object segmenters; LaMa and Attentive Eraser give similar results; moderate mixing probability ($p \approx 0.5$) is best.

## Limitations & Open Questions

- Requires an offline preprocessing pass with Grounded SAM + inpainting over the full training set — expensive for ImageNet-scale.
- Single-label assumption: ImageNet has one dominant class per image; multi-label datasets need adaptation.
- Inpainting failures (visible seams, hallucinated objects) can inject label noise into backgrounds.
- Evaluated mainly on classification; detection/segmentation transfer not shown.

## Connections

- Uses [[sources/lama-suvorov-2021|LaMa]] as its core inpainter — a direct dependency.
- Uses [[entities/grounded-sam|Grounded SAM]] for foreground extraction.
- Conceptually related to [[sources/simple-copy-paste-ghiasi-2021|Copy-Paste]] but for classification, and with background *replacement* rather than addition.
- Addresses [[concepts/shortcut-learning]] and texture/background bias in [[entities/vit|ViT]]s.
- Relevant to [[concepts/data-augmentation]] and [[concepts/robustness]].

---
title: "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [data-augmentation, instance-segmentation, copy-paste, object-detection]
sources: [raw/SimpleCopyPaste_Ghiasi_CVPR2021.pdf]
arxiv: "2012.07177"
venue: "CVPR 2021"
---

## Summary

Ghiasi et al. (2021; arXiv:2012.07177) show that a deliberately minimal version of copy-paste augmentation — randomly selecting annotated instances from one image and pasting them (with their masks) onto another, combined with **Large-Scale Jittering** (LSJ) — is an exceptionally strong augmentation for instance segmentation, rivaling or exceeding much more elaborate context-aware pasting schemes. With Mask R-CNN and an EfficientNet-B7 + NAS-FPN backbone, the method reaches **57.3 box AP / 49.1 mask AP on COCO**, a new SOTA at the time. Crucially, the gains are **additive with self-training** on unlabeled data, and the method transfers to LVIS with especially large gains on rare classes.

## Key Contributions

- Demonstrates that *context-aware* pasting (matching scene types, choosing plausible locations, blending shadows) is unnecessary: **random pasting works better** when combined with LSJ.
- Establishes copy-paste + LSJ as a default strong baseline for instance segmentation augmentation.
- Shows the augmentation is additive with semi-supervised self-training.
- Reports strong results on [[entities/lvis|LVIS]] long-tail, foreshadowing [[sources/x-paste-zhao-2023|X-Paste]].

## Technical Details

### The recipe

1. Sample two training images $\mathbf{x}^A, \mathbf{x}^B$.
2. Randomly pick a subset of instances from $\mathbf{x}^A$.
3. Paste each chosen instance (pixels gated by its mask) at a **uniformly random location** onto $\mathbf{x}^B$, with no scale change beyond what LSJ already provides.
4. Update the target masks: pasted instances occlude any previous target where they overlap.
5. Apply Gaussian blur on the paste boundary (optional, small effect).

No attempt is made to match lighting, shadows, or context.

### Large-Scale Jittering (LSJ)

Input images are resized by a random factor $s \sim \mathcal{U}(0.1, 2.0)$ (vs. the standard $\mathcal{U}(0.8, 1.25)$), then randomly cropped/padded to the training resolution. LSJ alone is a strong augmentation; copy-paste stacks on top of it.

### Training

Standard Mask R-CNN losses, long schedules (up to $576$ epochs for best numbers). No architectural changes.

## Results

- **COCO** (Mask R-CNN, EfficientNet-B7 + NAS-FPN, 1280 res, LSJ + Copy-Paste): **57.3 box AP**, **49.1 mask AP** — SOTA at publication.
- **+0.6 to +1.5 mask AP** over strong LSJ-only baselines across backbones.
- **LVIS**: substantial gains on rare and common classes.
- **Additive with self-training**: combining Copy-Paste with pseudo-labels on unlabeled COCO-unlabeled / Objects365 gives further gains.
- Head-to-head with context-aware pasting from prior work (Dwibedi et al., Dvornik et al.): random pasting wins.

## Limitations & Open Questions

- Restricted to instances already annotated in the dataset — cannot augment classes that are rare *everywhere*. This limitation motivates [[sources/x-paste-zhao-2023|X-Paste]].
- No lighting/harmonization — pasted instances look visibly cut-and-pasted. Surprisingly this does not hurt, but may limit gains at higher data regimes.
- Very long training schedules required to see full benefit.

## Connections

- Baseline method for all subsequent generative copy-paste work: [[sources/x-paste-zhao-2023|X-Paste]], [[sources/sos-huang-2025|SOS]].
- Complementary to [[sources/foraug-nauen-2025|ForAug]], which instead recombines foreground and background *regions* for classification.
- Uses LSJ, which became a standard strong augmentation.
- Relevant to [[concepts/copy-paste-augmentation]] and [[concepts/large-scale-jittering]].

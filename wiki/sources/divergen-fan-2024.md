---
title: "DiverGen: Improving Instance Segmentation by Learning Wider Data Distribution with More Diverse Generative Data"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, instance-segmentation, synthetic-data, lvis, long-tail, data-augmentation]
sources: [raw/DiverGen_Fan_CVPR2024.pdf]
arxiv: "2405.10185"
---

## Summary

DiverGen proposes a generative data augmentation framework for **long-tailed instance segmentation** (LVIS) that explicitly maximizes the *diversity* of synthetic data rather than its raw quantity. The authors analyze how generative data widens the training distribution, and introduce a Generative Data Diversity Enhancement (GDDE) strategy spanning category, prompt, and generative-model diversity, plus a SAM-background mask extraction (SAM-bg) and a CLIP inter-similarity filter to keep instances clean and varied. DiverGen substantially improves rare-category AP on LVIS and outperforms the previous state-of-the-art X-Paste pipeline.

## Key Contributions

- Systematic study of how synthetic data shifts the training distribution: *distribution discrepancy* between real and generative data is a feature, not a bug — it expands the model's effective support.
- **GDDE** strategy with three axes:
  1. **Category diversity** — augmenting LVIS with extra categories sampled from external taxonomies (ImageNet-21k) to enrich semantic coverage.
  2. **Prompt diversity** — combining template, LLM (ChatGPT)-generated, and caption-style prompts.
  3. **Generative model diversity** — mixing samples from Stable Diffusion and DeepFloyd-IF.
- **SAM-bg**: clean instance mask extraction by using SAM with a background point prompt to isolate foreground.
- **CLIP inter-similarity filter**: discards generated instances whose CLIP embedding is too far from the real-class centroid, removing semantic drift.
- New SOTA on LVIS rare-category instance segmentation under multiple backbones.

## Technical Details

**Pipeline.** For each target category $c$, prompts $\{p_i^c\}$ are sampled from three sources (templates, LLM, captions) and fed to Stable Diffusion or DeepFloyd-IF. SAM is then run with a single background point to separate foreground; the largest connected foreground region becomes the instance mask $m$. The instance is composited (Copy-Paste-style) onto LVIS training images.

**CLIP inter-similarity filter.** Let $\mathbf{e}_i^c$ be the CLIP image embedding of generated instance $i$ for class $c$ and $\bar{\mathbf{e}}^c$ the mean CLIP embedding over real LVIS instances of class $c$. Keep instance $i$ iff
$$
\cos\!\bigl(\mathbf{e}_i^c,\, \bar{\mathbf{e}}^c\bigr) \geq \tau_c,
$$
where $\tau_c$ is a per-class quantile threshold.

**Distribution discrepancy analysis.** The authors measure CLIP-embedding KDEs over real and generative pools; GDDE produces a *flatter, broader* distribution that overlaps the real one but extends into under-sampled regions, which they correlate with downstream rare-AP gains.

**Training.** Standard CenterNet2 / Mask2Former trained on the union of LVIS and pasted generative instances, with class-balanced sampling.

## Results

- LVIS v1 with CenterNet2 (R50): rare-AP improves by **+5.3** over baseline and **+1.1** over X-Paste; overall mask AP also improves.
- Stronger backbones (Swin-L, ConvNeXt-L): consistent gains across all AP buckets, with the largest improvements on rare and common categories.
- Ablations: each axis of GDDE contributes; mixing Stable Diffusion + DeepFloyd-IF beats either alone; SAM-bg outperforms U2Net-based mask extraction; CLIP inter-similarity filter is critical for rare classes.
- Diversity metrics (CLIP-feature variance, DINO-feature spread) correlate with rare-AP gains better than raw image count.

## Limitations

- Compositing-based aug inherits Copy-Paste artifacts (lighting/scale mismatch).
- SAM-bg assumes a single dominant foreground, which fails for cluttered generated images.
- CLIP filtering can over-prune visually unusual but valid instances.
- Requires running multiple large generators, which is compute-heavy.

## Connections

- Builds on / compares to X-Paste, [[sources/mosaicfusion-xie-2024]], [[sources/freemask-yang-2024]].
- Uses [[entities/sam]], [[entities/clip]], [[entities/stable-diffusion]], [[entities/deepfloyd-if]].
- Target benchmark: [[entities/lvis]]; detectors: [[entities/centernet2]], [[entities/mask2former]].

---
title: "FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, semantic-segmentation, synthetic-data, mask-conditioned-generation, ade20k, coco-stuff]
sources: [raw/FreeMask_Yang_NeurIPS2024.pdf]
arxiv: "2310.15160"
---

## Summary

FreeMask asks: given **free** semantic masks (the existing labels from segmentation datasets like ADE20K and COCO-Stuff), can we generate matching synthetic images that train better segmentation models? The authors use a mask-to-image diffusion model (FreestyleNet) to synthesize many photorealistic images per real mask, then introduce two filtering mechanisms — **hardness-aware mask resampling** and **pixel-level noisy-label filtering** — to control which masks to oversample and which synthetic pixels to trust. Pre-training plus joint training with FreeMask data substantially improves SegFormer, Mask2Former, and other segmentation models on ADE20K and COCO-Stuff.

## Key Contributions

- Demonstrates that mask-conditioned diffusion (FreestyleNet) can produce *training-useful* dense-label data without any new annotation cost.
- **Hardness-aware mask resampling**: oversample masks whose corresponding real images yield high baseline loss, focusing synthesis on hard scenes.
- **Pixel-level filtering of noisy synthetic labels** using a teacher's per-pixel loss/consistency.
- Two-stage training recipe (synthetic pre-training → joint real+synthetic fine-tuning) that consistently improves multiple segmentation architectures.

## Technical Details

**Generation.** FreestyleNet $G$ takes a semantic mask $M$ and an optional text prompt $c$ and outputs an image $\tilde{\mathbf{x}}=G(M,c)$ such that the segmentation layout matches $M$. For each real mask in ADE20K/COCO-Stuff, multiple $\tilde{\mathbf{x}}$ are sampled with varying prompts.

**Hardness-aware resampling.** Train a baseline segmenter $f_0$ on real data. For each mask $M_k$ with image $\mathbf{x}_k$, compute mean cross-entropy loss
$$
h_k = \frac{1}{|\Omega|}\sum_{p\in\Omega} \mathrm{CE}\bigl(f_0(\mathbf{x}_k)_p,\, M_k(p)\bigr).
$$
Sample synthesis count $n_k \propto h_k^\gamma$ so that hard masks (rare classes, cluttered scenes) get more synthetic variants.

**Pixel-level filtering.** For synthetic image $\tilde{\mathbf{x}}$ with mask $M$, run teacher $f_0$ to get prediction $\hat{p}=f_0(\tilde{\mathbf{x}})$. Mark a pixel as reliable iff
$$
\mathrm{CE}(\hat{p}_p, M(p)) < \tau_{\text{class}(M(p))},
$$
where $\tau$ is a per-class adaptive threshold. Unreliable pixels are ignored in the loss for downstream training, suppressing the influence of generation artifacts and class confusions.

**Training recipe.** (1) Pre-train segmenter on filtered FreeMask data only. (2) Fine-tune jointly on real + synthetic with the pixel filter active.

## Results

- **ADE20K**: SegFormer-B4 improves from 48.5 to **51.0** mIoU; Mask2Former (Swin-L) improves by ~+1.0 mIoU over a strong baseline.
- **COCO-Stuff-164k**: ~+1.5 mIoU consistent gains across SegFormer variants.
- Ablations: hardness resampling alone gives ~+0.5 mIoU; pixel filter alone gives ~+0.7; combined gives the full improvement; uniform sampling without filtering can *hurt* due to noisy labels.
- Synthetic-only pre-training already approaches real-only training, indicating high label fidelity.

## Limitations

- Tied to FreestyleNet's quality; failure modes (texture mismatch, class confusion) propagate.
- Pixel filter relies on a baseline trained on real data — bootstrapping in the no-real-data regime is unclear.
- Per-class thresholds add hyper-parameters that need tuning per dataset.
- Long-tail classes with few real masks still get limited diversity even after resampling.

## Connections

- Mask-conditioned analogue of [[sources/diffusionengine-zhang-2023]] (boxes) and [[sources/instagen-feng-2024]] (open-vocab boxes).
- Compare with instance-level pipelines: [[sources/divergen-fan-2024]], [[sources/mosaicfusion-xie-2024]].
- Uses [[concepts/mask-to-image-synthesis]], [[entities/segformer]], [[entities/mask2former]].

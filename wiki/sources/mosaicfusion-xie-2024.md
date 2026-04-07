---
title: "MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, instance-segmentation, lvis, training-free, cross-attention, multi-object]
sources: [raw/MosaicFusion_Xie_IJCV2024.pdf]
arxiv: "2309.13042"
---

## Summary

MosaicFusion is a **training-free** diffusion-based augmenter for large-vocabulary instance segmentation. Without finetuning Stable Diffusion or any auxiliary segmenter, it (i) divides the SD canvas into a **mosaic of regions** and runs region-conditioned denoising so that each tile generates a different object, and (ii) extracts **per-instance masks from cross-attention maps** aggregated across timesteps and layers. The result is multi-object images with pixel-accurate masks for many categories, including LVIS rare classes, which yield significant gains for Mask R-CNN, CenterNet2, and Mask2Former.

## Key Contributions

- A training-free pipeline producing *multi-object* synthetic images with instance masks — earlier methods generated one object per image and relied on Copy-Paste compositing.
- Mosaic canvas mechanism: spatially partition the latent canvas into $K$ regions, each conditioned on a different category prompt, with shared global context, all in a single SD denoise.
- Cross-attention-based mask extraction with multi-layer/multi-step aggregation and morphological refinement, no SAM or U2Net required.
- Demonstrates strong LVIS rare-AP improvements while being far cheaper than methods that train auxiliary networks.

## Technical Details

**Mosaic canvas.** Partition latent $\mathbf{z}_t \in \mathbb{R}^{c\times H\times W}$ into $K$ non-overlapping regions $\{\Omega_k\}$. For each denoising step, run $K+1$ U-Net forward passes: one for the global prompt and one per region prompt. Combine noise predictions with a region mask:
$$
\hat{\epsilon}(\mathbf{z}_t)_p = \begin{cases}
\hat{\epsilon}^{(k)}(\mathbf{z}_t, c_k)_p & p\in\Omega_k\\
\hat{\epsilon}^{(g)}(\mathbf{z}_t, c_g)_p & \text{otherwise}
\end{cases}
$$
yielding an image where region $k$ depicts an instance of category $c_k$ and the rest is a coherent global scene.

**Cross-attention mask extraction.** For each token $w_k$ corresponding to category name $c_k$, collect cross-attention maps $A^{(l,t)}_{w_k} \in \mathbb{R}^{H_l\times W_l}$ from selected U-Net layers $l$ and timesteps $t$. Upsample, average, and threshold:
$$
\bar A_{w_k} = \frac{1}{|L||T|} \sum_{l\in L,\, t\in T} \mathrm{Up}(A^{(l,t)}_{w_k}),\quad
m_k = \mathbb{1}[\bar A_{w_k} > \tau].
$$
Connected-component cleanup and CRF refinement give the final instance mask. A bounding box is read off from the mask.

**Filtering.** A CLIP score per crop and an area/aspect-ratio sanity check discard degenerate samples.

**Training.** Generated multi-object images and masks are added directly to LVIS training as full images (not paste compositing), preserving natural-looking scenes.

## Results

- **LVIS v1, Mask R-CNN R50**: +**5.5** rare-AP over baseline; clear gains across common/frequent buckets too.
- **CenterNet2, Mask2Former (Swin-L)**: consistent +1 to +3 rare-AP gains, competitive with X-Paste despite being training-free and simpler.
- Ablations: mosaic conditioning is essential (single-prompt baseline collapses to one object); multi-layer/multi-step attention aggregation outperforms single-layer; CLIP filter gives marginal additional gains.
- Compute: no auxiliary network training; per-image cost is modest compared to SAM-based pipelines.

## Limitations

- Cross-attention masks are coarse for thin/elongated objects; CRF helps but does not fully fix.
- Hard cap on number of regions $K$ — too many regions destroy global coherence.
- Inherits SD's biases toward iconic, centered objects.
- No control over object pose/scale beyond region size.

## Connections

- Most directly comparable to [[sources/divergen-fan-2024]] (which uses SAM for masks and focuses on diversity) and [[sources/freemask-yang-2024]] (which uses mask-conditioned generation).
- Cross-attention mask extraction relates to [[concepts/prompt-to-prompt]] and DAAM-style attention attribution.
- Complementary to box-level engines [[sources/diffusionengine-zhang-2023]] and [[sources/instagen-feng-2024]].

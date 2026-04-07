---
title: "SOS: Synthetic Object Segments for Scalable Training of Detection and Grounding"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [synthetic-data, instance-segmentation, grounding, diffusion, flux, harmonization, copy-paste]
sources: [raw/SOS_SyntheticObjectSegments_2025.pdf]
arxiv: "2510.09110"
---

## Summary

SOS (Huang et al., 2025; arXiv:2510.09110, UW / AI2) pushes the generative copy-paste paradigm substantially further by building a pipeline — called **SOC** in the paper — that produces **20M high-quality synthetic object segments** via [[entities/flux|FLUX]] text-to-image generation plus [[entities/dis|DIS]] matting, and composes them into training images using **3D-aware geometric layout augmentation**, **IC-Light relighting** for scene harmonization, and **camera configuration augmentation**. On [[entities/lvis|LVIS]] instance segmentation the method delivers **+10.9 AP**, and on [[entities/grefcoco|gRefCOCO]] grounding **+8.4 gIoU**, substantially outperforming [[sources/simple-copy-paste-ghiasi-2021|Copy-Paste]], [[sources/x-paste-zhao-2023|X-Paste]], SynGround, and SegGen.

## Key Contributions

- **Scale**: 20M synthetic segments, two orders of magnitude larger than prior generative instance pools.
- **Harmonization via IC-Light**: uses a relighting model with **mask-area-weighted blending** so pasted objects inherit scene illumination rather than looking pasted.
- **3D-aware layout augmentation**: samples plausible ground-plane positions and sizes using a pseudo-3D scene parameterization, avoiding the implausible stacks/overlaps that plague random pasting.
- **Camera configuration augmentation**: varies focal length / viewpoint priors during composition to diversify perspective.
- State-of-the-art results on both detection/segmentation and grounding, with consistent gains across backbones.

## Technical Details

### Segment generation

For each target class, FLUX is prompted to generate object-centric images; DIS produces high-quality alpha mattes (rather than binary masks), which is essential for clean paste boundaries. A CLIP-based filter discards low-agreement samples.

### 3D-aware layout augmentation

Rather than uniform random placement, SOS parameterizes a background image with an approximate ground plane and camera intrinsics (estimated or sampled), then samples object positions $(x,y,z)$ in 3D and projects them to 2D. Object scale is derived from projected depth, so closer objects are larger — yielding physically plausible compositions.

### IC-Light harmonization

After pasting, IC-Light predicts a relit version of the composite under the background's estimated lighting. The final image is a **mask-area-weighted blend**:

$$
\mathbf{x}_{\text{out}} = \alpha(A) \cdot \mathbf{x}_{\text{relit}} + (1 - \alpha(A)) \cdot \mathbf{x}_{\text{paste}},
$$

where $\alpha$ is an increasing function of the pasted mask area $A$. Small objects are left mostly as-is (IC-Light is unreliable on tiny regions), while large objects are fully relit.

### Camera configuration augmentation

Focal length and principal point are jittered per image, and backgrounds are resampled from diverse camera priors, broadening the training distribution beyond the source dataset's camera statistics.

### Training mixture

Synthetic composites are mixed with real data at a tuned ratio during detector / grounding model training.

## Results

- **LVIS instance segmentation**: **+10.9 AP** over real-only baseline; beats Copy-Paste, X-Paste, SynGround, SegGen.
- **gRefCOCO grounding**: **+8.4 gIoU**.
- Gains hold across backbones and scales; rare classes benefit most.
- Ablations: removing IC-Light, 3D layout, or camera aug each costs several AP; their effects are complementary.

## Limitations & Open Questions

- Heavy compute: 20M segment generation via FLUX is expensive.
- Pseudo-3D scene parameterization depends on a monocular depth / ground-plane estimator whose failures propagate.
- IC-Light can hallucinate lighting artifacts on complex materials; the mask-area blend is a heuristic mitigation.
- Still limited to classes FLUX can render well — ultra-rare or domain-specific categories are not covered (cf. [[sources/odgen-zhu-2024|ODGEN]]).

## Connections

- Direct successor to [[sources/x-paste-zhao-2023|X-Paste]] and [[sources/simple-copy-paste-ghiasi-2021|Simple Copy-Paste]]; adds harmonization and 3D layout.
- Shares generate-filter-compose philosophy with [[sources/odgen-zhu-2024|ODGEN]] but targets general-purpose benchmarks.
- Uses [[entities/flux|FLUX]] and [[entities/dis|DIS]]; builds on IC-Light relighting.
- Relevant to [[concepts/image-harmonization]], [[concepts/synthetic-data-for-detection]], [[concepts/visual-grounding]].

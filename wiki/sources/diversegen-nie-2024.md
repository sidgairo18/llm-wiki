---
title: "Generative Dataset Distillation with Semantic Coordination for Object Detection"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, object-detection, synthetic-data, semantic-coordination, ddim-inversion]
sources: [raw/DiverseGen_SemanticCoord_Nie_2024.pdf]
arxiv: "2408.02891"
---

## Summary

This work tackles two failure modes of naive diffusion-based detection data: (i) **inter-category semantic incoherence** — generated combinations of classes that never co-occur in the real world — and (ii) **intra-image surrounding-region inconsistency** — implausible backgrounds and scene context. The authors propose two complementary modules: a **Category Affinity Matrix (CAM)** that constrains prompt sampling to realistic class co-occurrences, and a **Surrounding Region Alignment (SRA)** module that regenerates backgrounds via DDIM inversion conditioned on retrieved real reference scenes. The combined pipeline yields generative detection data whose layouts are semantically coordinated with real data and produces consistent gains on COCO and VOC.

## Key Contributions

- Identifies semantic coordination (inter-class and scene-level) as a key bottleneck for generative detection augmentation, beyond image-quality and mask-quality issues.
- **Category Affinity Matrix (CAM)**: a co-occurrence prior estimated from real annotations that biases multi-object prompt sampling toward plausible class combinations.
- **Surrounding Region Alignment (SRA)**: reuses real scene context for synthetic objects via DDIM-inverted background regeneration, preserving the foreground while harmonizing surroundings with retrieved references.
- Plug-in module compatible with existing engines (DiffusionEngine, X-Paste) that improves their downstream detection AP.

## Technical Details

**Category Affinity Matrix.** From the real training set, count pairwise co-occurrences $N_{ij}$ of classes $i,j$ in the same image, normalize:
$$
A_{ij} = \frac{N_{ij}}{\sqrt{N_i N_j}}\,.
$$
When sampling a multi-class prompt, the next class is drawn from $p(j\mid i) \propto A_{ij}$, producing scenes like "person + bicycle" rather than "elephant + laptop".

**Surrounding Region Alignment via DDIM inversion.** Given a generated foreground instance with mask $m$, retrieve $k$ real background references whose semantic embedding is close to the prompt's scene type. For each reference $\mathbf{x}_{\text{ref}}$, run DDIM inversion to obtain latent $\mathbf{z}_T^{\text{ref}}$, then denoise jointly with the foreground latent under a mask-conditioned schedule:
$$
\mathbf{z}_{t-1} = (1-m)\odot \mathbf{z}_{t-1}^{\text{ref}} + m\odot \mathbf{z}_{t-1}^{\text{fg}},
$$
so that the surrounding region inherits real scene statistics while the foreground object is preserved.

**Integration.** Generated images and boxes are combined with real COCO/VOC data; standard detectors (Faster R-CNN, DINO) are trained.

## Results

- On COCO-DE-style augmentation: adding CAM+SRA improves Faster R-CNN AP by ~+1.5 over a strong DiffusionEngine baseline; DINO sees ~+1 AP.
- VOC: similar gains, more pronounced for context-dependent classes (e.g., boat, surfboard).
- Ablations: CAM alone helps multi-object scenes; SRA alone helps single-object scenes; both are complementary.
- Qualitative: generated images visibly resemble real COCO scene compositions.

## Limitations

- CAM is estimated from the same dataset that supplies real data, so it cannot introduce *new* co-occurrences (limits long-tail benefits).
- DDIM inversion is slow and the mask-blending denoising can cause boundary artifacts.
- Retrieval bank of real backgrounds is required, which may not be available for novel domains.

## Connections

- Builds on [[sources/diffusionengine-zhang-2023]], complements [[sources/instagen-feng-2024]] and [[sources/mosaicfusion-xie-2024]].
- Uses DDIM inversion (see [[concepts/ddim-inversion]]) and mask-based latent blending.
- Related to layout-conditioned generation (ControlNet, GLIGEN).

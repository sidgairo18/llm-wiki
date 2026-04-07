---
title: "X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [synthetic-data, instance-segmentation, copy-paste, diffusion, clip, data-augmentation]
sources: [raw/X-Paste_Zhao_ICML2023.pdf]
arxiv: "2212.03863"
venue: "ICML 2023"
---

## Summary

X-Paste (Zhao et al., 2023; arXiv:2212.03863) revisits the classic [[concepts/copy-paste-augmentation|copy-paste]] recipe for instance segmentation and shows that it scales dramatically when the source of instances is a text-to-image diffusion model rather than a closed dataset. The pipeline uses [[entities/stable-diffusion|Stable Diffusion]] to synthesize per-class object images from class-name prompts, extracts masks with an ensemble of salient-object / referring segmenters, filters by CLIP agreement, and pastes the resulting instances onto real backgrounds. On [[entities/lvis|LVIS]] with a strong CenterNet2 backbone, X-Paste yields **+2.6 box AP** and especially **+6.8 AP on rare classes**, demonstrating that generative copy-paste particularly helps the long tail.

## Key Contributions

- Shows copy-paste augmentation scales favorably with *generated* instance pools, reopening a line of work that had plateaued on dataset-internal pasting.
- A four-stage pipeline — **Object Instance Acquisition**, **Mask Generation**, **Filtering**, **Composition** — each modular and independently analyzed.
- An ensemble mask strategy combining [[entities/u2net|U2Net]], SelfReformer, UFO, and [[entities/clipseg|CLIPSeg]] with CLIP-max selection that outperforms any single segmenter.
- Strong LVIS gains, especially on rare classes, where real copy-paste pools are smallest.

## Technical Details

### Object Instance Acquisition

For each class $c$ in the LVIS vocabulary, prompt Stable Diffusion with templates like `"a photo of a {c}"` to generate a large pool of candidate object images. Retrieval from web image banks is also supported as an alternative acquisition source.

### Mask Generation (ensemble)

Each candidate image is passed through four segmenters: U2Net, SelfReformer, UFO (all salient-object), and CLIPSeg (referring). For candidate masks $\{M_1,\dots,M_4\}$, the final mask is selected by **CLIP-max**: render each masked crop, score CLIP similarity with the class prompt, and pick the mask whose crop scores highest. Formally,

$$
M^\star = \arg\max_{M_k} \; \text{sim}_{\text{CLIP}}\big(\text{crop}(\mathbf{x} \odot M_k),\; \text{"a photo of a } c\text{"}\big).
$$

### Filtering

Instances with $\text{sim}_{\text{CLIP}} < 0.21$ are discarded to remove failed generations and mis-segmentations.

### Composition

Surviving (image, mask) pairs are pasted onto real training images at random scales/positions following [[sources/simple-copy-paste-ghiasi-2021|Simple Copy-Paste]] conventions, with standard LSJ augmentation.

## Results

- **LVIS** with CenterNet2 + Swin-L: **+2.6 box AP**, **+6.8 AP$_r$** (rare), **+3.1 mask AP** over strong baselines.
- Gains persist across backbones (ResNet-50, Swin-T, Swin-L) and across detectors.
- Ablations: generated instances > retrieved only; ensemble masks > single segmenter; CLIP filtering essential.
- Demonstrates that, unlike real copy-paste, scaling the generated pool keeps improving AP, especially for rare classes.

## Limitations & Open Questions

- Pasted instances lack scene-consistent lighting/shadows — no harmonization step (addressed later by [[sources/sos-huang-2025|SOS]]).
- Diffusion generations are biased toward canonical viewpoints; rare poses remain under-represented.
- CLIP filter threshold is dataset-specific and tuned manually.
- Mask ensemble is expensive (4 segmenters + CLIP scoring per candidate).

## Connections

- Direct successor to [[sources/simple-copy-paste-ghiasi-2021|Simple Copy-Paste (Ghiasi et al., 2021)]]; replaces the dataset-internal instance pool with a generative one.
- Shares generate-segment-filter philosophy with [[sources/odgen-zhu-2024|ODGEN]] and [[sources/sos-huang-2025|SOS]].
- Uses [[entities/stable-diffusion|Stable Diffusion]], [[entities/clip|CLIP]], [[entities/u2net|U2Net]], [[entities/clipseg|CLIPSeg]].
- Relevant to [[concepts/long-tail-recognition]] and [[concepts/synthetic-data-for-detection]].

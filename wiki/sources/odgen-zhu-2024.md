---
title: "ODGEN: Domain-specific Object Detection Data Generation with Diffusion Models"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [synthetic-data, object-detection, diffusion, controlnet, data-augmentation]
sources: [raw/ODGEN_2024.pdf]
arxiv: "2405.15199"
venue: "NeurIPS 2024"
---

## Summary

ODGEN (Zhu et al., 2024; arXiv:2405.15199) is a pipeline for generating synthetic training data tailored to **domain-specific object detection**. Unlike prior generative augmentation methods that target general-purpose benchmarks (e.g., COCO), ODGEN addresses the regime where a practitioner has only a small labeled dataset in a narrow domain (e.g., underwater, aerial, medical) and wants to augment it with diffusion-generated images that respect both domain appearance and layout constraints. The method fine-tunes [[entities/stable-diffusion|Stable Diffusion]] on cropped foreground instances *and* entire scenes, then conditions a [[concepts/controlnet|ControlNet]] jointly on a **text list** (class names per box) and an **image list** (per-object visual prototypes), enabling accurate layout-respecting generation even in cluttered scenes with overlapping boxes.

## Key Contributions

- A domain-adaptation recipe for Stable Diffusion that uses both cropped-FG images and whole-scene images to avoid catastrophic forgetting of the scene prior while absorbing domain appearance.
- **Object-wise conditioning**: a novel ControlNet conditioning scheme that takes a per-box text prompt list plus a per-box image prototype list, disambiguating co-occurring classes under occlusion.
- A **foreground-reweighted diffusion loss** that upweights pixels inside bounding boxes during training, improving instance fidelity.
- A CLIP-based **corrupted-label filter** to discard generations where the requested box does not actually contain the requested class.
- Evaluation on 7 Roboflow domain-specific detection benchmarks (RF7), showing up to **+25.3 mAP@.50** over the real-only baseline and consistent gains over generic generative augmentation baselines.

## Technical Details

### Two-stage domain fine-tuning

Stable Diffusion is fine-tuned with LoRA on a mixture of (a) tightly cropped single-object patches $\{\mathbf{x}^{\text{fg}}_i\}$ labeled with their class names, and (b) entire annotated scene images $\{\mathbf{x}^{\text{scene}}_j\}$ with a templated caption listing all present classes. This dual diet teaches the model both *what the objects look like* in isolation and *how they arrange* in domain scenes.

### Object-wise ControlNet conditioning

Given a target layout $\{(b_k, c_k)\}_{k=1}^K$ with boxes $b_k$ and classes $c_k$, ODGEN constructs:

- a **text list** $T = [c_1, \dots, c_K]$ paired with the boxes,
- an **image list** $I = [\mathbf{v}_1, \dots, \mathbf{v}_K]$ of visual prototypes (cropped real exemplars) aligned to each box.

These are injected into ControlNet so that each box location is conditioned on *its own* prompt and exemplar, rather than the whole image sharing a single global caption. This resolves ambiguity when two classes overlap.

### Foreground-reweighted loss

Let $M(\mathbf{x})$ be the union mask of all boxes. The diffusion training loss is

$$
\mathcal{L}_{\text{ODGEN}} = \mathbb{E}_{t, \epsilon}\Big[\big(1 + \gamma \cdot M\big) \odot \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, c)\|_2^2\Big],
$$

with $\gamma > 0$ upweighting FG pixels. This concentrates model capacity on instance appearance rather than background.

### Corrupted-label filtering

After generation, each predicted box crop is scored with CLIP similarity to its requested class label and a domain-tuned threshold discards low-agreement samples before they enter the detector training set.

## Results

- **RF7 benchmark** (7 Roboflow domains: aquarium, underwater, aerial, chess, etc.): ODGEN improves YOLOv5/v7 and Faster R-CNN detectors by **up to +25.3 mAP@.50** vs. real-only training.
- Consistently beats prior generative augmentation baselines (naive SD prompting, GLIGEN, ControlNet without object-wise conditioning).
- Ablations show each component (dual fine-tuning, object-wise conditioning, FG reweighting, CLIP filter) contributes additively.

## Limitations & Open Questions

- Requires a nontrivial real seed set per domain for LoRA fine-tuning and for exemplar pools.
- The image-list conditioning leaks appearance from real exemplars, which can reduce sample diversity.
- Filtering uses CLIP, whose reliability in specialized domains (medical, underwater) is itself questionable.
- No evaluation on instance segmentation or on very dense scenes ($>20$ boxes).

## Connections

- Closely related to [[sources/x-paste-zhao-2023|X-Paste]] and [[sources/simple-copy-paste-ghiasi-2021|Simple Copy-Paste]], which also use generated/segmented instances for augmentation — ODGEN differs by generating *full scenes with layout control* rather than pasting.
- Shares the generate-then-filter philosophy with [[sources/sos-huang-2025|SOS]].
- Builds on [[concepts/controlnet|ControlNet]] and [[entities/stable-diffusion|Stable Diffusion]].
- Relevant to [[concepts/synthetic-data-for-detection]] and [[concepts/domain-adaptation]].

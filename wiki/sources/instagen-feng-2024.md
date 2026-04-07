---
title: "InstaGen: Enhancing Object Detection by Training on Synthetic Dataset"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [diffusion, object-detection, open-vocabulary, synthetic-data, grounding, self-training]
sources: [raw/InstaGen_Feng_CVPR2024.pdf]
arxiv: "2402.05937"
---

## Summary

InstaGen turns Stable Diffusion itself into a *grounded* image generator for object detection. The authors attach a lightweight **instance grounding head** to a frozen SD U-Net that predicts bounding boxes for objects mentioned in the prompt, trained on real detection data. A **self-training scheme** then bootstraps boxes for *novel* (open-vocabulary) categories that were never annotated. The resulting synthetic dataset trains detectors that beat strong open-vocabulary baselines (F-VLM, ViLD) on novel-class AP, and also boosts closed-set detection.

## Key Contributions

- Instance grounding head on top of frozen Stable Diffusion features — converts SD into a single-pass image+box generator for arbitrary text categories.
- Self-training procedure for novel categories: alignment between real-class box predictions and an open-vocabulary teacher detector enables pseudo-labeling for unseen classes.
- Synthetic dataset (InstaGen data) that, used as additional training data, improves closed-set COCO/LVIS detection and dramatically improves novel-class open-vocab detection.
- First demonstration that the diffusion model itself, not an external detector run post-hoc, can serve as the box annotator for its own samples.

## Technical Details

**Grounding head.** Multi-scale features $\{F_l\}$ are extracted from the SD U-Net at a chosen denoising step. A small transformer-based head $h_\phi$ takes $\{F_l\}$ and the per-token CLIP text embeddings of category names appearing in the prompt, and predicts a set of boxes per category in DETR style:
$$
\{(\hat b_i, \hat y_i)\} = h_\phi(\{F_l\},\, \{e_y\}).
$$
Training uses Hungarian matching against real boxes:
$$
\mathcal{L}_{\text{ground}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{L1}} + \lambda_2 \mathcal{L}_{\text{GIoU}}.
$$
SD weights are frozen; only $h_\phi$ is trained on COCO/LVIS images encoded into latents and partially noised.

**Self-training for novel categories.** For a novel class $y^\star$ never seen by $h_\phi$, generate images with prompts containing $y^\star$ along with base classes. Run an open-vocab teacher (e.g., a CLIP-based proposal scorer) to obtain candidate boxes; keep only those whose features align with both the SD grounding head's class-agnostic objectness and the CLIP text embedding of $y^\star$. Add high-confidence pseudo-labeled samples to the training pool and re-train $h_\phi$.

**Generation pipeline.** Templates produce prompts mixing base and novel classes; each sample yields image $+$ boxes in one forward pass through SD and $h_\phi$, with a CLIP filter discarding obvious failures.

## Results

- **COCO open-vocabulary** (base→novel split): +**5.8** novel AP over F-VLM and +4.0 over ViLD with the same backbone.
- **LVIS open-vocabulary**: large rare-AP gains, comparable or better than Detic without web-scale data.
- **Closed-set COCO**: training Faster R-CNN/DINO with InstaGen data adds +1–2 AP over real-only baselines.
- Ablations: grounding head trained on real data alone already produces usable boxes on synthetic images; self-training is what unlocks novel-class generalization; CLIP filter is mildly helpful.

## Limitations

- Quality of novel-class boxes depends on the open-vocab teacher; failure modes compound.
- SD struggles with small/cluttered objects, capping LVIS gains.
- Single-image, single-prompt generation limits scene complexity (compare with [[sources/mosaicfusion-xie-2024]]).
- Frozen SD bias (e.g., centered subjects) makes the box distribution differ from real datasets.

## Connections

- Closely related to [[sources/diffusionengine-zhang-2023]] (also a frozen-LDM detection adapter) but adds grounding-head training and self-training for open vocab.
- Complementary to [[sources/mosaicfusion-xie-2024]] (multi-object canvas) and [[sources/freemask-yang-2024]] (mask-conditioned).
- Uses [[entities/stable-diffusion]], [[entities/clip]]; baselines: [[entities/f-vlm]], [[entities/vild]], [[entities/detic]].

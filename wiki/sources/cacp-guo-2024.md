---
title: "CACP: Context-Aware Copy-Paste for Data Augmentation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [data-augmentation, copy-paste, vision-language, sam, blip, gradcam, segmentation]
sources: [raw/CACP_Guo_2024.pdf]
arxiv: "2407.08151"
---

## Summary

Guo, Wang, Chang, and Rambach (2024, DFKI) propose **Context-Aware Copy-Paste (CACP)**, an automated data-augmentation pipeline that addresses two core failure modes of vanilla Copy-Paste augmentation (Ghiasi et al. 2021): (i) **context neglect** — pasting objects into semantically incompatible scenes (e.g., a penguin in a desert) and (ii) **mask dependency** — requiring pre-existing instance masks that limit scalability. CACP combines [[entities/blip]] image captioning, BERT semantic similarity matching, [[entities/yolo]] object detection on Object365, and [[entities/sam]] mask extraction (with [[concepts/grad-cam]]-guided point prompts) into a fully automatic, label-free pipeline. The result is plug-and-play augmentation that improves classification, segmentation, and detection across Cat-Dog, CityPersons, and CamVid datasets.

## Key Contributions

- **Semantic source–target matching** via BLIP captions + BERT-embedding cosine similarity, replacing manual or random pairing.
- **Hybrid SAM prompting**: combine YOLO bounding boxes with Grad-CAM-derived points to obtain accurate object masks (3–5 CAM-derived points work best).
- **Annotation-free**: Object365 (365 classes, 2M images, 30M boxes) used as the target gallery; users can extend custom categories without any manual masks.
- **Empirical gains across tasks/architectures** (U-Net, FPN, PSPNet, DeepLabV3, DeepLabV3+, PAN, YOLOv5) and consistent **convergence speedup** on CamVid.

## Technical Details

### Pipeline

1. **Caption** the source image with BLIP: e.g. "a picture of a group of young men playing soccer."
2. **Match category**: compute BERT-embedding similarity between source caption $C(I_S)$ and each Object365 class name $C(I_{T_i})$. Pick $\arg\min_i \phi(I_i, I_T)$:
   $$I_S = \arg\min_{I_i}\phi(I_i, I_T).$$
   Empirically BERT similarity outperforms cosine on raw word embeddings (Table 2: 0.94 for "soccer" vs "two teams playing football", vs 0.41 spurious match).
3. **Detect** the matched object class in target image with YOLOv5 trained on Object365 → bounding boxes.
4. **Mask extraction**: feed each bounding box + Grad-CAM-derived points into SAM:
   - Grad-CAM is computed on the target image; high-value points inside the bounding box are sampled as positive prompts.
   - Hybrid box+point prompts give cleaner masks than box-only (Table 6: 0.734 mIoU bbox-only → 0.927 with 1 CAM point → 0.934 with 3 → 0.933 with 5).
5. **Composite**: $I_{\text{syn}} = I_S \otimes M + I_T \otimes (1 - M)$ with rescaling using a $\text{ratio}_{\min}^{\text{obj}_1,\text{obj}_2}, \text{ratio}_{\max}^{\text{obj}_1,\text{obj}_2}$ pre-computed from Object365 statistics for realistic scale.

### Why it helps

A Grad-CAM analysis (Figure 3) shows that vanilla Copy-Paste augmented images fail to activate person-related features in downstream classifiers, while CACP-augmented images produce clean person-shaped activations. The hypothesis: contextually compatible objects fall within the data manifold and thus contribute meaningful gradient signal during training, whereas incompatible pastes act as noise.

## Results

### CamVid (segmentation, mIoU per class) — gains over baselines

CACP improves nearly every backbone × class combination (Table 3). Examples:

| Backbone     | Pedestrian | Building | Tree | Overall ↑ |
|--------------|------------|----------|------|-----------|
| U-Net        | 0.447 → 0.481 | 0.764 → 0.783 | 0.834 → 0.841 | +0.016 |
| FPN          | 0.432 → 0.479 | 0.792 → 0.797 | 0.867 → 0.881 | +0.015 |
| PSPNet       | 0.445 → 0.487 | 0.792 → 0.799 | 0.843 → 0.862 | +0.012 |
| DeepLabV3    | 0.461 → 0.493 | 0.803 → 0.812 | 0.875 → 0.890 | +0.011 |
| DeepLabV3+   | 0.471 → 0.497 | 0.817 → 0.826 | 0.871 → 0.883 | +0.012 |
| PAN          | 0.501 → 0.513 | 0.806 → 0.821 | 0.868 → 0.891 | +0.012 |

### Cross-task (Table 4)

| Method     | Cat-Dog Acc | CityPersons mIoU | CityPersons mAP |
|------------|-------------|------------------|-----------------|
| Base       | 0.927       | 0.914            | 0.557           |
| + CP       | 0.941       | 0.897            | 0.561           |
| + aug      | 0.957       | 0.903            | 0.567           |
| + CP+aug   | 0.962       | 0.911            | 0.571           |
| + CACP     | 0.969       | 0.929            | 0.577           |
| **+ CACP+aug** | **0.974** | **0.938**     | **0.591**       |

CACP outperforms vanilla CP across all three tasks and stacks with traditional pixel-level augmentation.

### Convergence (Fig. 6)

DeepLabV3 on CamVid: dice loss with CACP plateaus around epoch 15; without CACP it has not converged by epoch 19.

### Robustness to partition (Table 5)

Augmenting only $1/n$ of training images: gains saturate after $n=2$ — augmenting half the images is sufficient.

## Limitations & Open Questions

- **Pipeline dependency**: relies on BLIP, BERT, YOLO-Object365, and SAM — many heavy models in the loop. No discussion of inference-time cost of augmentation generation.
- **Object365 class coverage** is the bottleneck for the target gallery; truly out-of-distribution categories require building custom galleries.
- **Limited evaluation scale**: experiments on small datasets (CamVid 701 imgs, Cat-Dog 25k, CityPersons 2,975); no ImageNet/COCO-scale validation of the *trained-on-CACP* models.
- **No comparison to diffusion-based augmentation** (e.g., X-Paste, Stable Diffusion synthesis) beyond a passing reference. Future work suggests diffusion integration.
- **No control for compute budget**: gains may partially come from longer effective training due to more diverse data.
- The Grad-CAM-prompt-count ablation shows mIoU saturating at 3 points; the choice of which CAM thresholds to sample is not justified.

## Connections

- Builds directly on **Copy-Paste** (Ghiasi et al. CVPR 2021) and **X-Paste** (Zhao et al. ICML 2023, which uses CLIP+Stable Diffusion).
- Heavy use of vision-language and foundation models: [[entities/blip]], [[entities/sam]], [[entities/clip]] (cited via MaskCLIP), [[entities/yolo]].
- Grad-CAM as a *prompt source* for SAM is an interesting cross-pollination of [[concepts/attribution-methods]] and [[concepts/promptable-segmentation]] — relevant to the wiki's interpretability focus.
- Related to broader [[concepts/data-augmentation]] literature: Mixup, CutMix, ClassMix, and Manifold Mixup.
- Connects to [[concepts/foundation-model-pipelines]] — a representative example of combining 4–5 pretrained models without any task-specific training.

> Note: the raw PDF `CACP_Guo_2024.pdf` actually contains two concatenated documents (CACP followed by an unrelated fairness-evaluation paper by Chen et al., arXiv:2407.08926). This source page covers only the CACP portion.

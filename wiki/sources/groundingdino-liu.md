---
title: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [object-detection, open-vocabulary, vision-language, detr, grounding]
sources: [raw/GroundingDINO_Liu.pdf]
arxiv: "2303.05499"
---

## Summary

Grounding DINO (Liu et al., 2023; arXiv:2303.05499) extends the closed-set [[entities/dino-detector|DINO]] detector into an open-set / open-vocabulary detector by tightly fusing a vision backbone with a BERT text encoder at three stages: a feature enhancer, a language-guided query selection module, and a cross-modality decoder. It treats detection as language grounding, predicting boxes for arbitrary noun phrases or category names provided as text prompts. With ~172M params and Objects365 + GoldG + Cap4M pretraining, it reaches 52.5 AP on COCO **zero-shot** and 26.1 AP on ODinW, setting SOTA among open-set detectors at the time.

## Key Contributions

- **Tight vision-language fusion** at three stages of a DETR-style detector (neck, query selection, decoder), as opposed to GLIP-style late fusion.
- **Sub-sentence text representation** using attention masks between category words to avoid unwanted inter-class attention while keeping per-token features.
- **Language-guided query selection**: anchors are initialized from image tokens with the highest similarity to text tokens.
- **Cross-modality decoder** with image cross-attention + text cross-attention layers per block.
- Strong zero-shot transfer to COCO, LVIS, ODinW; first DETR-style open-set detector at this scale.

## Technical Details

Architecture: Swin-T/L image backbone + BERT text encoder. The **feature enhancer** stacks deformable self-attention (image), vanilla self-attention (text), and bidirectional image↔text cross-attention.

Language-guided query selection picks the top-$N_q$ image tokens by max text similarity:

$$\mathrm{idx} = \mathrm{TopK}\left(\max_{j} X_I X_T^\top,\; N_q\right)$$

where $X_I \in \mathbb{R}^{N_I \times d}$ and $X_T \in \mathbb{R}^{N_T \times d}$ are enhanced image/text features. Selected tokens initialize positional queries; content queries remain learnable.

The contrastive grounding loss aligns each query with text tokens via dot-product logits, combined with L1 + GIoU box losses, plus a DINO-style denoising auxiliary task.

## Results

- **COCO zero-shot**: 52.5 AP (Swin-L, with Objects365+GoldG+Cap4M).
- **COCO fine-tuned**: 63.0 AP (test-dev).
- **LVIS MiniVal zero-shot**: 27.4 AP (Swin-L).
- **ODinW (35 datasets) zero-shot**: 26.1 AP average — SOTA over GLIP-L.
- **RefCOCO/+/g**: strong referring expression comprehension without REC-specific training.

## Limitations

- Requires paired grounding data; quality bottlenecked by Cap4M pseudo-labels.
- Inference cost from joint vision-language transformer is high.
- Sub-sentence trick assumes prompt formatting with explicit separators.

## Connections

- Builds directly on [[sources/deformable-detr-zhu-2021|Deformable DETR]] and DINO detector.
- Contrasts with GLIP (late-fusion grounding).
- Text encoder leverages BERT; image-text contrast resembles [[concepts/contrastive-learning|CLIP-style alignment]].
- Related to open-vocabulary detection lineage: OV-RCNN, ViLD, OWL-ViT.

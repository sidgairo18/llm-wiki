---
title: "DETRs with Collaborative Hybrid Assignments Training (Co-DETR)"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [object-detection, detr, label-assignment, training]
sources: [raw/Co-DETR_Zong_ICCV2023.pdf]
arxiv: "2211.12860"
---

## Summary

Co-DETR (Zong, Song, Liu; ICCV 2023; arXiv:2211.12860) is a training-time augmentation for DETR-style detectors that attaches **auxiliary heads with one-to-many label assignment** (e.g., ATSS, Faster R-CNN) on top of the encoder, and feeds **customized positive queries** derived from those heads' positives back into the decoder. This densifies supervision on the encoder, enriches decoder query training, and removes nothing at inference. With ViT-L backbones, Co-DETR achieves **66.0 AP on COCO test-dev** and **67.9 AP on LVIS**, the first DETR variant to top the COCO leaderboard at the time.

## Key Contributions

- Diagnoses that DETR's pure one-to-one Hungarian matching produces **sparse positive supervision** for the encoder, hurting feature learning.
- Introduces **collaborative hybrid assignments**: $K$ auxiliary heads with one-to-many assignment trained jointly with the main decoder.
- Generates **customized positive queries** $Q_i^{\text{pos}}$ from each auxiliary head's positive coordinates, processed by extra decoder branches with one-to-one matching to their assigned GT.
- All auxiliary branches are dropped at inference — **no overhead**.
- Plug-and-play across Deformable-DETR, DAB-DETR, DINO, H-DETR.

## Technical Details

Given encoder features $\mathcal{F}$, auxiliary head $\mathcal{H}_i$ produces predictions with one-to-many assignment loss $\mathcal{L}_i^{\text{aux}}$. Positives from $\mathcal{H}_i$ define coordinate priors that initialize a parallel set of queries $Q_i^{\text{pos}}$, decoded by a shared decoder, supervised by one-to-one matching:

$$\mathcal{L} = \mathcal{L}^{\text{dec}}_{1\text{-}1} + \sum_{i=1}^{K} \lambda_1 \mathcal{L}_i^{\text{aux}} + \lambda_2 \mathcal{L}_i^{\text{pos}}$$

Auxiliary heads tested: ATSS, Faster R-CNN RPN+RCNN. $K=1$ or $K=2$ typically suffices.

## Results

- **COCO val**: +5.8 AP over Deformable-DETR baseline; +1.5–2.4 AP over DINO baselines.
- **COCO test-dev**: 66.0 AP with ViT-L + Objects365 pretraining.
- **LVIS minival**: 67.9 AP.
- 2× faster convergence on Deformable-DETR.

## Limitations

- Training cost grows with $K$ (extra heads + queries), though inference is unchanged.
- Choice of auxiliary head and assigner is heuristic.
- Gains saturate as base detector becomes stronger.

## Connections

- Built on [[sources/deformable-detr-zhu-2021|Deformable DETR]] and DINO detector.
- Addresses the assignment dichotomy also tackled by H-DETR, Group-DETR.
- Relates to dense prediction lineage (ATSS, RetinaNet) that Co-DETR borrows from for auxiliary supervision.
- A key recipe in modern SOTA detectors including [[sources/rt-detrv4-2025|RT-DETRv4]] descendants.

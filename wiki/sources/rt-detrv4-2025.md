---
title: "RT-DETRv4: Real-Time Detection with Vision Foundation Model Distillation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [object-detection, detr, real-time, distillation, dinov3]
sources: [raw/RT-DETRv4_2025.pdf]
arxiv: "2510.25257"
---

## Summary

RT-DETRv4 (2025; arXiv:2510.25257) pushes the real-time DETR family forward by distilling representations from a frozen **DINOv3 ViT-B** vision foundation model into RT-DETR's encoder. Two components: a **Deep Semantic Injector (DSI)** that aligns the AIFI F5 feature of RT-DETR with DINOv3 patch features, and **Gradient-guided Adaptive Modulation (GAM)** that dynamically reweights the distillation loss based on the ratio of distillation-to-detection gradient norms. Achieves **49.7 / 53.5 / 55.4 / 57.0 AP** at 273 / 169 / 124 / 78 FPS on T4, a Pareto improvement over RT-DETRv3, YOLOv10/11, D-FINE.

## Key Contributions

- **VFM distillation for real-time detection**: first to inject DINOv3 semantics into a sub-100ms detector without inference overhead.
- **Deep Semantic Injector (DSI)**: lightweight projection + alignment loss between RT-DETR's deepest encoder feature (AIFI F5) and DINOv3 ViT-B patch tokens.
- **Gradient-guided Adaptive Modulation (GAM)**: balances multi-task optimization by tracking $\Vert\nabla \mathcal{L}_{\text{distill}}\Vert / \Vert\nabla \mathcal{L}_{\text{det}}\Vert$.
- New SOTA on the real-time speed-accuracy frontier of COCO.

## Technical Details

DSI alignment loss between student feature $F_s \in \mathbb{R}^{H_s W_s \times d_s}$ and teacher patch tokens $F_t \in \mathbb{R}^{H_t W_t \times d_t}$:

$$\mathcal{L}_{\text{DSI}} = 1 - \cos\bigl(\phi(F_s),\, \mathrm{resize}(F_t)\bigr)$$

where $\phi$ is a small MLP/conv projector matching dimensions.

GAM modulates the loss weight $\lambda_t$ at step $t$:

$$\lambda_t = \lambda_0 \cdot \mathrm{clip}\left(\frac{\Vert \nabla_\theta \mathcal{L}_{\text{det}}\Vert}{\Vert \nabla_\theta \mathcal{L}_{\text{DSI}}\Vert},\, \lambda_{\min}, \lambda_{\max}\right)$$

so when distillation gradients dominate, the weight shrinks, preventing the auxiliary loss from hijacking optimization.

Total objective:

$$\mathcal{L} = \mathcal{L}_{\text{det}}^{\text{Co-DETR}} + \lambda_t \,\mathcal{L}_{\text{DSI}}.$$

DINOv3 teacher is frozen; no inference cost. Distillation only on AIFI F5 (deepest level) — shallow features kept untouched to preserve localization detail.

## Results

| Variant | AP | FPS (T4) |
|---|---|---|
| RT-DETRv4-S | 49.7 | 273 |
| RT-DETRv4-M | 53.5 | 169 |
| RT-DETRv4-L | 55.4 | 124 |
| RT-DETRv4-X | 57.0 | 78 |

Outperforms RT-DETRv3, D-FINE, YOLOv10/11 across all sizes at matched FPS.

## Limitations

- Requires DINOv3 teacher during training (extra memory/compute).
- Distillation only on F5; shallower levels untouched — possibly leaves further gains.
- COCO-only evaluation in main tables; transfer to dense / open-vocab settings not shown.

## Connections

- Builds on RT-DETR / DINO detector / [[sources/co-detr-zong-2023|Co-DETR]] training.
- Teacher: **DINOv3** — foundation model from the [[concepts/self-supervised-learning|self-supervised representation learning]] line (DINO, DINOv2, DINOv3).
- Distillation paradigm shared with FitNets, ReviewKD, masked feature distillation.
- Continues the [[sources/deformable-detr-zhu-2021|Deformable DETR]] → DINO → RT-DETR lineage.

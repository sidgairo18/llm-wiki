---
title: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [object-detection, detr, attention, transformers]
sources: [raw/DeformableDETR_Zhu_ICLR2021.pdf]
arxiv: "2010.04159"
---

## Summary

Deformable DETR (Zhu et al., ICLR 2021; arXiv:2010.04159) addresses two pain points of the original DETR: **slow convergence** (500 epochs) and **poor small-object performance**. It replaces full-image attention with **deformable attention**, which for each query samples only $K$ learned offset locations around a reference point, naturally extends to **multi-scale features** without FPN, and enables **iterative bounding-box refinement** and a **two-stage variant**. It matches DETR's accuracy with **10× fewer training epochs** and is significantly better on small objects.

## Key Contributions

- **Deformable attention module**: linear-complexity attention that attends to a small set of learned key points per query.
- Natural **multi-scale extension** by sampling across feature levels without explicit FPN merging.
- **Iterative bbox refinement** across decoder layers: each layer refines the previous prediction.
- **Two-stage Deformable DETR**: encoder generates region proposals that initialize decoder queries.
- 10× faster convergence than DETR; better small-object AP.

## Technical Details

For a query feature $z_q$ with reference point $p_q$, deformable attention computes

$$\mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \sum_{k=1}^{K} A_{mqk} \cdot W'_m\, x\bigl(p_q + \Delta p_{mqk}\bigr)$$

where $\Delta p_{mqk}$ and $A_{mqk}$ (with $\sum_k A_{mqk}=1$) are predicted by linear projections of $z_q$. Complexity is $O(N_q M K C)$ — independent of spatial size $H W$, vs $O(H^2 W^2 C)$ for full self-attention.

Multi-scale variant samples $K$ points per level across $L$ levels:

$$\mathrm{MSDeformAttn}(z_q, \hat p_q, \{x^l\}) = \sum_{m,l,k} W_m A_{mlqk} W'_m x^l(\phi_l(\hat p_q) + \Delta p_{mlqk})$$

with normalized reference $\hat p_q \in [0,1]^2$.

Iterative refinement: layer $l$ predicts box deltas relative to layer $l{-}1$'s box. Two-stage: encoder outputs are scored as proposals; top-$N$ become decoder queries.

## Results

- **COCO val**: 43.8 AP at 50 epochs vs DETR's 42.0 AP at 500 epochs (R50).
- **Small-object AP**: substantial gains (~+5–6 APs).
- Two-stage + iterative refinement: 46.2 AP (R50).
- With ResNeXt-101+DCN: 52.3 AP.

## Limitations

- Custom CUDA kernel needed for efficient deformable attention.
- $K$, $M$ are hyperparameters; small $K$ may miss context.
- Decoder queries still randomly initialized in single-stage variant.

## Connections

- Foundation for nearly all modern DETR variants: DAB-DETR, DN-DETR, DINO detector, [[sources/groundingdino-liu|Grounding DINO]], [[sources/co-detr-zong-2023|Co-DETR]], RT-DETR.
- Deformable attention conceptually related to deformable convolutions (Dai et al., 2017).
- Multi-scale sampling replaces FPN-style feature pyramids used in Faster R-CNN, RetinaNet.

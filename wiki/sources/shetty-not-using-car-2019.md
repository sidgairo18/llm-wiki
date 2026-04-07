---
title: "Not Using the Car to See the Sidewalk: Object Removal for Context Debiasing"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [context-bias, data-augmentation, segmentation, object-removal, robustness]
sources: [raw/Shetty_NotUsingCar_CVPR2019.pdf]
arxiv: "1812.06707"
---

## Summary

Shetty et al. (CVPR 2019; arXiv:1812.06707) argue that classifiers and segmenters **cheat** by using co-occurring objects: e.g., predicting *sidewalk* pixels because there is a *car* nearby. They propose a simple but effective remedy — **remove objects from training images** using a GAN-based inpainter and retrain — and introduce robustness metrics $V^{\min}$, $V^{\text{mean}}$, and an **association rate** $\text{AR}(c_i,c_j)$ that quantify how much a model's prediction for class $c_i$ depends on the presence of class $c_j$.

## Key Contributions

- An **object-removal data augmentation** pipeline: for each training image and each labeled object, produce a version with that object inpainted away, paired with a label mask that also removes the object.
- Robustness metrics:
  - **Association rate** $\text{AR}(c_i,c_j)$: drop in $c_i$ prediction when $c_j$ is removed.
  - $V^{\min}$, $V^{\text{mean}}$: minimum and mean IoU under systematic object removals.
- Demonstrates that standard segmentation networks (DeepLab, PSPNet) have high AR between commonly co-occurring classes (car↔road, person↔sidewalk, etc.) and that removal-augmented training significantly reduces this.
- Shows the removal augmentation **improves standard mIoU** too, not just robustness.

## Technical Details

**Object removal.** Given image $x$ and instance mask $m$ of object class $c_j$, an inpainting GAN produces $\tilde x = G(x,m)$ in which the object region is replaced with contextually plausible content. The paired label map has $c_j$ pixels relabeled as the surrounding stuff class (from the annotation) or as "ignore."

**Training.** Standard segmentation loss is applied on both original and removed versions. Because removed versions contain none of class $c_j$, the network cannot shortcut via its presence.

**Association rate.**
$$
\text{AR}(c_i,c_j)=\frac{\text{IoU}(c_i\mid x)-\text{IoU}(c_i\mid \tilde x_{\setminus c_j})}{\text{IoU}(c_i\mid x)},
$$
where $\tilde x_{\setminus c_j}$ is $x$ with all $c_j$ instances removed. High $\text{AR}$ indicates the model's prediction of $c_i$ co-depends on $c_j$.

**Aggregate metrics.** For each class $c_i$:
$$
V^{\min}(c_i)=\min_{c_j}\text{IoU}(c_i\mid\tilde x_{\setminus c_j}),\quad V^{\text{mean}}(c_i)=\tfrac{1}{|C|}\sum_{c_j}\text{IoU}(c_i\mid\tilde x_{\setminus c_j}).
$$

## Results

- **Pascal VOC / COCO**: object-removal training reduces max AR across class pairs by roughly half.
- **mIoU improvements** of ~1-2 points on standard splits, with larger gains ($3-5$ points) on $V^{\min}$.
- **Qualitative**: models trained without augmentation hallucinate absent objects based on context; augmented models do not.
- Inpainting quality matters but is not critical — even mediocre fills provide useful signal.

## Limitations & Open Questions

- Requires instance-level masks, limiting applicability to datasets without such annotations.
- Inpainting artifacts could themselves become a shortcut.
- Only tackles **object-object** context bias, not background-scene bias.
- No theoretical account of why removal works; purely empirical.

## Connections

- Mitigation complement to [[sources/noise-or-signal-xiao-2021]] (backgrounds) and [[sources/singh-dont-judge-2020]] (CAM-based decorrelation).
- Precursor to modern synthetic-augmentation approaches like [[sources/conbias-chakraborty-2024]] that use generative models to rebalance concept combinations.
- Relevant to [[concepts/shortcut-learning]], [[concepts/context-bias]], and robust semantic segmentation.
- The $\text{AR}$ metric is a simple causal-style probe, related to counterfactual evaluation in interpretability.

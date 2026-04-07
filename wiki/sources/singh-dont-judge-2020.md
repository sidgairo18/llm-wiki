---
title: "Don't Judge an Object by Its Context: CAM-Based Context Decorrelation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [context-bias, cam, feature-decorrelation, multi-label, fairness]
sources: [raw/Singh_DontJudge_CVPR2020.pdf]
---

## Summary

Singh et al. (CVPR 2020) tackle **context bias** in visual recognition: when an object class co-occurs with a context class most of the time (e.g., *skateboard* with *person*), classifiers learn to predict the object from context cues and fail when the object appears out of context. They propose a CAM-based training scheme that splits features into **object** and **context** subspaces and **decorrelates** their contributions, improving recognition on rare (object-without-context) test samples across COCO-Stuff, DeepFashion, AwA, and UnRel.

## Key Contributions

- Identification of a **Most Biased Category (MBC) pair** $(c_i,c_j)$ per class: the context class whose presence most boosts the prediction of the target class.
- A feature-split training objective: for each biased pair, the last-layer feature $\phi(x)\in\mathbb{R}^d$ is partitioned into $\phi_o$ (object) and $\phi_c$ (context), with separate classifiers, and a penalty that **suppresses** the context head's contribution to the object class.
- Use of **Class Activation Maps (CAM)** to localize which spatial regions each sub-head attends to, enforcing that object prediction relies on object regions.
- Evaluations on four datasets with explicit *exclusive* (context-absent) test splits.

## Technical Details

**Bias identification.** For each class $c_i$, the MBC is
$$
c_j^\star = \arg\max_{c_j\ne c_i}\;\frac{p(c_i\mid c_j)}{p(c_i)} .
$$

**Feature split.** The CNN backbone produces feature map $F\in\mathbb{R}^{H\times W\times d}$. Split along channels: $F=[F_o;F_c]$, with classifiers $W_o,W_c$. Logits:
$$
z_{c_i} = \langle W_o^{(c_i)},\operatorname{GAP}(F_o)\rangle + \lambda\,\langle W_c^{(c_i)},\operatorname{GAP}(F_c)\rangle.
$$

**Decorrelation loss.** For MBC pair $(c_i,c_j)$ the paper adds a penalty of the form
$$
\mathcal{L}_{\text{dec}} = \sum_{(c_i,c_j)} \big|\langle W_c^{(c_i)},W_c^{(c_j)}\rangle\big|,
$$
discouraging the context head from coupling the two classes. CAM maps supervise spatial separation: $F_o$ activations should concentrate on object bounding boxes.

**Training.** End-to-end with cross-entropy + $\alpha\mathcal{L}_{\text{dec}}$ + CAM regularization.

## Results

- **COCO-Stuff (exclusive subset)**: +5-8 mAP on rare (object-without-context) categories over ERM.
- **DeepFashion**: improves attribute recognition when usual garment context is missing.
- **Animals with Attributes (AwA)**: improves unseen-background accuracy.
- **UnRel** (unusual relations): large gains, as expected — the benchmark is designed for out-of-context objects.

## Limitations & Open Questions

- Only pairwise (object, context) bias is modeled; higher-order combinations are ignored.
- MBC estimation requires labeled co-occurrence statistics.
- Feature-channel split is a strong architectural prior; the "right" split dimension is a hyperparameter.
- CAM supervision requires bounding boxes or comparable localization data.

## Connections

- A mitigation counterpart to the diagnostic findings in [[sources/noise-or-signal-xiao-2021]].
- Related to [[sources/shetty-not-using-car-2019]], which tackles context bias via data augmentation rather than feature splitting.
- Relevant to [[concepts/shortcut-learning]], [[concepts/context-bias]], and inherent interpretability via CAM.
- Closed-vocabulary predecessor to open-set frameworks like [[sources/mavias-sarridis-2025]].

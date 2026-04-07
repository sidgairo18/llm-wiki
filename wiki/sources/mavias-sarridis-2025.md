---
title: "MAVias: Open-Set Bias Discovery and Mitigation with Vision-Language Models"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [bias-mitigation, vision-language-models, open-set, fairness, spurious-correlations]
sources: [raw/MAVias_Sarridis_ICCV2025.pdf]
arxiv: "2412.06632"
---

## Summary

MAVias (Sarridis et al., ICCV 2025; arXiv:2412.06632) is an **open-set** bias mitigation framework that does not assume the set of spurious attributes is known in advance. It pipelines a generic image tagger, a large language model, and a vision-language encoder to automatically discover class-correlated "bias concepts" in free-form text, then trains the downstream classifier with a **logit-alignment loss** that discourages its predictions from being explainable by the bias concepts alone.

## Key Contributions

- **Open-set bias discovery** via (1) image tagging, (2) LLM filtering of task-irrelevant tags, (3) CLIP-style embedding of candidate bias phrases.
- A **bias-aware training objective** combining standard cross-entropy with a logit-alignment penalty that compares classifier logits to bias-concept similarities.
- No need for per-sample group labels, predefined attributes, or a held-out balanced validation set.
- Consistent improvements on Waterbirds, CelebA, UrbanCars, ImageNet-A/9 over LfF, JTT, DebiAN, BPA, and group-DRO.

## Technical Details

**Bias concept extraction.** Given training image $x$, a tagger returns tag set $T(x)$. Tags that are redundant with the class name or globally frequent are filtered by an LLM prompt, yielding a candidate bias vocabulary $\mathcal{B}$. Each concept $b\in\mathcal{B}$ is embedded as $\phi_{\text{text}}(b)$ with a frozen VL encoder; images are embedded as $\phi_{\text{img}}(x)$.

**Bias score.** For image $x$ and concept $b$:
$$
s_b(x)=\cos\!\big(\phi_{\text{img}}(x),\,\phi_{\text{text}}(b)\big).
$$
A per-image bias vector $\mathbf{s}(x)=[s_b(x)]_{b\in\mathcal{B}}$ is precomputed.

**Logit alignment loss.** Let $f_\theta(x)\in\mathbb{R}^C$ be classifier logits. MAVias learns a linear probe $W_b$ from $\mathbf{s}(x)$ to logits and penalizes their agreement:
$$
\mathcal{L}_{\text{MAV}} = \mathcal{L}_{\text{CE}}(f_\theta(x),y) + \lambda\,\big\|\operatorname{softmax}(f_\theta(x)) - \operatorname{softmax}(W_b\mathbf{s}(x))\big\|^2_{\text{neg}},
$$
where the second term is applied with a sign that **repels** classifier predictions from bias-explainable predictions (the paper details a contrastive form on worst-case concepts).

**Pipeline.** Tagging and embedding are one-off preprocessing; training cost is comparable to ERM.

## Results

- **Waterbirds / CelebA**: matches or exceeds group-DRO worst-group accuracy without group labels.
- **UrbanCars**: mitigates both BG and CoObj shortcuts simultaneously, unlike single-shortcut methods which suffer from the Whac-A-Mole effect.
- **ImageNet-9 / ImageNet-A**: improves background robustness and out-of-distribution accuracy.
- Bias vocabularies recovered by the LLM stage are human-interpretable, aiding dataset auditing.

## Limitations & Open Questions

- Relies on the quality of the VL encoder; biases present in CLIP can leak into the discovered vocabulary.
- LLM filtering is prompt-sensitive and may miss subtle biases (e.g., texture, lighting).
- Logit-alignment hyperparameter $\lambda$ needs tuning per dataset.
- Evaluated mostly on image classification; extensions to detection/segmentation remain open.

## Connections

- Open-set counterpart to [[sources/conbias-chakraborty-2024]], which uses closed-vocabulary concept graphs.
- Addresses the multi-shortcut failure mode documented in [[sources/whac-a-mole-li-2023]].
- Builds on [[entities/clip]]-style VL encoders for concept grounding.
- Related to [[concepts/spurious-correlations]] and [[concepts/shortcut-learning]].

---
title: "Common Sense Bias Modeling for Classification Tasks"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [bias-discovery, spurious-correlation, captioning, vision-language, fairness, debiasing, human-in-the-loop]
sources: [raw/CSBD_Wang_2024.pdf]
arxiv: "2401.13213"
authors: [Miao Zhang, Zee Fryer, Ben Colman, Ali Shahriyari, Gaurav Bharaj]
affiliation: "NYU; Reality Defender"
---

# Common Sense Bias Discovery (CSBD)

## Summary

Zhang et al. (arXiv:2401.13213, v5 Jan 2025) propose **Common Sense Bias Discovery (CSBD)**, a framework that uses **textual descriptions** of images (human captions or LVLM-generated) to systematically uncover spurious feature correlations that affect downstream classifiers. Instead of relying on labeled sensitive attributes or model-internal heuristics, CSBD extracts **noun-chunk clusters** as interpretable "features", computes pairwise $\phi$ correlations between feature presence indicators across the dataset, and then optionally invokes a **human-in-the-loop** to flag spurious vs. benign correlations. Identified spurious feature pairs are mitigated through a simple **data re-weighting** sampler that decorrelates target and spurious features. The approach is unsupervised w.r.t. sensitive attributes and outperforms LfF, DebiAN, PGD, and BPA on CelebA-Dialog and MS-COCO binary classification.

## Key Contributions

1. A novel **description-based bias discovery framework** (CSBD) that comprehensively analyzes spurious correlations in image datasets and produces de-biased training guidance.
2. A **common-sense reasoning approach** that builds interpretable feature clusters from noun chunks and quantifies pairwise correlations.
3. **Empirical evidence** that CSBD discovers static spurious features unaddressed by prior work and that mitigating them yields SOTA classifier debiasing on CelebA-Dialog and MS-COCO.
4. Demonstration that **LVLM-generated captions** (LLaVA-Next 7B) can substitute for human captions, enabling scaling to unannotated datasets.

## Technical Details

### Pipeline (Algorithm 1)

Input: image corpus $G=\{g_1,\dots,g_N\}$, description corpus $D=\{d_1,\dots,d_N\}$.

1. **Chunking + encoding.** Extract noun chunks $P=\{p_1,\dots,p_M\}$ via spaCy (`en_core_web_sm`); embed each chunk with the **Universal Sentence Encoder** to get $\mathbf{A}\in\mathbb{R}^{N\times 512}$.
2. **Agglomerative clustering** on $\mathbf{A}$ with complete linkage, stopping when the maximum Euclidean distance between any two cluster vectors exceeds threshold $z$ (default $z=1.0$). Each cluster $C_f$ is treated as a *feature*.
3. **Binary presence indicators** $\mathbf{t}_f\in\{0,1\}^N$: $t_{f,i}=1$ iff some chunk in $C_f$ appears in $d_i$.
4. **Pairwise $\phi$ correlation** between every two feature indicators:

$$\phi_{\mathbf{t}_f \mathbf{t}_{f'}} = \frac{x_{11}x_{00}-x_{10}x_{01}}{\sqrt{x_{1*}x_{0*}x_{*0}x_{*1}}},$$

with $x_{ab}=\|(\mathbf{t}_f^{(a)}\cdot\mathbf{t}_{f'}^{(b)})\|_1$ (a.k.a. Matthews correlation). Output: $(C_f,C_{f'},\phi_{\mathbf{t}_f\mathbf{t}_{f'}})$ if $|\phi|>\beta$.

5. **Human-in-the-loop**: a domain expert flags which highly-correlated pairs are spurious vs. benign (e.g., "teeth"-"smile" is benign).
6. **Bias mitigation by re-weighting.** Given target indicator $\mathbf{t}_y$ and spurious $\mathbf{t}_s$, sample weights are chosen so that

$$P'_D(\mathbf{t}_s=1\mid\mathbf{t}_y=0)=P'_D(\mathbf{t}_s=1\mid\mathbf{t}_y=1),$$

i.e., spurious feature presence is independent of the target across the resampled dataset. Augmentations (RandAugment, color jitter, resized crops) are applied alongside.

### Datasets

- **CelebA-Dialog**: CelebA images with 5-attribute captions (Bangs, Eyeglasses, Beard, Smiling, Age) plus pronouns.
- **MS-COCO 2014**: 1 caption per image; 9 (target, spurious) pairs picked from CSBD-discovered correlations (e.g., Cat–Couch, Cat–Keyboard, Dog–Frisbee, TV–Chair, Umbrella–Person, Kite–Person, Wine glasses–Person, Pizza–Oven).
- **LVLM captions**: LLaVA-Next 7B prompts ("Please describe face attributes..." / "Please describe the scene..."); SPICE 0.1583 against ground-truth captions.

### Implementation

CSBD itself is **CPU-only** (no model training). Downstream classifiers: ResNet-50 trained 100 epochs, SGD lr $10^{-4}$, momentum 0.9, weight decay 0.01, batch 256 (CelebA) / 32 (MS-COCO). PCA reduces USE embeddings before clustering ($>$90% variance retained); Chi-square test verifies $\phi$ significance.

## Results

### Bias mitigation on CelebA (Table 1, Worst / Avg group accuracy)

| Target / Spurious | None Wst | LfF | DebiAN | PGD | BPA | **CSBD** |
|---|---|---|---|---|---|---|
| Smiling / Male | 0.894 | 0.730 | 0.895 | 0.888 | 0.899 | **0.908** |
| Eyeglasses / Male | 0.962 | 0.912 | 0.967 | 0.966 | 0.971 | **0.973** |
| Young / Eyeglasses* | 0.652 | 0.640 | 0.609 | 0.646 | 0.638 | **0.655** |
| Receding hairline / Grey hair* | 0.186 | 0.174 | 0.225 | 0.198 | 0.351 | 0.337 |

(*) = mitigation guided by **LVLM-generated** captions.

### MS-COCO (Table 2)

CSBD attains the best Worst-group accuracy on 7 of 9 (target, spurious) pairs, e.g.:

- Cat / Couch: **0.906** Wst (vs PGD 0.884, BPA 0.853, no-mit 0.849).
- Cat / Keyboard: **0.953**.
- Dog / Frisbee: **0.940**.
- Umbrella / Person: **0.814**.

### Diagnostics

t-SNE on Cat-classifier embeddings (Fig. 4) shows that **only CSBD** produces a linearly separable decision boundary between Cat and No-cat samples regardless of the spurious "Couch" attribute; LfF / DebiAN / PGD / BPA fail to disentangle. Sensitivity analysis on $z$ (Table 3) shows results are robust over $z\in\{0.5,\dots,1.5\}$. Ablation on re-sampling weights (halving/doubling the increment) confirms the computed weights are critical, not merely the augmentations.

### Correlation discovery findings

CSBD surfaces both familiar (gender↔beard, $\phi=0.4498$) and novel correlations (city street↔bus, snowy slopes↔snowboard, kitchen↔scissors, bedroom↔chairs, beach↔kite). Tables 4–7 in the supplement list the full $\phi$-ranked feature pairs.

## Limitations & Open Questions

- Discovered biases are **limited to features expressible in the description corpus**; missing or low-quality captions cap the method's coverage (only 0.55 Pearson between caption-derived and label-derived feature presence on a small MS-COCO sample).
- Only **pairwise** correlations are modeled; higher-order interactions are unexplored.
- Currently demonstrated on **classification only**; extension to detection / segmentation is left to future work.
- Requires a **human-in-the-loop** to separate spurious from benign correlations, although this step is described as optional and lightweight.
- Re-weighting may slightly degrade Avg accuracy in some configurations (CelebA receding hairline case).

## Connections

- Strong methodological complement to [[sources/par-cooccurrencebias-ijcv2025]] (which targets per-attribute MI minimization given known attributes) — CSBD instead discovers the attributes themselves from text.
- Directly relevant to [[sources/vb-mitigator-2025]]: CSBD-style discovery could feed bias attributes into BLA methods (DI, BAdd, GroupDRO) currently requiring annotated $\mathcal A$.
- Conceptually adjacent to [[sources/mavias-sarridis-2025]] (foundation-model-derived bias tags) and the "Say My Name" framework.
- Pipeline relies on **noun chunking + Universal Sentence Encoder embeddings**; relates to [[concepts/text-embeddings]] and [[concepts/agglomerative-clustering]].
- The $\phi$ / Matthews correlation coefficient is a standard contingency-table statistic; see [[concepts/phi-correlation]].
- Compared baselines LfF, DebiAN, PGD (Bias Mimicking), BPA all appear in [[sources/vb-mitigator-2025]]'s benchmark — CSBD complements that benchmark by bringing description-aware discovery.
- LVLM captioning step relies on [[entities/llava-next]]; quality measured via SPICE.

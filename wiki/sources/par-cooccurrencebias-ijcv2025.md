---
title: "A Solution to Co-occurrence Bias in Pedestrian Attribute Recognition: Theory, Algorithms, and Improvements"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [pedestrian-attribute-recognition, bias-mitigation, mutual-information, disentanglement, fairness, spurious-correlation]
sources: [raw/PAR_CooccurrenceBias_IJCV2025.pdf]
arxiv: ""
doi: "10.1007/s11263-025-02405-7"
venue: "IJCV 2025"
authors: [Yibo Zhou, Hai-Miao Hu, Jinzuo Yu, Haotian Wu, Shiliang Pu, Hanzi Wang]
---

# A Solution to Co-occurrence Bias in Pedestrian Attribute Recognition

## Summary

Zhou et al. (IJCV 2025) tackle **attribute co-occurrence bias** in Pedestrian Attribute Recognition (PAR), arguing that deep models implicitly memorize dataset-level attribute correlations (e.g., *ShortSleeve* co-occurring with *Trouser* in summer-captured PA100k) and fail to generalize across scenes. They reframe co-occurrence as a *data selection bias* and propose **attributes-disentangled feature learning**: each attribute's specific feature must carry no information about other attributes, formalized via mutual-information minimization. They introduce two practical algorithms (DVMIM, PIL) plus refinements (NDSI, attributes-balanced re-sampling, causal-relations preservation), achieving SOTA on PA100k, PETA, RAP and especially the open-set/zero-shot variants PETAzs, RAPzs, UPAR, and MSP60k.

## Key Contributions

- A **novel perspective** on PAR: co-occurrence bias is conditioned statistical relevance, not causality, and should be actively disentangled rather than modeled.
- **DVMIM**: a Donsker-Varadhan representation based Mutual Information Minimizer that adversarially decouples per-attribute features, also providing a *quantitative measure* of inference interdependence.
- **PIL (Posterior-Invariant Learning)**: an optimization-friendly, non-adversarial alternative that enforces $P(\mathbf{y}^{/s} \mid \sum_l \mathbf{f}^l)$ invariance to in-distribution transforms of $\mathbf{f}^s$ via per-attribute feature interpolation.
- **NDSI (Norm-Direction Separated Interpolation)** to fully explore $\mathcal{F}_s$ by independently transforming feature norm (uncertainty) and direction (style/semantics).
- **Attributes-balanced re-sampling** via per-attribute memory banks for classifier fine-tuning under imbalanced labels.
- **Causal-relations preservation** by sharing interpolation weights across causally bound attributes (e.g., mutually exclusive age bins).
- Comprehensive experiments and a discussion of *infestation of attribute independencies* (off-diagonal contamination of the anchor-feature similarity matrix).

## Technical Details

### Setup

Given a backbone $\mathcal{E}_\phi$ producing $\mathbf{f}\in\mathbb{R}^K$, a decomposition layer $\mathcal{H}_\psi$ produces per-attribute features $\{\mathbf{f}^s\}_{s=1}^C$. The posterior $\widehat{\mathbf{y}}^s = P(\mathbf{a}^s=1\mid\mathbf{x})$ comes from a shared classifier $\mathcal{R}_\omega$ on $\sum_s \mathbf{f}^s$. Disentanglement requires:

$$\mathcal{I}(\mathbf{y}^{/s};\mathbf{f}^s)=0,\quad s=1,\dots,C,$$

where $\mathbf{y}^{/s}=(\mathbf{y}^1,\dots,\mathbf{y}^{s-1},\mathbf{y}^{s+1},\dots,\mathbf{y}^C)$.

### DVMIM (Donsker-Varadhan Mutual Information Minimizer)

Using the dual KL representation,

$$\mathcal{I}(\mathbf{y}^{/s};\mathbf{f}^s) = D_{KL}(P(\mathbf{y}^{/s},\mathbf{f}^s)\,\|\,P(\mathbf{y}^{/s})\!\times\!P(\mathbf{f}^s)),$$

a min-max game with per-attribute discriminators $\mathcal{D}_{\theta^s}$ is trained:

$$\mathcal{L}_{DV}=\mathcal{L}_{cls}+\mathcal{L}_{dis}+\lambda\,\mathcal{L}_{gen},$$

with classification cross-entropy $\mathcal{L}_{cls}$, discriminator loss $\mathcal{L}_{dis}$ approximating $-\mathcal{I}$, and generator loss $\mathcal{L}_{gen}=-\mathcal{L}_{dis}$. Joint samples come from $(\mathbf{f}^s,\widehat{\mathbf{y}}^{/s})$ on the same image; marginals are approximated by within-batch shuffling.

### PIL (Posterior-Invariant Learning)

PIL avoids the unstable adversarial loop. It constructs random per-attribute transforms $\mathcal{G}_i^s$ that map $\mathbf{f}_i^s$ to another in-distribution point $\widetilde{\mathbf{f}}_i^s$. Training optimizes:

$$\mathcal{L}_{PIL}=\min_{\psi,\phi,\omega}-\sum_{i=1}^N\sum_{s=1}^C(\alpha^s \mathbf{a}_i^s + (1-\alpha^s)\mathbf{a}_{r(i,s)}^s)\log\widehat{\widetilde{\mathbf{y}}}_i^s,$$

with $\widetilde{\mathbf{f}}_i^s = \alpha^s \mathbf{f}_i^s + (1-\alpha^s)\mathbf{f}_{r(i,s)}^s$, $\alpha^s\sim\mathcal{U}(0,1)$, and crucially $\alpha^k\neq\alpha^s,\,\forall k\neq s$. This is **per-attribute** feature MixUp, and provably enforces $\mathcal{I}(\mathbf{y}^{/s};\mathbf{f}^s)=0$.

### NDSI

To enrich the in-distribution manifold, norm and direction are interpolated independently:

$$\mathcal{G}_i^s(\mathbf{f}_i^s)=\big(\beta^s\|\mathbf{f}_i^s\|+(1-\beta^s)\|\mathbf{f}_{r(i,s)}^s\|\big)\cdot\Big(\alpha^s\tfrac{\mathbf{f}_i^s}{\|\mathbf{f}_i^s\|}+(1-\alpha^s)\tfrac{\mathbf{f}_{r(i,s)}^s}{\|\mathbf{f}_{r(i,s)}^s\|}\Big),$$

with $\alpha^s,\beta^s\sim\mathcal{U}(0,1)$, motivated by feature norm encoding data uncertainty (Shalev et al., 2018; Wang et al., 2018) and direction encoding style.

### Attributes-balanced re-sampling

Per attribute and per label, two memory banks $O_0^s, O_1^s$ buffer disentangled features post-convergence; the sampling function $r(i,s)$ is replaced by $b(i,s)$ that draws label 0/1 with equal probability. Only the classifier is fine-tuned afterwards.

### Causal-relations preservation

For groups of causally related attributes (e.g., $\text{Age}\!\le\!18,\,18\!<\!\text{Age}\!<\!60,\,60\!\le\!\text{Age}$), $\alpha^s$ is shared within the group so mutual exclusivity is preserved at inference, sacrificing some mA for higher Precision and rationality.

## Results

Backbones: **ResNet-50** and **ConvNeXt-base**. Datasets: PA100k, PETA, RAP, RAPzs, PETAzs, UPAR (re-labeled UPAR*), MSP60k.

Selected mA / F1 (Table 1) with Resnet-50 baseline at PA100k = 80.38 / 87.05:

| Method | PA100k mA | RAPzs mA | PETAzs mA | PETAzs F1 |
|---|---|---|---|---|
| Baseline ('21) | 80.38 | 72.32 | 71.62 | 71.68 |
| DAFL ('22) | 83.54 | – | – | – |
| Label2Label ('22) | 82.24 | 73.84 | 72.13 | 71.74 |
| DVMIM (Ours) | 83.14 | 75.54 | 74.26 | 72.80 |
| PIL (Ours) | **86.32** | **79.73** | **75.25** | 71.84 |
| PIL ConvNeXt-base | **88.70** | **82.04** | **79.73** | **75.81** |

On open-set **UPAR\*** PIL achieves **78.88 mA** vs PARFormer 72.58; on **MSP60k**, **62.47 mA / 69.84 F1** vs prior best ~59.75 / 65.82. Ablations (Table 2) confirm each technical addition (NDSI, ABR, CRP) is complementary. Fig. 9 shows DVMIM/PIL drastically reduce per-attribute estimated $\mathcal{I}(\mathbf{y}^{/s};\mathbf{f}^s)$ vs the baseline; Fig. 10 shows decorrelated anchor-feature similarity.

A notable diagnostic claim: the often-cited improvements of prior PAR work on PETA / RAP may be partially **information leakage** from training-test identity overlap (~31.5% / 57.7% per Jia et al. 2021b); on PETAzs, RAPzs, PA100k (no overlap) prior arts show only trivial gains over baseline, while PIL delivers large jumps.

## Limitations & Open Questions

- **DVMIM**: risk of degenerate "dummy" features that trivially zero MI; poor training efficiency; fails to reach SOTA alone.
- **PIL**: optimizes over interpolated labels, shifting the source label distribution, hurting close-set Precision; tends to predict more positives (higher Recall, lower Precision).
- Causal-relations preservation requires manual grouping and trades mA for rationality.
- The framework targets multi-label PAR; extension to other multi-attribute or multi-label vision tasks is mentioned but not evaluated.
- Anchor-feature similarity heatmaps still show residual stripes — bias is mitigated but not eliminated.

## Connections

- Related to [[concepts/spurious-correlation]] and [[concepts/co-occurrence-bias]] in computer vision; complements [[sources/conbias-chakraborty-2024]] and [[sources/gradient-starvation-pezeshki-2021]].
- The MI-based disentanglement objective relates to [[concepts/mutual-information-minimization]], FLAC ([[sources/vb-mitigator-2025]] integrates a similar idea), and Donsker-Varadhan / MINE estimators.
- Per-attribute feature MixUp links to [[concepts/mixup]] but operates in the *decoupled feature space* per attribute rather than image space.
- Connects to [[concepts/disentangled-representation-learning]] (InfoGAN, IB-GAN, $\beta$-VAE) but uniquely targets supervised multi-label classification rather than generative modeling.
- The Clever Hans framing connects to [[concepts/shortcut-learning]] and [[concepts/clever-hans-effect]] (Kauffmann et al., 2020).
- Relevant to fairness benchmarks evaluated by [[sources/vb-mitigator-2025]] and the bias-discovery view of [[sources/csbd-wang-2024]].

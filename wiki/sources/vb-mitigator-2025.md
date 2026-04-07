---
title: "VB-Mitigator: An Open-source Framework for Evaluating and Advancing Visual Bias Mitigation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [bias-mitigation, fairness, benchmark, framework, pytorch, spurious-correlation]
sources: [raw/VB-Mitigator_2025.pdf]
arxiv: "2507.18348"
authors: [Ioannis Sarridis, Christos Koutlis, Symeon Papadopoulos, Christos Diou]
code: "https://github.com/mever-team/vb-mitigator"
---

# VB-Mitigator

## Summary

Sarridis, Koutlis, Papadopoulos and Diou (CERTH / Harokopio Univ., 2025) introduce **VB-Mitigator**, an open-source PyTorch framework that unifies the development, evaluation and comparison of **visual bias mitigation** methods. Motivated by the fragmentation of the fairness/debiasing literature — disparate datasets, metrics, training pipelines — VB-Mitigator integrates **12 mitigation methods**, **7 datasets**, and a modular architecture (datasets, mitigators, models, metrics, tools/utilities, configuration via YACS). The paper additionally provides a standardized comparative evaluation across Biased-CelebA, Waterbirds, UrbanCars and ImageNet-9, recommending Worst-Group Accuracy (WGA) and Average Accuracy as the principal metrics.

## Key Contributions

- An **extensible PyTorch framework** with a `BaseTrainer` abstraction so new methods only need to override the pipeline stages they touch.
- A unified implementation of **12 mitigation methods** spanning Bias-Label-Aware (BLA) and Bias-Label-Unaware (BLU) families.
- Support for **7 datasets** (synthetic to general-purpose CV) covering single- and multi-attribute, foreground/background, demographic and unknown biases.
- A **standardized evaluation protocol** with WGA / AvgAcc and (for ImageNet-9) the seven test-set variants ORIGINAL, ONLY-BG-B/T, NO-FG, ONLY-FG, MIXED-RAND, MIXED-NEXT.
- A **comprehensive benchmark** of all 12 methods, identifying which approaches are robust across bias regimes.

## Technical Details

### Notation

Given $f(\mathbf{x};\boldsymbol\theta):\mathcal{X}\to\mathcal{Y}$ with feature extractor $\mathbf{z}=h(\mathbf{x};\boldsymbol\theta)$, dataset $\{(\mathbf{x}_i,y_i,a_i)\}$ where $a_i\in\mathcal{A}$ is a (possibly unknown) bias attribute, the vanilla objective is

$$\min_{\boldsymbol\theta}\frac{1}{N}\sum_{i=1}^N L(f(\mathbf{x}_i;\boldsymbol\theta),y_i).$$

### Integrated methods

**BLA (require bias labels):** GroupDRO, Domain Independent (DI), Entangling-and-Disentangling (EnD), Bias Balance (BB), Bias Addition (BAdd).

**BLU (no bias labels):** Learning from Failure (LfF), Spectral Decouple (SD), Just Train Twice (JTT), SoftCon, Debiasing Alternate Networks (Debian), FLAC / FLAC-B, MAVias.

Representative objectives:

- **GroupDRO**: $\min_{\boldsymbol\theta}\max_{g\in\mathcal G}\tfrac{1}{N_g}\sum_{i:g_i=g}L(f(\mathbf{x}_i;\boldsymbol\theta),y_i)$.
- **EnD**: $\min_{\boldsymbol\theta}\big[L(f(\mathbf{x};\boldsymbol\theta),y)+\lambda_{1}L_{dis}(\mathbf{z},\alpha)+\lambda_{2}L_{ent}(\mathbf{z},\alpha,y)\big]$.
- **BB**: $\min L(f(\mathbf{x};\boldsymbol\theta)+\mathbf{p},y)$ with prior $\mathbf{p}$.
- **BAdd**: $\min L(f(\mathbf{x};\boldsymbol\theta),\mathbf{b},y)$ with bias features $\mathbf{b}$ from a bias-capturing model.
- **LfF / Debian**: $\min\frac{1}{N}\sum w_i L(\cdot)$ with $w_i$ from an auxiliary biased predictor.
- **SD**: $L(\cdot)+\lambda_{SD}\|\hat y\|^2$ on logits.
- **SoftCon**: SupCon weights $w_{i,j}=1-\frac{\mathbf{b}_i\mathbf{b}_j}{\|\mathbf{b}_i\|\|\mathbf{b}_j\|}$.
- **FLAC**: $\min L+\lambda_{\text{FLAC}}I(\mathbf{z},\alpha)$ — minimizes mutual information between representations and a bias-capturing feature.
- **MAVias**: foundation-model-derived tags + VLM embeddings as a regularization term $L_{reg}(\mathbf{x},\mathbf{b},\lambda_{\text{MAVias},2})$.

### Datasets

Biased-MNIST (FG color, 99–99.9%), FB-Biased-MNIST (FG+BG, 90–99%), Biased-UTKFace (race/age, 90%), Biased-CelebA (gender, 90%), Waterbirds (background, 95%), UrbanCars (background + co-occurring object, 95%), ImageNet-9 (unknown background/texture biases).

### Metrics

Recommended: **Worst Group Accuracy** $\text{WGA}=\min_g \text{Acc}(g)$ and Average Accuracy $\text{AvgAcc}=\tfrac{1}{|\mathcal G|}\sum_g \text{Acc}(g)$. Bias-Conflict Accuracy (BCA) is supported but criticized for not capturing multi-bias interactions. For ImageNet-9, accuracy on the 7 test variants is reported.

### Implementation

Built on PyTorch with **YACS** configurations, W&B / TensorBoard logging, deterministic seeding (subject to CUDA non-determinism). Modular components: `datasets/`, `mitigators/`, `models/` (CNNs, ResNets, EfficientNets, ViTs, Swin), `metrics/`, `tools/`. Experiments use NVIDIA A100s, 5 seeds.

## Results

From Table 3 (Worst-Group / Average Accuracy on CelebA, Waterbirds, UrbanCars):

| Method | CelebA WGA | Waterbirds WGA | UrbanCars WGA |
|---|---|---|---|
| GroupDRO | 48.88 | 66.38 | 20.00 |
| DI | 84.88 | 91.64 | **86.13** |
| EnD | 54.56 | **94.72** | 47.52 |
| BB | 85.68 | 93.24 | 66.24 |
| BAdd | **88.98** | 94.00 | 84.64 |
| LfF | 58.58 | 94.42 | 46.88 |
| SD | 52.10 | 94.10 | 58.56 |
| JTT | 74.68 | 94.10 | 72.48 |
| SoftCon | 34.34 | 47.04 | 34.72 |
| Debian | 49.22 | 94.74 | 48.96 |
| FLAC | 84.32 | 91.08 | 62.56 |
| MAVias | 84.88 | **95.90** | 80.80 |

Key takeaways: **BLA methods (DI, BAdd) lead** on most single-attribute settings; **GroupDRO, EnD, BB struggle on UrbanCars** because of its multi-attribute structure; **MAVias is the most consistent BLU method**; **SoftCon is unstable** due to dependence on the auxiliary bias model. On ImageNet-9 (Table 4), SD and MAVias generalize best across the 7 test variants while SoftCon fails to converge.

## Limitations & Open Questions

- Bias mitigation remains an open problem; the included methods do not guarantee perfectly fair models.
- The benchmark currently focuses on classification; detection/segmentation are out of scope.
- Future work: integrate foundation-model-based bias discovery (links to [[sources/csbd-wang-2024]], "Say My Name", etc.) so that biases need not be pre-annotated for general-purpose datasets.
- Reproducibility caveat: full determinism is impossible due to CUDA / hardware variation.

## Connections

- Sister project to [[sources/mavias-sarridis-2025]] (the MAVias method evaluated here).
- Concept page: [[concepts/visual-bias-mitigation]], [[concepts/worst-group-accuracy]].
- Datasets used overlap with [[sources/conbias-chakraborty-2024]] (Waterbirds), [[sources/gradient-starvation-pezeshki-2021]] (Spectral Decouple), and the broader spurious-correlation literature.
- Methodologically complementary to the per-attribute disentanglement of [[sources/par-cooccurrencebias-ijcv2025]] and the description-based bias discovery of [[sources/csbd-wang-2024]].
- Useful reference framework for future entries under [[concepts/fairness-aware-learning]].

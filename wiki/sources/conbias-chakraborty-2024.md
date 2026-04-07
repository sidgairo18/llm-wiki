---
title: "ConBias: Concept Graphs for Contextual Bias Mitigation"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [bias-mitigation, spurious-correlations, concept-graphs, data-augmentation, vision]
sources: [raw/ConBias_Chakraborty_NeurIPS2024.pdf]
arxiv: "2409.18055"
---

## Summary

ConBias (Chakraborty et al., NeurIPS 2024; arXiv:2409.18055) introduces a concept-graph based framework for detecting and mitigating **contextual bias** in visual datasets. Training images frequently pair foreground objects with a narrow set of co-occurring "context" concepts (e.g., cows-on-grass, boats-on-water), causing classifiers to rely on spurious background cues. ConBias represents the dataset as a multipartite concept graph whose nodes are class labels and concept tags, discovers **imbalanced cliques** over concept combinations, and then generates synthetic images (via a text-to-image diffusion model) that fill the under-represented cliques, yielding a balanced concept distribution across classes.

## Key Contributions

- A **concept-graph representation** of the training distribution that makes class-concept co-occurrence statistics explicit and auditable.
- A clique-analysis algorithm that identifies **under-represented higher-order concept combinations** (not just pairwise correlations).
- A **targeted augmentation pipeline** using text-to-image generation to synthesize images for minority cliques, avoiding naive oversampling.
- Empirical gains on Waterbirds, UrbanCars, and COCO-GB over prior debiasing baselines, especially for worst-group accuracy.

## Technical Details

**Concept graph.** Let $\mathcal{C}$ denote the set of class labels and $\mathcal{K}$ the set of concepts extracted via an off-the-shelf tagger (e.g., RAM / CLIP-based). A multipartite graph $G=(V,E)$ is built with $V = \mathcal{C}\cup\mathcal{K}$ and edges weighted by empirical co-occurrence counts $w(c,k)=\#\{x : y(x)=c,\, k\in K(x)\}$.

**Clique imbalance.** For each class $c$, the method enumerates $k$-cliques of concepts co-occurring with $c$ and computes a per-class clique distribution $p_c(\mathcal{Q})$ for cliques $\mathcal{Q}\subseteq\mathcal{K}$. A clique is flagged as **biased** if
$$
\operatorname{Var}_{c\in\mathcal{C}} p_c(\mathcal{Q}) > \tau,
$$
i.e., the concept combination is strongly class-correlated.

**Balanced augmentation.** For each under-represented (class, clique) pair, ConBias prompts a diffusion model with a template such as *"a photo of a {class} with {concept_1}, {concept_2}, ..."* to synthesize $N$ new images. The augmented dataset is re-trained with standard ERM; no architectural changes are required.

## Results

- **Waterbirds**: improves worst-group accuracy over ERM and JTT while matching group-DRO without needing group labels at training time.
- **UrbanCars**: reduces the BG and CoObj shortcut gaps relative to LfF and DebiAN; particularly effective when both shortcuts are present.
- **COCO-GB**: closes a portion of the gender-object bias gap and reduces biased-clique frequency by construction.

## Limitations & Open Questions

- Depends on the quality of the external concept tagger; missed or noisy tags propagate into the graph.
- Diffusion-generated images can inherit their own biases and artifacts; no guarantee the synthetic samples contain the requested clique.
- Clique enumeration scales combinatorially with concept vocabulary; paper restricts to small $k$.
- Only tested on relatively small benchmarks; unclear how it scales to ImageNet-level vocabularies.

## Connections

- Related to [[concepts/spurious-correlations]] and [[concepts/shortcut-learning]].
- Alternative to group-label-free debiasing methods such as JTT, LfF, DebiAN.
- Complementary to [[sources/mavias-sarridis-2025]] which uses open-set VLM bias discovery instead of a closed concept tagger.
- Uses text-to-image diffusion (see [[concepts/latent-diffusion]]) as a data source.

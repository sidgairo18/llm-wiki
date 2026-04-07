---
title: "AttentionDrop: A Novel Regularization Method for Transformer Models"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [regularization, transformers, attention, generalization]
sources: [raw/DG-DETR_Hwang_2025.pdf]
arxiv: "2504.12088"
---

> Note: the raw filename `DG-DETR_Hwang_2025.pdf` is misleading — the actual content is **AttentionDrop** by Baig et al. (arXiv:2504.12088), a transformer regularization paper, not a DETR variant. Filed under the filename-derived slug for traceability.

## Summary

AttentionDrop (Baig et al., 2025; arXiv:2504.12088) introduces a family of stochastic regularizers that perturb the **attention probability distribution** inside transformer self-attention layers, instead of dropping activations as in standard Dropout/DropPath. Three variants are proposed — Hard Attention Masking, Blurred Attention Smoothing, and Consistency-Regularized AttentionDrop — analyzed via a PAC-Bayes generalization bound and shown to improve accuracy and robustness on CIFAR-10/100, ImageNet-1K, and WMT14 EN-DE.

## Key Contributions

- Identifies attention-map overconfidence as a generalization failure mode distinct from feature co-adaptation.
- Three plug-in regularizers operating on softmax attention weights $A \in \mathbb{R}^{N \times N}$.
- PAC-Bayes-style bound showing reduced KL between perturbed and prior posteriors.
- Empirical gains on vision (ViT, DeiT) and NMT (Transformer-base) benchmarks; improved calibration and adversarial robustness.

## Technical Details

Standard attention: $A = \mathrm{softmax}(QK^\top/\sqrt{d_k})$.

**Hard Attention Masking (HAM)**: zero-out top-$k$ entries per row with probability $p$, then renormalize:

$$\tilde A_{ij} = \frac{m_{ij} A_{ij}}{\sum_{j'} m_{ij'} A_{ij'}},\quad m_{ij}\sim\mathrm{Bernoulli}(1-p)\;\text{on top-}k.$$

**Blurred Attention Smoothing (BAS)**: convolve attention rows with a 1-D Gaussian kernel $g_\sigma$ to spread mass to neighbors:

$$\tilde A_i = (1-\alpha)\,A_i + \alpha\,(g_\sigma * A_i).$$

**Consistency-Regularized AttentionDrop (CRAD)**: two stochastic forward passes with different masks; add a consistency loss $\mathcal{L}_{\text{cons}} = \mathrm{KL}(p_1 \Vert p_2)$ to the task loss.

PAC-Bayes bound (sketch): expected risk

$$\mathbb{E}[R(h)] \le \hat R(h) + \sqrt{\frac{\mathrm{KL}(Q\Vert P) + \log(2\sqrt{n}/\delta)}{2(n-1)}}$$

argues attention perturbation tightens $\mathrm{KL}(Q\Vert P)$ vs vanilla dropout.

## Results

- **CIFAR-100 (ViT-S)**: +0.8–1.4% top-1 over Dropout/DropPath baselines.
- **ImageNet-1K (DeiT-S)**: small but consistent gains.
- **WMT14 EN-DE**: +0.4–0.6 BLEU.
- Improved ECE calibration and robustness to FGSM/PGD perturbations.

## Limitations

- Gains modest on large pretrained models.
- Adds hyperparameters ($p$, $k$, $\sigma$, $\alpha$) per layer.
- CRAD doubles forward cost during training.

## Connections

- Variant of stochastic regularization lineage: Dropout, DropPath, Stochastic Depth, Attention Dropout.
- Consistency regularization echoes R-Drop and Mean Teacher.
- Tangentially related to interpretability work on attention sparsity / entropy.

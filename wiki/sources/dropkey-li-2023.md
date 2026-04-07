---
title: "DropKey for Vision Transformer"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [regularization, attention, vision-transformer, dropout, vit]
sources: [raw/DropKey_Li_CVPR2023.pdf]
arxiv: "2208.02646"
venue: "CVPR 2023"
---

## Summary

Li, Hu, Nie et al. (CVPR 2023) revisit dropout in the self-attention layers of Vision Transformers (ViTs). They argue the community has largely ignored dropout *inside* attention, and analyze three core questions: (i) **what** to drop, (ii) **how** to schedule the drop ratio, and (iii) **whether** structured (block-wise) drop helps. Their answer is **DropKey**: instead of dropping post-softmax attention weights (vanilla DropAttention), drop the *Keys* by setting selected $QK^\top$ entries to $-\infty$ *before* softmax. Combined with a *decreasing* drop schedule across layers, this yields a simple, drop-in replacement that consistently improves T2T-ViT, VOLO, CeiT, and DeiT on ImageNet, COCO detection (DETR), HICO-DET (QPIC), and HUMBI human-mesh recovery.

## Key Contributions

- **Dropout-before-softmax scheme**: drop Keys rather than attention weights, so the softmax denominator is automatically renormalized over surviving keys (no manual rescaling).
- **Theoretical analysis** showing DropKey acts as an *adaptive smoothing* prior on the attention operator via a Lagrangian formulation, penalizing weight peaks.
- **Decreasing (Scheduled↓) drop ratio** across self-attention stack: high drop in shallow layers (low-level features), low drop in deep layers (preserve high-level semantics).
- **Empirical demonstration that structured drop is unnecessary for ViTs** (DropKey-Block and DropKey-Cross degrade performance), unlike CNNs where DropBlock helps.
- New SOTAs across multiple ViT architectures and tasks with a 2-line code change.

## Technical Details

### Vanilla self-attention with patch dropout

Standard ViT self-attention output for patch $i$:
$$o = \sum_{j=1}^{n_h n_w}\!\Big(\frac{d_j p_j}{\sum_{j'} d_{j'} p_{j'}}\Big) v_j, \quad p_j = \frac{\exp(qk_j^\top/\sqrt{n_c})}{\sum_{j'}\exp(qk_{j'}^\top/\sqrt{n_c})},$$
where $d_j \sim \text{Bernoulli}(1-d)$ is the post-softmax dropout mask. The averaging over non-dropout units after softmax breaks the probability distribution and is the source of overfitting to specific patterns.

### DropKey: drop before softmax

Generate a mask matrix $D$ with
$$d_j = \begin{cases} 0 & \text{w.p. } 1-d \\ -\infty & \text{w.p. } d \end{cases}$$
applied to similarity scores so that
$$p_j = \frac{\exp(d_j + qk_j^\top/\sqrt{n_c})}{\sum_{j'}\exp(d_{j'} + qk_{j'}^\top/\sqrt{n_c})}.$$
Implementation is two lines:

```python
m_r = torch.ones_like(attn) * mask_ratio
attn = attn + torch.bernoulli(m_r) * -1e12
```

A separate masked-key map is generated *per query* (not shared across queries).

### Adaptive smoothing analysis

Treating self-attention as solving $\min_{p_j, v_j} \tfrac{1}{2}\|\sum p_j v_j - y\|^2$ s.t. $p_j > 0,\ \sum p_j = 1$, the authors decompose $v_j = \beta_j e + \alpha_j$ along/perpendicular to $y$, write the Lagrangian, and show via partial derivatives $\partial L/\partial p_j$ and $\partial L/\partial v_j$ that DropKey introduces a smoothing coefficient $c$ (a Lagrange multiplier) that reduces the gradient gap between high- and low-attention patches:
$$\frac{\partial L}{\partial p_s} - \frac{\partial L}{\partial p_t} = (\sum p_j \beta_j - M)(\beta_s - \beta_t) + (\sum p_j \alpha_j)^\top(\alpha_s - \alpha_t).$$
The conclusion is that DropKey implicitly acts as an adaptive smoother — a property the explicit re-normalization in DropAttention only crudely approximates.

### Scheduled drop ratio

Three schedules tested: Constant, Scheduled↑ (linearly increasing with depth), Scheduled↓ (linearly decreasing). On CIFAR with T2T-ViT and VOLO at base drop ratio 0.3:

| Schedule       | T2T19+DropKey CIFAR-10 | T2T19+DropKey CIFAR-100 |
|----------------|------------------------|--------------------------|
| Scheduled↑     | 96.6                   | 77.2                     |
| Constant       | 97.4                   | 78.6                     |
| **Scheduled↓** | **97.6**               | **79.2**                 |

Increasing drop with depth hurts (deep layers carry semantics needed for classification).

### Structured DropKey

DropKey-Block (drop contiguous square windows of keys, à la DropBlock) and DropKey-Cross (drop entire row/column stripes) degrade performance for all window sizes. ViTs already model long-range context, so structured noise destroys signal rather than regularizing it.

### Inference alignment

To address train/test expectation mismatch, two methods compared: (a) Monte Carlo averaging at inference, (b) brief finetuning without DropKey. Finetuning works better and is preferred.

## Results

**ImageNet-1K Top-1** (selected):

| Backbone     | + Dropout (best) | + DropKey (best)  |
|--------------|------------------|-------------------|
| T2T-14       | 81.65 (dr=0.05)  | **82.04** (dr=0.05) |
| T2T-19       | 82.68 (dr=0.1)   | **82.94** (dr=0.1)  |
| VOLO-d1      | 84.35 (dr=0.05)  | **84.53** (dr=0.05) |
| VOLO-d2      | 85.22 (dr=0.05)  | **85.38** (dr=0.05) |
| CeiT-T       | 81.90            | **82.27**           |
| DeiT-B       | 81.91            | **82.25**           |

**COCO val (DETR)**: AP 41.9 (base) → 42.2 (+Dropout) → **42.9** (+DropKey).
**HICO-DET (QPIC-ResNet101)**: full mAP 32.42 → 33.21 (+DropKey).
**HUMBI mesh recovery (METRO)**: improvements of 0.7/0.5/0.4 on mPVE/mPJPE/PA-mPJPE.

Entropy analysis: DropKey increases per-head attention entropy more than vanilla Dropout, confirming the smoothing interpretation.

## Limitations & Open Questions

- The theoretical analysis relies on a simplified single-layer Lagrangian; the multi-layer interaction is not analyzed.
- Hyperparameter sweeping shows that *the optimal drop ratio depends on backbone weight*: heavyweight models (T2T-19, VOLOd2) want larger drop (0.1) than lightweight ones (0.05). No principled selection rule is provided.
- Train/test expectation mismatch is acknowledged; the finetuning fix adds an extra training stage.
- Tested only on encoder-style ViTs; not evaluated on causal/decoder transformers or NLP.
- Relationship to attention temperature is not explored — both produce smoother attention.

## Connections

- Direct alternative to vanilla **DropAttention** (Zehui et al. 2019) and the approach in [[sources/attentiondrop-baig-2025]] — DropKey is essentially "soft hard masking" applied uniformly rather than to top-$k$.
- The decreasing drop schedule is the *opposite* of [[concepts/stochastic-depth]] (Huang et al. 2016), which conventionally increases drop with depth.
- Adaptive smoothing interpretation links DropKey to [[concepts/label-smoothing]] and entropy-regularization methods.
- Structured-drop negative result is interesting next to **DropBlock** (Ghiasi et al. 2018), which is a strong CNN regularizer — supports the view that ViTs and CNNs have qualitatively different inductive biases.
- Relevant entity pages: [[entities/vit]], [[entities/t2t-vit]], [[entities/volo]], [[entities/deit]], [[entities/detr]].

---
title: "Gradient Starvation: A Learning Proclivity in Neural Networks"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [optimization, shortcut-learning, ntk, regularization, generalization]
sources: [raw/GradientStarvation_Pezeshki_ICML2021.pdf]
arxiv: "2011.09468"
---

## Summary

Pezeshki et al. (NeurIPS 2021; arXiv:2011.09468) formalize **Gradient Starvation (GS)**: under cross-entropy training, neural networks latch onto a small subset of statistically dominant features that are sufficient to reduce the loss, **starving** the gradients that would otherwise drive learning of other, potentially more robust, features. Using a neural-tangent-kernel (NTK) linearization the authors derive a coupled-ODE description of feature learning and show that highly correlated features suppress each other's updates. They propose **Spectral Decoupling (SD)**, a simple L2 penalty on logits, as a principled remedy.

## Key Contributions

- A precise definition of gradient starvation via NTK eigen-decomposition.
- A coupled-ODE analysis showing that learning dynamics along different kernel eigen-directions **interact through the softmax**, and that dominant directions choke out minor ones.
- **Spectral Decoupling**: replace weight decay with an L2 penalty on the network's logits, which provably decouples the ODEs and restores learning of minority features.
- Experiments on synthetic 2D, Colored-MNIST, CIFAR, and CelebA showing improved robustness to spurious correlations.

## Technical Details

**Setup.** Binary classification with logits $z(x)=f_\theta(x)\in\mathbb{R}$, loss $\mathcal{L}=\sum_i \log(1+e^{-y_i z_i})$. In the NTK regime, $z$ evolves linearly:
$$
\dot z_i = -\sum_j \Theta_{ij}\,\partial_{z_j}\mathcal{L},
$$
where $\Theta$ is the NTK Gram matrix. Diagonalizing $\Theta=U\Lambda U^\top$ gives feature coordinates $\tilde z=U^\top z$ whose dynamics are
$$
\dot{\tilde z}_k = -\lambda_k\,\tilde g_k(\tilde z),
$$
with $\tilde g$ the softmax gradient expressed in the eigenbasis.

**Starvation.** Because softmax-gradient magnitudes decay once the margin is positive, once the top-eigenvalue direction fits the data the gradient on **all** directions collapses. Minor features are never learned — "starved."

**Spectral Decoupling.** Adding
$$
\mathcal{L}_{\text{SD}}=\mathcal{L}+\tfrac{\lambda}{2}\|z\|^2
$$
changes each ODE to $\dot{\tilde z}_k=-\lambda_k(\tilde g_k+\lambda \tilde z_k)$, which **decouples the components**: each direction has its own linear drag, preventing the winner-take-all dynamic. Unlike weight decay (which shrinks parameters uniformly in parameter space) SD acts in function space.

## Results

- **Synthetic 2D spirals**: ERM learns a staircase boundary leveraging a single spurious direction; SD recovers the correct curved boundary.
- **Colored-MNIST**: SD substantially outperforms ERM and matches IRM without requiring environment labels.
- **CelebA (blonde-hair vs. gender bias)**: SD improves worst-group accuracy.
- **CIFAR-10/100**: SD is competitive with weight decay as a general regularizer even outside bias settings.

## Limitations & Open Questions

- Analysis is exact only in the NTK/lazy regime; finite-width, feature-learning regimes are only heuristically covered.
- Hyperparameter $\lambda$ is critical and dataset-specific.
- The theoretical story is cleanest for binary classification; multi-class softmax coupling is sketched but not fully characterized.
- Does not tell us **which** features are starved — only that some are.

## Connections

- Provides an optimization-level explanation for [[concepts/shortcut-learning]] and [[concepts/spurious-correlations]].
- Complementary to data-side fixes such as [[sources/conbias-chakraborty-2024]] and [[sources/mavias-sarridis-2025]].
- Related to NTK literature (Jacot et al. 2018) and IRM (Arjovsky et al. 2019).
- The multi-shortcut failures in [[sources/whac-a-mole-li-2023]] can be viewed as GS over multiple dominant directions.

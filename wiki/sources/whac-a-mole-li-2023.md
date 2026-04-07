---
title: "Whac-A-Mole: Multi-Shortcut Robustness and the UrbanCars Benchmark"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [shortcut-learning, benchmarks, spurious-correlations, robustness, datasets]
sources: [raw/WHAC-A-Mole_Li_ICLR2023.pdf]
---

## Summary

Li et al. (ICLR 2023) introduce **UrbanCars**, a diagnostic benchmark with **two controlled shortcuts** — background (BG) and co-occurring object (CoObj) — and empirically demonstrate a **Whac-A-Mole effect**: debiasing methods that mitigate one shortcut often **amplify reliance on the other**. They argue that single-shortcut benchmarks (Waterbirds, CelebA) have lulled the community into a false sense of progress, and call for multi-shortcut evaluation as standard practice.

## Key Contributions

- **UrbanCars dataset**: urban/country car classification where each image has both a controlled background (urban/country scene) and a controlled co-occurring object (fire hydrant, cow, etc.), with tunable correlation strengths.
- **BG-Gap and CoObj-Gap metrics**: per-shortcut worst-group accuracy deltas that expose which shortcut a model relies on.
- Systematic evaluation of ~10 SOTA debiasing methods (ERM, group-DRO, JTT, LfF, DebiAN, EIIL, LCA, etc.) showing the Whac-A-Mole phenomenon.
- Diagnosis that methods using **single-attribute group labels** cannot handle multi-shortcut regimes by construction.

## Technical Details

**Dataset construction.** A car foreground is composited onto a background and paired with a co-occurring object. The joint distribution over (class, BG, CoObj) is controlled by two correlation levels $\rho_{\text{BG}},\rho_{\text{CoObj}}\in[0,1]$; the default uses $0.95$ for both.

**Metrics.** Let $\text{Acc}$ be overall accuracy and $\text{Acc}^{\text{wg-BG}}$ be accuracy on the worst BG group:
$$
\text{BG-Gap} = \text{Acc} - \text{Acc}^{\text{wg-BG}},\qquad
\text{CoObj-Gap} = \text{Acc} - \text{Acc}^{\text{wg-CoObj}}.
$$
A model relying on BG has a large BG-Gap; mitigating BG typically shrinks BG-Gap but **enlarges** CoObj-Gap.

**Finding.** Across methods, the sum $\text{BG-Gap}+\text{CoObj-Gap}$ stays roughly constant, suggesting the network has a fixed "shortcut budget" that migrates between features when one is regularized.

## Results

- **ERM** relies strongly on BG, mildly on CoObj.
- **group-DRO on BG** closes BG-Gap to near zero but nearly **doubles** CoObj-Gap.
- **LfF / JTT**: similar migration behavior; no method using a single bias-label achieves low gaps on both.
- **DebiAN** (which learns the bias without labels) does better but still not uniformly.
- **Oracle multi-attribute group-DRO** is the only method that closes both gaps — highlighting the value of explicit multi-shortcut supervision.

## Limitations & Open Questions

- Only two shortcuts are controlled; real datasets have many.
- Synthetic compositing may not reflect natural shortcut structures.
- Does not propose a new mitigation method; it is primarily diagnostic.
- Leaves open whether open-set bias discovery (cf. [[sources/mavias-sarridis-2025]]) can escape the Whac-A-Mole trap.

## Connections

- Provides the benchmark used by [[sources/conbias-chakraborty-2024]] and [[sources/mavias-sarridis-2025]].
- Empirical counterpart to the theoretical [[sources/gradient-starvation-pezeshki-2021]] story about competing feature directions.
- Related to background-robustness work in [[sources/noise-or-signal-xiao-2021]].
- Core to [[concepts/shortcut-learning]] and [[concepts/spurious-correlations]].

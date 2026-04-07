---
title: "Noise or Signal: The Role of Image Backgrounds in Object Recognition"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [background-bias, robustness, imagenet, datasets, shortcut-learning]
sources: [raw/Noise_or_Signal_Xiao_NeurIPS2021.pdf]
arxiv: "2006.09994"
---

## Summary

Xiao et al. (NeurIPS 2021; arXiv:2006.09994) ask how much ImageNet classifiers rely on image **backgrounds** versus foreground objects. They release **ImageNet-9**, a controlled 9-class subset with foreground/background-swapped variants, and introduce the **BG-Gap** metric. They find that modern CNNs use background signal **significantly**, and that simply training on backgrounds alone yields non-trivial accuracy — indicating backgrounds carry class signal that models happily exploit.

## Key Contributions

- **ImageNet-9 (IN-9)** benchmark with seven variants: Original, Only-FG, Mixed-Same, Mixed-Rand, Mixed-Next, Only-BG-T (tiled), Only-BG-B (blank).
- **BG-Gap**: accuracy drop when the background is replaced with one drawn from a different class:
  $$\text{BG-Gap}=\text{Acc}(\text{Mixed-Same})-\text{Acc}(\text{Mixed-Rand}).$$
- Demonstration that **Only-BG** (no foreground) achieves 40-50% top-1 on IN-9, far above chance (11%), proving backgrounds are independently predictive.
- Evidence that **larger / more accurate models rely more on backgrounds**, not less — accuracy gains partly come from better background exploitation.
- Shows adversarial training reduces BG reliance at the cost of clean accuracy.

## Technical Details

**Dataset construction.** IN-9 uses nine coarse ImageNet super-classes (dog, bird, vehicle, reptile, carnivore, insect, instrument, primate, fish). Foreground masks are obtained via bounding boxes + GrabCut. Variants swap foreground/background combinations:

- **Only-FG**: foreground on black background.
- **Mixed-Same**: foreground on a random background **of the same class**.
- **Mixed-Rand**: foreground on a random background **of a different class**.
- **Only-BG-T**: foreground region filled by tiling surrounding background.
- **Only-BG-B**: foreground region blacked out.

**Background signal quantification.** Accuracy on Only-BG measures how much a classifier can recover from backgrounds alone. A model is "background-reliant" if BG-Gap $\gg 0$.

**Model suite.** ResNet-50 variants, including standard, adversarially trained (various $\epsilon$), and SIN (Stylized-ImageNet).

## Results

- Standard ResNet-50: ~88% on Original, but Mixed-Rand drops ~14 points $\Rightarrow$ large BG-Gap.
- Only-BG-T accuracy ~45%: backgrounds alone carry substantial signal.
- Higher-capacity / better-accuracy models (EfficientNet, etc.) have **larger** BG-Gaps.
- Adversarially trained models have smaller BG-Gap but lower clean accuracy.
- Stylized-ImageNet training reduces texture bias but not background bias.

## Limitations & Open Questions

- BG removal via bounding boxes is coarse; finer masks could change the picture.
- Only nine super-classes; unclear how results scale to 1000 classes.
- "Background" and "foreground" are not always well-defined (scenes, multi-object images).
- Does not distinguish context that is **causally** relevant from purely spurious background.

## Connections

- Foundational evidence for [[concepts/shortcut-learning]] and [[concepts/background-bias]].
- Complementary to [[sources/whac-a-mole-li-2023]], which generalizes to multi-shortcut.
- [[sources/singh-dont-judge-2020]] and [[sources/shetty-not-using-car-2019]] propose mitigations for context bias of this kind.
- Related to texture-vs-shape bias work (Geirhos et al. 2019).

---
title: "Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa)"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [inpainting, fourier-convolutions, generative-models, perceptual-loss]
sources: [raw/LaMa_Suvorov.pdf]
arxiv: "2109.07161"
venue: "WACV 2022"
---

## Summary

LaMa (Suvorov et al., 2021; Samsung AI) is a feed-forward image inpainting model designed specifically for **large missing regions** and **high-resolution generalization**. Its three core ingredients are (1) **Fast Fourier Convolutions (FFCs)** in the generator, which give every layer a global receptive field from the first block and enable extrapolation to resolutions far above the training size; (2) a **high receptive field (HRF) perceptual loss** using a pretrained segmentation network with dilated convolutions; and (3) an **aggressive large-mask training regime** where most training masks cover large contiguous areas. With only ~27M parameters, LaMa matches or beats much larger adversarial inpainters on Places and CelebA-HQ and generalizes zero-shot to 2K+ resolutions.

## Key Contributions

- Brings **Fast Fourier Convolutions** (Chi et al., 2020) into inpainting, giving an image-wide receptive field at every layer — critical for large holes and for high-res generalization.
- A **high receptive field perceptual loss** using a dilated-conv segmentation backbone rather than the standard VGG features.
- A **large-mask training protocol**: masks sampled as wide strokes, boxes, and irregular unions covering up to ~50% of the image, forcing the model to synthesize structure, not just texture.
- State-of-the-art or competitive results with a **small (~27M parameter) feed-forward model**, and surprising **zero-shot generalization to resolutions** well above training.

## Technical Details

### Fast Fourier Convolution (FFC)

A standard convolution has a local receptive field. FFC splits channels into a **local branch** (ordinary convs) and a **global branch** that performs a real-valued 2D FFT, applies a 1x1 conv on the spectral coefficients, and inverse-FFTs back:

$$
\text{FFC}(\mathbf{x}) = \text{Conv}_{\text{local}}(\mathbf{x}_L) \; \| \; \mathcal{F}^{-1}\big(\text{Conv}_{1\times 1}(\mathcal{F}(\mathbf{x}_G))\big).
$$

Because the FFT touches every spatial location, the effective receptive field is the whole image from layer one. This is both statistically efficient (periodic structures are captured compactly) and crucial at test time when the hole is larger than any conventional receptive field.

### HRF perceptual loss

Standard perceptual losses use VGG, whose receptive field is limited and which is trained on classification. LaMa uses a **dilated-conv segmentation network** (ResNet50-based) as the feature extractor for the perceptual loss:

$$
\mathcal{L}_{\text{HRF}} = \sum_\ell \big\| \phi_\ell(\mathbf{x}) - \phi_\ell(\hat{\mathbf{x}}) \big\|_1,
$$

with $\phi_\ell$ the dilated segmentation features. This penalizes semantic structural errors visible only at large scales.

### Large-mask training

Training masks are sampled from a mixture of wide brush strokes, axis-aligned boxes, and their unions, with parameters tuned so that the expected occluded area is large (often $>40\%$). This shifts the learned prior from texture completion toward global structure inference.

### Full loss

$$
\mathcal{L} = \mathcal{L}_{\text{adv}} + \lambda_{\text{HRF}} \mathcal{L}_{\text{HRF}} + \lambda_{R_1} R_1 + \lambda_{\text{disc}} \mathcal{L}_{\text{disc-perc}},
$$

combining a non-saturating adversarial loss, HRF perceptual loss, $R_1$ gradient penalty, and a discriminator-based perceptual term.

## Results

- **Places (512x512)**: beats CoModGAN, DeepFillv2, EdgeConnect on LPIPS and FID under wide-mask protocols.
- **CelebA-HQ**: competitive with much larger models.
- **High-res generalization**: LaMa trained at 256 inpaints 1024+ and 2K images zero-shot with no retiling; competitors collapse.
- **Parameter count**: ~27M — an order of magnitude smaller than contemporaneous GAN inpainters.

## Limitations & Open Questions

- Non-stochastic (deterministic generator), so cannot produce diverse completions for the same hole.
- Struggles with holes that require strong semantic content (faces, text) where a diffusion-based inpainter would do better.
- Adversarial training is still touchy; results depend on careful discriminator tuning.
- Superseded on quality metrics (but not on speed/size) by diffusion inpainters post-2022.

## Connections

- Core dependency of [[sources/foraug-nauen-2025|ForAug]], which uses LaMa to erase foregrounds and build its background bank.
- Builds on Fast Fourier Convolutions (Chi et al., NeurIPS 2020).
- Contrast with diffusion-based inpainters ([[entities/stable-diffusion|Stable Diffusion]] inpaint, RePaint) — LaMa is a single feed-forward pass and far cheaper.
- Relevant to [[concepts/image-inpainting]] and [[concepts/global-receptive-field]].

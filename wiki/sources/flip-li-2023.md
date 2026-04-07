---
title: "Scaling Language-Image Pre-training via Masking (FLIP)"
type: source
created: 2026-04-07
updated: 2026-04-07
tags: [vision-language, clip, masking, scaling, contrastive-learning, vlm]
sources: [raw/PatchDropout_Liu_2023.pdf]
arxiv: "2212.00794"
venue: "CVPR 2023"
note: "Filename in raw/ refers to PatchDropout_Liu_2023, but the actual PDF content is FLIP (Li, Fan, Hu, Feichtenhofer, He; Meta FAIR). Filed under the correct paper identity."
---

## Summary

Li, Fan, Hu, Feichtenhofer, and He (Meta FAIR, CVPR 2023) propose **FLIP (Fast Language-Image Pre-training)**, a strikingly simple modification to [[entities/clip]]: during contrastive image-text pretraining, randomly mask out and *remove* a large fraction (50–75%) of image patches from the ViT encoder, MAE-style. Unlike MAE, no reconstruction is performed — only the contrastive InfoNCE loss is used. The 2–4× FLOPs reduction enables 2–4× larger batches at the same memory, and the larger batch + more samples-per-iteration trade-off **improves both accuracy and wall-clock time** over standard CLIP. Pretrained on LAION-400M, FLIP outperforms OpenCLIP and an in-house CLIP reproduction across zero-shot classification, retrieval, captioning, VQA, and robustness benchmarks, and enables practical scaling experiments along model, data, and schedule axes.

## Key Contributions

- **Patch masking as a CLIP accelerator**: drop 50–75% of image patches at the encoder input; reduce per-step compute and memory enough to scale batch size.
- **No reconstruction**: unlike MAE, FLIP keeps only the contrastive objective. Adding a reconstruction head is *neutral or slightly negative* for zero-shot.
- **Inference unmasking**: while pretrained on masked images, the model is applied to *intact* images at inference; a brief "unmasked tuning" stage closes the small remaining train/test gap.
- **3.7× wall-clock speedup** to reach a given accuracy on LAION-400M ViT-L/16, vs. the 0%-mask CLIP baseline.
- **Large-scale scaling study** along three axes: model size (ViT-L → ViT-H), data size (400M → 2B LAION), and schedule length (12.8B → 25.6B sampled pairs). Data scaling improves zero-shot transfer; model scaling improves transfer learning; combining all three is complementary.
- **New SOTA on public-data ImageNet-1K zero-shot**: 78.8% (FLIP-H/14 with model+data scaling), beating OpenCLIP-H (78.0%) at ~5× lower training cost.

## Technical Details

### Architecture

Following CLIP: a ViT image encoder and a Transformer text encoder. Differences from CLIP:
- **No extra LayerNorm** after patch embedding.
- **Global average pooling** at the end of the image encoder (not class token).
- **Non-autoregressive** text encoder (easier to mask), WordPiece tokenizer, sequence length 32 (vs CLIP's BPE/length 77).

### Masking

A grid of non-overlapping $16\times 16$ (or $14\times 14$) patches is divided; a mask ratio $r \in \{0.5, 0.75\}$ is sampled and the corresponding patches are *removed* (not just zeroed) from the input sequence. The image encoder runs on the visible patches only, similar to MAE. This reduces image-encoder FLOPs to $1/2$ (or $1/4$) and allows $2{-}4\times$ larger batches at the same memory.

### Contrastive objective

Standard InfoNCE / symmetric cross-entropy on cosine-similarity embeddings, with a learnable temperature $\tau$:
$$\mathcal{L}_{\text{FLIP}} = -\frac{1}{2N}\sum_{i=1}^N\Big[\log\frac{\exp(\langle\mathbf{z}^I_i,\mathbf{z}^T_i\rangle/\tau)}{\sum_j\exp(\langle\mathbf{z}^I_i,\mathbf{z}^T_j\rangle/\tau)} + \log\frac{\exp(\langle\mathbf{z}^I_i,\mathbf{z}^T_i\rangle/\tau)}{\sum_j\exp(\langle\mathbf{z}^I_j,\mathbf{z}^T_i\rangle/\tau)}\Big].$$

### Text masking (ablation only)

Symmetric text masking (50%) is studied but gives a marginal speedup (text encoder is small) and a 2.2% accuracy drop. Prioritized masking on non-pad tokens reduces the drop to 0.4%. Not used in main results.

### Unmasked tuning

After main pretraining, run a brief (~0.32 epoch) phase with mask ratio set to 0% on the same data. This closes the train/inference distribution gap, improving 75%-mask zero-shot from 68.2 → 69.5 and 50%-mask from 69.6 → 70.1.

### Hyperparameters

AdamW, LR 4e-6 (linear scaled with batch / 256), weight decay 0.2, $(\beta_1,\beta_2) = (0.9, 0.95)$, cosine decay, 51.2M-256M warmup samples, float32. Trained on TPU-v3 (256 cores for B/L, 256 for H).

## Results

### Zero-shot ImageNet-1K (LAION-400M, 32 epochs)

| Model           | Data        | ViT  | Zero-shot |
|-----------------|-------------|------|-----------|
| OpenCLIP        | LAION-400M  | L/14 | 72.8      |
| CLIP (repro.)   | LAION-400M  | L/14 | 73.1      |
| **FLIP**        | LAION-400M  | L/14 | **74.6**  |
| **FLIP**        | LAION-400M  | H/14 | **75.5**  |

FLIP exceeds OpenCLIP by **1.8%** and the in-house CLIP reproduction by **1.5%** on the same data. With LAION-2B + 25.6B sampled data (joint scaling), FLIP-H/14 reaches **78.8%**, vs OpenCLIP-H 78.0%.

### Transfer

- **Linear probing** (LAION-400M, ViT-L/16): FLIP 83.6% > CLIP repro 82.6% > OpenCLIP 82.1%.
- **Fine-tuning**: FLIP 86.9% > 86.3 / 86.2.
- **Zero-shot retrieval (Flickr30k, COCO)**: FLIP dominates CLIP baselines on both image and text retrieval R@1/5/10.
- **Zero-shot robustness (IN-V2/A/R/Sketch/ObjectNet)**: FLIP > CLIP-LAION on all sets, but still trailing the original WIT-trained CLIP on IN-A/R (clear systematic gap from pretraining data, not method).
- **COCO captioning**: BLEU-4 37.4 vs CLIP-repro 36.4; CIDEr 127.7 vs 125.6.
- **VQAv2**: 74.7 vs 74.5 (parity).

### Scaling behavior (Table 8)

| Setting             | Sampled | IN-1K | COCO retrieval | Captioning |
|---------------------|---------|-------|-----------------|------------|
| Baseline L/16       | 12.8B   | 74.3  | 75.0            | 127.7      |
| + Model scaling H/16| 12.8B   | 75.5  | 76.4            | 130.3      |
| + Data scaling 2B   | 12.8B   | 75.8  | 78.2            | 128.9      |
| + Schedule 25.6B    | 25.6B   | 73.9  | 75.5            | 130.2      |
| + Model + Data      | 12.8B   | 77.6  | 79.9            | 130.4      |
| + Joint (all 3)     | 25.6B   | **78.8** | **80.9**     | 130.2      |

Data scaling helps zero-shot transfer most; model scaling helps fine-tuning transfer most; schedule scaling has rapidly diminishing returns.

## Limitations & Open Questions

- **Inference distribution shift**: model is trained on masked images but applied to intact images. Mitigated by unmasked tuning but not fully eliminated.
- **No reconstruction signal**: ablation shows reconstruction is "small negative or neutral" for zero-shot, but the question of whether a properly tuned MAE-style head could *help* fine-tuning remains open.
- **Text masking**: dismissed because text encoder is small, but for larger text encoders (e.g., LLM-style) the trade-off may flip.
- **Schedule scaling has diminishing returns** — implies that simply training longer cannot substitute for more data.
- **Pretraining data still matters more than method**: WIT-trained original CLIP retains a ~20% advantage on IN-A even when LAION-trained FLIP outperforms LAION-CLIP. The community's switch to LAION incurs a quality cost.

## Connections

- FLIP combines [[entities/clip]] (contrastive objective) with [[concepts/masked-autoencoders]] (sparse ViT encoder) — but *without* the reconstruction head. The asymmetry vs MAE is notable.
- Closely related to **MAE** (He et al. 2022) and **MaskCLIP** (Dong et al. 2022), which uses masked self-distillation rather than masking the encoder input.
- Predecessor / analog of methods that drop image patches for efficiency (e.g., literal "PatchDropout" by Liu et al.) — the underlying intuition is the same: contrastive learning is robust to losing a large fraction of input patches because the alignment objective operates at the *image-level* feature.
- Important for scaling discussions in [[concepts/contrastive-learning]] and [[concepts/vision-language-pretraining]].
- Relevant entity pages: [[entities/clip]], [[entities/openclip]], [[entities/laion]], [[entities/vit]], [[entities/mae]].

> Note on filename mismatch: the file `raw/PatchDropout_Liu_2023.pdf` contains the FLIP paper, not the literal "PatchDropout" paper by Liu et al. Filed here under the correct paper identity (Li et al. 2023, FLIP).

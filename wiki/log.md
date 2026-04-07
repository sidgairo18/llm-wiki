---
title: "Wiki Log"
type: log
created: 2026-04-06
updated: 2026-04-07
---

# Wiki Log

## [2026-04-06] init | Wiki initialized
Focus areas: vision-language models, representation learning, generative diffusion (image/video), mechanistic interpretability. Schema configured for Claude Code agent. Obsidian as frontend IDE.

## [2026-04-07] batch-ingest | Bulk ingest of 34 source PDFs
Batch-ingested 34 PDFs from `raw/` in parallel across four thematic groups. The corpus shifts the wiki's de facto scope toward **object detection, synthetic data generation, and spurious-correlation/bias mitigation** — adjacent to but distinct from the original VLM/diffusion/interpretability focus. See `overview.md` for updated synthesis.

Groups processed:
- **Detection / DETR (6):** Grounding DINO, Co-DETR, Deformable DETR, RT-DETRv4, S3OD, plus one mislabeled file.
- **Synthetic data / augmentation (14):** DiffusionEngine, DiverGen, DiverseGen, ControllableDiffusionAug, FreeMask, InstaGen, MosaicFusion, ODGEN, X-Paste, SimpleCopyPaste, SOS, ForAug, LaMa, CACP.
- **Bias / spurious correlations (10):** ConBias, MAVias, Gradient Starvation, Whac-A-Mole, Noise or Signal, Singh (Don't Judge), Shetty (Not Using Car), PAR, VB-Mitigator, CSBD.
- **Attention regularization (3):** AttentionDrop, DropKey, FLIP.

**Failed / mismatched files** (see index.md "Missing / Failed Ingest"):
- 3 empty/unreadable: L-CRP, ObjectNet, (plus DSA extraction failed).
- 4 filename/content mismatches: DG-DETR (was AttentionDrop), PatchDropout (was FLIP), BackgroundNoMore, WeatherCausal, DSA.

Deferred: concept/entity page extraction, cross-linking pass, lint. Recommend a subsequent lint pass to extract recurring concepts (copy-paste augmentation, spurious correlation, DETR lineage, diffusion-based data synthesis) into dedicated concept pages.

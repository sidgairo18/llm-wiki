---
title: "Overview"
type: overview
created: 2026-04-06
updated: 2026-04-07
tags: [meta, synthesis]
---

# Research Overview

*Evolving synthesis of everything in the wiki. Rewritten as sources accumulate.*

## Scope Update (2026-04-07)

After the first batch ingest of 34 sources, the working scope is broader than the originally stated focus areas. The corpus clusters around four interconnected threads — all touching on **how models learn *what they should* vs. *what the data lets them*** — and the wiki now reflects this shift:

1. **Object Detection & DETR lineage** — query-based detectors and their evolution toward real-time, open-vocabulary, and distilled variants.
2. **Synthetic Data & Generative Augmentation** — using diffusion models, inpainting, and copy-paste to manufacture training data for detection and segmentation.
3. **Spurious Correlations & Bias Mitigation** — diagnosing and removing reliance on background, co-occurrence, and contextual shortcuts.
4. **Attention / Regularization** — making ViT attention less brittle via structured dropout and PAC-Bayes-motivated regularizers.

The originally stated pillars (VLMs, representation learning, diffusion, mech-interp) remain in scope but are currently under-sourced; they enter the wiki mainly as *tools* (DINOv3 as a feature encoder in S3OD/RT-DETRv4; FLUX/Stable Diffusion as data engines; CLIP/BLIP as open-vocabulary priors).

## Four Threads

### 1. Object Detection / DETR lineage
From the original DETR's slow convergence, [[sources/deformable-detr-zhu-2021|Deformable DETR]] introduced sparse multi-scale deformable attention (10× faster training), [[sources/co-detr-zong-2023|Co-DETR]] added auxiliary one-to-many heads for richer supervision, and [[sources/groundingdino-liu|Grounding DINO]] fused language into the encoder/decoder for open-set detection. [[sources/rt-detrv4-2025|RT-DETRv4]] closes the real-time/accuracy gap by distilling DINOv3 features into RT-DETR. [[sources/s3od-kupyn-2025|S3OD]] applies the same "DINOv3 as prior + generative data" recipe to salient object detection.

### 2. Synthetic Data & Generative Augmentation
The central question: **can generative models close the long tail of real-world detection/segmentation datasets?**

- **Diffusion as a data engine:** [[sources/diffusionengine-zhang-2023|DiffusionEngine]], [[sources/divergen-fan-2024|DiverGen]], [[sources/diversegen-nie-2024|DiverseGen]], [[sources/controllable-diffusion-aug-fang-2024|ControllableDiffusionAug]], [[sources/freemask-yang-2024|FreeMask]], [[sources/instagen-feng-2024|InstaGen]], [[sources/mosaicfusion-xie-2024|MosaicFusion]], [[sources/odgen-zhu-2024|ODGEN]], [[sources/s3od-kupyn-2025|S3OD]] — each exploits a different property (T2I controllability, cross-attention masks, mask-conditioned generation, mosaic compositing).
- **Copy-paste lineage:** [[sources/simple-copy-paste-ghiasi-2021|Simple Copy-Paste]] → [[sources/x-paste-zhao-2023|X-Paste]] (foundation-model-extracted segments) → [[sources/cacp-guo-2024|CACP]] (BLIP+SAM+GradCAM context-aware pipeline) → [[sources/sos-huang-2025|SOS]] (synthetic segments) → [[sources/foraug-nauen-2025|ForAug]] (foreground-focused augmentation).
- **Inpainting:** [[sources/lama-suvorov-2021|LaMa]] provides the backbone technology for background replacement and object removal used by several augmentation methods.

### 3. Spurious Correlations & Bias Mitigation
Evidence that models exploit backgrounds/context instead of objects: [[sources/noise-or-signal-xiao-2021|Noise or Signal]] (ImageNet backgrounds), [[sources/singh-dont-judge-2020|Singh et al.]] ("Don't judge an object by its context"), [[sources/shetty-not-using-car-2019|Shetty et al.]] (object removal as a test), [[sources/par-cooccurrencebias-ijcv2025|PAR]] (co-occurrence bias in detection).

Mitigation strategies: [[sources/conbias-chakraborty-2024|ConBias]] (concept-level debiasing), [[sources/mavias-sarridis-2025|MAVias]] (multi-attribute), [[sources/whac-a-mole-li-2023|Whac-A-Mole]] (fixing one spurious feature reveals another), [[sources/gradient-starvation-pezeshki-2021|Gradient Starvation]] (optimization-level explanation), [[sources/vb-mitigator-2025|VB-Mitigator]] (toolkit), [[sources/csbd-wang-2024|CSBD]] (class-specific discovery).

**Cross-thread tension:** synthetic data (Thread 2) is one of the most promising bias mitigators — it can manufacture counterfactual backgrounds and rare contexts — yet diffusion models inherit biases from their own training data. Several papers here (MosaicFusion, FreeMask, InstaGen) implicitly rely on this being acceptable. An open question worth tracking: *when does synthetic augmentation reduce vs. entrench spurious correlations?*

### 4. Attention Regularization
[[sources/attentiondrop-baig-2025|AttentionDrop]] (three stochastic attention regularizers with a PAC-Bayes bound), [[sources/dropkey-li-2023|DropKey]] (drop Keys pre-softmax with depth-decreasing schedule), [[sources/flip-li-2023|FLIP]] (mask 50-75% of patches in CLIP pretraining for speedup). All three target the brittleness of ViT attention but from different angles — regularization, optimization speedup, and a compute-efficiency angle respectively.

## Cross-Cutting Observations

- **DINOv3 as the universal prior.** Both RT-DETRv4 and S3OD use DINOv3 features as a strong geometric/semantic prior, distilled into efficient task-specific heads. Representation learning has become infrastructure for detection.
- **Copy-paste never died.** Despite diffusion-based augmentation, a sizable fraction of the corpus still leans on paste-style compositing — often combined with foundation-model-extracted segments. The two paradigms are converging.
- **Bias literature is operationalizing.** Early work (Noise or Signal, Singh, Shetty) diagnosed the problem; the 2024-2025 cluster (ConBias, MAVias, VB-Mitigator, CSBD) is building reusable tooling and benchmarks.
- **The detection/generation/bias triangle.** These three threads are not independent: detection benefits from synthetic data, synthetic data risks amplifying bias, and bias analysis evaluates detection. Future synthesis should probably treat them as one system.

## Open Questions
- When does synthetic augmentation *reduce* vs. *amplify* spurious correlations?
- Is DINOv3 distillation the new default for efficient detection, or a transitional approach?
- Can the "Whac-A-Mole" problem (fixing one spurious feature reveals another) be avoided structurally, or only empirically?
- How should co-occurrence bias benchmarks (PAR) interact with generative augmentation pipelines (InstaGen, ODGEN)?

## Gaps
- No dedicated concept pages yet (e.g., `concepts/copy-paste-augmentation.md`, `concepts/detr-lineage.md`, `concepts/spurious-correlation.md`). Recommended next step.
- No entity pages for recurring models (DINOv3, Stable Diffusion, FLUX, SAM, BLIP, CLIP).
- The originally-stated focus on mechanistic interpretability has no sources yet.
- Video diffusion remains uncovered.

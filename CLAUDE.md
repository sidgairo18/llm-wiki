# LLM Wiki — Schema & Agent Instructions

> This file configures Claude Code as the wiki maintainer.
> It is the authoritative reference for structure, conventions, and workflows.

## Overview

This is a personal research wiki on **computer vision and machine learning**, with focus areas:

- **Vision-Language Models** (CLIP, SigLIP, LLaVA, Florence, PaLI, etc.)
- **Representation Learning** (contrastive, self-supervised, masked autoencoders, DINOv2, etc.)
- **Generative Diffusion Models** — image (DDPM, LDM, Stable Diffusion, DiT, flow matching) and video (SVD, Sora-class, CogVideo, etc.)
- **Mechanistic Interpretability** (probing, feature visualization, SAEs, circuit analysis, causal tracing)
- **Interpretability** (heatmap based attributions, inherent interpretability, concept discovery, concept)

The human is a PhD researcher comfortable with advanced ML. Use precise technical language, cite arXiv IDs where known, and include mathematical notation (LaTeX in `$...$` or `$$...$$`) when it aids clarity. Prefer PyTorch conventions when discussing implementation.

---

## Directory Structure

```
llm-wiki/
├── CLAUDE.md              ← this file (schema)
├── raw/                   ← immutable source documents (any format)
│   ├── assets/            ← images downloaded from clipped web articles (not PDFs)
│   ├── *.pdf              ← arXiv papers, reports (most common)
│   ├── *.md               ← web articles via Obsidian Web Clipper, personal notes
│   └── *.txt / *.html     ← other text sources
├── wiki/                  ← LLM-maintained knowledge base
│   ├── index.md           ← master catalog of all wiki pages
│   ├── log.md             ← chronological record of operations
│   ├── overview.md        ← high-level synthesis of all knowledge
│   ├── sources/           ← one summary page per ingested source
│   ├── concepts/          ← topic pages (e.g., flow-matching.md)
│   ├── entities/          ← people, orgs, models, datasets, benchmarks
│   ├── comparisons/       ← structured comparisons (e.g., DiT-vs-UNet.md)
│   └── explorations/      ← filed Q&A, analyses, deep-dives
└── tools/                 ← optional CLI utilities
```

---

## Conventions

### Filenames
- Lowercase, hyphenated: `latent-diffusion.md`, `openai.md`
- No spaces, no special characters beyond hyphens

### Page Format

Every wiki page starts with YAML frontmatter:

```yaml
---
title: "Latent Diffusion Models"
type: concept              # one of: source, concept, entity, comparison, exploration
created: 2026-04-06
updated: 2026-04-06
tags: [diffusion, generative-models, image-synthesis]
sources: [raw/ldm-rombach-2022.md]     # which raw sources inform this page
arxiv: "2112.10752"                     # if applicable
---
```

### Cross-References
- Use Obsidian-style wiki links: `[[concept-name]]` or `[[concept-name|display text]]`
- Link liberally — every mention of a concept/entity that has its own page should be linked on first occurrence in a section.

### Mathematical Notation
- Inline: `$\mathcal{L}_{\text{CLIP}} = ...$`
- Display: fenced `$$` blocks
- Use standard ML notation: $\mathbf{x}$ for vectors, $\mathbf{W}$ for weight matrices, $p_\theta$ for parameterized distributions, $\mathbb{E}$ for expectations.

### Citations
- Always include arXiv ID when known: `(Rombach et al., 2022; arXiv:2112.10752)`
- For non-arXiv sources, include URL or DOI

---

## Workflows

### 1. INGEST (adding a new source)

When the human says "ingest [source]" or drops a file into `raw/`:

**Supported source formats:**
- `.pdf` — read directly (arXiv papers, reports). Most common case.
- `.md` — read as markdown (web clips, notes).
- `.txt` / `.html` — read as text.
- If the source contains inline images, reference them from `raw/assets/`.

**Naming convention for raw files:** `<short-slug>-<first-author>-<year>.<ext>`
Examples: `clip-radford-2021.pdf`, `mae-he-2022.pdf`, `scaling-vlms-blog-2024.md`

**Steps:**

1. **Read** the source fully.
2. **Discuss** key takeaways with the human — ask what to emphasize if ambiguous.
3. **Create** `wiki/sources/<source-slug>.md` with:
   - Frontmatter (type: source)
   - One-paragraph summary
   - Key contributions (bulleted)
   - Technical details (methods, architectures, loss functions — include equations)
   - Results summary (key benchmarks, SOTA claims)
   - Limitations & open questions
   - Connections to existing wiki pages
4. **Update** existing wiki pages:
   - Add/update relevant `wiki/concepts/` pages
   - Add/update relevant `wiki/entities/` pages (models, datasets, people)
   - Update `wiki/overview.md` if the source changes the big picture
   - Flag contradictions with existing content explicitly: `> ⚠️ CONTRADICTION: ...`
5. **Update** `wiki/index.md` — add entry under correct category.
6. **Append** to `wiki/log.md`.

#### Batch Ingest

When the human says "ingest all new files in raw/" or lists multiple files:

1. **Detect** unprocessed sources: list files in `raw/` that have no corresponding page in `wiki/sources/`.
2. **Report** the list to the human and confirm before proceeding.
3. **Two modes:**
   - **Interactive** (default): process one source at a time, discuss each with the human before moving to the next.
   - **Batch** (human says "batch mode" or "don't wait"): process all sources sequentially without pausing. Write a brief summary of each to the log. Present a full summary at the end.
4. In batch mode, **defer large wiki restructuring** — create source pages and update entities/concepts, but save `overview.md` rewrite for the end (one rewrite after all sources, not one per source).
5. In batch mode, **update `index.md` incrementally** (after each source) so progress is visible in Obsidian.

### 2. QUERY (answering questions)

When the human asks a question:

1. **Read** `wiki/index.md` to locate relevant pages.
2. **Read** the relevant wiki pages (not raw sources, unless wiki is insufficient).
3. **Synthesize** an answer with `[[wiki-links]]` as citations.
4. **Offer to file** the answer as `wiki/explorations/<slug>.md` if it contains novel synthesis.

### 3. LINT (health check)

When the human says "lint" or "health check":

1. Scan all wiki pages for:
   - **Contradictions**: claims that conflict across pages
   - **Stale content**: superseded by newer sources
   - **Orphan pages**: no inbound links from other pages
   - **Missing pages**: concepts/entities mentioned but lacking their own page
   - **Missing cross-references**: related pages that don't link to each other
   - **Data gaps**: important topics in the focus areas with no sources yet
2. Report findings as a checklist.
3. Suggest new sources to look for and new questions to investigate.

### 4. COMPARE

When the human asks to compare methods/models:

1. Create `wiki/comparisons/<slug>.md` with a structured table + prose analysis.
2. Include: architecture, training objective, data requirements, key metrics, compute cost, limitations.
3. Update `wiki/index.md`.

---

## Index File Format

`wiki/index.md` uses this structure:

```markdown
# Wiki Index

## Sources
| Page | Summary | Date Added |
|------|---------|------------|
| [[sources/clip-radford-2021]] | Contrastive vision-language pretraining | 2026-04-06 |

## Concepts
| Page | Summary |
|------|---------|
| [[concepts/contrastive-learning]] | InfoNCE, NT-Xent, and related objectives |

## Entities
| Page | Type | Summary |
|------|------|---------|
| [[entities/clip]] | model | OpenAI's contrastive VLM |

## Comparisons
| Page | Summary |
|------|---------|
| [[comparisons/dit-vs-unet]] | Transformer vs U-Net architectures for diffusion |

## Explorations
| Page | Summary | Date |
|------|---------|------|
```

---

## Log File Format

`wiki/log.md` uses this structure — each entry starts with `##` for parseability:

```markdown
# Wiki Log

## [2026-04-06] init | Wiki initialized
Focus areas: VLMs, representation learning, diffusion models, interpretability.
```

---

## Tips for the Agent

- **Err on the side of creating pages.** If a concept is mentioned more than once across sources, it deserves its own page.
- **Be opinionated in the overview.** The overview should reflect the current synthesis, not just list sources.
- **Note temporal context.** Methods evolve fast — always note when a result was published and whether it's been superseded.
- **Preserve nuance.** Don't flatten disagreements — if the field is split, say so.
- **Use the explorations/ directory.** Good Q&A answers should be filed, not lost in chat history.

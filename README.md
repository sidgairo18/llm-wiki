# LLM Wiki — Quick Start

A personal, LLM-maintained knowledge base for machine learning research. You curate the sources and ask the questions; the LLM does all the summarizing, cross-referencing, and bookkeeping.

## How It Works

You drop source documents (papers, articles, notes) into `raw/`. You tell your LLM agent to ingest them. The agent reads each source, writes structured summary pages, creates and updates concept/entity pages, maintains cross-references, and keeps a master index — all as interlinked markdown files in `wiki/`. You browse the result in Obsidian and direct the agent's work. Over time the wiki compounds: every source and every question you ask makes it richer.

## Prerequisites

- **Claude Code** (terminal) — the LLM agent that maintains the wiki
- **Obsidian** — the frontend for browsing the wiki (AppImage works without sudo, see below)

## Setup

```bash
# Unpack the starter kit
tar xzf llm-wiki.tar.gz
cd llm-wiki

# Open in Obsidian: launch Obsidian → "Open folder as vault" → select llm-wiki/

# Open Claude Code in the same directory
claude
```

Claude Code will read `CLAUDE.md` automatically and know how to operate the wiki.

## Directory Structure

```
llm-wiki/
├── CLAUDE.md                ← schema & agent instructions (edit to customize)
├── obsidian-setup-guide.md  ← detailed Obsidian tutorial
├── raw/                     ← your source documents (immutable)
│   └── assets/              ← images from clipped web articles
├── wiki/                    ← LLM-maintained knowledge base
│   ├── index.md             ← master catalog (agent reads this first)
│   ├── log.md               ← chronological operation record
│   ├── overview.md          ← evolving synthesis of all knowledge
│   ├── sources/             ← one summary page per ingested source
│   ├── concepts/            ← topic pages (e.g., flow-matching.md)
│   ├── entities/            ← models, datasets, people, orgs, benchmarks
│   ├── comparisons/         ← structured side-by-side analyses
│   └── explorations/        ← filed Q&A, deep-dives, analyses
└── tools/                   ← optional CLI utilities (search, etc.)
```

## Core Workflows

**Ingest a source:**
```
# Drop a PDF into raw/, then tell the agent:
ingest raw/clip-radford-2021.pdf

# Or batch ingest everything new:
ingest all new files in raw/

# Autonomous batch (no per-paper discussion):
ingest all new files in raw/ — batch mode, don't wait for my input
```

**Ask a question:**
```
How does flow matching differ from the DDPM forward process?
```
The agent searches the wiki index, reads relevant pages, and synthesizes an answer with citations. Ask it to file the answer as an exploration page if it's worth keeping.

**Run a health check:**
```
lint
```
Finds contradictions, orphan pages, missing cross-references, stale claims, and suggests new sources to look for.

**Compare methods:**
```
compare DiT vs U-Net for diffusion backbones
```
Creates a structured comparison page in `wiki/comparisons/`.

## Naming Convention for Sources

`<short-slug>-<first-author>-<year>.<ext>`

Examples: `clip-radford-2021.pdf`, `mae-he-2022.pdf`, `dit-peebles-2023.pdf`, `scaling-vlms-blog-2024.md`

## Obsidian Tips

- Install via AppImage (no sudo): `chmod +x Obsidian-*.AppImage && ./Obsidian-*.AppImage`
- Install the **Dataview** plugin for live tables over page metadata
- Install **Obsidian Web Clipper** (browser extension) to clip articles directly into `raw/`
- Use **Graph View** (`Ctrl+G`) to see the shape of your wiki
- Use **Quick Switcher** (`Ctrl+O`) to jump to any page
- See `obsidian-setup-guide.md` for full setup instructions

## FAQ

**Should I make a separate wiki for each project?**
Not if your projects overlap. Interconnected research areas (e.g., VLMs, diffusion, representation learning) benefit from shared cross-references. Use one wiki and rely on folders, tags, and the index for internal organization. Make a separate wiki only for genuinely disjoint domains.

**Can I put PDFs directly in `raw/`?**
Yes. `raw/` accepts any format: `.pdf`, `.md`, `.txt`, `.html`. PDFs go directly in `raw/`, not in `raw/assets/`. The `assets/` subdirectory is specifically for images downloaded from clipped web articles.

**What if my wiki outgrows the index file?**
At small-to-medium scale (~100 sources, ~hundreds of pages), the flat `index.md` works well — the LLM reads it and navigates from there. When it gets too large, move to hierarchical indices (`index-diffusion.md`, `index-vlm.md`, etc.) or add a search tool like qmd (hybrid BM25/vector search with LLM re-ranking).

**Should I edit wiki pages manually?**
Generally no — that's the LLM's job. If you want something changed, tell the agent. You *read* the wiki; the LLM *writes* it. The exception is `CLAUDE.md`, which you and the LLM co-evolve as you figure out what conventions work for your domain.

**How do I back up the wiki?**
It's just a directory of files. `git init` and commit regularly — you get full version history for free. This also lets you revert if the LLM makes a bad edit.

**Can I use a different LLM agent?**
Yes. The pattern is agent-agnostic. Rename `CLAUDE.md` to whatever your agent reads (e.g., `AGENTS.md` for Codex) and adjust agent-specific instructions. The wiki structure and conventions stay the same.

**What about embedding-based RAG?**
You probably don't need it at first. The index-based approach (LLM reads `index.md`, drills into pages) works surprisingly well up to a few hundred pages. Add vector search when navigation by index becomes a bottleneck, not before.

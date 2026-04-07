---
title: "Obsidian Setup Guide for LLM Wiki"
---

# Obsidian Setup Guide

A quick-start guide for using Obsidian as the frontend for your LLM Wiki.

## 1. Install Obsidian

Download from [obsidian.md](https://obsidian.md). Available on Linux, macOS, Windows, iOS, Android. Free for personal use.

## 2. Open Your Wiki as a Vault

Obsidian operates on **vaults** — each vault is just a folder on disk.

1. Launch Obsidian → "Open folder as vault"
2. Select your `llm-wiki/` directory
3. That's it — Obsidian now indexes all `.md` files in the tree

Everything Claude Code writes to this folder appears in Obsidian in real time. No sync step needed.

## 3. Essential Settings

Open Settings (gear icon, bottom-left) and configure:

### Files & Links
- **New link format** → "Shortest path when possible" (keeps `[[wiki-links]]` clean)
- **Use [[Wikilinks]]** → ON (this is what the wiki uses)
- **Default location for new attachments** → "In subfolder under current folder" or set to `raw/assets/`
- **Detect all file extensions** → ON

### Editor
- **Show frontmatter** → ON (so you can see the YAML metadata)
- **Readable line length** → personal preference, but ON is usually easier on the eyes

### Appearance
- Pick a theme you like. "Minimal" and "Prism" are popular for research use.

## 4. Recommended Community Plugins

Go to Settings → Community plugins → Turn off restricted mode → Browse.

### Must-Have

**Dataview**
Runs queries over your page frontmatter. Once installed, you can embed live tables in any page:
```dataview
TABLE title, type, updated
FROM "wiki/concepts"
SORT updated DESC
```
This auto-generates a table of all concept pages sorted by last update. Very powerful for navigating a growing wiki.

**Graph View** (built-in, just enable it)
Click the graph icon in the left sidebar. Shows all pages as nodes, all `[[wiki-links]]` as edges. Invaluable for seeing the shape of your knowledge base — which pages are hubs, which are orphans, where clusters form.

Tips for graph view:
- Filter by folder (e.g., show only `wiki/concepts/`)
- Color nodes by tag or folder
- Use "local graph" on any page to see its neighborhood

### Nice-to-Have

**Marp Slides**
Renders Marp-format markdown as slide decks directly in Obsidian. When you ask Claude Code to generate a presentation from wiki content, it can output Marp format and you view it here.

**Obsidian Web Clipper** (browser extension)
Converts web articles to markdown and saves them to your vault. Perfect for getting sources into `raw/`. Install the browser extension, configure it to save to `raw/`, and clip articles with one click.

After clipping, you can download embedded images locally:
- Settings → Files and links → Attachment folder path → `raw/assets/`
- Settings → Hotkeys → search "Download attachments" → bind to e.g. `Ctrl+Shift+D`
- After clipping an article, hit the hotkey to localize all images

**Calendar**
Shows a calendar sidebar; clicking a date opens/creates a daily note. Useful if you combine the wiki with a research journal.

**Templater**
Lets you create templates with dynamic content (dates, prompts). Can be useful for standardizing source summary pages, though Claude Code handles most of this.

## 5. Workflow: Side-by-Side

The recommended setup:
- **Left monitor / left half**: Terminal with Claude Code open in `llm-wiki/`
- **Right monitor / right half**: Obsidian viewing the same `llm-wiki/` vault

When you tell Claude Code to ingest a paper, you watch the wiki pages update in real time in Obsidian. You can follow links, check the graph view, read the updated pages, and guide the agent.

## 6. Key Obsidian Shortcuts

| Action | Shortcut |
|--------|----------|
| Quick switcher (jump to any page) | `Ctrl+O` / `Cmd+O` |
| Search across all files | `Ctrl+Shift+F` / `Cmd+Shift+F` |
| Toggle graph view | `Ctrl+G` / `Cmd+G` |
| Open command palette | `Ctrl+P` / `Cmd+P` |
| Follow a `[[link]]` | `Ctrl+Click` / `Cmd+Click` |
| Go back | `Ctrl+Alt+←` / `Cmd+Opt+←` |
| Toggle edit/preview | `Ctrl+E` / `Cmd+E` |

## 7. Tips

- **Don't edit wiki pages manually** — that's the LLM's job. If you want something changed, tell Claude Code.
- **Do browse and read** — the whole point is that the wiki is human-readable. Follow links, explore the graph, spot gaps.
- **Star important pages** — click the star icon to bookmark pages you reference often.
- **Use search liberally** — `Ctrl+Shift+F` searches full text across all files. Faster than asking the LLM for simple lookups.
- **Obsidian is local-first** — your data never leaves your machine unless you set up Obsidian Sync (paid) or use git.

<p align="center">
  <img src="assets/Readme_Logo.png" alt="Sommelier" width="200">
</p>

<h1 align="center">Sommelier</h1>

<p align="center"><strong>Teach AI your taste. Apply it to everything.</strong></p>

You know quality when you see it -- but writing down *why* is the hard part. Sommelier figures it out for you. Show it examples you like and it reverse-engineers your standards into a reusable profile: a human-readable style guide that doubles as an executable classifier.

Sort 1,000 family photos in 10 minutes for $1.30. Extract coding standards from your best files. Grade 200 essays against criteria you never had to write by hand.

---

## How It Works

1. **Classify** -- AI evaluates each file against your profile's criteria and sorts it into categories
2. **Learn** -- Feed the best results back in, and Sommelier synthesizes what makes them good
3. **Apply** -- The generated profile becomes a reusable classifier for future batches

Each cycle sharpens the profile. What starts as "sort my photos" becomes a rich, nuanced document that captures exactly how you think about quality -- then enforces it automatically.

---

## What People Use It For

### Family Photos

The original use case. Ships with a built-in profile that classifies photos into Share / Storage / Review / Ignore. Run it on a year of camera roll and get the 200 worth sharing for about a dollar.

<details>
<summary><strong>Example: AI-generated taste profile from a family photo library</strong></summary>

Nobody wrote this by hand -- it was synthesized from examples:

> **Philosophy:** Photos worth sharing capture genuine moments of connection and joy between baby and loved ones. They should highlight the baby's expressions, engagement with the world, and the loving interactions they share, all presented in a visually clear and appealing manner.
>
> **Top priorities (ranked):**
> 1. Baby's face is visible, well-lit, and in focus
> 2. Baby actively engaging with people or objects
> 3. Clear and expressive facial expressions (joy, curiosity, mischief)
> 4. Positive interactions between baby and parent/caregiver
> 5. Well-framed photos that focus on the baby and minimize distractions
> 6. Photos that evoke positive emotions -- joy, love, wonder, humor
> 7. Good lighting, sharpness, and minimal blur
>
> **Must-have:** Baby's face clearly visible and in focus. Baby engaged in the activity or with another person. Well-lit and sharp. Expressive and genuine emotion conveyed.
>
> **Highly valued:** Interactions between baby and parent (especially eye contact or physical touch). Baby displaying positive emotions. Photos that tell a story or capture a special moment.
>
> **Deal-breakers:** Baby's face not visible or obscured. Blurry or out of focus. Poor lighting. Baby displaying negative expression without context. Child too small in frame.
>
> **Burst guidance:** Prioritize the clearest expressions, best focus on the baby's face, and minimal distractions. Choose the photo that best captures the moment's essence.
>
> **Specific guidance:**
> - *"Does this photo capture the baby's personality and current stage of development?"*
> - *"If both parent and baby are in the photo, ensure both have engaged and positive expressions."*
> - *"Reject photos even if technically good if the baby looks unhappy (unless comically so)."*

That profile is both a human-readable style guide *and* an executable classifier. Every future photo is evaluated against those exact criteria.

</details>

### Grading at Scale

Grade 200 student submissions against *your* standards. Create a profile from a handful of A-grade and C-grade examples. Sommelier extracts what your grading decisions had in common and applies them to the full batch -- each result includes a brief explanation of why the submission landed where it did, grounded in the profile's criteria.

### Code Standards

Point Sommelier at your `src/` directory with a code quality profile. It classifies every file into tiers (Exemplary / Solid / Needs Work) with explanations. Feed the Exemplary bucket back into profile generation and Sommelier synthesizes *your* standards -- error handling patterns, naming conventions, architectural choices you actually follow. That profile becomes both a style guide and an automated reviewer for future code.

### Any Collection

Research papers, product photos, legal documents, design assets. Anything where you "know it when you see it" but can't write the rules upfront. Sommelier closes the gap between your tacit taste and explicit, repeatable judgment.

---

## Taste Profiles

Profiles define *what* you're classifying, *how* you want it categorized, and *what criteria* to use.

### Example Profile

```json
{
  "name": "student-essays",
  "categories": [
    {"name": "Strong", "description": "Excellent work, minimal feedback needed"},
    {"name": "Developing", "description": "Shows promise, needs specific feedback"},
    {"name": "Needs Work", "description": "Significant gaps to address"}
  ],
  "top_priorities": ["Clear thesis supported by evidence", "Logical structure", "Engagement with sources"],
  "philosophy": "Reward critical thinking and clear communication over polish.",
  "thresholds": {"Strong": 0.70, "Developing": 0.40}
}
```

Profiles are stored as JSON in `profiles/`. Sommelier ships with starter profiles for common use cases -- family photos, code review, student essays, and product photography. See the full schema in `profiles/default-photos.json`.

### Ways to Create a Profile

| Method | Input | Time | Best for |
|--------|-------|------|----------|
| Quick Profile | Plain English description | ~10s | Getting started fast |
| Generate from Examples | Good/bad example folders | ~2 min | When you have sorted examples |
| **Pairwise Training** | 15-50+ side-by-side comparisons | 15-30 min | **Highest accuracy** |

- **Quick profile:** Ask Claude *"Create a profile for sorting research papers into Keep, Skim, and Skip"* and it generates one from your description. Great for a first pass.
- **From examples:** Point `sommelier_generate_profile` at a folder of good examples and a folder of bad examples. Sommelier analyzes both and synthesizes criteria.
- **Pairwise training:** Launch the Gradio trainer (`sommelier train <folder>`) for the highest-fidelity option. Compare photos side-by-side, pick keepers from burst galleries, and synthesize a profile that captures exactly how you think about quality.
- **By hand:** Write a JSON file directly in `profiles/`.

### Training and Refinement

Profiles improve over time:

1. **Pairwise training** -- Run `sommelier train <folder>` to launch a Gradio UI. Compare photos side-by-side, pick keepers from burst galleries, and synthesize a profile when you have enough labels (15+).

2. **Corrective refinement** -- After classifying a folder, correct the results you disagree with and call `sommelier_refine_profile` in Claude Desktop. Sommelier analyzes the gap between its predictions and your corrections and adjusts criteria, thresholds, and priorities. Repeat each batch to continuously sharpen the profile -- this produces the highest fidelity over time.

3. **Simple feedback** -- Submit individual corrections via `sommelier_submit_feedback` for lightweight feedback without full refinement.

---

## Getting Started

Sommelier runs as an MCP server inside Claude Desktop, or as a standalone CLI / REST API.

**Prerequisites:** Python 3.12+, at least one AI provider API key.

### Install

```bash
pip install sommelier[gemini]        # Gemini (recommended -- cheapest)
# or: pip install sommelier[openai]  # OpenAI
# or: pip install sommelier[all]     # Everything
```

<details>
<summary><strong>Install from source</strong></summary>

```bash
git clone https://github.com/bweller-lgtm/Sommelier.git
cd Sommelier
pip install -e ".[gemini]"
```

</details>

### First-run setup

```bash
sommelier init
```

Creates your config directory, prompts for API keys, and optionally wires up Claude Desktop. Config files are stored in a platform-appropriate location:
- **Windows:** `%APPDATA%\sommelier\`
- **macOS:** `~/Library/Application Support/sommelier/`
- **Linux:** `$XDG_CONFIG_HOME/sommelier/` (default `~/.config/sommelier/`)

### Claude Desktop (Recommended)

Connect Sommelier to Claude Desktop and use it conversationally. No command-line needed after setup.

If `sommelier init` configured Claude Desktop for you, just restart Claude Desktop. Otherwise, add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "sommelier",
      "args": ["serve"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

<details>
<summary><strong>Claude Desktop config locations</strong></summary>

- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `$XDG_CONFIG_HOME/Claude/claude_desktop_config.json`

</details>

<details>
<summary><strong>Running from a cloned repo (alternative)</strong></summary>

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/Sommelier",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

</details>

Restart Claude Desktop. Ask it: *"Check my Sommelier status"* to verify everything is connected. Then just talk to it:
- *"Sort the photos in my Camera Roll folder"*
- *"Create a profile for grading student essays"*
- *"Generate a taste profile from my best code examples in src/"*
- *"Refine my photo profile -- I disagreed with some of the results"*

<details>
<summary><strong>All MCP tools</strong> (13 tools)</summary>

| Tool | What it does |
|------|-------------|
| `sommelier_status` | Check setup status (API keys, providers, profiles) |
| `sommelier_list_profiles` | List all taste profiles |
| `sommelier_get_profile` | Get profile details with human-readable summary |
| `sommelier_create_profile` | Create a new profile with full control |
| `sommelier_update_profile` | Update specific fields of an existing profile |
| `sommelier_delete_profile` | Delete a profile (with confirmation) |
| `sommelier_quick_profile` | AI-generate a profile from a plain English description |
| `sommelier_generate_profile` | AI-generate a profile from example files |
| `sommelier_classify_folder` | Classify all files in a folder |
| `sommelier_classify_files` | Classify specific files by path |
| `sommelier_submit_feedback` | Submit classification corrections |
| `sommelier_view_feedback` | Review all feedback and stats |
| `sommelier_refine_profile` | Refine profile from classification corrections (AI) |

Pairwise training (side-by-side photo comparison and profile synthesis) is handled by the standalone Gradio trainer: `sommelier train <folder>`.

</details>

<details>
<summary><strong>CLI</strong></summary>

```bash
# Classify a folder (auto-detects media types and provider)
sommelier classify ~/Photos/MyFolder

# Use a specific taste profile
sommelier classify ~/Photos/MyFolder --profile default-photos

# Force a specific AI provider
sommelier classify ~/Photos/MyFolder --provider openai

# Dry run (test without moving files)
sommelier classify ~/Photos/MyFolder --dry-run

# Check setup status
sommelier status
```

Files are sorted into `MyFolder_sorted/` with subfolders matching your profile's categories.

</details>

<details>
<summary><strong>REST API</strong></summary>

```bash
python serve.py                          # Start on localhost:8000
python serve.py --host 0.0.0.0 --port 9000  # Custom host/port
python serve.py --reload                 # Development mode
```

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/profiles/` | List all profiles |
| GET | `/api/profiles/{name}` | Get profile details |
| POST | `/api/profiles/` | Create new profile |
| PUT | `/api/profiles/{name}` | Update profile |
| DELETE | `/api/profiles/{name}` | Delete profile |
| POST | `/api/classify/folder` | Classify a local folder |
| GET | `/api/classify/{job_id}` | Get job status |
| GET | `/api/classify/{job_id}/results` | Get results |
| GET | `/api/results/{job_id}/export` | Export as CSV |
| POST | `/api/training/feedback` | Submit feedback |
| GET | `/api/training/stats` | Get training stats |

</details>

---

## AI Providers

Sommelier supports three AI providers. Install only the SDK(s) you need.

| Feature | Gemini | OpenAI (GPT-4o/4.1) | Anthropic (Claude) |
|---------|--------|---------------------|-------------------|
| Images | Native | Base64 | Base64 |
| Videos | Native upload | Frame extraction | Frame extraction |
| PDFs | Native upload | Page-to-image | Native |
| Relative cost | Cheapest | Mid | Most expensive |
| Env var | `GEMINI_API_KEY` | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` |

Gemini is the recommended default -- cheapest and has native video/PDF support. The system auto-detects which key is available (gemini > openai > anthropic), or you can force a provider with `--provider` or in `config.yaml`.

---

## Cost Estimates

Gemini 3 Flash pricing (cheapest provider):

| Item | Cost |
|------|------|
| Photo | ~$0.0013 |
| Burst photo | ~$0.00043 (shared prompt) |
| Video | ~$0.011 per minute |
| Document | ~$0.002-0.01 (varies by size) |

**Example:** 1,000 photos = ~$1.30 (first run), ~$0 (cached re-run).

OpenAI and Anthropic cost more per call. See provider pricing pages for current rates.

---

## Configuration

All settings are in `config.yaml`. Key sections:

```yaml
model:
  provider: null          # null = auto-detect. Options: gemini, openai, anthropic
  name: "gemini-3-flash-preview"
  openai_model: "gpt-4.1"
  anthropic_model: "claude-sonnet-4-20250514"

classification:
  share_threshold: 0.60       # Confidence threshold for top category
  review_threshold: 0.35      # Below this → lowest category
  classify_videos: true
  parallel_video_workers: 10

profiles:
  profiles_dir: "profiles"
  active_profile: "default-photos"
  auto_detect_media_type: true

burst_detection:
  time_window_seconds: 10
  embedding_similarity_threshold: 0.92

document:
  max_file_size_mb: 50.0
  max_pages_per_document: 20
  enable_text_embeddings: true
  similarity_threshold: 0.85
```

See `config.yaml` for all options with inline documentation.

---

## Supported Formats

| Type | Extensions |
|------|-----------|
| **Images** | `.jpg` `.jpeg` `.png` `.gif` `.webp` `.heic` `.tif` `.tiff` `.bmp` |
| **Videos** | `.mp4` `.mov` `.avi` `.mkv` `.m4v` `.3gp` `.wmv` `.flv` `.webm` |
| **Documents** | `.pdf` `.docx` `.xlsx` `.pptx` `.html` `.htm` `.txt` `.md` `.csv` `.rtf` |
| **Source Code** | `.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.go` `.rs` `.rb` `.cpp` `.c` `.cs` `.swift` `.kt` `.php` `.lua` `.scala` + more |
| **Config / IaC** | `.yaml` `.yml` `.toml` `.json` `.xml` `.tf` `.hcl` `.dockerfile` |
| **Shell** | `.sh` `.bash` `.zsh` `.ps1` `.sql` |

PDFs and videos are uploaded natively to Gemini. For OpenAI and Anthropic, videos are frame-extracted and PDFs are rendered to images automatically.

---

## Output Structure

```
MyFolder_sorted/
├── Share/                       # Share-worthy photos/videos
├── Storage/                     # Keep but don't share
├── Review/                      # Needs manual review
├── Ignore/                      # Not relevant
└── Reports/                     # Classification logs + summary
```

For custom profiles, output folders match the profile's category names (e.g., `Strong/`, `Maybe/`, `Pass/`).

---

## Reference

<details>
<summary><strong>Error Handling</strong></summary>

Automatic retry with exponential backoff for API failures.

| Error Type | Retriable | Description |
|------------|-----------|-------------|
| `api_error` | Yes | General API failures |
| `timeout` | Yes | Request timeout |
| `rate_limit` | Yes | API rate limiting |
| `invalid_response` | Yes | Malformed API response |
| `safety_blocked` | No | Content blocked by safety filters |
| `load_error` | No | Failed to load file |

Reprocess failures:
```bash
python clear_failed_for_retry.py "path/to/sorted" --dry-run  # Preview
python clear_failed_for_retry.py "path/to/sorted"            # Clear and re-run
```

</details>

<details>
<summary><strong>Architecture</strong></summary>

```
sommelier/
├── core/                  # Configuration, caching, AI clients, profiles
│   ├── ai_client.py       # AIClient ABC + AIResponse
│   ├── provider_factory.py # Auto-detect provider from API keys
│   ├── media_prep.py      # Video frame extraction, PDF rendering, base64 encoding
│   └── providers/         # Gemini, OpenAI, Anthropic implementations
├── features/              # Quality scoring, burst detection, embeddings
├── classification/        # Prompt building, AI classification, confidence routing
├── pipelines/             # Photo, document, and mixed orchestration
├── api/                   # REST API (FastAPI)
├── training/              # Pairwise training & profile synthesis
└── mcp/                   # MCP server (Claude Desktop)
```

| Command | Purpose |
|---------|---------|
| `sommelier classify <folder>` | Classify files against a profile |
| `sommelier train <folder>` | Launch Gradio pairwise trainer |
| `sommelier serve` | Start MCP server for Claude Desktop |
| `sommelier init` | Interactive first-run setup |
| `sommelier status` | Show config, profiles, API key status |

</details>

<details>
<summary><strong>Development</strong></summary>

```bash
pip install -e ".[gemini,dev]"               # Install for development
pytest tests/ -v                              # Run tests
pytest tests/ --cov=sommelier --cov-report=html     # With coverage
```

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

**"No AI provider configured"** -- Create `.env` with at least one API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`).

**"No module named 'sommelier'"** -- Install with `pip install -e .` or run from the project root directory.

**Low share rate (<20%)** -- Lower `classification.share_threshold` in `config.yaml` (e.g., 0.50).

**Document extraction issues** -- Install document support: `pip install sommelier[documents]`.

</details>

---

Built with Google Gemini, OpenAI, Anthropic, OpenCLIP, sentence-transformers, FastAPI, MCP SDK, Gradio, and Claude Code.

**Version:** 3.0.0 | **Last Updated:** February 2026

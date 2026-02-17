# Sommelier

**Teach AI your taste. Apply it to everything.**

Sort 1,000 family photos in 10 minutes for $1.30. Extract coding standards from your best code. Screen 500 resumes against criteria you didn't have to write by hand.

### You don't need to know your criteria.

Point Sommelier at examples you like and it figures out what they have in common, then applies your judgment at scale. The generated profile is both a human-readable style guide and an executable classifier -- your documented standards, synthesized from examples. This is what makes it different from a one-shot prompt: your taste compounds over time.

Works with photos, videos, documents, and source code. Supports **Gemini**, **OpenAI**, and **Anthropic** -- just set an API key and go.

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

Here's a real profile Sommelier generated from a family photo library. Nobody wrote this by hand -- it was synthesized from examples:

<details>
<summary><strong>Generated taste profile: baby photos</strong></summary>

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

</details>

That profile is both a human-readable style guide *and* an executable classifier. Every future photo is evaluated against those exact criteria.

### Screening at Scale

Resumes, grant proposals, student papers, vendor submissions. Create a profile from the last 10 candidates you actually hired. Sommelier extracts what *your* hiring decisions had in common and applies your judgment to the next 500 applicants.

Classification uses confidence tiers:
- **High confidence** -- clearly strong or clearly weak candidates are sorted automatically
- **Medium confidence** -- borderline candidates go to a review pile for human attention
- **Low confidence** -- flagged for manual review rather than guessing

You set the thresholds per category. A resume screener might use `Strong: 0.70, Maybe: 0.40` -- anything above 70% confidence goes to Strong, 40-70% to Maybe, below 40% to Pass. The review pile shrinks as the profile gets sharper.

### Code Standards

Point Sommelier at your `src/` directory with a code quality profile. It classifies every file into tiers (Exemplary / Solid / Needs Work). Feed the Exemplary bucket back into profile generation and Sommelier synthesizes *your* standards -- error handling patterns, naming conventions, architectural choices you actually follow. That profile becomes both a style guide and an automated reviewer for future code.

### Any Collection

Research papers, product photos, legal documents, design assets. Anything where you "know it when you see it" but can't write the rules upfront. Sommelier closes the gap between your tacit taste and explicit, repeatable judgment.

---

## Taste Profiles

Profiles define *what* you're classifying, *how* you want it categorized, and *what criteria* to use.

### Example Profile

```json
{
  "name": "linkedin-profiles",
  "description": "Sort LinkedIn profile PDFs by hire potential",
  "media_types": ["document"],
  "categories": [
    {"name": "Strong", "description": "Excellent candidates to interview"},
    {"name": "Maybe", "description": "Worth a second look"},
    {"name": "Pass", "description": "Not a match for this role"}
  ],
  "top_priorities": [
    "Relevant experience in the target domain",
    "Track record of increasing responsibility",
    "Technical skills alignment"
  ],
  "positive_criteria": {
    "must_have": ["Relevant industry experience", "Clear career progression"],
    "highly_valued": ["Leadership experience", "Quantified achievements"]
  },
  "negative_criteria": {
    "deal_breakers": ["No relevant experience", "Frequent job hopping without growth"]
  },
  "philosophy": "Focus on demonstrated impact and growth potential over credentials.",
  "thresholds": {"Strong": 0.70, "Maybe": 0.40}
}
```

Profiles are stored as JSON in the `profiles/` directory.

### Ways to Create a Profile

| Method | Input | Time | Fidelity |
|--------|-------|------|----------|
| Quick Profile | Plain English description | ~10s | Low |
| Generate from Examples | Good/bad example folders | ~2 min | Medium |
| **Pairwise Training** | 15-50+ side-by-side comparisons | 15-30 min | **High** |
| **Corrective Refinement** | Post-classification corrections | Ongoing | **Highest** |

- **Quick profile:** Ask Claude *"Create a profile for sorting research papers into Keep, Skim, and Skip"* and it generates one from your description.
- **From examples:** Point `sommelier_generate_profile` at a folder of good examples and a folder of bad examples. Sommelier analyzes both and synthesizes criteria.
- **Pairwise training:** Start a training session on a photo folder. Sommelier presents side-by-side comparisons and burst galleries. After 15-50+ choices, it synthesizes a high-fidelity profile from your decisions. Run entirely through MCP tools in Claude Desktop.
- **Corrective refinement:** Classify a folder, correct the results you disagree with, then call `sommelier_refine_profile`. Sommelier analyzes the gap between its predictions and your corrections and adjusts the profile. Repeat each batch to continuously sharpen the profile -- this produces the highest fidelity over time.
- **By hand:** Write a JSON file directly in `profiles/`.

### Training and Feedback

The primary training workflow runs through MCP tools in Claude Desktop:

1. **Pairwise training** -- `sommelier_start_training` scans a folder, detects bursts, and begins a session. Use `sommelier_get_comparison` and `sommelier_submit_comparison` to work through side-by-side choices. For burst groups, `sommelier_submit_gallery` lets you pick keepers. When done, `sommelier_synthesize_profile` generates a profile from your decisions.

2. **Corrective refinement** -- After classifying a folder, correct any misclassified files and call `sommelier_refine_profile` with the corrections. The AI analyzes what the profile got wrong and adjusts criteria, thresholds, and priorities to match your actual preferences.

3. **Simple feedback** -- Submit individual corrections via `sommelier_submit_feedback` for lightweight feedback without full refinement.

> **Legacy scripts:** `taste_trainer_pairwise_v4.py` (Gradio UI) and `learn_from_reviews.py` still exist in the repo for standalone use.

---

## Getting Started

Sommelier runs as an MCP server inside Claude Desktop, or as a standalone CLI / REST API.

### Claude Desktop (Recommended)

Connect Sommelier to Claude Desktop and use it conversationally. No command-line needed after setup.

**Prerequisites:** Python 3.12+, at least one AI provider API key.

**Step 1.** Clone the repo and install dependencies:
```bash
git clone https://github.com/bweller-lgtm/Sommelier.git
cd Sommelier
py -3.12 -m pip install -r requirements.txt
```

**Step 2.** Create a `.env` file in the project root with at least one key:
```
GEMINI_API_KEY=your_key_here      # Recommended (cheapest, native video/PDF)
OPENAI_API_KEY=your_key_here      # GPT-4o / GPT-4.1
ANTHROPIC_API_KEY=your_key_here   # Claude
```

**Step 3.** Add to your Claude Desktop config (`claude_desktop_config.json`):
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

**Step 4.** Restart Claude Desktop. Ask it: *"Check my Sommelier status"* to verify everything is connected.

From there, just talk to it:
- *"Sort the photos in my Camera Roll folder"*
- *"Create a profile for screening resumes"*
- *"Generate a taste profile from my best code examples in src/"*

**MCP Tools:**

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
| | **Training & Refinement** |
| `sommelier_start_training` | Start a pairwise training session on a photo folder |
| `sommelier_get_comparison` | Get next photo comparison from active session |
| `sommelier_submit_comparison` | Record pairwise choice with reasoning |
| `sommelier_submit_gallery` | Record burst gallery keeper selections |
| `sommelier_training_status` | Get session progress or list all sessions |
| `sommelier_synthesize_profile` | Generate profile from training data (AI) |
| `sommelier_refine_profile` | Refine profile from classification corrections (AI) |

<details>
<summary><strong>CLI</strong></summary>

```bash
# Classify a folder (auto-detects media types and provider)
py -3.12 taste_classify.py "C:\Photos\MyFolder"

# Use a specific taste profile
py -3.12 taste_classify.py "C:\Photos\MyFolder" --profile default-photos

# Force a specific AI provider
py -3.12 taste_classify.py "C:\Photos\MyFolder" --provider openai

# Dry run (test without moving files)
py -3.12 taste_classify.py "C:\Photos\MyFolder" --dry-run
```

Files are sorted into `MyFolder_sorted/` with subfolders matching your profile's categories.

</details>

<details>
<summary><strong>REST API</strong></summary>

```bash
py -3.12 serve.py                          # Start on localhost:8000
py -3.12 serve.py --host 0.0.0.0 --port 9000  # Custom host/port
py -3.12 serve.py --reload                 # Development mode
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
| **Images** | `.jpg` `.jpeg` `.png` `.webp` `.heic` `.tif` `.tiff` `.bmp` |
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
├── ImprovementCandidates/       # Gray zone photos (if enabled)
│   └── improvement_candidates.csv
├── Improved/                    # AI-enhanced photos
└── Reports/                     # Classification logs (CSV)
```

For custom profiles, output folders match the profile's category names (e.g., `Strong/`, `Maybe/`, `Pass/`).

---

## Photo Improvement (Gray Zone)

Detects meaningful family moments with technical issues (blur, noise, bad exposure) and uses Gemini AI to enhance them.

1. **During classification** -- Gray zone photos are copied to `ImprovementCandidates/`
2. **Review** -- Approve candidates via Gradio UI or CSV editing
3. **Improve** -- `py -3.12 improve_photos.py "path/to/sorted_folder"`

Cost: ~$0.134 per image (Gemini 3 Pro) or ~$0.039 (Flash). Originals are always preserved.

---

## Error Handling

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
py -3.12 clear_failed_for_retry.py "path/to/sorted" --dry-run  # Preview
py -3.12 clear_failed_for_retry.py "path/to/sorted"            # Clear and re-run
```

---

## Architecture

```
src/
├── core/                  # Configuration, caching, AI clients, profiles
│   ├── ai_client.py       # AIClient ABC + AIResponse
│   ├── provider_factory.py # Auto-detect provider from API keys
│   ├── media_prep.py      # Video frame extraction, PDF rendering, base64 encoding
│   ├── providers/         # AI provider implementations
│   │   ├── gemini.py      # Gemini (native video/PDF)
│   │   ├── openai_provider.py   # OpenAI (GPT-4o/4.1)
│   │   └── anthropic_provider.py # Anthropic (Claude)
│   ├── config.py          # Type-safe YAML configuration
│   ├── cache.py           # Unified caching system
│   ├── models.py          # Gemini API client (extends AIClient)
│   ├── file_utils.py      # File type detection (images, videos, documents, code)
│   ├── profiles.py        # Taste profile management
│   └── logging_config.py  # Logging utilities
├── features/              # Feature extraction
│   ├── quality.py         # Photo quality scoring
│   ├── burst_detector.py  # Photo burst detection
│   ├── embeddings.py      # CLIP visual embeddings
│   └── document_features.py  # Document text/metadata extraction
├── classification/        # AI classification
│   ├── prompt_builder.py  # Dynamic prompt generation (any media + profile)
│   ├── classifier.py      # AI classification (photos, videos, documents)
│   └── routing.py         # Category routing with confidence thresholds
├── pipelines/             # Orchestration
│   ├── base.py            # Abstract pipeline interface
│   ├── photo_pipeline.py  # Photo/video pipeline
│   ├── document_pipeline.py  # Document pipeline
│   └── mixed_pipeline.py  # Auto-detect and dispatch
├── api/                   # REST API (FastAPI)
│   ├── app.py             # Application factory
│   ├── models.py          # Pydantic request/response models
│   ├── routers/           # Endpoint definitions
│   └── services/          # Business logic layer
├── training/              # Pairwise training & profile synthesis
│   ├── session.py         # Training session state & persistence
│   ├── sampler.py         # Smart comparison selection (burst-aware)
│   └── synthesizer.py     # AI-powered profile synthesis & refinement
├── mcp/                   # MCP server (Claude Desktop)
│   └── server.py          # Tool definitions
└── improvement/           # Photo improvement (gray zone)
    ├── improver.py        # Gemini image enhancement
    └── review_ui.py       # Gradio review interface
```

| Entry Point | Purpose |
|-------------|---------|
| `mcp_server.py` | MCP server for Claude Desktop |
| `taste_classify.py` | CLI classification |
| `serve.py` | REST API server |
| `improve_photos.py` | Photo improvement |
| `generate_taste_profile.py` | Standalone profile generation |

---

## Development

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Install dependencies
py -3.12 -m pip install -r requirements.txt
```

---

## Troubleshooting

**"No AI provider configured"** -- Create `.env` with at least one API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`).

**"No module named 'src'"** -- Run from the project root directory.

**Low share rate (<20%)** -- Lower `classification.share_threshold` in `config.yaml` (e.g., 0.50).

**Document extraction issues** -- Install the relevant parser: `py -3.12 -m pip install pypdf python-docx openpyxl python-pptx beautifulsoup4`.

---

## Credits

Built with Google Gemini, OpenAI, Anthropic (AI), OpenCLIP (visual embeddings), sentence-transformers (text embeddings), FastAPI (REST API), MCP SDK (Claude Desktop), Gradio (training UI), and Claude Code (development).

---

**Version:** 3.2.0 | **Last Updated:** February 2026 | **Status:** Production Ready

# Sommelier

**Teach AI your taste. Apply it to everything.**

Sort 1,000 family photos in 10 minutes for $1.30. Or extract coding standards from your best code. Or screen 500 resumes against criteria you didn't have to write by hand.

Sommelier learns what "good" looks like from your examples, turns that into a reusable taste profile, then applies your judgment at scale. Works with photos, videos, documents, and source code. Supports **Gemini**, **OpenAI**, and **Anthropic** -- just set an API key and go.

> **Don't know your criteria yet?** Point Sommelier at examples you like and it'll figure it out. The generated profile *is* your documented standards -- and it's executable.

---

## What People Use It For

**Sort family photos** -- the original use case. Ship with a built-in profile that classifies photos into Share / Storage / Review / Ignore. Run it on a year of camera roll, get the 200 worth sharing, for about a dollar.

**Extract coding standards** -- point it at your `src/` directory with a code quality profile. It classifies every file into quality tiers. Then feed the "Exemplary" bucket back in and Sommelier synthesizes *what makes those files good* into a reusable profile -- error handling patterns, naming conventions, architectural choices. That profile becomes both a style guide and an automated reviewer for future code.

**Screen applications at scale** -- resumes, grant proposals, student papers, vendor submissions. Create a profile from the last 10 candidates you actually hired. Sommelier extracts what they had in common and applies it to the next 500 applicants. Your criteria, not a generic model's.

**Curate any collection** -- research papers, product photos, legal documents, design assets. Anything where you "know it when you see it" but can't write the rules upfront. Sommelier closes the gap between tacit taste and explicit, repeatable judgment.

---

## How It Works

1. **Classify** -- AI evaluates each file against your profile's criteria and sorts it into categories
2. **Learn** -- Feed exemplary results back in, and Sommelier synthesizes what makes them good
3. **Apply** -- The generated profile becomes a reusable classifier for future runs

This is the loop that makes Sommelier different from a one-shot classifier. Your taste improves over time.

---

## Quick Start

**Requires Python 3.12+**

### 1. Install Dependencies

```bash
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file with **any one** of these:
```
GEMINI_API_KEY=your_key_here      # Recommended (cheapest, native video/PDF)
OPENAI_API_KEY=your_key_here      # GPT-4o / GPT-4.1
ANTHROPIC_API_KEY=your_key_here   # Claude
```

The system auto-detects which key is available. If you have multiple keys, it prefers Gemini > OpenAI > Anthropic (or use `--provider` to override).

### 3. Run Classification

```bash
# Classify a folder (auto-detects media types and provider)
py -3.12 taste_classify.py "C:\Photos\MyFolder"

# Use a specific taste profile
py -3.12 taste_classify.py "C:\Photos\MyFolder" --profile default-photos

# Force a specific AI provider
py -3.12 taste_classify.py "C:\Photos\MyFolder" --provider openai

# Classify documents
py -3.12 taste_classify.py "C:\Docs\Reports" --profile default-documents

# Dry run (test without moving files)
py -3.12 taste_classify.py "C:\Photos\MyFolder" --dry-run
```

That's it! Files will be sorted into `C:\Photos\MyFolder_sorted/`.

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

Gemini is recommended as the default -- it is the cheapest and has native support for video and PDF uploads. OpenAI and Anthropic work by extracting video frames and rendering PDF pages as images, so they work with all media types but at slightly lower fidelity for video.

Provider selection (in priority order):
1. `--provider` CLI flag or `config.yaml > model > provider`
2. Auto-detect from environment variables (gemini > openai > anthropic)

### Configuration

```yaml
model:
  provider: null          # null = auto-detect. Options: gemini, openai, anthropic
  name: "gemini-3-flash-preview"
  openai_model: "gpt-4.1"
  anthropic_model: "claude-sonnet-4-20250514"
  video_frame_count: 8    # Frames extracted for non-Gemini video
  pdf_render_dpi: 150     # DPI for OpenAI PDF rendering
```

---

## Complete Usage

### CLI Classification

```bash
# Basic classification (auto-detects media type and provider)
py -3.12 taste_classify.py "path/to/media"

# With a specific profile
py -3.12 taste_classify.py "path/to/media" --profile my-custom-profile

# Force a specific AI provider
py -3.12 taste_classify.py "path/to/media" --provider openai

# Disable video classification
py -3.12 taste_classify.py "path/to/photos" --no-classify-videos

# Custom output directory
py -3.12 taste_classify.py "path/to/media" --output "path/to/output"

# Use custom configuration
py -3.12 taste_classify.py "path/to/media" --config custom_config.yaml

# More parallel video workers
py -3.12 taste_classify.py "path/to/media" --parallel-videos 20
```

### API Server

```bash
# Start the API server
py -3.12 serve.py

# With custom host/port
py -3.12 serve.py --host 127.0.0.1 --port 9000

# Development mode (auto-reload)
py -3.12 serve.py --reload
```

Then access the API at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**API Endpoints:**
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

### MCP Server (Claude Desktop)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "taste-cloner": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/taste-cloner"
    }
  }
}
```

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `taste_cloner_status` | Check setup status (API keys, providers, profiles) |
| `taste_cloner_list_profiles` | List all taste profiles |
| `taste_cloner_get_profile` | Get profile details with human-readable summary |
| `taste_cloner_create_profile` | Create a new profile with full control |
| `taste_cloner_update_profile` | Update specific fields of an existing profile |
| `taste_cloner_delete_profile` | Delete a profile (with confirmation) |
| `taste_cloner_quick_profile` | AI-generate a profile from a plain English description |
| `taste_cloner_generate_profile` | AI-generate a profile from example files |
| `taste_cloner_classify_folder` | Classify all files in a folder (supports batching) |
| `taste_cloner_classify_files` | Classify specific files by path |
| `taste_cloner_submit_feedback` | Submit classification corrections |
| `taste_cloner_view_feedback` | Review all feedback and stats |

---

## Taste Profiles

Profiles define *what* you're classifying, *how* you want it categorized, and *what criteria* to use.

### Create a Custom Profile

Profiles are stored as JSON in the `profiles/` directory:

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

### Profile Management

```bash
# Via CLI (uses --profile flag)
py -3.12 taste_classify.py "path/to/docs" --profile linkedin-profiles

# Via API
curl -X POST http://localhost:8000/api/profiles/ -H "Content-Type: application/json" -d @profile.json

# Via MCP (Claude Desktop)
# Just ask: "Create a taste profile for sorting research papers"
```

### Migrating Existing Preferences

If you have an existing `taste_preferences.json`, the system uses it automatically as a fallback. To explicitly migrate it to the new profile format, use the ProfileManager:

```python
from src.core.profiles import ProfileManager
pm = ProfileManager("profiles")
pm.migrate_from_taste_preferences(Path("taste_preferences_generated.json"), "my-photos")
```

---

## Document and Code Classification

Classify documents and source code alongside photos and videos.

### Supported Formats

| Format | Extensions | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Native Gemini analysis, text extraction, metadata |
| Word | `.docx` | Text + metadata extraction |
| Excel | `.xlsx` | Sheet names, sample data extraction |
| PowerPoint | `.pptx` | Slide text, notes, metadata |
| HTML | `.html`, `.htm` | Tag stripping, text extraction |
| Plain Text | `.txt`, `.md`, `.csv`, `.rtf` | Direct text reading |
| Source Code | `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.rb`, `.cpp`, `.c`, `.cs`, `.swift`, `.kt`, `.php`, `.lua`, `.scala` + more | Read as text, classified by content |
| Config/IaC | `.yaml`, `.yml`, `.toml`, `.json`, `.xml`, `.tf`, `.hcl`, `.dockerfile` | Read as text |
| Shell | `.sh`, `.bash`, `.zsh`, `.ps1`, `.sql` | Read as text |

### How It Works

1. **Feature Extraction** -- Text, metadata, and embeddings are extracted from each document
2. **Similarity Grouping** -- Documents with similar content are grouped (like burst detection for photos)
3. **Classification** -- Gemini evaluates documents against profile criteria
4. **Routing** -- Documents are sorted into profile-defined categories

### Configuration

```yaml
document:
  max_file_size_mb: 50.0
  max_pages_per_document: 20
  text_extraction_max_chars: 50000
  enable_text_embeddings: true
  similarity_threshold: 0.85
```

### Codebase Analysis

Use Sommelier to extract best practices from a codebase:

**Step 1: Classify code by quality.** Create a profile and point it at your `src/` folder:
```
# In Claude Desktop (MCP):
# "Create a code quality profile that sorts Python files into Exemplary, Solid, and Needs-Work"
# → taste_cloner_quick_profile

# Then classify:
# "Classify all files in my project's src directory"
# → taste_cloner_classify_folder
```

**Step 2: Generate best practices.** Feed the "Exemplary" bucket back into profile generation:
```
# "Generate a profile from my exemplary code examples"
# → taste_cloner_generate_profile with good_examples_folder pointing to Exemplary/
```

The generated profile **is** your best practices document -- its `top_priorities`, `positive_criteria`, and `philosophy` fields are synthesized from what the AI found in common across your best code. And that profile is then reusable as a classifier for future code reviews.

---

## Photo Improvement (Gray Zone)

Automatically detects meaningful family moments with technical issues and uses Gemini AI to enhance them.

### What Gets Flagged?

Photos with **high contextual value** (rare family groupings, interactions, milestones) **plus** technical issues (blur, noise, exposure, etc.).

### Workflow

1. **During Classification** -- Gray zone photos are copied to `ImprovementCandidates/`
2. **Review** -- Approve candidates via Gradio UI or CSV editing
3. **Improvement** -- Process approved candidates:
   ```bash
   py -3.12 improve_photos.py "path/to/sorted_folder"
   ```

### Cost

- **Per image:** ~$0.134 (Gemini 3 Pro Image) or ~$0.039 (Gemini 3 Flash Image)
- Originals are always preserved

---

## Error Tracking and Retry

Automatic retry with exponential backoff for API failures. Failed classifications are tracked with error metadata.

| Error Type | Retriable | Description |
|------------|-----------|-------------|
| `api_error` | Yes | General API failures |
| `timeout` | Yes | Request timeout |
| `rate_limit` | Yes | API rate limiting |
| `invalid_response` | Yes | Malformed API response |
| `safety_blocked` | No | Content blocked by safety filters |
| `load_error` | No | Failed to load file |

### Reprocessing Failed Items

```bash
# See what would be cleared
py -3.12 clear_failed_for_retry.py "path/to/sorted" --dry-run

# Clear failed items and re-run
py -3.12 clear_failed_for_retry.py "path/to/sorted"
py -3.12 taste_classify.py "path/to/source"
```

---

## Configuration

All settings are in `config.yaml`. Key sections:

### Classification Thresholds
```yaml
classification:
  share_threshold: 0.60
  review_threshold: 0.35
  burst_rank_consider: 2
  classify_videos: true
  parallel_video_workers: 10
```

### Document Processing
```yaml
document:
  max_pages_per_document: 20
  enable_text_embeddings: true
  similarity_threshold: 0.85
```

### Profile Management
```yaml
profiles:
  profiles_dir: "profiles"
  active_profile: "default-photos"
  auto_detect_media_type: true
```

### Burst Detection
```yaml
burst_detection:
  time_window_seconds: 10
  embedding_similarity_threshold: 0.92
```

See `config.yaml` for all options with inline documentation.

---

## Architecture

```
src/
+-- core/                  # Configuration, caching, AI clients, profiles
|   +-- ai_client.py       # AIClient ABC + AIResponse
|   +-- provider_factory.py # Auto-detect provider from API keys
|   +-- media_prep.py      # Video frame extraction, PDF rendering, base64 encoding
|   +-- providers/         # AI provider implementations
|   |   +-- gemini.py      # Gemini (native video/PDF)
|   |   +-- openai_provider.py   # OpenAI (GPT-4o/4.1)
|   |   +-- anthropic_provider.py # Anthropic (Claude)
|   +-- config.py          # Type-safe YAML configuration
|   +-- cache.py           # Unified caching system
|   +-- models.py          # Gemini API client (extends AIClient)
|   +-- file_utils.py      # File type detection (images, videos, documents)
|   +-- profiles.py        # Taste profile management system
|   +-- logging_config.py  # Logging utilities
+-- features/              # Feature extraction
|   +-- quality.py         # Photo quality scoring
|   +-- burst_detector.py  # Photo burst detection
|   +-- embeddings.py      # CLIP visual embeddings
|   +-- document_features.py  # Document text/metadata extraction
+-- classification/        # AI classification
|   +-- prompt_builder.py  # Dynamic prompt generation (any media + profile)
|   +-- classifier.py      # AI classification (photos, videos, documents)
|   +-- routing.py         # Category routing with profile support
+-- pipelines/             # Orchestration
|   +-- base.py            # Abstract pipeline interface
|   +-- photo_pipeline.py  # Photo/video pipeline
|   +-- document_pipeline.py  # Document pipeline
|   +-- mixed_pipeline.py  # Auto-detect and dispatch
+-- api/                   # REST API (FastAPI)
|   +-- app.py             # Application factory
|   +-- models.py          # Pydantic request/response models
|   +-- routers/           # Endpoint definitions
|   +-- services/          # Business logic layer
+-- mcp/                   # MCP server (Claude Desktop)
|   +-- server.py          # Tool definitions
+-- improvement/           # Photo improvement (gray zone)
    +-- improver.py        # Gemini image enhancement
    +-- review_ui.py       # Gradio review interface
```

### Entry Points

| File | Purpose |
|------|---------|
| `taste_classify.py` | Main CLI for classification |
| `serve.py` | FastAPI server |
| `mcp_server.py` | MCP server for Claude Desktop |
| `improve_photos.py` | Photo improvement |
| `taste_trainer_pairwise_v4.py` | Interactive training UI |
| `generate_taste_profile.py` | AI profile generation |

---

## Training Your Classifier

### Option 1: Interactive Training (Recommended)
```bash
py -3.12 taste_trainer_pairwise_v4.py
```
Opens Gradio UI. Compare photos and train your preferences.

### Option 2: Generate Taste Profile from Examples
```bash
py -3.12 generate_taste_profile.py
```
Analyzes your Share/Storage folders and creates `taste_preferences_generated.json`.

### Option 3: Manual Corrections
```bash
py -3.12 learn_from_reviews.py
```
Learns from photos you manually moved to correct folders.

### Option 4: API Feedback Loop
Submit corrections via API or MCP, then regenerate profiles from accumulated feedback.

---

## Cost Estimates

Costs vary by provider. Gemini is the cheapest option. The estimates below use Gemini 3 Flash.

**Gemini 3 Flash Pricing:**
- **Input:** $0.50 per 1M tokens
- **Output:** $3.00 per 1M tokens

**Per Item (Classification):**
- **Photo:** ~$0.0013
- **Burst Photo:** ~$0.00043 (shared prompt)
- **Video:** ~$0.011 per minute
- **Document:** ~$0.002-0.01 (varies by size)

**Photo Improvement:**
- ~$0.134 per image (Gemini 3 Pro) or ~$0.039 (Flash)

**Example:** 1,000 photos = ~$1.30 (first run), ~$0 (cached re-run)

---

## Output Structure

```
MyFolder_sorted/
+-- Share/                       # Share-worthy photos/videos
+-- Storage/                     # Keep but don't share
+-- Review/                      # Needs manual review
+-- Ignore/                      # No children or inappropriate
+-- ImprovementCandidates/       # Gray zone photos
|   +-- improvement_candidates.csv
+-- Improved/                    # AI-enhanced photos
+-- Reports/                     # Classification logs (CSV)
```

For document profiles, output folders match the profile's category names (e.g., `Exemplary/`, `Acceptable/`, `Discard/`).

---

## Development

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Install New Dependencies
```bash
py -3.12 -m pip install -r requirements.txt
```

### Project Dependencies

| Category | Packages |
|----------|----------|
| AI Providers | google-generativeai (Gemini), openai (OpenAI), anthropic (Anthropic) |
| ML | torch, open-clip-torch, sentence-transformers |
| Images | Pillow, pillow-heif, opencv-python |
| Documents | pypdf, python-docx, openpyxl, python-pptx, beautifulsoup4 |
| API | fastapi, uvicorn |
| MCP | mcp |
| UI | gradio, tqdm |

---

## Troubleshooting

### "No module named 'src'"
Run from the project root directory:
```bash
cd "path/to/sommelier"
py -3.12 taste_classify.py ...
```

### "No AI provider configured"
Create `.env` file in project root with at least one key:
```
GEMINI_API_KEY=your_key_here      # Recommended
OPENAI_API_KEY=your_key_here      # Alternative
ANTHROPIC_API_KEY=your_key_here   # Alternative
```

### "Config file not found"
Ensure `config.yaml` exists in the project root.

### Low Share Rate (<20%)
```yaml
classification:
  share_threshold: 0.50  # Lower = more photos in Share
```

### Document extraction issues
Install the relevant parser:
```bash
py -3.12 -m pip install pypdf python-docx openpyxl python-pptx beautifulsoup4
```

---

## Performance

**Typical Run (1,000 photos):**
- Burst Detection: ~30 seconds
- Embedding Extraction: ~2 minutes (cached: instant)
- Gemini Classification: ~5-10 minutes
- **Total:** ~10-15 minutes (first run), ~5 minutes (cached)

**Document Classification (100 documents):**
- Feature Extraction: ~30 seconds
- Similarity Grouping: ~10 seconds
- Gemini Classification: ~2-5 minutes
- **Total:** ~3-6 minutes

---

## Credits

**Built with:**
- Google Gemini, OpenAI, Anthropic (AI classification)
- OpenCLIP (visual embeddings)
- sentence-transformers (text embeddings)
- FastAPI (REST API)
- MCP SDK (Claude Desktop integration)
- Gradio (training UI)
- Claude Code (AI-assisted development)

---

**Version:** 3.1.0
**Last Updated:** February 2026
**Status:** Production Ready

# Taste Cloner Photo Sorter

**AI-Powered Photo & Video Classification System**

Automatically sort family photos and videos using Google Gemini AI. Learns your personal taste and organizes thousands of images with 75-80% automation.

---

## 🎯 What It Does

Classifies photos/videos into 4 categories:
- **Share** - Worth sharing with family (target: 25-30%)
- **Storage** - Keep but don't share
- **Review** - Uncertain, needs manual review
- **Ignore** - No children or inappropriate

**NEW: Gray Zone Photo Improvement** - Detects meaningful family moments with technical issues (blur, noise, etc.) and uses Gemini AI to enhance them while preserving authenticity.

**NEW: Error Tracking & Retry** - Automatic retry with exponential backoff for API failures. Failed classifications are tracked with error metadata for easy reprocessing.

---

## 🚀 Quick Start (30 Seconds)

**Requires Python 3.12**

### 1. Install Dependencies
```bash
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run Classification
```bash
py -3.12 taste_classify.py "C:\Photos\MyFolder"
```

That's it! Photos will be sorted into `C:\Photos\MyFolder_sorted/`

---

## 📚 Complete Usage

### Basic Classification
```bash
# Classify photos only
py -3.12 taste_classify.py "path/to/photos"

# Classify photos and videos (videos are ON by default)
py -3.12 taste_classify.py "path/to/photos" --classify-videos

# Disable video classification
py -3.12 taste_classify.py "path/to/photos" --no-classify-videos

# Dry run (test without moving files)
py -3.12 taste_classify.py "path/to/photos" --dry-run

# Custom output directory
py -3.12 taste_classify.py "path/to/photos" --output "path/to/output"

# Use custom configuration
py -3.12 taste_classify.py "path/to/photos" --config custom_config.yaml
```

### Video Classification Options
```bash
# Use 20 parallel workers for faster video processing
py -3.12 taste_classify.py "path/to/photos" --parallel-videos 20

# Videos are classified by default (10 workers)
# Use --no-classify-videos to just copy them without AI classification
```

---

## 🎨 Photo Improvement (Gray Zone)

**NEW in v2.1** - Automatically detects "gray zone" photos: meaningful family moments with technical issues that can be improved by AI.

### What Gets Flagged?

Photos with **high contextual value** (rare family groupings, parent-child interactions, emotional moments) **plus** technical issues:
- Motion blur / focus blur
- Noise / grain
- Under/overexposure
- White balance issues
- Low resolution

### Workflow

1. **During Classification** - Gray zone photos are copied to `ImprovementCandidates/` with metadata saved to `improvement_candidates.csv`

2. **Review** - After classification, you're prompted to review candidates:
   - **Gradio UI** - Opens interactive review interface
   - **Manual** - Edit CSV directly (set `approved` column to Y/N)

3. **Improvement** - Process approved candidates:
   ```bash
   py -3.12 improve_photos.py "path/to/sorted_folder"
   ```

### Improve Photos Script

```bash
# Review in Gradio UI first, then improve
py -3.12 improve_photos.py "path/to/sorted" --review

# Dry run - see what would happen
py -3.12 improve_photos.py "path/to/sorted" --dry-run

# Improve all approved candidates
py -3.12 improve_photos.py "path/to/sorted"

# Use more parallel workers
py -3.12 improve_photos.py "path/to/sorted" --parallel 10
```

### Cost

- **Per image:** ~$0.134 (Gemini 3 Pro Image - best quality)
- **Budget option:** ~$0.039 (Gemini 3 Flash Image - faster)
- **Example:** 10 approved photos = ~$1.34 (Pro) or ~$0.39 (Flash)
- Cost estimate shown before processing with confirmation prompt

### Configuration

```yaml
photo_improvement:
  enabled: true                    # Enable gray zone detection
  contextual_value_threshold: high # "high" or "medium"
  min_issues_for_candidate: 1      # Minimum technical issues
  cost_per_image: 0.134            # For estimates
  parallel_workers: 5              # Concurrent processing
  model_name: "gemini-3-pro-image-preview"  # Best quality
  review_after_sort: true          # Prompt to review after classification
```

### Important Notes

- **Originals preserved** - Original photos are never modified
- **Both versions saved** - `Improved/` contains `{name}_original.jpg` and `{name}_improved.jpg`
- **Conservative prompts** - AI is instructed to preserve faces, expressions, and authenticity
- **Manual approval required** - No photos are improved without explicit approval

---

## 🔄 Error Tracking & Retry

**NEW in v2.2** - Robust error handling with automatic retry and tracking for failed classifications.

### How It Works

When API calls fail (timeouts, rate limits, connection errors), the system:
1. **Automatic Retry** - Retries up to 2 times with exponential backoff (2s, 4s delays)
2. **Error Tracking** - Failed photos are marked with `is_error_fallback=True` in reports
3. **Review Fallback** - Photos that fail all retries go to `Review/` with low confidence (0.3)
4. **Burst Awareness** - When reprocessing, entire bursts are cleared to maintain grouping context

### Error Types

| Error Type | Retriable | Description |
|------------|-----------|-------------|
| `api_error` | Yes | General API failures (connection, server errors) |
| `timeout` | Yes | Request timeout |
| `rate_limit` | Yes | API rate limiting |
| `invalid_response` | Yes | Malformed API response |
| `safety_blocked` | No | Content blocked by safety filters |
| `load_error` | No | Failed to load/process image |

### Reprocessing Failed Photos

After a classification run, you can identify and reprocess photos that failed due to errors:

```bash
# See what would be cleared (dry run)
py -3.12 clear_failed_for_retry.py "path/to/sorted" --dry-run

# Clear failed photos from sorted folders and cache
py -3.12 clear_failed_for_retry.py "path/to/sorted"

# Then re-run classification - only cleared photos will be processed
py -3.12 taste_classify.py "path/to/source"
```

The `clear_failed_for_retry.py` script:
- Reads the latest classification report to find error fallbacks
- Clears entire bursts when any photo in the burst failed
- Removes photos from all sorted folders (Share, Storage, Review, Ignore)
- Clears corresponding cache entries so photos will be re-classified

### Configuration

```yaml
classification:
  classification_retries: 2        # Number of retry attempts
  retry_delay_seconds: 2.0         # Initial delay (doubles on each retry)
  retry_on_errors:                 # Error types that trigger retry
    - api_error
    - timeout
    - rate_limit
    - invalid_response
```

---

## ⚙️ Configuration

All settings are in `config.yaml`. Key settings:

### Classification Thresholds
```yaml
classification:
  share_threshold: 0.60       # 60% confidence required for Share
  review_threshold: 0.35      # 35-59% → Review, <35% → Storage
  burst_rank_consider: 2      # Only rank 1-2 considered for sharing
  classify_videos: true       # Enable video classification
  parallel_video_workers: 10  # Number of parallel video workers
```

### Burst Detection
```yaml
burst_detection:
  time_window_seconds: 10     # Max time between burst photos
  embedding_similarity_threshold: 0.92  # Visual similarity threshold
  min_burst_size: 2           # Minimum photos to form burst
```

### Quality Scoring
```yaml
quality:
  sharpness_weight: 0.8       # Weight for sharpness
  brightness_weight: 0.2      # Weight for brightness
  sharpness_threshold: 200.0  # Laplacian variance threshold
```

See `config.yaml` for all configurable options.

---

## 🏗️ New Architecture (v2.0)

The project has been **completely refactored** with a clean, modular architecture:

### Core Infrastructure
- **`src/core/config.py`** - Type-safe configuration management
- **`src/core/cache.py`** - Unified caching system (embeddings, quality, API responses)
- **`src/core/models.py`** - Gemini API client with retry logic
- **`src/core/file_utils.py`** - File type detection and image utilities

### Feature Extraction
- **`src/features/quality.py`** - Quality scoring and face detection
- **`src/features/burst_detector.py`** - Burst detection (temporal + visual)
- **`src/features/embeddings.py`** - CLIP embedding extraction

### Classification
- **`src/classification/prompt_builder.py`** - Unified prompt generation
- **`src/classification/classifier.py`** - Media classification (photos/videos)
- **`src/classification/routing.py`** - Burst-aware routing with diversity checking

### Photo Improvement
- **`src/improvement/improver.py`** - Photo improvement using Gemini image generation
- **`src/improvement/review_ui.py`** - Gradio UI for reviewing candidates

### Entry Points
- **`taste_classify.py`** - Main classification script
- **`improve_photos.py`** - Photo improvement script
- **`clear_failed_for_retry.py`** - Clear failed classifications for reprocessing
- **`taste_trainer_pairwise_v4.py`** - Training UI (Gradio)
- **`generate_taste_profile.py`** - Generate AI taste profile

---

## 🎓 Training Your Classifier

### Option 1: Interactive Training (Recommended)
```bash
py -3.12 taste_trainer_pairwise_v4.py
```

Opens Gradio UI in browser. Compare photos and train your preferences.

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

**Note:** The included `taste_preferences.json` and `taste_preferences_generated.json` files serve as templates. Customize them for your own preferences using any of the methods above.

---

## 📊 How It Works

### 1. Burst Detection
Groups similar photos taken in quick succession (within 10 seconds).

### 2. Classification
Each photo/burst is analyzed by Gemini AI based on your taste preferences.

### 3. Routing
- **Singletons:** Route based on confidence threshold
- **Bursts:** Only top 1-2 photos considered for Share, rest go to Storage
- **Diversity:** Prevents similar photos from both being shared

### 4. Cost Optimization
- **Caching:** Saves ~90% on re-runs
- **Batch Processing:** Efficient API usage
- **Smart Chunking:** Handles large bursts gracefully

---

## 💰 Cost Estimates

**Gemini 3 Flash Pricing (Dec 2025):**
- **Input:** $0.50 per 1M tokens
- **Output:** $3.00 per 1M tokens

**Per Item (Classification):**
- **Photo:** ~$0.0013 per photo
- **Burst Photo:** ~$0.00043 per photo (shared prompt cost)
- **Video:** ~$0.011 per minute

**Photo Improvement (Gemini Image Generation):**
- **Per improved photo:** ~$0.134 (Gemini 3 Pro) or ~$0.039 (Gemini 3 Flash)
- **Example:** 10 improved photos ≈ $1.34 (Pro) or $0.39 (Flash)

**Example:** 1,000 photos ≈ $1.30 classification (first run), ~$0 (cached re-run)

---

## 📁 Output Structure

```
MyFolder_sorted/
├── Share/                    # 📸 Share-worthy photos/videos (25-30% target)
├── Storage/                  # 💾 Keep but don't share
├── Review/                   # 🤔 Uncertain, needs manual review
├── Ignore/                   # 🚫 No children or inappropriate
├── Videos/                   # 🎥 Videos (if classification disabled)
├── ImprovementCandidates/    # 🎨 Gray zone photos (high value + technical issues)
│   └── improvement_candidates.csv  # Metadata & approval status
├── Improved/                 # ✨ AI-enhanced photos (after processing)
│   └── improvement_results.csv     # Processing results
└── Reports/                  # 📊 Classification logs (CSV)
```

---

## 🔧 Development

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_classification.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Current Status:** 130+ passing tests ✅

### Project Structure
```
src/
├── core/              # Configuration, caching, Gemini client
├── features/          # Quality, bursts, embeddings
├── classification/    # Prompts, classifier, routing
└── improvement/       # Photo improvement (gray zone)

tests/                 # Unit + integration tests
config.yaml            # All configuration in one place
taste_classify.py      # Main classification entry point
improve_photos.py      # Photo improvement entry point
```

---

## 📖 Documentation

- **`VIDEO_CLASSIFICATION_GUIDE.md`** - Video-specific documentation
- **`examples/`** - Sample outputs and taste profiles
- **`config.yaml`** - All configuration options with inline documentation

---

## 🎯 Key Features

✅ **Unified Infrastructure** - Clean, modular codebase
✅ **Consistent Handling** - Same logic for singletons, bursts, videos
✅ **Burst-Aware Routing** - Only top photos from bursts shared
✅ **Diversity Checking** - Prevents duplicate-looking shares
✅ **Video Classification** - ON by default, 10 parallel workers
✅ **Cost Optimization** - Smart caching saves ~90% on re-runs
✅ **Type-Safe Config** - YAML configuration with validation
✅ **Comprehensive Tests** - Unit and integration tests ensure quality
✅ **Taste Profile Integration** - Uses both manual and AI-generated preferences
✅ **Gray Zone Detection** - Identifies valuable photos with technical issues
✅ **AI Photo Improvement** - Enhance gray zone photos with Gemini image generation
✅ **Gradio Review UI** - Interactive interface for approving improvement candidates
✅ **Error Tracking & Retry** - Automatic retry with exponential backoff for API failures
✅ **Failed Reprocessing** - Easy identification and reprocessing of error fallbacks

---

## 🐛 Troubleshooting

### "No module named 'src'"
```bash
# Run from project root directory
cd "path/to/Taste Cloner Photo Sorter"
py -3.12 taste_classify.py ...
```

### "GEMINI_API_KEY not set"
Create `.env` file in project root:
```
GEMINI_API_KEY=your_key_here
```

### "Config file not found"
Make sure `config.yaml` exists in project root.

### Low Share Rate (<20%)
Decrease `share_threshold` in `config.yaml`:
```yaml
classification:
  share_threshold: 0.50  # Lower = more photos in Share
```

### High Share Rate (>35%)
Increase `share_threshold`:
```yaml
classification:
  share_threshold: 0.70  # Higher = fewer photos in Share
```

---

## 📈 Performance

**Typical Run (1,000 photos):**
- **Burst Detection:** ~30 seconds
- **Quality Scoring:** ~1 minute
- **Embedding Extraction:** ~2 minutes (cached: instant)
- **Gemini Classification:** ~5-10 minutes
- **Total:** ~10-15 minutes (first run), ~5 minutes (cached)

**With 10 Videos:**
- Add ~2-5 minutes for video classification

---

## 🏆 Results

**Automation Rate:** 75-80% of photos correctly classified
**Share Rate:** Typically 25-30% of photos (configurable)
**Accuracy:** Based on your training examples and taste profile

---

## 🤝 Contributing

This is a personal project, but suggestions welcome!

1. Run tests: `pytest tests/`
2. Check code: All 65 tests should pass
3. Update docs if needed

---

## 📝 License

Personal project - Use at your own risk

---

## 🙏 Credits

**Built with:**
- Google Gemini 3 Flash (AI classification)
- OpenCLIP (visual embeddings)
- Gradio (training UI)
- Claude Code (AI-assisted development was transformative!)

**Key Insight:** For subjective preference tasks, skip traditional ML pipelines and use LLMs directly.

---

## 📞 Support

**Issues?** Check:
1. Is `GEMINI_API_KEY` set in `.env`?
2. Is `config.yaml` present?
3. Are you running from project root?
4. Do you have all dependencies? (`py -3.12 -m pip install -r requirements.txt`)

For video-specific issues, see `VIDEO_CLASSIFICATION_GUIDE.md`.

---

**Version:** 2.2.0 (Error Tracking & Retry)
**Last Updated:** December 25, 2025
**Status:** Production Ready ✅

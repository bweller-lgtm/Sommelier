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

---

## 🚀 Quick Start (30 Seconds)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run Classification
```bash
python taste_classify.py "C:\Photos\MyFolder"
```

That's it! Photos will be sorted into `C:\Photos\MyFolder_sorted/`

---

## 📚 Complete Usage

### Basic Classification
```bash
# Classify photos only
python taste_classify.py "path/to/photos"

# Classify photos and videos (videos are ON by default)
python taste_classify.py "path/to/photos" --classify-videos

# Disable video classification
python taste_classify.py "path/to/photos" --no-classify-videos

# Dry run (test without moving files)
python taste_classify.py "path/to/photos" --dry-run

# Custom output directory
python taste_classify.py "path/to/photos" --output "path/to/output"

# Use custom configuration
python taste_classify.py "path/to/photos" --config custom_config.yaml
```

### Video Classification Options
```bash
# Use 20 parallel workers for faster video processing
python taste_classify.py "path/to/photos" --parallel-videos 20

# Videos are classified by default (10 workers)
# Use --no-classify-videos to just copy them without AI classification
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

### Entry Points
- **`taste_classify.py`** - Main classification script
- **`taste_trainer_pairwise_v4.py`** - Training UI (Gradio)
- **`generate_taste_profile.py`** - Generate AI taste profile

---

## 🎓 Training Your Classifier

### Option 1: Interactive Training (Recommended)
```bash
python taste_trainer_pairwise_v4.py
```

Opens Gradio UI in browser. Compare photos and train your preferences.

### Option 2: Generate Taste Profile from Examples
```bash
python generate_taste_profile.py
```

Analyzes your Share/Storage folders and creates `taste_preferences_generated.json`.

### Option 3: Manual Corrections
```bash
python learn_from_reviews.py
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

**Gemini 2.5 Flash Pricing (2025):**
- **Input:** $0.30 per 1M tokens
- **Output:** $2.50 per 1M tokens

**Per Item:**
- **Photo:** ~$0.0008 per photo
- **Burst Photo:** ~$0.00026 per photo (shared prompt cost)
- **Video:** ~$0.0067 per minute

**Example:** 1,000 photos ≈ $0.80 (first run), $0.08 (cached re-run)

---

## 📁 Output Structure

```
MyFolder_sorted/
├── Share/          # 📸 Share-worthy photos/videos (25-30% target)
├── Storage/        # 💾 Keep but don't share
├── Review/         # 🤔 Uncertain, needs manual review
├── Ignore/         # 🚫 No children or inappropriate
├── Videos/         # 🎥 Videos (if classification disabled)
└── Reports/        # 📊 Classification logs (CSV)
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

**Current Status:** 65 passing tests ✅

### Project Structure
```
src/
├── core/              # Configuration, caching, Gemini client
├── features/          # Quality, bursts, embeddings
└── classification/    # Prompts, classifier, routing

tests/                 # 65 passing unit + integration tests
config.yaml            # All configuration in one place
taste_classify.py      # Main entry point
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
✅ **Comprehensive Tests** - 65 tests ensure quality
✅ **Taste Profile Integration** - Uses both manual and AI-generated preferences

---

## 🐛 Troubleshooting

### "No module named 'src'"
```bash
# Run from project root directory
cd "path/to/Taste Cloner Photo Sorter"
python taste_classify.py ...
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
- Google Gemini 2.5 Flash (AI classification)
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
4. Do you have all dependencies? (`pip install -r requirements.txt`)

For video-specific issues, see `VIDEO_CLASSIFICATION_GUIDE.md`.

---

**Version:** 2.0.0 (Refactored)
**Last Updated:** November 11, 2025
**Status:** Production Ready ✅

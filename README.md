# LLM Taste Cloner - Photo Classification System

A machine learning-powered photo classification system that learns your photo preferences and automatically sorts your photos into Share, Storage, Review, and Ignore categories.

## Quick Start (30 Seconds)

### 1. Setup (First Time Only)
```powershell
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
# Add this line: GOOGLE_API_KEY=your_api_key_here
```

### 2. Classify Photos
**PowerShell:**
```powershell
cd "E:\OneDrive\Evie and Elle\LLM Taste Cloner"
python taste_classify_gemini_v4.py "path\to\your\photos"
```

**Or use the batch file:**
```cmd
classify_folder.bat "path\to\your\photos"
```

### 3. Important: For Video Classification
```powershell
# CRITICAL: Use these settings for videos
python taste_classify_gemini_v4.py "path\to\your\photos" --classify-videos --parallel-videos 10
```

**Without these flags:**
- Videos will NOT be classified (just copied to Videos/ folder)
- Processing will be slower (default is 10 workers, don't change this)

## Output Structure

Your photos will be sorted into:
```
<input_folder>_sorted/
├── Share/          # Photos worth sharing with family
├── Storage/        # Keep but not share-worthy
├── Review/         # Uncertain, needs manual review
├── Ignore/         # No children or inappropriate content
└── Videos/         # Videos (classified if --classify-videos used)
```

## Main File: taste_classify_gemini_v4.py

This is your primary workflow file for classifying photos.

### Command-Line Options

```
python taste_classify_gemini_v4.py <input_folder> [options]

Required:
  input_folder          Path to folder containing photos to classify

Optional:
  --dry-run            Test run without moving files
  --classify-videos    Classify videos using AI (costs more tokens)
                       Default: just copy videos to Videos/ folder
  --parallel-videos N  Number of parallel workers for video classification
                       Default: 10 (RECOMMENDED - do not change)
                       Use 1 for sequential processing
```

### Examples

**Basic photo classification:**
```powershell
python taste_classify_gemini_v4.py "E:\Photos\2024\January"
```

**With video classification (RECOMMENDED SETTINGS):**
```powershell
python taste_classify_gemini_v4.py "E:\Photos\2024\January" --classify-videos --parallel-videos 10
```

**Dry run (test without moving files):**
```powershell
python taste_classify_gemini_v4.py "E:\Photos\2024\January" --dry-run
```

## Working with Claude Code

### Opening Claude Code
1. Open PowerShell
2. Navigate to project: `cd "E:\OneDrive\Evie and Elle\LLM Taste Cloner"`
3. Launch Claude Code: `claude-code` (or your preferred method)

### Common Claude Code Tasks

**"Help me classify photos in [folder]"**
```
Claude will help you run:
python taste_classify_gemini_v4.py "[folder]" --classify-videos --parallel-videos 10
```

**"Review the classification results"**
```
Claude can help analyze the output statistics and suggest improvements
```

**"Update my taste preferences"**
```
Claude can help modify taste_preferences.json or retrain the model
```

## How It Works

1. **Burst Detection**: Groups similar photos taken in quick succession
2. **AI Classification**: Uses Google Gemini to evaluate each photo/video
3. **Preference Learning**: Learns from your previous choices in `taste_preferences.json`
4. **Routing**: Applies quality thresholds and burst ranking to sort photos

### Classification Categories

- **Share** (≥60% confidence): Worth sharing with family
  - Best moments, genuine expressions, special interactions
  - Clear, well-composed, emotionally engaging

- **Storage** (35-59% confidence OR lower-ranked in burst): Keep but not share-worthy
  - Decent quality but not exceptional
  - Duplicates or near-duplicates
  - Lower-ranked photos in bursts

- **Review** (uncertain cases): Manual review needed
  - Mid-tier burst photos that might be share-worthy
  - Uncertain classifications
  - Photos blocked by safety filters

- **Ignore**: Not relevant
  - No children/babies visible
  - Inappropriate content
  - Screenshots, documents, food-only photos

- **Videos**: Separate folder
  - If `--classify-videos` used: Classified into Share/Storage/Review/Ignore
  - Otherwise: All videos copied here without classification

## Key Configuration Files

- **taste_classify_gemini_v4.py** - Main classification script (use this!)
- **taste_preferences.json** - Your learned photo preferences
- **burst_detector.py** - Burst detection algorithm
- **.env** - API keys (GOOGLE_API_KEY)
- **requirements.txt** - Python dependencies

## Additional Files

- **generate_taste_profile.py** - Generate taste profile from examples
- **learn_from_reviews.py** - Learn from your manual reviews
- **taste_trainer_pairwise_v4.py** - Train the model with pairwise comparisons
- **quick_start_v4.py** - Quick start utility script

## Project Structure

```
.
├── taste_classify_gemini_v4.py  # Main classification script
├── burst_detector.py            # Burst detection logic
├── burst_features.py            # Burst feature computation
├── taste_sort_win.py            # Core sorting utilities
├── classify_folder.bat          # Windows batch helper
├── requirements.txt             # Dependencies
├── .env                         # API keys
├── taste_preferences.json       # Your preferences
│
├── old/                         # Old/archived files
│   ├── tests/                   # Test and debug scripts
│   ├── utilities/               # Utility scripts
│   └── docs/                    # Old documentation
│
├── DATA_SCIENCE_LEARNINGS.md    # Technical insights
└── VIDEO_CLASSIFICATION_GUIDE.md # Video classification guide
```

## Cache Files

The system caches API responses to save costs:
- Location: `.taste_cache/`
- Automatically created
- Safe to delete if you want fresh classifications

## Cost Estimation

- Photos: ~$0.00125 per photo (Gemini 2.5 Flash)
- Videos: ~$0.05 per video (larger input)
- Burst detection uses CLIP embeddings (free, local computation)

## Troubleshooting

### "No API key found"
- Ensure `.env` file exists with: `GOOGLE_API_KEY=your_key_here`

### "Videos not being classified"
- Add `--classify-videos` flag

### "Processing too slow"
- For videos, ensure `--parallel-videos 10` is set
- Default of 10 workers is optimal

### "Too many photos going to Share"
- Check `SHARE_THRESHOLD` in taste_classify_gemini_v4.py:236
- Current: 0.60 (60% confidence required)

### "Not enough photos going to Share"
- Lower `SHARE_THRESHOLD` or adjust taste preferences
- Review `taste_preferences.json`

## Next Steps After Classification

1. **Review the Review folder**: Manually sort uncertain photos
2. **Learn from reviews**: Run `learn_from_reviews.py` to update preferences
3. **Share from Share folder**: Upload to family album, social media, etc.
4. **Archive Storage folder**: Move to long-term backup

## Important Notes

- **CRITICAL**: Always use `--parallel-videos 10` for video classification
- **CRITICAL**: Use `--classify-videos` if you want videos classified (not just copied)
- Photos are not deleted, only organized into folders
- Use `--dry-run` to test before committing to file operations
- The system learns from your choices over time

## Resources

- **DATA_SCIENCE_LEARNINGS.md**: Technical insights about the classification system
- **VIDEO_CLASSIFICATION_GUIDE.md**: Detailed video classification documentation
- **old/docs/**: Additional archived documentation

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review this README for common issues
3. Check `DATA_SCIENCE_LEARNINGS.md` for technical details
4. Ask Claude Code for help with debugging

---

**Last Updated**: Phase I Complete (November 2024)
**Next Review**: One month from pause (December 2024)

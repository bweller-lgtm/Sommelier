# Phase I Status Report - LLM Taste Cloner

**Project Pause Date**: November 3, 2024
**Expected Resume**: December 3, 2024 (1 month)
**Status**: Phase I Complete - Ready for Pause

---

## Project Overview

The LLM Taste Cloner is a machine learning-powered photo classification system that learns your photo preferences and automatically sorts photos into Share, Storage, Review, and Ignore categories using Google Gemini AI.

---

## Phase I Accomplishments

### Core Features Implemented
- ✅ Photo classification using Gemini 2.5 Flash AI
- ✅ Video classification support with parallel processing
- ✅ Burst detection and grouped evaluation
- ✅ Diversity checking to avoid duplicate sharing
- ✅ Taste preference learning system
- ✅ Four-tier classification (Share/Storage/Review/Ignore)
- ✅ Windows batch file helpers
- ✅ Cost-saving API response cache
- ✅ Comprehensive error handling and safety filters

### Key Technical Achievements
- **Burst Detection**: Temporal and visual similarity grouping using CLIP embeddings
- **Grouped Evaluation**: Evaluates entire bursts together for better context
- **Rank-Based Routing**: Top 1-2 photos from bursts considered for sharing
- **Absolute Quality Thresholds**: 60% confidence required for Share classification
- **Diversity Checking**: Prevents multiple similar photos from same burst going to Share
- **Parallel Video Processing**: 10 workers for efficient video classification
- **Safety Filters**: Handles Gemini safety blocks gracefully

### Classification Thresholds (Current Settings)
```python
SHARE_THRESHOLD = 0.60           # 60% confidence required for Share
REVIEW_THRESHOLD = 0.35          # 35-59% → Review
BURST_RANK_CONSIDER_THRESHOLD = 2  # Only rank 1-2 for potential sharing
BURST_RANK_REVIEW_THRESHOLD = 4    # Rank 3-4 → Review, 5+ → Storage
```

### Cost Structure
- **Photos**: ~$0.00125 per photo (Gemini 2.5 Flash)
- **Videos**: ~$0.05 per video
- **Cache Savings**: 10 cached classification sets (saves on re-runs)

---

## Current File Structure

### Core Workflow Files (Root Directory)
```
taste_classify_gemini_v4.py      # Main classification script [PRIMARY FILE]
burst_detector.py                # Burst detection logic
burst_features.py                # Burst feature computation
taste_sort_win.py                # Core sorting utilities
classify_folder.bat              # Windows batch helper
requirements.txt                 # Python dependencies
.env                             # API keys (GOOGLE_API_KEY)
taste_preferences.json           # Manual taste preferences
taste_preferences_generated.json # AI-generated taste profile
```

### Supporting Files
```
generate_taste_profile.py        # Generate taste profile from examples
learn_from_reviews.py            # Learn from manual reviews
taste_trainer_pairwise_v4.py     # Train with pairwise comparisons
quick_start_v4.py                # Quick start utility
```

### Documentation
```
README.md                        # Comprehensive usage guide [START HERE]
DATA_SCIENCE_LEARNINGS.md        # Technical insights and findings
VIDEO_CLASSIFICATION_GUIDE.md    # Video classification details
PHASE_I_STATUS.md                # This file
```

### Archived Files (old/)
```
old/tests/                       # Test scripts and debugging utilities
old/utilities/                   # One-off utility scripts
old/docs/                        # Old technical documentation
old/run_classification.bat       # Deprecated batch file
old/[various old versions]       # Previous implementation versions
```

### Cache Structure
```
.taste_cache/
├── emb/                         # CLIP embeddings for burst detection
├── labeling_pairwise/           # Cached Gemini classification results
├── faces/                       # Face detection cache
├── features/                    # Feature extraction cache
├── phash/                       # Perceptual hash cache
└── quality/                     # Quality assessment cache

Current cache: ~10 classification sets cached
```

---

## Taste Preferences Configuration

### Two Preference Files

**taste_preferences.json** (Manual, Explicit)
- Hand-crafted preference definitions
- Focus on parent-child interaction quality
- Emphasizes parent expressions and engagement
- Philosophy: "Interaction quality matters most"

**taste_preferences_generated.json** (AI-Generated)
- Generated from training examples
- More detailed technical criteria
- Structured guidance for the AI
- Philosophy: "Genuine moments of connection and joy"

### Key Preference Themes (Common to Both)
1. **Baby's face must be clearly visible and in focus**
2. **Parent expressions matter as much as baby's**
3. **Interaction quality > Solo shots**
4. **Genuine emotional moments > Posed photos**
5. **Technical quality is necessary but not sufficient**
6. **Be selective in bursts** (only 1-2 best photos)

### Target Share Rate
- Calibrated for ~25-30% of photos to be Share-worthy
- Current thresholds designed to hit this target

---

## Git Repository Status

### Recent Commits
```
c2d21540 Fix Windows console encoding for emoji support
ae051726 Add .env file support for API key management
32c48e29 Remove hardcoded API key for security
ec1b39ed Remove cache files from version control
ced36f43 Initial commit: LLM photo taste cloner project
```

### Files Ready to Commit
- Project reorganization (old/ folder structure)
- README.md (comprehensive guide)
- DATA_SCIENCE_LEARNINGS.md
- VIDEO_CLASSIFICATION_GUIDE.md
- classify_folder.bat
- PHASE_I_STATUS.md
- Core Python files with latest improvements

### Files in .gitignore (Intentionally Excluded)
- .taste_cache/ (local API cache)
- .env (API keys - NEVER commit)
- old/ (archived files)
- __pycache__/
- taste_preferences*.json (may contain personal data)

---

## Known Issues & Edge Cases

### Working Well
- Photo classification accuracy is good
- Burst detection reliably groups similar photos
- Diversity checking prevents duplicate sharing
- Cost optimization through caching works well
- Safety filters handle inappropriate content

### Areas for Future Improvement
- **Video classification**: More expensive, may need refinement
- **Review folder**: Can accumulate uncertain cases (manual review needed)
- **Burst thresholds**: May need tuning based on your specific photo collection
- **Share rate**: Monitor if 25-30% target is being hit

### Known Limitations
- Requires Google API key (costs money per classification)
- Video processing is slower than photos (even with 10 workers)
- Safety filters occasionally block appropriate family photos
- Cache can become stale if preferences change significantly

---

## Critical Usage Notes

### MUST Remember When Resuming

**For Video Classification:**
```bash
python taste_classify_gemini_v4.py "folder" --classify-videos --parallel-videos 10
```
- **--classify-videos**: Required or videos just get copied to Videos/ folder
- **--parallel-videos 10**: Critical for performance (default is 10, don't change)

**Quick Classification (Photos Only):**
```bash
python taste_classify_gemini_v4.py "folder"
```
or
```cmd
classify_folder.bat "folder"
```

### Output Folders Created
```
<input_folder>_sorted/
├── Share/          # Worth sharing with family (~25-30% of photos)
├── Storage/        # Keep but not share-worthy
├── Review/         # Uncertain, needs manual review
├── Ignore/         # No children or inappropriate
└── Videos/         # Videos (classified if --classify-videos used)
```

---

## System Requirements

### Python Dependencies
```
google-generativeai
gradio
python-dotenv
```
Plus standard libraries: PIL, torch, open_clip, numpy, pandas, tqdm

### Environment Setup
```
.env file must contain:
GOOGLE_API_KEY=your_api_key_here
```

### Platform
- Developed on: Windows (win32)
- Batch files: Windows-specific (classify_folder.bat)
- Python scripts: Cross-platform compatible

---

## Resuming After One Month

### Quick Resume Checklist

1. **Navigate to project**
   ```powershell
   cd "E:\OneDrive\Evie and Elle\LLM Taste Cloner"
   ```

2. **Read README.md** (refresh your memory on usage)

3. **Verify .env file** (ensure API key is still valid)

4. **Test with dry run**
   ```powershell
   python taste_classify_gemini_v4.py "test_folder" --dry-run
   ```

5. **Run actual classification**
   ```powershell
   python taste_classify_gemini_v4.py "folder" --classify-videos --parallel-videos 10
   ```

6. **Review results**
   - Check Share/ folder for quality
   - Review Review/ folder manually
   - Verify Storage/ folder isn't missing good photos

7. **Adjust if needed**
   - Modify thresholds in taste_classify_gemini_v4.py:236-242
   - Update taste_preferences.json if preferences changed
   - Re-run with adjusted settings

### Files to Check First
1. **README.md** - Complete usage guide
2. **PHASE_I_STATUS.md** - This file
3. **taste_preferences.json** - Your current preferences
4. **DATA_SCIENCE_LEARNINGS.md** - Technical insights

---

## Performance Metrics

### Expected Processing Speed
- **Photos**: ~2-3 seconds per photo (with caching)
- **Videos**: ~10-15 seconds per video (with 10 parallel workers)
- **Burst detection**: Nearly instant (local CLIP embeddings)
- **Diversity checking**: ~1-2 seconds per pair

### Cache Hit Benefits
- First run: Full API costs
- Re-runs: ~90% cache hits (10x cost savings)
- Cache location: .taste_cache/

---

## Next Phase Ideas (Phase II)

When you resume, consider:
1. **Fine-tune thresholds** based on real usage
2. **Review and learn** from Review/ folder classifications
3. **Train with more examples** using taste_trainer_pairwise_v4.py
4. **Optimize video classification** (reduce costs if needed)
5. **Batch process old photo libraries**
6. **Export statistics** on classification patterns
7. **A/B test threshold adjustments**

---

## Emergency Reference

### If Something Goes Wrong

**"Can't find API key"**
- Check .env file exists
- Verify: `GOOGLE_API_KEY=your_key_here`

**"Videos not classified"**
- Add: `--classify-videos` flag

**"Too slow"**
- Check: `--parallel-videos 10` is set for videos
- Photos are always sequential (by design)

**"Too many/few Share photos"**
- Adjust SHARE_THRESHOLD in taste_classify_gemini_v4.py:240
- Current: 0.60 (60%)
- Lower = more Share, Higher = fewer Share

**"Files not moving"**
- Check if `--dry-run` flag is accidentally set
- Verify destination folders exist (auto-created normally)

### Support Resources
- README.md - Usage guide
- DATA_SCIENCE_LEARNINGS.md - Technical details
- VIDEO_CLASSIFICATION_GUIDE.md - Video-specific info
- Ask Claude Code for help debugging

---

## Final Notes

This project has reached a stable, usable state. The core classification system works well, and the codebase is organized for easy resumption. The README.md contains everything needed to get started quickly.

**Key Success Factors:**
- The AI learns your preferences over time
- Burst detection prevents sharing too many duplicates
- Cost optimization through caching works effectively
- Four-tier classification gives you control

**Remember:** This tool is meant to assist, not replace, your judgment. Always review the Share/ folder before actually sharing photos, and use the Review/ folder to train the system on edge cases.

**Good luck when you return!** 🚀

---

**Status**: ✅ Ready for One-Month Pause
**Next Action**: Read README.md when resuming
**Primary File**: taste_classify_gemini_v4.py

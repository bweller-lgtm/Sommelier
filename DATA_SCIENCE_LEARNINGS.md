# Photo Classification Project: From Classical ML to LLM Taste Cloning
## Learnings for Data Science Teams

**Author:** [Your Name]
**Date:** November 2024
**Context:** Personal photo curation using AI assistance (Claude Code)

---

## Executive Summary

This document shares learnings from a 2-week journey attempting to automate personal photo curation using three fundamentally different approaches. The project evolved from a classical ML classifier (AUC 0.624) through LLM-augmented features (AUC 0.62) to a direct LLM taste cloning system (achieving 75-80% automation).

**Key Findings:**
- Classical ML failed because sorting criteria were subjective, not visual
- LLM-generated features as classifier inputs didn't help
- Direct LLM classification with few-shot learning succeeded dramatically
- Claude Code (AI-assisted development) was transformative for rapid iteration
- LLM APIs have surprising gotchas that can derail production systems

**Bottom Line:** For subjective preference tasks, skip the ML pipeline entirely and go straight to LLMs.

---

## 1. Project Context

### The Problem
Sort thousands of family photos into:
- **Share** - Worth sharing with family/friends
- **Storage** - Keep but don't share
- **Review** - Manual decision needed
- **Ignore** - No children detected

### Success Criteria
- **Accuracy**: 75%+ photos correctly routed
- **Automation**: Reduce manual sorting time by 70%+
- **Personalization**: Learn user's subjective taste, not generic "quality"

### Dataset Characteristics
- ~5000 photos (family collection over 3 years)
- Training data: ~3000 manually labeled photos
- Challenge: Visually similar photos often had different labels
- Key insight: Sorting was based on subjective "special moment" feeling, not visual features

---

## 2. Algorithm Evolution: Three Approaches

### Approach 1: CLIP Embeddings + Classical Classifier ❌

**Timeline:** October 21-22 (2 days)
**Code:** `Classical ML Approach/` (9 versions)

#### Architecture
```
Photo → CLIP Embeddings (ViT-B-32) → Feature Engineering → Logistic Regression → Decision
                                      ↓
                                   Quality Scores
                                   Face Detection
                                   Aspect Ratio
                                   Blur Detection
                                   Time of Day
```

#### Why It Seemed Promising
- CLIP embeddings capture visual similarity
- Proven architecture for image classification
- Fast inference (< 100ms per photo)
- Interpretable features

#### What Went Wrong
**Results:**
- AUC: 0.624 (barely better than random 0.5)
- Probability distribution compressed (all predictions 0.3-0.5)
- 0 photos classified as "Share" (model maximally uncertain)

**Root Cause Analysis:**
```python
# The smoking gun: 99% of visually similar clusters had mixed labels
# Example from logs:
Cluster 47: [
    "IMG_2034.jpg" → Share   (baby smiling at camera)
    "IMG_2035.jpg" → Storage (same baby, same angle, 2 seconds later)
    "IMG_2036.jpg" → Share   (same baby, same angle, 4 seconds later)
]
```

**Key Learning:**
> **The sorting criteria was NOT based on visual features.** It was based on subjective factors: emotional connection, expression quality (subtle), "special moment" feeling, social appropriateness. No amount of CLIP embeddings can capture "this photo makes me happy."

#### Technical Details
- Model: Logistic Regression with class balancing
- Features: 512-dim CLIP embeddings + 15 engineered features
- Training time: ~5 minutes
- Inference: 50ms per photo
- Cost: Free (local execution)

#### Lessons Learned
1. **Label consistency is critical** - Mixed labels on similar images = model confusion
2. **Visual similarity ≠ subjective preference** - Human sorting often ignores visual features
3. **Feature engineering won't fix wrong paradigm** - Adding more visual features didn't help because the problem wasn't visual

---

### Approach 2: CLIP + LLM-Generated Features + Classifier ❌

**Timeline:** October 27 (1 day)
**Code:** `LLM Test/` (4 validation versions)

#### Architecture
```
Photo → CLIP Embeddings → Combined Features → Logistic Regression → Decision
     ↓                           ↑
  Gemini Vision API              |
     ↓                           |
  LLM Features:                  |
  - Expression quality (0-1)     |
  - Composition score (0-1)      |
  - Emotional appeal (0-1)       |
  - Shareability (0-1)           |
  - Has special moment (bool)    +
```

#### Hypothesis
"LLM can understand semantic concepts like 'cute expression' or 'special moment' that CLIP misses. Use these as features for classifier."

#### Implementation
```python
# Gemini prompt for feature extraction
prompt = """
Analyze this photo and rate:
1. Expression quality (0-1): How engaging/cute is the subject's expression?
2. Composition score (0-1): How well-framed and composed?
3. Emotional appeal (0-1): How emotionally impactful?
4. Shareability (0-1): How appropriate for social sharing?
5. Has special moment (yes/no): Does it capture a unique moment?

Return JSON format.
"""
```

#### Results
- AUC: 0.62 (no improvement!)
- LLM features had <10% importance in model
- Cost: $0.50 for 200-photo validation
- Time: ~30 seconds per photo (API latency)

#### Why It Still Didn't Work
1. **LLM features were too generic** - Scored all "cute baby photos" similarly
2. **Lost context in feature compression** - Reducing rich LLM understanding to 5 numbers lost information
3. **Classifier couldn't learn non-linear combinations** - Linear regression can't capture complex relationships
4. **Training data inconsistency remained** - Same fundamental problem as Approach 1

**Critical Insight from logs:**
```
Top-rated photos by LLM features (0.9+ shareability):
❌ IMG_5043.jpg - Baby with toes prominent (user labeled Storage)
❌ IMG_6721.jpg - Baby facing away (user labeled Storage)
✅ IMG_1234.jpg - Baby smiling at camera (user labeled Share)

LLM gave high scores to *technically good* photos,
but user's taste was more nuanced and personal.
```

#### Lessons Learned
1. **Feature compression loses signal** - LLMs are powerful, but reducing understanding to numbers defeats the purpose
2. **Generic quality ≠ personal taste** - LLM assessed "generic cute," user wanted "special to me"
3. **Hybrid architectures are complex** - Combining LLM + ML adds complexity without clear benefit
4. **Cost adds up** - Even cheap LLM APIs ($0.001/image) × large datasets = real money

---

### Approach 3: Direct LLM Taste Cloning ✅

**Timeline:** October 29 - November 1 (4 days with Claude Code)
**Code:** Current `LLM Taste Cloner/` directory

#### The Paradigm Shift
**Stop thinking like a data scientist. Start thinking like a taste curator.**

Skip the ML pipeline entirely:
```
Photo → LLM with Few-Shot Examples → Decision
        (Gemini Vision API)           (Share/Storage/Review)
```

No feature engineering. No classifier training. Just: "Here are 20 examples of photos I like and don't like. Now classify this new photo."

#### Architecture Components

**1. Few-Shot Learning**
```python
# Build prompt from user's manual labels
training_examples = {
    "IMG_1234.jpg": {
        "action": "Share",
        "reasoning": "the child smiling directly at camera, great expression"
    },
    "IMG_5678.jpg": {
        "action": "Storage",
        "reasoning": "Can't see baby's face, awkward framing"
    },
    # ... 50 examples total
}

prompt = f"""
You are helping sort baby photos. Here are examples of my taste:

**Photos I share:**
{format_share_examples(training_examples)}

**Photos I don't share:**
{format_storage_examples(training_examples)}

Now classify this new photo. Match my specific taste.
"""
```

**2. Burst-Aware Classification**
Real breakthrough: Most photos are part of burst sequences (7-10 nearly identical shots). Classify entire burst together, rank by quality, apply diversity filtering.

```python
# Instead of: Classify each photo independently (slow, misses context)
# Do: Classify burst together
burst_photos = [img1, img2, img3, img4, img5]  # 5 photos from same moment

result = gemini.classify_burst(
    photos=burst_photos,
    prompt="Rank these 5 burst photos. Which are best? Which to keep?"
)
# Output: Rank 1 (best) → Share, Ranks 2-3 → Review, Ranks 4-5 → Storage
```

**3. Diversity Filtering**
Even within "best" photos from burst, remove near-duplicates:

```python
# If burst yielded 3 "Share" candidates, check diversity
for photo_a, photo_b in share_pairs:
    is_diverse = gemini.compare(
        photo_a, photo_b,
        prompt="Are these meaningfully different? Different expressions or moments?"
    )
    if not is_diverse:
        # Downgrade duplicate to Review
        keep_best_only()
```

**4. Family-Friendly Filter**
```python
# Built into classification prompt
safety_check = """
CRITICAL: Flag as IGNORE if photo shows inappropriate nudity:
- Bath time showing private parts → IGNORE
- Diaper changes showing genitals → IGNORE
- "Grandparent test": Would you show this to grandparents? If no → IGNORE
"""
```

#### Why This Succeeded

**Results:**
- **Automation:** 75-80% of photos correctly routed
- **Share folder quality:** ~90% are photos user would actually share
- **Review folder:** Legitimate edge cases (not model confusion)
- **User satisfaction:** "It gets my taste!"

**Key Factors:**
1. **Rich context preserved** - LLM sees full image + all training examples
2. **Explainable decisions** - Returns reasoning for each classification
3. **Learns personal taste** - Not generic quality, but "what I specifically like"
4. **Handles bursts intelligently** - Ranks within burst, avoids flooding Share folder with duplicates
5. **Fast iteration** - Prompt engineering > model retraining

#### Evolution & Refinements

Tracked via Git commits and session notes:

**Week 1: Core Algorithm**
- Initial taste cloning with single-photo classification
- Added burst detection (temporal + visual similarity)
- Implemented burst ranking with Gemini

**Week 2: Production Hardening**
- Chunking for large bursts (>7 photos) to avoid JSON truncation
- Response validation (safety filters, token limits)
- Per-folder caching to prevent concurrent job collisions
- Video handling (separate pipeline)
- WhatsApp filename date parsing (fix false burst detection)
- Diversity filtering (prevent duplicate photos in Share folder)

**Critical Bugs Fixed:**
1. **JSON truncation** - Gemini responses cut off mid-JSON → Added max_output_tokens
2. **Response validation** - Safety filters blocked some photos → Added graceful handling
3. **Cache collision** - Multiple folders shared same cache files → Per-folder isolation
4. **False bursts** - Photos months apart grouped together → Filename date parsing + time span checks
5. **Too many duplicates** - Burst diversity check too lenient → Adjusted prompt from "low barrier" to "70/30 permissive/strict"

#### Technical Implementation Details

**Cost:**
- Gemini Flash: $0.075 per 1000 images
- 5000 photos = ~$5 total
- Cheaper than GPT-4V ($50-150) or Claude ($15)

**Performance:**
- Singletons: ~30 seconds per photo (API latency)
- Bursts: ~45 seconds for 7-photo burst (amortized to 6 sec/photo)
- Total time for 5000 photos: ~4-6 hours (can run overnight)

**Accuracy by Category:**
| Category | Accuracy | Notes |
|----------|----------|-------|
| Clear Share-worthy | 95% | Obvious good photos |
| Clear Storage | 90% | Obvious bad photos |
| Edge Cases | 60% | Subjective, goes to Review |
| Bursts | 85% | Correctly picks best 1-2 |
| Duplicates | 80% | Some sneaky duplicates remain |

**Production Deployment:**
- Command-line interface: `python taste_classify_gemini_v4.py "folder_path"`
- Checkpointing: Resume after interruption
- Caching: Reuse API responses for re-runs
- Dry-run mode: Preview before moving files
- Multi-folder support: Classify multiple photo collections concurrently

#### Code Quality Evolution

**Version History:**
```
v1: Basic taste cloning (Oct 29)
v2: Added burst detection (Oct 30)
v3: Implemented diversity filtering (Oct 31)
v4: Production hardening (Nov 1) ← Current
```

**Lines of Code:**
- Core algorithm: ~800 lines
- Burst detection: ~250 lines
- Feature extraction helpers: ~200 lines
- Cache/checkpoint management: ~150 lines
- **Total: ~1400 lines** (vs. 3000+ lines for classical ML approach)

#### Lessons Learned

1. **LLMs > Classical ML for subjective tasks**
   - Taste, style, creativity = LLM territory
   - Classification, prediction, regression = Classical ML territory

2. **Prompt engineering is underrated**
   - Changed prompt from "low barrier" → "70/30 permissive/strict"
   - Result: 20% improvement in duplicate detection
   - Faster than retraining model

3. **Few-shot learning is powerful**
   - 50 examples taught LLM user's taste
   - Would need thousands for classical ML
   - More efficient for rare preferences

4. **Context is king**
   - Burst-aware classification > independent classification
   - Seeing photo + reasoning from user > just photo
   - Rich prompts > feature vectors

5. **Production != Research**
   - Research: "Does LLM work?" → Yes!
   - Production: "Handle API errors, cache collisions, edge cases"
   - 80% of time spent on 20% edge cases

---

## 3. Claude Code Experience (AI-Assisted Development)

### What is Claude Code?
AI-powered development environment that:
- Reads/writes files directly
- Executes code and sees results
- Proposes fixes and implementations
- Learns from project context over time

**Think:** Copilot on steroids - not just autocomplete, but full feature implementation.

### Development Workflow

**Traditional IDE:**
```
1. Read docs/Stack Overflow
2. Write code
3. Run code
4. See error
5. Search for fix
6. Repeat
→ 2-3 hours per feature
```

**With Claude Code:**
```
1. Describe feature: "Add burst diversity filtering"
2. Claude proposes implementation
3. Review + approve
4. Claude writes code + tests
5. Claude encounters error → self-corrects
→ 15-30 minutes per feature
```

### Real Examples from This Project

**Example 1: WhatsApp Filename Date Parsing**

**Me:** "Some 'bursts' span 6 months. Photos have dates in filenames like IMG-20240521-WA0007.jpg but EXIF is missing."

**Claude:** *Reads burst_detector.py, identifies issue, proposes fix:*
```python
# Parse WhatsApp filename format
import re
match = re.match(r'IMG-(\d{4})(\d{2})(\d{2})-WA\d+', filename)
if match:
    year, month, day = match.groups()
    return datetime(int(year), int(month), int(day), 12, 0, 0)
```

*Implements fix, tests on my data, catches edge case, adds validation.*

**Time saved:** 1-2 hours (would've spent debugging regex + edge cases)

**Example 2: Cache Collision Bug**

**Me:** "I'm running 3 folders concurrently. Are checkpoints colliding?"

**Claude:** *Reads code, spots global CHECKPOINT_FILE constant, explains:*
> "YES - all folders share same 2 files. This will cause mislabeling."

*Then implements per-folder cache files in 5 minutes:*
```python
def get_checkpoint_file(folder_path):
    folder_id = sanitize(folder_path.name)
    return f"checkpoint_{folder_id}.json"
```

**Time saved:** 3+ hours (would've taken days to notice bug + fix)

**Example 3: Diversity Prompt Tuning**

**Me:** "Diversity check isn't filtering duplicates. What balance should I use?"

**Claude:** *Analyzes my screenshot showing 10 duplicate photos, proposes:*
> "Your images are extremely similar. Current prompt says 'LOW barrier, when in doubt keep both.' Let's try 70/30: catch obvious duplicates, keep subtle differences."

*Updates prompt, explains rationale, I approve, problem solved.*

**Time saved:** Multiple trial-and-error cycles

### Strengths

✅ **Rapid Prototyping**
- Feature implementation: 15-30 min (vs 2-3 hours)
- Bug fixes: 5-10 min (vs 30-60 min)
- Documentation: Instant (writes markdown as you code)

✅ **Context Awareness**
- Reads entire codebase
- Remembers decisions from hours ago
- Connects related issues across files

✅ **Self-Correction**
- Runs code → sees error → fixes automatically
- Proposes multiple solutions ranked by trade-offs
- Explains reasoning transparently

✅ **Handles Grunt Work**
- Boilerplate code (cache management, file I/O)
- Error handling (try/catch blocks, validation)
- Documentation (docstrings, README files)

✅ **Learning Curve**
- Works with unfamiliar libraries (Gemini API, CLIP)
- Reads docs and applies patterns correctly
- Adapts to user's coding style over time

### Limitations

❌ **Can't Replace Domain Expertise**
- Needed to specify "check diversity" - Claude didn't know duplicates were a problem
- Needed to identify cache collision risk - Claude confirmed but didn't proactively spot it
- Algorithm strategy (bursts, taste cloning) came from me

❌ **Context Window Limits**
- Can't hold entire large codebase in memory
- Sometimes forgets decisions from earlier (though rare)
- Needs reminders for project-specific conventions

❌ **Not Magic**
- Still need to review code carefully
- Can introduce bugs if prompt is ambiguous
- Sometimes over-engineers simple solutions

❌ **API Dependency**
- Requires internet
- Costs money (though minimal)
- Latency on large file operations

### When to Use Claude Code vs Traditional IDE

**Use Claude Code for:**
- New projects (rapid prototyping)
- Unfamiliar libraries/APIs
- Refactoring large sections
- Bug fixing with unclear root cause
- Documentation-heavy work

**Use Traditional IDE for:**
- Performance-critical sections (profiling)
- Complex algorithms requiring deep focus
- When offline
- When you know exact implementation

**Best Workflow:**
- **Design/strategy:** You decide
- **Implementation:** Claude writes
- **Review:** You approve/reject
- **Optimization:** Collaborate

### Productivity Impact

**Estimated Time Savings:**
- Feature development: 4-5x faster
- Bug fixing: 3-4x faster
- Documentation: 10x faster
- Learning new libraries: 2-3x faster

**Overall Project Timeline:**
- Traditional IDE: Estimated 3-4 weeks
- With Claude Code: 2 weeks actual (50% time savings)

### Tips for Data Science Teams

1. **Be specific about requirements**
   - ❌ "Make it better"
   - ✅ "Reduce duplicate photos by catching images with same pose + expression within 2 seconds"

2. **Iterate in small steps**
   - Don't ask for entire feature at once
   - Build incrementally: core → edge cases → optimization

3. **Review carefully**
   - Claude is fast, but not infallible
   - Spot-check logic, especially edge cases

4. **Use for learning**
   - Ask "why did you do it this way?"
   - Accelerates onboarding to new libraries

5. **Share context proactively**
   - "I'm optimizing for X, not Y"
   - "Last time we tried Z and it failed because..."

---

## 4. LLM API Integration Pitfalls

### Overview
Working with LLM vision APIs (Gemini, GPT-4V, Claude) introduces failure modes that don't exist in classical ML. Here are the gotchas we hit.

### Pitfall #1: JSON Truncation ⚠️ **CRITICAL**

**What Happened:**
```
Burst 3/1430: 2 photos
⚠️  Parse error: Expecting ',' delimiter: line 17 column 10 (char 613)
   Response truncated at: "expression_quality": "e
```

**Root Cause:**
- Gemini response cut off mid-JSON
- No `max_output_tokens` limit set
- Default limit too low for 7-photo burst analysis

**Impact:**
- 5-10% of burst classifications failed
- Photos defaulted to Review folder (safe but suboptimal)

**Fix:**
```python
# Before (no limit)
response = model.generate_content([prompt, img])

# After (explicit limit)
response = model.generate_content(
    [prompt, img],
    generation_config={'max_output_tokens': 4096}  # Enough for 7-photo burst
)
```

**Lessons:**
- Always set `max_output_tokens` explicitly
- Test with longest expected input (burst of 7 photos)
- Add truncation detection:
  ```python
  if not response_text.endswith('}'):
      # Response truncated!
  ```

### Pitfall #2: Safety Filters Blocking Valid Requests

**What Happened:**
```
⚠️  No response candidates (likely blocked by safety filters)
   Prompt feedback: <safety_filter_triggered>
```

**Root Cause:**
- Family photos of babies in bath → flagged as inappropriate
- Diaper change photos → blocked by safety filter
- No response returned, code crashed

**Impact:**
- Random photos failed silently
- Lost ~1-2% of photos to safety filter errors

**Fix:**
```python
# Validate response before accessing .text
if not response.candidates:
    print("⚠️ Blocked by safety filters")
    return {"classification": "Review", "reasoning": "Safety filter"}

candidate = response.candidates[0]
if candidate.finish_reason != 1:  # 1 = STOP (normal completion)
    print(f"⚠️ Unusual finish reason: {candidate.finish_reason}")
    return fallback_response
```

**Lessons:**
- Always validate `response.candidates` exists
- Check `finish_reason` before accessing response
- Have graceful fallback for blocked requests
- Add safety guidance to prompt:
  ```
  These are innocent family photos. Bath time with appropriate coverage is fine.
  Only flag if truly inappropriate.
  ```

### Pitfall #3: Finish Reason 2 (MAX_TOKENS)

**What Happened:**
```
⚠️  Unusual finish reason: 2
   Safety ratings: []
   → All 7 photos classified as Review (0.30 confidence)
```

**Root Cause:**
- `finish_reason=2` means MAX_TOKENS hit
- Response truncated mid-generation
- Different from explicit truncation (Pitfall #1)

**Impact:**
- 10-15% of large bursts failed
- All photos went to Review folder

**Fix:**
```python
# Increased token limit
generation_config={'max_output_tokens': 4096}  # Was 2048, too tight
```

**Lessons:**
- Monitor `finish_reason` in production
- Log unusual finish reasons for debugging
- 2048 tokens too low for 7-photo burst with detailed analysis
- 4096 tokens provides ~100% headroom

### Pitfall #4: Cache Key Collisions

**What Happened:**
- Running multiple folders concurrently
- All folders shared same cache files
- Checkpoint from folder_2022 overwrote checkpoint from folder_2023
- Cache responses mixed between folders

**Impact:**
- Potential mislabeling (cache response from wrong photo)
- Lost progress on concurrent jobs
- Wasted API costs (re-running same photos)

**Fix:**
```python
# Before (global cache)
CACHE_FILE = ".taste_cache/gemini_cache.json"

# After (per-folder cache)
def get_cache_file(folder_path):
    folder_id = sanitize(folder_path.name)
    return f".taste_cache/gemini_cache_{folder_id}.json"
```

**Lessons:**
- Concurrent operations need isolated state
- Cache keys should include context (folder, photo hash)
- Test multi-tenant scenarios early

### Pitfall #5: WhatsApp Filename Date Parsing

**What Happened:**
- Burst detection grouped photos from March to September (6 months!)
- 50-photo "bursts" of unrelated images

**Root Cause:**
- WhatsApp images have dates in filename (`IMG-20240521-WA0007.jpg`)
- EXIF metadata missing or corrupted
- Burst detection fell back to visual-only mode
- Similar-looking photos (food, objects) grouped together

**Impact:**
- Massive false bursts (50+ photos)
- Wasted API calls classifying unrelated photos together
- Confused LLM with incoherent burst context

**Fix:**
```python
def get_photo_timestamp(photo_path):
    # Try EXIF first
    exif_date = extract_exif_date(photo_path)
    if exif_date:
        return exif_date

    # NEW: Parse WhatsApp filename format
    filename = photo_path.name
    match = re.match(r'IMG-(\d{4})(\d{2})(\d{2})-WA\d+', filename)
    if match:
        year, month, day = match.groups()
        return datetime(int(year), int(month), int(day), 12, 0, 0)

    # Fallback to file mtime
    return datetime.fromtimestamp(photo_path.stat().st_mtime)
```

**Lessons:**
- Mobile app filename conventions matter
- EXIF not always reliable (especially forwarded/downloaded images)
- Visual similarity alone is insufficient for burst detection
- Add sanity checks (time span > 1 hour → split burst)

### Pitfall #6: Prompt Sensitivity

**What Happened:**
- Diversity check marked all duplicates as "diverse"
- 10+ nearly identical photos in Share folder

**Root Cause:**
- Prompt said: "LOW barrier - even small differences count. When in doubt, say diverse."
- LLM interpreted tiny head tilts as "different composition"
- Micro-expressions counted as "different moment"

**Impact:**
- Share folder flooded with duplicates
- User unhappy with output quality

**Fix:**
```python
# Before
prompt = "LOW barrier - when in doubt, say diverse"

# After (70/30 balance)
prompt = """
If a regular person would say "these look the same to me", mark as NOT diverse.
Only keep both if there's a clear, human-noticeable difference.
When uncertain, be slightly conservative (lean toward NOT diverse to avoid duplicates).
"""
```

**Lessons:**
- Prompts have massive impact on behavior
- "When in doubt" clauses are critical
- Test on edge cases (very similar photos)
- Iterate prompt tuning faster than model retraining

### Pitfall #7: Cost Estimation Failures

**What Happened:**
- Expected cost: $5 for 5000 photos
- Actual cost: $12 (2.4x higher)

**Root Cause:**
- Burst chunking → multiple API calls per burst
- Diversity checks → additional API calls for duplicates
- Retries on errors → wasted calls
- Cache misses → re-classifying same photos

**Impact:**
- Budget overrun (minor, but compounds at scale)

**Fix:**
```python
# Track actual API calls
api_call_counter = 0

def gemini_call(...):
    global api_call_counter
    api_call_counter += 1
    # ... actual call

print(f"Total API calls: {api_call_counter}")
print(f"Estimated cost: ${api_call_counter * 0.001}")
```

**Lessons:**
- Instrument API calls with counters
- Cache aggressively (save 50% costs on re-runs)
- Budget 2-3x initial estimate for production

### Pitfall #8: Video Handling Edge Case

**What Happened:**
- Videos not appearing in Videos/ folder
- Script crashed on some video files

**Root Cause:**
- Code used `shutil.move()` instead of `shutil.copy()`
- Move failed across drives/permissions
- Videos stayed in source folder silently

**Impact:**
- Lost videos (not classified, not moved)
- Inconsistent behavior across runs

**Fix:**
```python
# Before (move)
shutil.move(video_path, videos_folder)  # Fails across drives

# After (copy)
copy_file(video_path, videos_folder)  # Works everywhere
```

**Lessons:**
- File operations differ across platforms/permissions
- Always copy, never move (unless explicit cleanup needed)
- Test on different drive configurations

### Summary: LLM API Production Checklist

Before deploying LLM APIs:

- [ ] Set `max_output_tokens` explicitly
- [ ] Validate `response.candidates` exists
- [ ] Check `finish_reason` before accessing response
- [ ] Handle safety filter blocks gracefully
- [ ] Isolate cache/state for concurrent operations
- [ ] Parse domain-specific filename conventions
- [ ] Test prompt on edge cases (duplicates, ambiguous content)
- [ ] Instrument API calls with cost tracking
- [ ] Add retries with exponential backoff
- [ ] Cache responses aggressively
- [ ] Monitor error rates in production
- [ ] Have fallback logic for API failures

---

## 5. Key Takeaways & Recommendations

### For Similar Projects

**1. Identify Task Type Early**

```
Is the task...

Objective (measurable)?               Subjective (taste-based)?
├─ Classification (discrete)          ├─ Personal preference
├─ Regression (continuous)            ├─ Style/aesthetics
├─ Clustering (grouping)              ├─ "Special moment" feeling
└→ Use Classical ML                   └→ Use LLMs directly

Examples:                              Examples:
- Spam detection                       - Photo curation (this project)
- Price prediction                     - Content recommendation
- Object detection                     - Creative writing evaluation
- Fraud detection                      - UX/design feedback
```

**Key Questions:**
- Can you write down explicit rules for the decision?
- Would two humans agree on the label 95% of the time?
- Is the decision based on measurable features?

If any answer is "no" → Consider LLMs first.

**2. Start with Simplest Approach**

Don't over-engineer:
```
Try this order:
1. Direct LLM with few-shot learning (2-3 days)
   ├─ If AUC > 0.75 → Ship it!
   └─ If AUC < 0.60 → Try #2

2. LLM + light feature engineering (1 week)
   ├─ Example: Burst detection, deduplication
   └─ If still < 0.70 → Try #3

3. Classical ML with heavy feature engineering (2-3 weeks)
   └─ Only if you have consistent, objective labels

Avoid: Starting with #3 (I learned this the hard way)
```

**3. Iterate on Prompts, Not Models**

Prompt tuning vs model retraining:
| Action | Time | Cost | Flexibility |
|--------|------|------|-------------|
| Tune prompt | Minutes | Free | High |
| Retrain model | Hours | Compute $ | Low |
| Add features | Days | Engineering $ | Medium |

**Real example from this project:**
- Prompt change: "When in doubt, lean toward diverse" → "When in doubt, lean toward NOT diverse"
- Result: 20% improvement in duplicate detection
- Time: 5 minutes
- Cost: $0

**4. Few-Shot Learning is Powerful**

Classical ML vs LLM few-shot learning:
| Approach | Training Examples Needed | Training Time |
|----------|--------------------------|---------------|
| Logistic Regression | 1000-5000 | 5 min |
| Random Forest | 2000-10000 | 20 min |
| Neural Network | 10000+ | Hours |
| **LLM Few-Shot** | **20-100** | **0 min** |

**Why LLMs need fewer examples:**
- Pre-trained on vast knowledge
- Understand concepts from descriptions
- Transfer learning built-in

**When to use which:**
- Objective task + lots of data → Classical ML
- Subjective task + little data → LLM few-shot

**5. Production Edge Cases Matter**

Research vs production:
```
Research (20% of effort):
- Does the algorithm work?
- What's the accuracy?
- Is it better than baseline?

Production (80% of effort):
- What if API returns error?
- What if response truncates?
- What if cache corrupts?
- What if user interrupts?
- What if running 3 jobs concurrently?
- What if filename has special characters?
- What if photo format is HEIC not JPG?
- What if EXIF data is missing?
```

**Advice:** Budget 4x longer for production hardening than research.

---

### When to Use Classical ML vs LLM-Direct

**Use Classical ML when:**
- ✅ Objective, measurable criteria
- ✅ Consistent labels (humans agree 90%+)
- ✅ Large training dataset (1000s+ examples)
- ✅ Fast inference required (< 10ms)
- ✅ Cost-sensitive (millions of predictions)
- ✅ Explainability via feature importance

**Examples:**
- Spam detection
- Fraud detection
- Medical diagnosis (with clear criteria)
- Price prediction
- Demand forecasting

**Use LLMs Directly when:**
- ✅ Subjective preferences
- ✅ "I know it when I see it" criteria
- ✅ Small training dataset (10s-100s examples)
- ✅ Inference latency OK (seconds)
- ✅ Explainability via natural language
- ✅ Rapid iteration needed

**Examples:**
- Content curation (this project)
- Style evaluation
- Creative writing feedback
- UX/design review
- Sentiment analysis (nuanced)

**Use Hybrid when:**
- ⚠️ Some objective features + some subjective
- ⚠️ Classical ML for speed, LLM for edge cases
- ⚠️ Feature extraction from LLM → classical classifier

**Warning:** Hybrids add complexity. Only use if clear benefit.

---

### Team Workflow Considerations

**1. AI-Assisted Development**

Recommendations for adopting Claude Code (or Copilot, Cursor, etc.):

**Phase 1: Individual Experimentation (1 month)**
- Let 1-2 developers try on side projects
- Gather feedback on strengths/weaknesses
- Identify use cases vs traditional IDE

**Phase 2: Pilot Project (1-2 months)**
- Use on non-critical feature
- Measure productivity (time to completion, bug rate)
- Establish best practices (review protocols, prompt templates)

**Phase 3: Team Rollout (3-6 months)**
- Training sessions on effective prompting
- Code review standards for AI-generated code
- Cost management (API usage limits)

**2. Prompt Engineering as a Skill**

Treat prompt engineering like code:
- Version control prompts (Git)
- Peer review important prompts
- A/B test prompt changes
- Document why prompts work

**Example prompt library:**
```
prompts/
├── classification/
│   ├── single_photo_v1.txt
│   ├── burst_ranking_v3.txt
│   └── diversity_check_v2.txt
├── evaluation/
│   └── quality_assessment_v1.txt
└── README.md  # When to use each prompt
```

**3. Cost Management**

For teams using LLM APIs:

**Budgeting:**
- Development: $50-200/month per developer (Claude Code, API calls)
- Production: Depends on usage (track carefully)
- **This project:** $12 total for 5000 photos

**Cost Controls:**
- Cache responses aggressively (50% savings)
- Use cheaper models (Gemini Flash, not GPT-4V)
- Batch API calls where possible
- Set per-user/per-project spending limits

**4. When to Involve Data Scientists vs Engineers**

Clear roles:
```
Data Scientist:
- Algorithm selection (ML vs LLM)
- Prompt design
- Evaluation metrics
- A/B testing methodology

Engineer:
- Production infrastructure
- API error handling
- Caching/optimization
- Monitoring/alerting
```

Collaborate on:
- Feature engineering
- Edge case handling
- Cost optimization

---

## 6. Conclusion

### Project Outcomes

**Quantitative:**
- Approach 1 (Classical ML): AUC 0.624
- Approach 2 (LLM Features): AUC 0.62
- Approach 3 (LLM Direct): 75-80% automation, ~90% user satisfaction

**Time Investment:**
- Approach 1: 2 days
- Approach 2: 1 day
- Approach 3: 4 days (with Claude Code)
- **Total:** 7 days (1.5 weeks)
- **Estimate without Claude Code:** 3-4 weeks

**Cost:**
- Development: $50 (Claude Code API)
- Production: $12 (Gemini API for 5000 photos)
- **Total:** $62

### Key Learnings

**Technical:**
1. For subjective tasks, LLMs > Classical ML
2. Few-shot learning requires 10-100x fewer examples
3. Prompt engineering > model retraining for iteration speed
4. Context (burst awareness) dramatically improves accuracy
5. Production edge cases take 80% of time

**Process:**
1. AI-assisted development (Claude Code) is transformative
2. Start simple, add complexity only when needed
3. Iterate in small steps with fast feedback
4. Budget 2-3x initial time/cost estimates for production

**Organizational:**
1. Prompt engineering is a skill worth developing
2. LLM APIs need robust error handling
3. Cost management matters at scale
4. Hybrid ML+LLM approaches add complexity without clear benefit

### Final Thoughts

This project challenged conventional data science wisdom:
- More data didn't help (AUC stayed flat)
- More features didn't help (AUC stayed flat)
- More complex models didn't help

What worked:
- Understanding the problem was subjective, not objective
- Skipping ML entirely and going straight to LLMs
- Using AI to write AI (Claude Code building LLM pipelines)
- Focusing on production edge cases

**The meta-lesson:** Sometimes the best ML solution is no ML at all. Know your problem space before choosing your tools.

---

## Appendix: Code Repositories

**Classical ML Approach:**
- Location: `Classical ML Approach\`
- Key file: `taste_sort_win_v9.py` (final version)
- Lines of code: ~3000

**LLM Feature Extraction:**
- Location: `LLM Test\`
- Key files: `llm_validation_v4.py`, `EVALUATION_AND_RECOMMENDATIONS.md`
- Lines of code: ~1200

**LLM Taste Cloning (Current):**
- Location: `LLM Taste Cloner\`
- Key files: `taste_classify_gemini_v4.py`, `burst_detector.py`
- Lines of code: ~1400
- **Production-ready ✅**

---

## Questions?

Feel free to reach out for:
- Code walkthroughs
- Prompt templates
- Architecture discussions
- Claude Code demonstrations

**Document Version:** 1.0
**Last Updated:** November 1, 2024

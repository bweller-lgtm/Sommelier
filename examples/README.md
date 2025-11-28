# Examples

This folder contains sample outputs to help new users understand what the classifier generates.

## Files

### sample_taste_preferences.json

A sample AI-generated taste profile created by `generate_taste_profile.py`. This file demonstrates:

- **share_philosophy**: Overall guiding principle for what makes a photo worth sharing
- **top_priorities**: Key factors to consider when classifying photos
- **share_criteria**: Specific criteria that make photos share-worthy
- **reject_criteria**: Factors that indicate a photo shouldn't be shared
- **contextual_preferences**: Nuanced preferences for specific scenarios
- **specific_guidance**: Actionable tips for classification decisions

## Creating Your Own

Generate your own taste profile by running:

```bash
python generate_taste_profile.py
```

This analyzes photos in your Share and Storage folders to learn your preferences.

## Customization

Team members can customize the taste preferences by either:

1. **Running the generator** - Analyze your own Share/Storage examples
2. **Manual editing** - Modify `taste_preferences.json` or `taste_preferences_generated.json` directly
3. **Interactive training** - Use `python taste_trainer_pairwise_v4.py` for pairwise comparison training

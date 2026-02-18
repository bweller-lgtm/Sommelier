"""Smart comparison selection for pairwise training sessions."""

import random
from typing import Optional

from .session import TrainingSession


class ComparisonSampler:
    """
    Select next comparison from a training session using ratio-based strategy.

    Sampling mix:
    - 50% within-burst pairs (weight 2.0) — quality refinement
    - 30% between-burst pairs (weight 1.0) — preference learning
    - 20% gallery reviews for bursts with 5+ photos (weight 1.5)

    Prefers unlabeled photos first, then photos with fewest labels.
    Gracefully degrades when certain types are unavailable.
    """

    WITHIN_RATIO = 0.5
    BETWEEN_RATIO = 0.3
    GALLERY_RATIO = 0.2
    LARGE_BURST_THRESHOLD = 5

    def __init__(self, session: TrainingSession, seed: int = 1337):
        self.session = session
        self.rng = random.Random(seed)

        # Separate bursts (2+ photos) from singletons
        self.multi_bursts = [b for b in session.bursts if len(b) >= 2]
        self.large_bursts = [
            b for b in session.bursts if len(b) >= self.LARGE_BURST_THRESHOLD
        ]
        self.small_bursts = [
            b for b in session.bursts
            if 2 <= len(b) < self.LARGE_BURST_THRESHOLD
        ]

    def get_next(self) -> Optional[dict]:
        """
        Get the next comparison to present.

        Returns dict with:
        - type: "pairwise_within" | "pairwise_between" | "gallery"
        - For pairwise: photo_a, photo_b, comparison_type, context
        - For gallery: photos, context
        Returns None if no more comparisons are available.
        """
        # Decide what type of comparison to serve based on ratios
        # and what's available
        available_types = self._available_types()
        if not available_types:
            return None

        # Weighted random selection based on ratios
        roll = self.rng.random()
        cumulative = 0.0
        total_weight = sum(available_types.values())

        # Normalize weights
        for comp_type, weight in available_types.items():
            cumulative += weight / total_weight
            if roll <= cumulative:
                if comp_type == "gallery":
                    return self._gallery_review()
                elif comp_type == "within":
                    return self._within_burst_pair()
                else:
                    return self._between_burst_pair()

        # Fallback: return first available type
        first_type = next(iter(available_types))
        if first_type == "gallery":
            return self._gallery_review()
        elif first_type == "within":
            return self._within_burst_pair()
        return self._between_burst_pair()

    def _available_types(self) -> dict[str, float]:
        """Determine which comparison types are available with their weights."""
        labeled = self.session.get_labeled_photos()
        available = {}

        # Check within-burst availability
        has_within = any(
            sum(1 for p in burst if p not in labeled) >= 2
            for burst in self.multi_bursts
        )
        if has_within:
            available["within"] = self.WITHIN_RATIO

        # Check between-burst availability (need 2+ bursts/singletons with unlabeled)
        sources_with_unlabeled = 0
        for burst in self.multi_bursts:
            if any(p not in labeled for p in burst):
                sources_with_unlabeled += 1
        for s in self.session.singletons:
            if s not in labeled:
                sources_with_unlabeled += 1
        if sources_with_unlabeled >= 2:
            available["between"] = self.BETWEEN_RATIO

        # Check gallery availability
        has_gallery = any(
            sum(1 for p in burst if p not in labeled) >= 3
            for burst in self.large_bursts
        )
        if has_gallery:
            available["gallery"] = self.GALLERY_RATIO

        # If no bursts at all, fall back to random singleton pairs
        if not available and len(self.session.singletons) >= 2:
            unlabeled_singletons = [
                s for s in self.session.singletons if s not in labeled
            ]
            if len(unlabeled_singletons) >= 2:
                available["between"] = 1.0

        return available

    def _within_burst_pair(self) -> Optional[dict]:
        """Select a pair from within the same burst."""
        labeled = self.session.get_labeled_photos()

        # Find bursts with unlabeled photos, prefer least-labeled
        candidates = []
        for burst in self.multi_bursts:
            unlabeled = [p for p in burst if p not in labeled]
            if len(unlabeled) >= 2:
                candidates.append((len(unlabeled), burst, unlabeled))

        if not candidates:
            # Fall back to bursts with any 2 photos (even if labeled)
            for burst in self.multi_bursts:
                if len(burst) >= 2:
                    candidates.append((0, burst, burst))

        if not candidates:
            return None

        # Prefer bursts with more unlabeled photos
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[: max(3, len(candidates) // 3)]
        _, burst, pool = self.rng.choice(top_candidates)

        pair = self.rng.sample(pool, 2)
        return {
            "type": "pairwise_within",
            "photo_a": pair[0],
            "photo_b": pair[1],
            "comparison_type": "within_burst",
            "context": f"Same burst ({len(burst)} photos). Which is better?",
        }

    def _between_burst_pair(self) -> Optional[dict]:
        """Select a pair from different bursts or singletons."""
        labeled = self.session.get_labeled_photos()

        # Collect one representative from each source
        sources = []
        for burst in self.multi_bursts:
            unlabeled = [p for p in burst if p not in labeled]
            if unlabeled:
                sources.append(self.rng.choice(unlabeled))
            elif burst:
                sources.append(self.rng.choice(burst))

        for s in self.session.singletons:
            if s not in labeled:
                sources.append(s)

        if len(sources) < 2:
            # Include labeled singletons as fallback
            for s in self.session.singletons:
                if s not in sources:
                    sources.append(s)
            if len(sources) < 2:
                return None

        pair = self.rng.sample(sources, 2)
        return {
            "type": "pairwise_between",
            "photo_a": pair[0],
            "photo_b": pair[1],
            "comparison_type": "between_burst",
            "context": "Different scenes. Which do you prefer overall?",
        }

    def _gallery_review(self) -> Optional[dict]:
        """Select a large burst for gallery review."""
        labeled = self.session.get_labeled_photos()

        # Find large bursts with enough unlabeled photos
        candidates = []
        for burst in self.large_bursts:
            unlabeled = [p for p in burst if p not in labeled]
            if len(unlabeled) >= 3:
                candidates.append((len(unlabeled), burst))

        if not candidates:
            return None

        # Prefer bursts with most unlabeled
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, burst = self.rng.choice(
            candidates[: max(2, len(candidates) // 2)]
        )

        # Show up to 20 photos
        photos = burst[:20]
        return {
            "type": "gallery",
            "photos": photos,
            "context": (
                f"Burst of {len(burst)} photos. "
                f"Select ALL keepers (photos worth sharing)."
            ),
        }

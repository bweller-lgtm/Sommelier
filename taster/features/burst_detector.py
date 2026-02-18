"""Burst detection using temporal proximity and visual similarity."""
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
from PIL.ExifTags import TAGS

from ..core.config import BurstDetectionConfig


class BurstDetector:
    """
    Detect photo bursts using temporal proximity and visual similarity.

    A burst is a sequence of photos taken in quick succession that are visually similar.
    """

    def __init__(self, config: BurstDetectionConfig):
        """
        Initialize burst detector.

        Args:
            config: Burst detection configuration.
        """
        self.config = config

    def detect_bursts(
        self,
        photos: List[Path],
        embeddings: np.ndarray
    ) -> List[List[Path]]:
        """
        Detect bursts in photo collection.

        Args:
            photos: List of photo paths.
            embeddings: CLIP embeddings (normalized, shape: [N, D]).

        Returns:
            List of bursts, where each burst is a list of photo paths.
            Singletons are returned as single-item lists.
        """
        # Validate inputs
        if len(photos) == 0:
            return []

        if embeddings.shape[0] != len(photos):
            raise ValueError(
                f"Embeddings array length ({embeddings.shape[0]}) does not match "
                f"photos list length ({len(photos)}). Inputs must be aligned."
            )

        if not self.config.enabled:
            # Return all photos as singletons
            return [[p] for p in photos]

        print(f"\n🔍 Detecting bursts (temporal + visual)...")
        print(f"   Time window: {self.config.time_window_seconds}s")
        print(f"   Similarity threshold: {self.config.embedding_similarity_threshold}")

        # Extract timestamps
        photo_times = self._extract_timestamps(photos)

        if not photo_times:
            # Fall back to visual-only clustering
            print("   No timestamps found, using visual-only clustering")
            return self._detect_visual_only(photos, embeddings)

        # Sort by timestamp
        photo_times.sort(key=lambda x: x[2])
        print(f"   Extracted timestamps from {len(photo_times)}/{len(photos)} photos")

        # Create temporal groups
        temporal_groups = self._create_temporal_groups(photo_times)
        print(f"   Found {len(temporal_groups)} temporal groups")

        # Cluster each temporal group by visual similarity
        bursts, singletons = self._cluster_temporal_groups(
            temporal_groups,
            embeddings,
            photos
        )

        # Report statistics
        self._print_statistics(bursts, singletons)

        # Return bursts + singletons (as single-item lists)
        return bursts + [[p] for p in singletons]

    def _extract_timestamps(
        self,
        photos: List[Path]
    ) -> List[Tuple[int, Path, datetime]]:
        """
        Extract timestamps from photos.

        Returns list of (index, path, timestamp) tuples.
        """
        photo_times = []

        for i, photo in enumerate(photos):
            timestamp = self._get_photo_timestamp(photo)
            if timestamp:
                photo_times.append((i, photo, timestamp))

        return photo_times

    def _get_photo_timestamp(self, photo_path: Path) -> Optional[datetime]:
        """
        Extract photo timestamp from EXIF, filename, or file metadata.

        Tries in order:
        1. EXIF DateTimeOriginal (most accurate)
        2. EXIF DateTime
        3. WhatsApp filename format
        4. File modification time

        Args:
            photo_path: Path to photo.

        Returns:
            Datetime object or None if timestamp couldn't be extracted.
        """
        # Try EXIF data first (most accurate)
        try:
            with Image.open(photo_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    # Try DateTimeOriginal (36867) - when photo was taken
                    datetime_original = exif_data.get(36867)
                    if datetime_original:
                        return datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")

                    # Try DateTime (306) - when file was modified
                    datetime_tag = exif_data.get(306)
                    if datetime_tag:
                        return datetime.strptime(datetime_tag, "%Y:%m:%d %H:%M:%S")
        except Exception:
            # EXIF extraction failed, continue to other methods
            pass

        # Try parsing WhatsApp filename format (IMG-YYYYMMDD-WA####.jpg)
        filename = photo_path.name
        match = re.match(
            r'(?:IMG|VID|AUD)-(\d{4})(\d{2})(\d{2})-',
            filename,
            re.IGNORECASE
        )
        if match:
            year, month, day = match.groups()
            try:
                # Use noon as default time for date-only timestamps
                return datetime(int(year), int(month), int(day), 12, 0, 0)
            except ValueError:
                pass

        # Fallback to file modification time
        try:
            return datetime.fromtimestamp(photo_path.stat().st_mtime)
        except Exception:
            return None

    def _create_temporal_groups(
        self,
        photo_times: List[Tuple[int, Path, datetime]]
    ) -> List[List[Tuple[int, Path, datetime]]]:
        """
        Group photos by temporal proximity.

        Args:
            photo_times: List of (index, path, timestamp) tuples (sorted by time).

        Returns:
            List of temporal groups.
        """
        if not photo_times:
            return []

        temporal_groups = []
        current_group = [photo_times[0]]

        for i in range(1, len(photo_times)):
            prev_time = current_group[-1][2]
            curr_time = photo_times[i][2]

            time_diff = (curr_time - prev_time).total_seconds()

            if time_diff <= self.config.time_window_seconds:
                current_group.append(photo_times[i])
            else:
                if len(current_group) >= self.config.min_burst_size:
                    temporal_groups.append(current_group)
                current_group = [photo_times[i]]

        # Add last group
        if len(current_group) >= self.config.min_burst_size:
            temporal_groups.append(current_group)

        return temporal_groups

    def _cluster_temporal_groups(
        self,
        temporal_groups: List[List[Tuple[int, Path, datetime]]],
        embeddings: np.ndarray,
        all_photos: List[Path]
    ) -> Tuple[List[List[Path]], List[Path]]:
        """
        Cluster each temporal group by visual similarity.

        Returns tuple of (bursts, singletons).
        """
        bursts = []
        singletons = []
        photos_in_bursts = set()
        photos_in_groups = set()  # Track all photos in temporal groups

        for group in temporal_groups:
            # Track all photos in this group
            group_photos_set = {item[1] for item in group}
            photos_in_groups.update(group_photos_set)

            if len(group) < self.config.min_burst_size:
                singletons.extend([item[1] for item in group])
                continue

            # Get embeddings for this group
            group_indices = [item[0] for item in group]
            group_photos = [item[1] for item in group]
            group_embeddings = embeddings[group_indices]

            # Compute pairwise similarity
            similarity_matrix = cosine_similarity(group_embeddings)

            # Cluster using similarity threshold
            visited = set()

            for i in range(len(group_photos)):
                if i in visited:
                    continue

                # Start new cluster
                cluster = [i]
                visited.add(i)

                # Find all photos similar to this one
                for j in range(i + 1, len(group_photos)):
                    if j in visited:
                        continue

                    # Check if j is similar to any photo in current cluster
                    max_sim = max(similarity_matrix[k, j] for k in cluster)

                    if max_sim >= self.config.embedding_similarity_threshold:
                        cluster.append(j)
                        visited.add(j)

                # Create burst if large enough
                if len(cluster) >= self.config.min_burst_size:
                    burst_photos = [group_photos[idx] for idx in cluster]

                    # Split if too large
                    if self.config.enable_chunking and len(burst_photos) > self.config.max_burst_size:
                        for chunk_start in range(0, len(burst_photos), self.config.max_burst_size):
                            chunk = burst_photos[chunk_start:chunk_start + self.config.max_burst_size]
                            if len(chunk) >= self.config.min_burst_size:
                                bursts.append(chunk)
                                photos_in_bursts.update(chunk)
                    else:
                        bursts.append(burst_photos)
                        photos_in_bursts.update(burst_photos)
                else:
                    singletons.extend([group_photos[idx] for idx in cluster])

        # Add photos not in any temporal group as singletons
        for photo in all_photos:
            if photo not in photos_in_groups:
                singletons.append(photo)

        return bursts, singletons

    def _detect_visual_only(
        self,
        photos: List[Path],
        embeddings: np.ndarray
    ) -> List[List[Path]]:
        """
        Fallback: detect bursts using only visual similarity.

        Args:
            photos: List of photo paths.
            embeddings: CLIP embeddings.

        Returns:
            List of bursts (including singletons as single-item lists).
        """
        similarity_matrix = cosine_similarity(embeddings)
        visited = set()
        bursts = []

        for i in range(len(photos)):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, len(photos)):
                if j in visited:
                    continue

                max_sim = max(similarity_matrix[k, j] for k in cluster)

                if max_sim >= self.config.embedding_similarity_threshold:
                    cluster.append(j)
                    visited.add(j)

            burst_photos = [photos[idx] for idx in cluster]
            bursts.append(burst_photos)

        return bursts

    def _print_statistics(self, bursts: List[List[Path]], singletons: List[Path]):
        """Print burst detection statistics."""
        print(f"\n✅ Burst detection complete:")
        print(f"   Bursts: {len(bursts)}")
        if bursts:
            sizes = [len(b) for b in bursts]
            print(f"   Burst sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
            print(f"   Total photos in bursts: {sum(sizes)}")
        print(f"   Singletons: {len(singletons)}")

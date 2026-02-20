"""File utilities for media handling."""
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image, ImageOps


class FileTypeRegistry:
    """Registry of supported file types."""

    # Supported image extensions
    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".webp", ".heic",
        ".tif", ".tiff", ".bmp", ".gif"
    }

    # Supported video extensions
    VIDEO_EXTENSIONS = {
        ".mp4", ".mov", ".avi", ".mkv", ".m4v",
        ".3gp", ".wmv", ".flv", ".webm"
    }

    # Supported audio extensions
    AUDIO_EXTENSIONS = {
        ".mp3", ".wav", ".flac", ".aac", ".ogg",
        ".m4a", ".wma", ".opus", ".aiff"
    }

    # Supported document extensions
    DOCUMENT_EXTENSIONS = {
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".html", ".htm", ".txt", ".md", ".csv", ".rtf",
    }

    # Supported code extensions (treated as documents for classification)
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".rb", ".php",
        ".c", ".cpp", ".h", ".hpp", ".cs",
        ".swift", ".kt", ".scala", ".lua",
        ".sh", ".bash", ".zsh", ".ps1",
        ".sql", ".r", ".m", ".pl",
        ".yaml", ".yml", ".toml", ".json", ".xml",
        ".dockerfile", ".tf", ".hcl",
    }

    @classmethod
    def is_image(cls, path: Path) -> bool:
        """
        Check if file is an image.

        Args:
            path: Path to file.

        Returns:
            True if image, False otherwise.
        """
        return path.suffix.lower() in cls.IMAGE_EXTENSIONS

    @classmethod
    def is_video(cls, path: Path) -> bool:
        """
        Check if file is a video.

        Args:
            path: Path to file.

        Returns:
            True if video, False otherwise.
        """
        return path.suffix.lower() in cls.VIDEO_EXTENSIONS

    @classmethod
    def is_audio(cls, path: Path) -> bool:
        """Check if file is an audio file."""
        return path.suffix.lower() in cls.AUDIO_EXTENSIONS

    @classmethod
    def is_code(cls, path: Path) -> bool:
        """Check if file is a code/config file."""
        return path.suffix.lower() in cls.CODE_EXTENSIONS

    @classmethod
    def is_document(cls, path: Path) -> bool:
        """Check if file is a document (includes code files)."""
        ext = path.suffix.lower()
        return ext in cls.DOCUMENT_EXTENSIONS or ext in cls.CODE_EXTENSIONS

    @classmethod
    def is_media(cls, path: Path) -> bool:
        """
        Check if file is any supported media type.

        Args:
            path: Path to file.

        Returns:
            True if media, False otherwise.
        """
        return cls.is_image(path) or cls.is_video(path) or cls.is_audio(path)

    @classmethod
    def list_images(cls, directory: Path, recursive: bool = False) -> List[Path]:
        """
        List all images in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            List of image paths.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        pattern = "**/*" if recursive else "*"

        images = []
        for path in directory.glob(pattern):
            if path.is_file() and cls.is_image(path):
                images.append(path)

        return sorted(images)

    @classmethod
    def list_videos(cls, directory: Path, recursive: bool = False) -> List[Path]:
        """
        List all videos in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            List of video paths.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        pattern = "**/*" if recursive else "*"

        videos = []
        for path in directory.glob(pattern):
            if path.is_file() and cls.is_video(path):
                videos.append(path)

        return sorted(videos)

    @classmethod
    def list_audio(cls, directory: Path, recursive: bool = False) -> List[Path]:
        """
        List all audio files in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            List of audio paths.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        pattern = "**/*" if recursive else "*"

        audio = []
        for path in directory.glob(pattern):
            if path.is_file() and cls.is_audio(path):
                audio.append(path)

        return sorted(audio)

    @classmethod
    def list_documents(cls, directory: Path, recursive: bool = False) -> List[Path]:
        """
        List all documents in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            List of document paths.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        pattern = "**/*" if recursive else "*"

        documents = []
        for path in directory.glob(pattern):
            if path.is_file() and cls.is_document(path):
                documents.append(path)

        return sorted(documents)

    @classmethod
    def list_media(cls, directory: Path, recursive: bool = False) -> Dict[str, List[Path]]:
        """
        List all media files in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            Dictionary with 'images' and 'videos' keys containing lists of paths.
        """
        return {
            "images": cls.list_images(directory, recursive),
            "videos": cls.list_videos(directory, recursive),
            "audio": cls.list_audio(directory, recursive),
        }

    @classmethod
    def list_all_media(cls, directory: Path, recursive: bool = False) -> Dict[str, List[Path]]:
        """
        List all media files including documents and audio in directory.

        Args:
            directory: Directory to search.
            recursive: Search recursively.

        Returns:
            Dictionary with 'images', 'videos', 'audio', and 'documents' keys.
        """
        return {
            "images": cls.list_images(directory, recursive),
            "videos": cls.list_videos(directory, recursive),
            "audio": cls.list_audio(directory, recursive),
            "documents": cls.list_documents(directory, recursive),
        }

    @classmethod
    def list_bundles(cls, directory: Path) -> List[Dict]:
        """Discover subfolder bundles for multi-file classification.

        Each immediate subdirectory with classifiable files is one bundle.

        Args:
            directory: Parent directory to scan.

        Returns:
            List of dicts: {"name": str, "path": Path,
            "files": {"images": [...], "videos": [...], "audio": [...], "documents": [...]}}
        """
        bundles = []
        if not directory.is_dir():
            return bundles
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            media = cls.list_all_media(child)
            total = sum(len(v) for v in media.values())
            if total > 0:
                bundles.append({
                    "name": child.name,
                    "path": child,
                    "files": media,
                })
        return bundles

    @classmethod
    def detect_media_type(cls, directory: Path, recursive: bool = False) -> str:
        """
        Detect the predominant media type in a directory.

        Args:
            directory: Directory to scan.
            recursive: Search recursively.

        Returns:
            "image", "video", "document", or "mixed".
        """
        media = cls.list_all_media(directory, recursive)
        has_images = len(media["images"]) > 0 or len(media["videos"]) > 0
        has_audio = len(media.get("audio", [])) > 0
        has_documents = len(media["documents"]) > 0

        type_count = sum([has_images, has_audio, has_documents])
        if type_count > 1:
            return "mixed"
        elif has_documents:
            return "document"
        elif has_audio:
            return "audio"
        elif len(media["videos"]) > 0 and len(media["images"]) == 0:
            return "video"
        else:
            return "image"


class ImageUtils:
    """Utilities for image processing."""

    @staticmethod
    def load_and_fix_orientation(
        image_path: Path,
        max_size: int = 1024
    ) -> Optional[Image.Image]:
        """
        Load image, fix EXIF orientation, and resize if needed.

        This function:
        1. Opens the image
        2. Applies EXIF orientation correction
        3. Resizes to max_size if larger (maintains aspect ratio)

        Args:
            image_path: Path to image file.
            max_size: Maximum dimension (width or height).

        Returns:
            PIL Image object or None if loading fails.
        """
        try:
            # Open image
            img = Image.open(image_path)

            # Fix EXIF orientation
            img = ImageOps.exif_transpose(img)

            # Resize if needed
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return img

        except Exception as e:
            # Try opening without processing as fallback
            try:
                return Image.open(image_path)
            except Exception:
                # Complete failure
                print(f"Warning: Failed to load image {image_path}: {e}")
                return None

    @staticmethod
    def get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions without loading full image.

        Args:
            image_path: Path to image file.

        Returns:
            Tuple of (width, height) or None if fails.
        """
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    @staticmethod
    def ensure_rgb(img: Image.Image) -> Image.Image:
        """
        Ensure image is in RGB mode.

        Args:
            img: PIL Image object.

        Returns:
            RGB PIL Image object.
        """
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    @staticmethod
    def save_image(img: Image.Image, output_path: Path, quality: int = 95) -> bool:
        """
        Save image to file.

        Args:
            img: PIL Image object.
            output_path: Path to save image.
            quality: JPEG quality (1-100).

        Returns:
            True if successful, False otherwise.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure RGB for JPEG
            if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                img = ImageUtils.ensure_rgb(img)
                img.save(output_path, "JPEG", quality=quality)
            else:
                img.save(output_path)

            return True
        except Exception as e:
            print(f"Warning: Failed to save image {output_path}: {e}")
            return False


def setup_heif_support():
    """Setup HEIF/HEIC image support if available."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return True
    except ImportError:
        return False


# Auto-setup HEIF support on import
_HEIF_AVAILABLE = setup_heif_support()

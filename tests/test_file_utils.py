"""Tests for file utilities."""
import pytest
from pathlib import Path
from PIL import Image
from src.core.file_utils import FileTypeRegistry, ImageUtils


class TestFileTypeRegistry:
    """Tests for FileTypeRegistry."""

    def test_is_image(self):
        """Test image detection."""
        assert FileTypeRegistry.is_image(Path("photo.jpg"))
        assert FileTypeRegistry.is_image(Path("photo.JPEG"))
        assert FileTypeRegistry.is_image(Path("photo.png"))
        assert FileTypeRegistry.is_image(Path("photo.heic"))
        assert not FileTypeRegistry.is_image(Path("video.mp4"))
        assert not FileTypeRegistry.is_image(Path("document.pdf"))

    def test_is_video(self):
        """Test video detection."""
        assert FileTypeRegistry.is_video(Path("video.mp4"))
        assert FileTypeRegistry.is_video(Path("video.MOV"))
        assert FileTypeRegistry.is_video(Path("video.avi"))
        assert not FileTypeRegistry.is_video(Path("photo.jpg"))
        assert not FileTypeRegistry.is_video(Path("document.pdf"))

    def test_is_media(self):
        """Test media detection."""
        assert FileTypeRegistry.is_media(Path("photo.jpg"))
        assert FileTypeRegistry.is_media(Path("video.mp4"))
        assert not FileTypeRegistry.is_media(Path("document.pdf"))

    def test_list_images(self, tmp_path):
        """Test listing images."""
        # Create test files
        (tmp_path / "photo1.jpg").touch()
        (tmp_path / "photo2.png").touch()
        (tmp_path / "video.mp4").touch()
        (tmp_path / "document.txt").touch()

        images = FileTypeRegistry.list_images(tmp_path)

        assert len(images) == 2
        assert any(p.name == "photo1.jpg" for p in images)
        assert any(p.name == "photo2.png" for p in images)

    def test_list_videos(self, tmp_path):
        """Test listing videos."""
        # Create test files
        (tmp_path / "photo.jpg").touch()
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mov").touch()

        videos = FileTypeRegistry.list_videos(tmp_path)

        assert len(videos) == 2
        assert any(p.name == "video1.mp4" for p in videos)
        assert any(p.name == "video2.mov" for p in videos)

    def test_list_media(self, tmp_path):
        """Test listing all media."""
        # Create test files
        (tmp_path / "photo.jpg").touch()
        (tmp_path / "video.mp4").touch()
        (tmp_path / "document.txt").touch()

        media = FileTypeRegistry.list_media(tmp_path)

        assert len(media["images"]) == 1
        assert len(media["videos"]) == 1

    def test_list_recursive(self, tmp_path):
        """Test recursive listing."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "photo1.jpg").touch()
        (subdir / "photo2.jpg").touch()

        # Non-recursive
        images = FileTypeRegistry.list_images(tmp_path, recursive=False)
        assert len(images) == 1

        # Recursive
        images = FileTypeRegistry.list_images(tmp_path, recursive=True)
        assert len(images) == 2


class TestImageUtils:
    """Tests for ImageUtils."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color="red")
        img.save(img_path)
        return img_path

    def test_load_and_fix_orientation(self, sample_image):
        """Test loading image."""
        img = ImageUtils.load_and_fix_orientation(sample_image)

        assert img is not None
        assert isinstance(img, Image.Image)
        assert img.size == (800, 600)

    def test_load_with_resize(self, sample_image):
        """Test loading image with resize."""
        img = ImageUtils.load_and_fix_orientation(sample_image, max_size=400)

        assert img is not None
        # Should be resized to fit within 400x400
        assert max(img.size) == 400
        assert min(img.size) == 300  # Aspect ratio preserved

    def test_load_nonexistent_image(self, tmp_path):
        """Test loading non-existent image."""
        img = ImageUtils.load_and_fix_orientation(tmp_path / "nonexistent.jpg")
        assert img is None

    def test_get_image_dimensions(self, sample_image):
        """Test getting image dimensions."""
        dimensions = ImageUtils.get_image_dimensions(sample_image)

        assert dimensions == (800, 600)

    def test_get_dimensions_nonexistent(self, tmp_path):
        """Test getting dimensions of non-existent image."""
        dimensions = ImageUtils.get_image_dimensions(tmp_path / "nonexistent.jpg")
        assert dimensions is None

    def test_ensure_rgb(self):
        """Test ensuring RGB mode."""
        # Create RGBA image
        img_rgba = Image.new("RGBA", (100, 100), color="red")
        img_rgb = ImageUtils.ensure_rgb(img_rgba)

        assert img_rgb.mode == "RGB"

        # Already RGB
        img_rgb2 = ImageUtils.ensure_rgb(img_rgb)
        assert img_rgb2.mode == "RGB"

    def test_save_image(self, tmp_path):
        """Test saving image."""
        # Create image
        img = Image.new("RGB", (100, 100), color="blue")

        # Save
        output_path = tmp_path / "output.jpg"
        success = ImageUtils.save_image(img, output_path, quality=90)

        assert success
        assert output_path.exists()

        # Verify it can be loaded
        loaded = Image.open(output_path)
        assert loaded.size == (100, 100)

    def test_save_image_creates_directory(self, tmp_path):
        """Test that save_image creates parent directories."""
        # Save to nested path that doesn't exist
        output_path = tmp_path / "nested" / "dir" / "output.jpg"
        img = Image.new("RGB", (100, 100), color="green")

        success = ImageUtils.save_image(img, output_path)

        assert success
        assert output_path.exists()

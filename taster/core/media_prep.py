"""Media preprocessing for non-native provider formats.

Provides frame extraction for video, page rendering for PDFs,
and base64 encoding for images — used by OpenAI and Anthropic providers.
"""
import base64
import io
from pathlib import Path
from typing import List, Optional

from PIL import Image


class ImageEncoder:
    """Resize and base64-encode images for API payloads."""

    @staticmethod
    def to_base64(
        image: Image.Image,
        format: str = "JPEG",
        max_size: int = 1024,
    ) -> str:
        """Return a data-URI-ready base64 string (no prefix)."""
        img = image.copy()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=format, quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def image_to_bytes(
        image: Image.Image,
        format: str = "JPEG",
        max_size: int = 1024,
    ) -> bytes:
        """Return raw image bytes after resize."""
        img = image.copy()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=format, quality=85)
        return buf.getvalue()

    @staticmethod
    def media_type_for(format: str = "JPEG") -> str:
        """Return the MIME type for an image format."""
        return {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(format.upper(), "image/jpeg")


class VideoFrameExtractor:
    """Extract evenly-spaced frames from a video file."""

    @staticmethod
    def extract_frames(
        path: Path,
        max_frames: int = 8,
    ) -> List[Image.Image]:
        """Return up to *max_frames* PIL Images sampled uniformly."""
        try:
            import cv2
        except ImportError:
            print("Warning: opencv-python not installed, cannot extract video frames")
            return []

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # Compute evenly-spaced indices
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            indices = [int(step * i) for i in range(max_frames)]

        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # OpenCV uses BGR; convert to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        cap.release()
        return frames


class PDFPageRenderer:
    """Render PDF pages to images."""

    @staticmethod
    def render_pages(
        path: Path,
        max_pages: int = 5,
        dpi: int = 150,
    ) -> List[Image.Image]:
        """Render the first *max_pages* pages as PIL Images.

        Tries pymupdf (fitz) first, then falls back to pdf2image/poppler.
        """
        # Try pymupdf
        try:
            import fitz  # pymupdf

            doc = fitz.open(str(path))
            pages: List[Image.Image] = []
            for i in range(min(len(doc), max_pages)):
                page = doc[i]
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            doc.close()
            return pages
        except ImportError:
            pass

        # Try pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(path), dpi=dpi, first_page=1, last_page=max_pages
            )
            return images[:max_pages]
        except ImportError:
            pass

        print(
            "Warning: Neither pymupdf nor pdf2image available. "
            "Install one for PDF rendering: pip install pymupdf"
        )
        return []

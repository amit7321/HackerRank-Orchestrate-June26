"""Image path resolution and real-mime-type detection from magic bytes.

Several images in this dataset carry a `.jpg` extension but are actually
WebP-encoded. Gemini needs the correct mime type, so we sniff file
signatures instead of trusting the extension.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config


@dataclass
class LoadedImage:
    image_id: str
    path: Path
    data: Optional[bytes]
    mime_type: Optional[str]
    load_ok: bool
    error: Optional[str] = None


def sniff_mime_type(data: bytes) -> Optional[str]:
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[0:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[4:12] in (b"ftypavif", b"ftypavis"):
        return "image/avif"
    return None


def image_id_from_path(rel_path: str) -> str:
    return Path(rel_path).stem


def parse_image_paths(image_paths_field: str) -> List[str]:
    return [p.strip() for p in (image_paths_field or "").split(";") if p.strip()]


def resolve_image_path(rel_path: str) -> Path:
    rel_path = rel_path.strip()
    candidate = config.REPO_ROOT / rel_path
    if candidate.exists():
        return candidate
    # Fall back to dataset/ prefix in case the CSV path omits it.
    candidate = config.DATASET_DIR / rel_path
    return candidate


def load_image(rel_path: str) -> LoadedImage:
    image_id = image_id_from_path(rel_path)
    path = resolve_image_path(rel_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return LoadedImage(image_id=image_id, path=path, data=None, mime_type=None, load_ok=False, error=str(exc))

    mime_type = sniff_mime_type(data)
    if mime_type is None:
        return LoadedImage(
            image_id=image_id,
            path=path,
            data=None,
            mime_type=None,
            load_ok=False,
            error="unrecognized image format (magic-byte sniff failed)",
        )
    return LoadedImage(image_id=image_id, path=path, data=data, mime_type=mime_type, load_ok=True)


def load_images(image_paths_field: str) -> List[LoadedImage]:
    return [load_image(p) for p in parse_image_paths(image_paths_field)]

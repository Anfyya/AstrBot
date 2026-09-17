"""Media file utilities.

Provides shared media reference materialization, format conversion, duration
probing, and image compression helpers.
"""

import asyncio
import base64
import binascii
import ctypes
import errno
import hashlib
import io
import math
import mimetypes
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias
from urllib.parse import unquote, urlparse, urlsplit
from urllib.request import url2pathname

from aiohttp import ClientError
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError

from astrbot import logger
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.datetime_utils import generate_timestamp_id
from astrbot.core.utils.io import DownloadFileHTTPError, download_file
from astrbot.core.utils.tencent_record_helper import (
    tencent_silk_to_wav,
    wav_to_tencent_silk,
)

IMAGE_COMPRESS_DEFAULT_MAX_SIZE = 1280
IMAGE_COMPRESS_DEFAULT_QUALITY = 95
IMAGE_COMPRESS_DEFAULT_OPTIMIZE = True
IMAGE_COMPRESS_DEFAULT_MIN_FILE_SIZE_MB = 1.0
# Model image inputs larger than this are skipped before decoding.
MODEL_IMAGE_MAX_INPUT_BYTES = 32 * 1024 * 1024
# Original encoded bytes are reused only for small stills; larger inputs are
# re-encoded so the output stays bounded by pixel size and quality.
MODEL_IMAGE_REUSE_MAX_BYTES = 2 * 1024 * 1024
IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES = 4 * 1024 * 1024

_WEBP_PRESERVE = object()
_WEBP_DECODER = None
_WEBP_DECODER_UNAVAILABLE = False
_WEBP_ADVANCED_DECODER_AVAILABLE = False
_WEBP_DECODER_ABI_VERSION = 0x0210


class _WebPRgbaBuffer(ctypes.Structure):
    """ctypes view of libwebp's externally-owned packed pixel buffer."""

    _fields_ = [
        ("rgba", ctypes.POINTER(ctypes.c_ubyte)),
        ("stride", ctypes.c_int),
        ("size", ctypes.c_size_t),
    ]


class _WebPYuvaBuffer(ctypes.Structure):
    """ctypes view of libwebp's YUVA buffer union arm."""

    _fields_ = [
        ("y", ctypes.POINTER(ctypes.c_ubyte)),
        ("u", ctypes.POINTER(ctypes.c_ubyte)),
        ("v", ctypes.POINTER(ctypes.c_ubyte)),
        ("a", ctypes.POINTER(ctypes.c_ubyte)),
        ("y_stride", ctypes.c_int),
        ("u_stride", ctypes.c_int),
        ("v_stride", ctypes.c_int),
        ("a_stride", ctypes.c_int),
        ("y_size", ctypes.c_size_t),
        ("u_size", ctypes.c_size_t),
        ("v_size", ctypes.c_size_t),
        ("a_size", ctypes.c_size_t),
    ]


class _WebPBufferUnion(ctypes.Union):
    """ctypes view of libwebp's decoded buffer union."""

    _fields_ = [
        ("rgba", _WebPRgbaBuffer),
        ("yuva", _WebPYuvaBuffer),
    ]


class _WebPDecBuffer(ctypes.Structure):
    """ctypes view of libwebp's decoder output descriptor."""

    _fields_ = [
        ("colorspace", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("is_external_memory", ctypes.c_int),
        ("u", _WebPBufferUnion),
        ("pad", ctypes.c_uint32 * 4),
        ("private_memory", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class _WebPBitstreamFeatures(ctypes.Structure):
    """ctypes view of libwebp's bitstream feature structure."""

    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("has_alpha", ctypes.c_int),
        ("has_animation", ctypes.c_int),
        ("format", ctypes.c_int),
        ("pad", ctypes.c_uint32 * 5),
    ]


class _WebPDecoderOptions(ctypes.Structure):
    """ctypes view of libwebp's decoder options structure."""

    _fields_ = [
        (name, ctypes.c_int)
        for name in (
            "bypass_filtering",
            "no_fancy_upsampling",
            "use_cropping",
            "crop_left",
            "crop_top",
            "crop_width",
            "crop_height",
            "use_scaling",
            "scaled_width",
            "scaled_height",
            "use_threads",
            "dithering_strength",
            "flip",
            "alpha_dithering_strength",
        )
    ] + [("pad", ctypes.c_uint32 * 5)]


class _WebPDecoderConfig(ctypes.Structure):
    """ctypes view of libwebp's advanced decoder configuration."""

    _fields_ = [
        ("input", _WebPBitstreamFeatures),
        ("output", _WebPDecBuffer),
        ("options", _WebPDecoderOptions),
    ]


class ImagePayloadTooLargeError(ValueError):
    """Raised when an encoded image cannot fit the preparation budget."""


def _get_webp_decoder():
    """Load the decoder already shipped with Pillow when it exposes C APIs.

    Returns:
        A configured ctypes library, or ``None`` when the Pillow build does
        not expose the decoder symbols.
    """
    global _WEBP_ADVANCED_DECODER_AVAILABLE
    global _WEBP_DECODER, _WEBP_DECODER_UNAVAILABLE
    if _WEBP_DECODER_UNAVAILABLE:
        return None
    if _WEBP_DECODER is not None:
        return _WEBP_DECODER
    try:
        from PIL import _webp

        decoder = ctypes.CDLL(_webp.__file__)
        get_info = decoder.WebPGetInfo
        decode_rgb = decoder.WebPDecodeRGBInto
        decode_rgba = decoder.WebPDecodeRGBAInto
        get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        get_info.restype = ctypes.c_int
        for decode in (decode_rgb, decode_rgba):
            decode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            decode.restype = ctypes.c_void_p
    except (AttributeError, ImportError, OSError):
        _WEBP_DECODER_UNAVAILABLE = True
        return None
    try:
        init_config = decoder.WebPInitDecoderConfigInternal
        decode = decoder.WebPDecode
        free_buffer = decoder.WebPFreeDecBuffer
        init_config.argtypes = [
            ctypes.POINTER(_WebPDecoderConfig),
            ctypes.c_int,
        ]
        init_config.restype = ctypes.c_int
        decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_WebPDecoderConfig),
        ]
        decode.restype = ctypes.c_int
        free_buffer.argtypes = [ctypes.POINTER(_WebPDecBuffer)]
        free_buffer.restype = None
    except AttributeError:
        _WEBP_ADVANCED_DECODER_AVAILABLE = False
    else:
        _WEBP_ADVANCED_DECODER_AVAILABLE = True
    _WEBP_DECODER = decoder
    return decoder


def _webp_container_properties(
    data: bytes,
) -> tuple[int, int, bool, bool, bool] | None:
    """Read WebP dimensions and flags without constructing a Pillow decoder.

    Args:
        data: Complete WebP file bytes.

    Returns:
        Width, height, alpha flag, animation flag, and EXIF flag.
        ``None`` when the container needs the optional decoder to expose its
        dimensions and that decoder is unavailable.

    Raises:
        ValueError: The RIFF/WebP container is malformed.
    """
    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("invalid WebP container")
    width = height = None
    has_alpha = False
    has_animation = False
    has_exif = False
    offset = 12
    while offset + 8 <= len(data):
        chunk_type = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        end = start + chunk_size
        if end > len(data):
            raise ValueError("truncated WebP chunk")
        chunk = data[start:end]
        if chunk_type == b"VP8X" and len(chunk) >= 10:
            has_alpha = bool(chunk[0] & 0x10)
            has_animation = bool(chunk[0] & 0x02)
            width = 1 + int.from_bytes(chunk[4:7] + b"\0", "little")
            height = 1 + int.from_bytes(chunk[7:10] + b"\0", "little")
        elif chunk_type in {b"ANIM", b"ANMF"}:
            has_animation = True
        elif chunk_type == b"ALPH":
            has_alpha = True
        elif chunk_type == b"EXIF":
            has_exif = True
        elif chunk_type == b"VP8L" and len(chunk) >= 5 and chunk[0] == 0x2F:
            has_alpha = has_alpha or bool(
                int.from_bytes(chunk[1:5], "little") & (1 << 28)
            )
        offset = end + (chunk_size & 1)
    if width is None or height is None:
        decoder = _get_webp_decoder()
        if decoder is None:
            return None
        get_info = decoder.WebPGetInfo
        width_value, height_value = ctypes.c_int(), ctypes.c_int()
        if not get_info(
            ctypes.c_char_p(data),
            len(data),
            ctypes.byref(width_value),
            ctypes.byref(height_value),
        ):
            raise ValueError("invalid WebP bitstream")
        width, height = width_value.value, height_value.value
    return width, height, has_alpha, has_animation, has_exif


def _decode_webp_with_advanced_config(
    decoder,
    data: bytes,
    width: int,
    height: int,
    target: tuple[int, int],
    has_alpha: bool,
):
    """Decode a scaled static WebP into one caller-owned pixel buffer.

    Args:
        decoder: Configured libwebp shared library.
        data: Complete encoded WebP bytes.
        width: Original image width.
        height: Original image height.
        target: Requested output dimensions.
        has_alpha: Whether the decoded image needs an alpha channel.

    Returns:
        A Pillow image when the advanced ABI is available and decoding succeeds;
        otherwise ``None`` so the caller can use the basic decoder path.

    Raises:
        MemoryError: The caller-owned output buffer cannot be allocated.
    """
    if not _WEBP_ADVANCED_DECODER_AVAILABLE or target == (width, height):
        return None

    channels = 4 if has_alpha else 3
    mode = "RGBA" if has_alpha else "RGB"
    pixels = bytearray(target[0] * target[1] * channels)
    pixel_buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
    config = _WebPDecoderConfig()
    if not decoder.WebPInitDecoderConfigInternal(
        ctypes.byref(config), _WEBP_DECODER_ABI_VERSION
    ):
        return None
    config.output.colorspace = 1 if has_alpha else 0
    config.output.is_external_memory = 1
    config.output.u.rgba.rgba = ctypes.cast(
        pixel_buffer, ctypes.POINTER(ctypes.c_ubyte)
    )
    config.output.u.rgba.stride = target[0] * channels
    config.output.u.rgba.size = len(pixels)
    config.options.use_scaling = 1
    config.options.scaled_width, config.options.scaled_height = target
    try:
        if (
            decoder.WebPDecode(ctypes.c_char_p(data), len(data), ctypes.byref(config))
            != 0
        ):
            return None
        if (config.output.width, config.output.height) != target:
            return None
    finally:
        decoder.WebPFreeDecBuffer(ctypes.byref(config.output))
    del pixel_buffer
    image = PILImage.frombuffer(mode, target, pixels, "raw", mode, 0, 1)
    image.format = "WEBP"
    return image


def _open_static_webp(
    source: bytes | Path,
    max_size: int,
    max_encoded_bytes: int,
    *,
    preserve_dimensions: bool = False,
):
    """Decode static WebP directly into one caller-owned pixel buffer.

    Args:
        source: Encoded WebP bytes or a local WebP path.
        max_size: Maximum edge length for the eventual preparation step.
        max_encoded_bytes: Base64 payload budget.
        preserve_dimensions: Whether the eventual preparation step must retain
            the original dimensions.

    Returns:
        A Pillow image backed by one RGB/RGBA buffer, ``_WEBP_PRESERVE`` for a
        compliant image, or ``None`` to use Pillow's compatibility path.

    Raises:
        ImagePayloadTooLargeError: An animation already exceeds the budget.
        ValueError: The WebP container is invalid.
        MemoryError: The target pixel buffer cannot be allocated.
    """
    data = source if isinstance(source, bytes) else source.read_bytes()
    try:
        properties = _webp_container_properties(data)
    except ValueError:
        return None
    if properties is None:
        return None
    width, height, has_alpha, has_animation, has_exif = properties
    PILImage._decompression_bomb_check((width, height))
    encoded_size = 4 * ((len(data) + 2) // 3)
    if has_animation:
        if encoded_size > max_encoded_bytes:
            raise ImagePayloadTooLargeError(
                f"Animated image exceeds the {max_encoded_bytes}-byte encoding limit"
            )
        return _WEBP_PRESERVE
    if encoded_size <= max_encoded_bytes and (
        preserve_dimensions or max(width, height) <= max_size
    ):
        return _WEBP_PRESERVE
    # The direct route cannot carry EXIF orientation without loading Pillow's
    # normal WebP plugin. Keep correctness for metadata-bearing images.
    if has_exif:
        return None
    decoder = _get_webp_decoder()
    if decoder is None:
        return None
    target = (width, height)
    if not preserve_dimensions:
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            target = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
        elif encoded_size > max_encoded_bytes * 2 and max(width, height) >= max_size:
            target = (
                max(1, width * 3 // 4),
                max(1, height * 3 // 4),
            )
    scaled = _decode_webp_with_advanced_config(
        decoder,
        data,
        width,
        height,
        target,
        has_alpha,
    )
    if scaled is not None:
        del data
        return scaled

    channels = 4 if has_alpha else 3
    pixels = bytearray(width * height * channels)
    pixel_buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
    decode = decoder.WebPDecodeRGBAInto if has_alpha else decoder.WebPDecodeRGBInto
    if not decode(
        ctypes.c_char_p(data),
        len(data),
        pixel_buffer,
        len(pixels),
        width * channels,
    ):
        raise ValueError("invalid WebP bitstream")
    del pixel_buffer, data
    mode = "RGBA" if has_alpha else "RGB"
    image = PILImage.frombuffer(mode, (width, height), pixels, "raw", mode, 0, 1)
    image.format = "WEBP"
    return image


@dataclass(slots=True)
class ImagePreparationOptions:
    """Options for the single provider-facing image preparation boundary."""

    enabled: bool = True
    max_size: int = IMAGE_COMPRESS_DEFAULT_MAX_SIZE
    quality: int = IMAGE_COMPRESS_DEFAULT_QUALITY
    optimize: bool = IMAGE_COMPRESS_DEFAULT_OPTIMIZE
    max_encoded_bytes: int | None = IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES
    preserve_dimensions: bool = False


def get_image_preparation_options(
    provider_settings: dict | None,
) -> ImagePreparationOptions:
    """Build image preparation options from provider settings.

    Args:
        provider_settings: Provider-level image preparation configuration.

    Returns:
        Validated image preparation options using the standard defaults.
    """
    if not isinstance(provider_settings, dict):
        return ImagePreparationOptions()

    enabled = provider_settings.get("image_compress_enabled", True)
    if not isinstance(enabled, bool):
        enabled = True
    raw_options = provider_settings.get("image_compress_options", {})
    options = raw_options if isinstance(raw_options, dict) else {}

    max_size = options.get("max_size", IMAGE_COMPRESS_DEFAULT_MAX_SIZE)
    if not isinstance(max_size, int) or isinstance(max_size, bool):
        max_size = IMAGE_COMPRESS_DEFAULT_MAX_SIZE

    quality = options.get("quality", IMAGE_COMPRESS_DEFAULT_QUALITY)
    if not isinstance(quality, int) or isinstance(quality, bool):
        quality = IMAGE_COMPRESS_DEFAULT_QUALITY

    max_encoded_bytes = options.get(
        "max_encoded_bytes", IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES
    )
    if not isinstance(max_encoded_bytes, int) or isinstance(max_encoded_bytes, bool):
        max_encoded_bytes = IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES
    optimize = options.get("optimize", IMAGE_COMPRESS_DEFAULT_OPTIMIZE)
    if not isinstance(optimize, bool):
        optimize = IMAGE_COMPRESS_DEFAULT_OPTIMIZE

    return ImagePreparationOptions(
        enabled=enabled,
        max_size=max(max_size, 1),
        quality=min(max(quality, 1), 100),
        optimize=optimize,
        max_encoded_bytes=max(max_encoded_bytes, 1),
    )


MEDIA_MIME_EXTENSIONS = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/amr": ".amr",
    "audio/silk": ".silk",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

IMAGE_FORMAT_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "AVIF": "image/avif",
}

ANIMATED_MONTAGE_GRID = 3
"""Animated images become a grid x grid frame montage (contact sheet)."""

ANIMATED_MONTAGE_FRAME_COUNT = ANIMATED_MONTAGE_GRID * ANIMATED_MONTAGE_GRID

CONVERT_CACHE_DIR_NAME = "media_convert_cache"
"""Cache directory (under the AstrBot temp dir) for converted images and frames."""

AUDIO_FORMAT_MIME_TYPES = {
    "aac": "audio/aac",
    "amr": "audio/amr",
    "flac": "audio/flac",
    "mp3": "audio/mp3",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "silk": "audio/silk",
    "tencent_silk": "audio/silk",
    "wav": "audio/wav",
}

DEFAULT_MEDIA_SUFFIXES = {
    "audio": ".wav",
    "image": ".bin",
    "video": ".mp4",
    "file": ".bin",
}


MediaRefStr: TypeAlias = str
"""
A media reference string accepted by MediaResolver: local path, file URI, HTTP(S),
base64://, data URI, or legacy bare base64.

Examples:
    Local path: ``/tmp/image.png``
    File URI: ``file:///tmp/image.png``
    HTTP(S) URL: ``https://example.com/image.png``
    base64:// payload: ``base64://iVBORw0KGgo...``
    Data URI: ``data:image/png;base64,iVBORw0KGgo...``
    Legacy bare base64: ``iVBORw0KGgo...``
"""


@dataclass(frozen=True, slots=True)
class ImagePreparationInput:
    """Describe an image entering the shared preparation boundary.

    Args:
        value: Image path, URL, data URI, base64 reference, or raw bytes.
        source_kind: Logical producer used for diagnostics and experiments.
        cleanup_paths: Temporary paths owned by the caller and released after
            preparation finishes.
    """

    value: MediaRefStr | bytes
    source_kind: str = "unknown"
    cleanup_paths: tuple[Path, ...] = ()


@dataclass(slots=True)
class ResolvedMediaData:
    """Base64 media bytes plus the metadata needed by provider payloads.

    Attributes:
        base64_data: Raw base64 payload without a ``data:`` URI prefix.
        mime_type: MIME type to send with provider payloads.
        format: Optional normalized media format, such as ``wav`` for audio.
    """

    base64_data: str
    mime_type: str
    format: str | None = None
    byte_size: int | None = None

    def to_bytes(self) -> bytes:
        """Decode the base64 payload, accepting missing padding."""
        return _decode_base64_payload(
            self.base64_data,
            error_message="invalid resolved media base64 data",
        )

    def to_data_url(self) -> str:
        """Return a ``data:<mime>;base64,...`` URL for multimodal providers."""
        return f"data:{self.mime_type};base64,{self.base64_data}"


@dataclass(slots=True)
class _LocalMediaFile:
    path: Path
    mime_type: str | None = None
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class ResolvedMediaFile:
    """A media reference resolved to a local path.

    ``cleanup_paths`` contains temporary files owned by the resolver. Callers that
    use ``MediaResolver.as_path()`` get automatic cleanup; callers that need to
    keep a path after the resolver returns should use ``MediaResolver.to_path()``.
    """

    source_ref: MediaRefStr
    media_type: str
    path: Path
    mime_type: str | None = None
    format: str | None = None
    cleanup_paths: list[Path] = field(default_factory=list)

    def read_bytes(self) -> bytes:
        """Read the resolved local file."""
        return self.path.read_bytes()

    def to_base64(self) -> str:
        """Read the resolved local file and return raw base64 data."""
        return _encode_file_to_base64(self.path)

    def to_data_url(self) -> str:
        """Read the resolved local file and return a data URL."""
        mime_type = self.mime_type or "application/octet-stream"
        return f"data:{mime_type};base64,{self.to_base64()}"

    def open(self, mode: str = "rb"):
        """Open the resolved local file."""
        return self.path.open(mode)

    def detach(self) -> None:
        """Keep temporary files alive after resolver cleanup would normally run."""

        self.cleanup_paths.clear()

    def cleanup(self) -> None:
        _cleanup_paths(self.cleanup_paths)


def is_file_uri(value: object) -> bool:
    """Return whether a value is a ``file:`` URI.

    Args:
        value: Candidate media reference or local path.

    Returns:
        ``True`` only for string values whose parsed URI scheme is ``file``.
    """

    if not isinstance(value, str):
        return False
    try:
        return urlsplit(value).scheme.lower() == "file"
    except ValueError:
        return False


def file_uri_to_path(file_uri: MediaRefStr) -> str:
    """Normalize file URIs to local filesystem paths.

    Args:
        file_uri: A ``file:`` URI or a plain filesystem path.

    Returns:
        The local filesystem path decoded with standard-library URL path rules.
        Non-``file:`` inputs are returned unchanged for convenience.
    """

    if not is_file_uri(file_uri):
        return file_uri

    parsed = urlparse(file_uri)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    if netloc and netloc.lower() != "localhost":
        if len(netloc) == 2 and netloc[1] == ":" and netloc[0].isalpha():
            return str(Path(url2pathname(f"{netloc}{path}")))
        return str(Path(url2pathname(f"//{netloc}{path}")))

    path = url2pathname(path)
    # url2pathname keeps "/" on POSIX but converts it to "\" on Windows, so
    # accept both prefixes before the drive colon.
    if (
        len(path) >= 4
        and path[0] in ("/", "\\")
        and path[2] == ":"
        and path[1].isalpha()
    ):
        path = path[1:]
    elif os.name != "nt" and path.startswith("//"):
        # Older AstrBot builds generated file:////path for POSIX absolute paths.
        path = "/" + path.lstrip("/")
    return str(Path(path))


def _extension_from_mime_type(mime_type: str | None) -> str | None:
    """Return a filesystem suffix for a MIME type, if one is known."""
    if not mime_type:
        return None
    normalized = mime_type.split(";", 1)[0].strip().lower()
    if not normalized:
        return None
    return MEDIA_MIME_EXTENSIONS.get(normalized) or mimetypes.guess_extension(
        normalized
    )


def _temp_media_path(media_type: str, suffix: str) -> Path:
    """Create a unique path under AstrBot's temp directory for materialized media."""
    temp_dir = Path(get_astrbot_temp_path())
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_media_type = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in media_type
    )
    return temp_dir / f"media_{safe_media_type}_{generate_timestamp_id()}{suffix}"


def _parse_base64_data_uri(data_uri: str) -> tuple[str | None, bytes]:
    """Parse a base64 data URI and return ``(mime_type, decoded_bytes)``."""
    header, separator, payload = data_uri.partition(",")
    if not separator or not header.lower().startswith("data:"):
        raise ValueError("invalid data URI")

    header_body = header[5:]
    header_parts = header_body.split(";") if header_body else []
    mime_type = header_parts[0].strip() if header_parts and header_parts[0] else None
    if not any(part.lower() == "base64" for part in header_parts[1:]):
        raise ValueError("data URI is not base64 encoded")

    return mime_type, _decode_base64_payload(
        payload,
        error_message="invalid base64 data URI payload",
    )


def _decode_base64_payload(
    payload: str,
    *,
    error_message: str,
    validate: bool = False,
) -> bytes:
    """Decode a base64 payload while tolerating omitted padding.

    Args:
        payload: Base64 payload without a data URI header.
        error_message: Message to use when decoding fails.
        validate: Whether to ask ``base64.b64decode`` to reject non-base64
            characters.

    Returns:
        Decoded bytes.

    Raises:
        ValueError: Raised when the payload cannot be decoded.
    """
    payload = "".join(payload.split())
    missing_padding = len(payload) % 4
    if missing_padding:
        payload += "=" * (4 - missing_padding)

    try:
        return base64.b64decode(payload, validate=validate)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(error_message) from exc


def _encode_file_to_base64(path: Path) -> str:
    """Encode a file without retaining a second full raw-image copy.

    Args:
        path: Local file to read.

    Returns:
        Base64 text without a data-URI prefix or line breaks.

    Raises:
        OSError: The file cannot be opened or read.
    """
    encoded = io.StringIO()
    remainder = b""
    chunk = b""
    block = b""
    chunk_size = 1023 * 1024
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            block = remainder + chunk if remainder else chunk
            complete_size = len(block) - len(block) % 3
            if complete_size:
                encoded.write(base64.b64encode(block[:complete_size]).decode("ascii"))
            remainder = block[complete_size:]
    del chunk, block
    if remainder:
        encoded.write(base64.b64encode(remainder).decode("ascii"))
    return encoded.getvalue()


def describe_media_ref(media_ref: object | None) -> str:
    """Return a log-safe description of a media reference.

    Args:
        media_ref: Original media reference from a platform, plugin, or provider
            request. It may contain a signed URL or a large base64 payload.

    Returns:
        A short description that avoids logging query strings, tokens, and base64
        payload contents.
    """

    if not media_ref:
        return "<empty media ref>"
    if not isinstance(media_ref, str):
        return f"media ref type={type(media_ref).__name__}"

    ref_len = len(media_ref)
    if media_ref.startswith("data:"):
        header, _, payload = media_ref.partition(",")
        mime_type = header[5:].split(";", 1)[0] or "unknown"
        return f"data URI mime={mime_type!r} payload_len={len(payload)}"

    if media_ref.startswith("base64://"):
        return f"base64 media payload_len={len(media_ref.removeprefix('base64://'))}"

    parsed = urlparse(media_ref)
    if parsed.scheme in {"http", "https"}:
        filename = Path(unquote(parsed.path or "")).name
        suffix = f" file={filename!r}" if filename else ""
        return f"{parsed.scheme} URL host={parsed.netloc!r}{suffix} len={ref_len}"

    if is_file_uri(media_ref):
        filename = Path(file_uri_to_path(media_ref)).name
        return f"file URI name={filename!r} len={ref_len}"

    media_path_exists = False
    try:
        media_path_exists = Path(media_ref).exists()
    except OSError:
        pass
    if not media_path_exists:
        compact = "".join(media_ref.split())
        if compact:
            try:
                _decode_base64_payload(
                    compact,
                    error_message="invalid bare base64 media payload",
                    validate=True,
                )
            except ValueError:
                pass
            else:
                return f"bare base64 media payload_len={len(compact)}"

    return f"local media path name={Path(media_ref).name!r} len={ref_len}"


def detect_image_mime_type(
    image_source: bytes | str | Path,
    *,
    default_mime_type: str | None = "image/jpeg",
) -> str | None:
    """Detect an image MIME type from encoded bytes or a local path.

    Args:
        image_source: Encoded image bytes or a local image path to inspect.
        default_mime_type: MIME type to return when detection fails.

    Returns:
        The detected MIME type, ``application/octet-stream`` for a recognized
        format without a registered MIME, or ``default_mime_type`` when detection
        fails.
    """

    try:
        image_file = (
            io.BytesIO(image_source)
            if isinstance(image_source, bytes)
            else image_source
        )
        with PILImage.open(image_file) as image:
            image.verify()
            image_format = str(image.format or "").upper()
    except Exception as exc:
        if not is_recoverable_image_error(exc):
            raise
        return default_mime_type

    # A decoded format must never inherit an unverified input MIME hint.
    return IMAGE_FORMAT_MIME_TYPES.get(
        image_format, PILImage.MIME.get(image_format, "application/octet-stream")
    )


async def detect_image_mime_type_async(
    image_source: bytes | str | Path,
    *,
    default_mime_type: str | None = "image/jpeg",
) -> str | None:
    """Detect an image MIME type without blocking the event loop.

    Args:
        image_source: Encoded image bytes or a local image path to inspect.
        default_mime_type: MIME type to return when detection fails.

    Returns:
        The detected MIME type, or ``default_mime_type`` when detection fails or
        the format is unknown.
    """

    return await asyncio.to_thread(
        detect_image_mime_type,
        image_source,
        default_mime_type=default_mime_type,
    )


def _guess_mime_type(path: Path, fallback: str | None = None) -> str | None:
    """Guess a MIME type from a filename, with an optional fallback."""
    return mimetypes.guess_type(path.name)[0] or fallback


def _cleanup_paths(cleanup_paths: list[Path] | None) -> None:
    """Best-effort cleanup for temporary files created by the resolver."""
    for cleanup_path in cleanup_paths or []:
        try:
            cleanup_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to cleanup %s: %s", cleanup_path, exc)


async def _materialize_media_ref(
    media_ref: MediaRefStr,
    *,
    media_type: str = "file",
    default_suffix: str | None = None,
) -> _LocalMediaFile:
    """Resolve a plugin-facing media reference to a local file.

    Supported references: local paths, file:// URIs, http(s) URLs, base64://,
    data:*;base64,... URIs, and legacy bare base64 payloads.

    Args:
        media_ref: Original media reference from a platform, plugin, or history.
        media_type: Logical media family used for temp filenames and defaults.
        default_suffix: Suffix to use when the reference does not carry one.
    """

    cleanup_paths: list[Path] = []
    suffix = default_suffix or DEFAULT_MEDIA_SUFFIXES.get(media_type, ".bin")

    if media_ref.startswith(("http://", "https://")):
        if media_type == "image":
            target_path = _temp_media_path("image", ".bin")
        else:
            parsed = urlparse(media_ref)
            target_suffix = Path(parsed.path).suffix or suffix
            target_path = _temp_media_path(media_type, target_suffix)
        cleanup_paths.append(target_path)
        try:
            await download_file(media_ref, str(target_path))
        except Exception:
            _cleanup_paths(cleanup_paths)
            raise
        mime_type = _guess_mime_type(target_path)
        if media_type == "image":
            detected_mime_type = await detect_image_mime_type_async(
                target_path,
                default_mime_type=None,
            )
            if detected_mime_type:
                mime_type = detected_mime_type
                detected_suffix = _extension_from_mime_type(detected_mime_type)
                if detected_suffix and target_path.suffix.lower() != detected_suffix:
                    detected_path = _temp_media_path("image", detected_suffix)
                    await asyncio.to_thread(target_path.rename, detected_path)
                    cleanup_paths[-1] = detected_path
                    target_path = detected_path
        return _LocalMediaFile(
            path=target_path,
            mime_type=mime_type,
            cleanup_paths=cleanup_paths,
        )

    if is_file_uri(media_ref):
        path = Path(file_uri_to_path(media_ref))
        return _LocalMediaFile(path=path, mime_type=_guess_mime_type(path))

    if media_ref.startswith("data:"):
        mime_type, media_bytes = _parse_base64_data_uri(media_ref)
        target_suffix = _extension_from_mime_type(mime_type) or suffix
        if media_type == "image" and target_suffix == suffix:
            detected_mime_type = await detect_image_mime_type_async(
                media_bytes,
                default_mime_type=None,
            )
            if detected_mime_type:
                mime_type = detected_mime_type
                target_suffix = _extension_from_mime_type(detected_mime_type) or suffix
        target_path = _temp_media_path(media_type, target_suffix)
        cleanup_paths.append(target_path)
        try:
            await asyncio.to_thread(target_path.write_bytes, media_bytes)
        except Exception:
            _cleanup_paths(cleanup_paths)
            raise
        return _LocalMediaFile(
            path=target_path,
            mime_type=mime_type,
            cleanup_paths=cleanup_paths,
        )

    if media_ref.startswith("base64://"):
        media_bytes = _decode_base64_payload(
            media_ref.removeprefix("base64://"),
            error_message="invalid base64 media payload",
        )
        mime_type = None
        target_suffix = suffix
        if media_type == "image":
            mime_type = await detect_image_mime_type_async(
                media_bytes,
                default_mime_type=None,
            )
            target_suffix = _extension_from_mime_type(mime_type) or suffix
        target_path = _temp_media_path(media_type, target_suffix)
        cleanup_paths.append(target_path)
        try:
            await asyncio.to_thread(target_path.write_bytes, media_bytes)
        except Exception:
            _cleanup_paths(cleanup_paths)
            raise
        return _LocalMediaFile(
            path=target_path,
            mime_type=mime_type,
            cleanup_paths=cleanup_paths,
        )

    path = Path(media_ref)
    path_exists = False
    try:
        path_exists = path.exists()
    except OSError:
        pass
    if path_exists:
        return _LocalMediaFile(path=path, mime_type=_guess_mime_type(path))

    compact_media_ref = "".join(media_ref.split())
    if compact_media_ref:
        try:
            media_bytes = _decode_base64_payload(
                compact_media_ref,
                error_message="invalid bare base64 media payload",
                validate=True,
            )
        except ValueError:
            pass
        else:
            mime_type = None
            target_suffix = suffix
            if media_type == "image":
                mime_type = await detect_image_mime_type_async(
                    media_bytes,
                    default_mime_type=None,
                )
                target_suffix = _extension_from_mime_type(mime_type) or suffix
            target_path = _temp_media_path(media_type, target_suffix)
            cleanup_paths.append(target_path)
            try:
                await asyncio.to_thread(target_path.write_bytes, media_bytes)
            except Exception:
                _cleanup_paths(cleanup_paths)
                raise
            return _LocalMediaFile(
                path=target_path,
                mime_type=mime_type,
                cleanup_paths=cleanup_paths,
            )

    return _LocalMediaFile(path=path, mime_type=_guess_mime_type(path))


class MediaResolver:
    """Resolve, convert, and export media references.

    The resolver accepts local paths, file:// URIs, http(s) URLs, base64:// payloads,
    data:*;base64,... URIs, and legacy bare base64 payloads. Temporary paths are
    cleaned when using as_path(), while to_path() intentionally leaves returned
    paths alive for callers that need to hand them to platform SDKs.

    Args:
        media_ref: Source media reference. It may be a local path, ``file://`` URI,
            HTTP(S) URL, ``base64://`` payload, base64 data URI, or legacy bare
            base64 payload.
        media_type: Logical media family. ``audio`` enables format conversion and
            defaults to WAV output; ``image`` enables image MIME detection.
        default_suffix: Fallback suffix for temporary files when the source does
            not expose one.
    """

    def __init__(
        self,
        media_ref: MediaRefStr,
        *,
        media_type: str = "file",
        default_suffix: str | None = None,
    ) -> None:
        self.media_ref = media_ref
        self.media_type = media_type
        self.default_suffix = default_suffix

    async def _resolve_path(
        self,
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ) -> ResolvedMediaFile:
        """Materialize the source and apply media-type-specific conversion.

        For audio, ``target_format`` controls the output format, including the
        QQ / Wechat / Wecom ``tencent_silk`` upload format. When it is not set, audio
        resolves to WAV unless ``preserve_mp3`` is true and the source already
        appears to be MP3.
        """
        local_file = await _materialize_media_ref(
            self.media_ref,
            media_type=self.media_type,
            default_suffix=self.default_suffix,
        )
        cleanup_paths = list(local_file.cleanup_paths)
        resolved_path = local_file.path
        mime_type = local_file.mime_type or _guess_mime_type(resolved_path)
        resolved_format = resolved_path.suffix.lower().lstrip(".") or None

        try:
            if self.media_type == "audio":
                audio_format = target_format
                if not audio_format:
                    audio_format = (
                        "mp3" if preserve_mp3 and resolved_format == "mp3" else "wav"
                    )

                if audio_format == "tencent_silk":
                    intermediate_cleanup_paths = list(cleanup_paths)
                    silk_path = _temp_media_path("audio", ".silk")
                    try:
                        wav_path = Path(await ensure_wav(str(resolved_path)))
                        if wav_path != resolved_path:
                            intermediate_cleanup_paths.append(wav_path)
                        duration = await wav_to_tencent_silk(
                            str(wav_path), str(silk_path)
                        )
                        if duration <= 0:
                            raise ValueError(
                                "Tencent Silk conversion returned empty duration"
                            )
                    except Exception:
                        _cleanup_paths([*intermediate_cleanup_paths, silk_path])
                        raise

                    _cleanup_paths(intermediate_cleanup_paths)
                    cleanup_paths = [silk_path]
                    resolved_path = silk_path
                    resolved_format = audio_format
                    mime_type = AUDIO_FORMAT_MIME_TYPES[resolved_format]
                else:
                    if audio_format == "wav":
                        converted_audio_path = Path(
                            await ensure_wav(str(resolved_path))
                        )
                    elif resolved_format == audio_format:
                        converted_audio_path = resolved_path
                    else:
                        converted_audio_path = Path(
                            await convert_audio_format(
                                str(resolved_path),
                                output_format=audio_format,
                            )
                        )

                    if converted_audio_path != resolved_path:
                        cleanup_paths.append(converted_audio_path)
                    resolved_path = converted_audio_path
                    resolved_format = audio_format
                    mime_type = AUDIO_FORMAT_MIME_TYPES.get(
                        resolved_format, "audio/wav"
                    )
        except Exception:
            _cleanup_paths(cleanup_paths)
            raise

        return ResolvedMediaFile(
            source_ref=self.media_ref,
            media_type=self.media_type,
            path=resolved_path,
            mime_type=mime_type,
            format=resolved_format,
            cleanup_paths=cleanup_paths,
        )

    @asynccontextmanager
    async def as_path(
        self,
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ) -> AsyncIterator[ResolvedMediaFile]:
        """Yield a resolved local file and clean resolver-owned temp files on exit.

        Use this when the consumer only needs the file during the context manager.
        For audio, pass ``target_format`` to force a format such as ``wav`` or
        ``tencent_silk``.
        """
        resolved = await self._resolve_path(
            target_format=target_format,
            preserve_mp3=preserve_mp3,
        )
        try:
            yield resolved
        finally:
            resolved.cleanup()

    async def to_path(
        self,
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ) -> str:
        """Return a resolved local path and keep temporary files alive.

        This is for message components and platform SDK calls that need a path
        after the resolver method returns. Callers or event cleanup should remove
        the returned temp file later.
        """
        resolved = await self._resolve_path(
            target_format=target_format,
            preserve_mp3=preserve_mp3,
        )
        resolved.detach()
        return str(resolved.path.resolve())

    async def to_bytes(
        self,
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ) -> bytes:
        """Resolve media, read bytes, and clean resolver-owned temp files."""
        async with self.as_path(
            target_format=target_format,
            preserve_mp3=preserve_mp3,
        ) as resolved:
            return resolved.read_bytes()

    async def to_base64(
        self,
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ) -> str:
        """Resolve media to raw base64 data without a data URI prefix."""
        return base64.b64encode(
            await self.to_bytes(
                target_format=target_format,
                preserve_mp3=preserve_mp3,
            )
        ).decode("utf-8")

    async def to_base64_data(
        self,
        *,
        strict: bool = False,
        target_format: str | None = None,
        preserve_mp3: bool = False,
        default_mime_type: str | None = "image/jpeg",
    ) -> ResolvedMediaData | None:
        """Resolve media to base64 data plus MIME metadata.

        Args:
            strict: Raise on invalid or unreadable media instead of returning
                ``None`` where the resolver can safely ignore the reference.
            target_format: Optional output format for audio conversion.
            preserve_mp3: Keep existing MP3 audio as MP3 when no target format is
                provided; otherwise audio defaults to WAV.
            default_mime_type: Fallback MIME type for legacy image base64 payloads
                whose bytes cannot be identified.
        """
        if self.media_type == "image":
            async with self.as_path(target_format=target_format) as resolved:
                try:
                    media_bytes = await asyncio.to_thread(resolved.read_bytes)
                except OSError as exc:
                    if strict or not is_recoverable_image_error(exc):
                        raise
                    return None

                mime_type = await detect_image_mime_type_async(
                    media_bytes,
                    default_mime_type=None,
                )
                if (
                    not mime_type
                    and resolved.mime_type
                    and resolved.mime_type.startswith("image/")
                ):
                    mime_type = resolved.mime_type
                is_legacy_base64_ref = self.media_ref.startswith("base64://")
                is_remote_or_data_ref = self.media_ref.startswith(
                    ("http://", "https://", "data:")
                ) or is_file_uri(self.media_ref)
                if not is_legacy_base64_ref and not is_remote_or_data_ref:
                    try:
                        _decode_base64_payload(
                            "".join(self.media_ref.split()),
                            error_message="invalid bare base64 media payload",
                            validate=True,
                        )
                    except ValueError:
                        is_legacy_base64_ref = False
                    else:
                        is_legacy_base64_ref = True
                if not mime_type and is_legacy_base64_ref:
                    mime_type = default_mime_type
                if not mime_type:
                    if strict:
                        raise ValueError(
                            f"Invalid image file: {describe_media_ref(self.media_ref)}"
                        )
                    return None

                return ResolvedMediaData(
                    base64_data=base64.b64encode(media_bytes).decode("utf-8"),
                    mime_type=mime_type,
                )

        async with self.as_path(
            target_format=target_format,
            preserve_mp3=preserve_mp3,
        ) as resolved:
            try:
                media_bytes = resolved.read_bytes()
            except OSError:
                if strict:
                    raise
                return None

            mime_type = resolved.mime_type or "application/octet-stream"
            return ResolvedMediaData(
                base64_data=base64.b64encode(media_bytes).decode("utf-8"),
                mime_type=mime_type,
                format=resolved.format,
            )

    async def to_data_url(
        self,
        *,
        strict: bool = False,
        target_format: str | None = None,
        preserve_mp3: bool = False,
        default_mime_type: str | None = "image/jpeg",
    ) -> str | None:
        """Resolve media directly to a provider-ready data URL."""
        resolved = await self.to_base64_data(
            strict=strict,
            target_format=target_format,
            preserve_mp3=preserve_mp3,
            default_mime_type=default_mime_type,
        )
        return resolved.to_data_url() if resolved else None

    @asynccontextmanager
    async def open(
        self,
        mode: str = "rb",
        *,
        target_format: str | None = None,
        preserve_mp3: bool = False,
    ):
        """Open resolved media as a file object inside a cleanup context."""
        async with self.as_path(
            target_format=target_format,
            preserve_mp3=preserve_mp3,
        ) as resolved:
            with resolved.open(mode) as file_obj:
                yield file_obj


async def resolve_image_ref_to_base64_data(
    image_ref: MediaRefStr | bytes,
    *,
    strict: bool = False,
    default_mime_type: str | None = "image/jpeg",
    options: ImagePreparationOptions | None = None,
) -> ResolvedMediaData | None:
    """Resolve and prepare an image reference for a provider request.

    ``strict=False`` returns ``None`` for invalid images so provider payload
    assembly can skip bad image refs without failing the whole request. Resource
    and size-limit errors still propagate because returning the original image
    would allow an invalid or oversized payload to reach a provider.

    Args:
        image_ref: Local path, URL, data URI, base64 reference, or image bytes.
        strict: Raise ordinary resolution errors instead of returning ``None``.
        default_mime_type: MIME fallback for otherwise unidentified images.
        options: Dimension, encoding, and cleanup policy for image preparation.

    Returns:
        Prepared provider-ready image data, or ``None`` for a safely skippable
        invalid image when ``strict`` is false.

    Raises:
        ImagePayloadTooLargeError: The image cannot fit the configured budget.
        MemoryError: Image preparation exhausts process resources.
        OSError: The source cannot be read or the prepared output cannot be written.
    """
    try:
        return await prepare_image_source(
            image_ref,
            options=options,
            default_mime_type=default_mime_type,
        )
    except (ImagePayloadTooLargeError, MemoryError):
        raise
    except (OSError, ValueError):
        is_legacy_base64 = isinstance(image_ref, str) and image_ref.startswith(
            "base64://"
        )
        if isinstance(image_ref, str) and not is_legacy_base64:
            is_reference_scheme = image_ref.startswith(
                ("http://", "https://", "data:")
            ) or is_file_uri(image_ref)
            try:
                path_exists = Path(image_ref).exists()
            except OSError:
                path_exists = False
            if not is_reference_scheme and not path_exists:
                try:
                    _decode_base64_payload(
                        "".join(image_ref.split()),
                        error_message="invalid bare base64 media payload",
                        validate=True,
                    )
                except ValueError:
                    pass
                else:
                    is_legacy_base64 = True
        if is_legacy_base64:
            # Preserve the historical fallback for opaque legacy base64 refs,
            # while still enforcing the configured request budget.
            legacy = await MediaResolver(
                image_ref,
                media_type="image",
                default_suffix=".bin",
            ).to_base64_data(
                strict=strict,
                default_mime_type=default_mime_type,
            )
            if (
                legacy
                and options
                and options.max_encoded_bytes is not None
                and len(legacy.base64_data) > options.max_encoded_bytes
            ):
                raise ImagePayloadTooLargeError(
                    f"Image exceeds the {options.max_encoded_bytes}-byte encoding limit"
                )
            return legacy
        if strict:
            raise
        return None


def _image_convert_cache_dir() -> Path:
    # Reading and encoding must work even when the optional cache is unavailable.
    return Path(get_astrbot_temp_path()) / CONVERT_CACHE_DIR_NAME


def is_recoverable_image_error(error: Exception) -> bool:
    """Identify ordinary input, decoder, network and cache failures.

    Args:
        error: Exception raised while processing an image.

    Returns:
        Whether the image may be skipped. Resource exhaustion, Pillow's image
        safety limit and programming errors must propagate to the caller.
    """
    if isinstance(error, OSError) and error.errno in {
        errno.ENOMEM,
        errno.EMFILE,
        errno.ENFILE,
    }:
        return False
    return isinstance(
        error,
        (
            OSError,
            ValueError,
            SyntaxError,
            EOFError,
            struct.error,
            ClientError,
            DownloadFileHTTPError,
        ),
    )


def _image_convert_cache_key(source_bytes: bytes, params: str) -> str:
    """Build a content-addressed key for derived image bytes.

    Args:
        source_bytes: Original encoded image content.
        params: Algorithm version, output category and effective size.

    Returns:
        A filesystem-safe content and parameter digest.
    """
    source_digest = hashlib.sha256(source_bytes).hexdigest()[:32]
    params_digest = hashlib.sha256(params.encode()).hexdigest()[:8]
    return f"{source_digest}_{params_digest}"


def _image_has_alpha(image: PILImage.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _inspect_image(image_bytes: bytes) -> int:
    """Inspect decodable image bytes.

    Args:
        image_bytes: Encoded image bytes.

    Returns:
        Number of frames in the validated image.

    Raises:
        Exception: Raised by Pillow when the bytes are not a decodable image.
    """
    with PILImage.open(io.BytesIO(image_bytes)) as image:
        frame_count = getattr(image, "n_frames", 1)
        image.verify()
    if frame_count == 1:
        # verify() checks structure, not pixels. Reopen before loading; compatible
        # still images otherwise bypass every operation that decodes their data.
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            image.load()
    # Animation frames are decoded while building the montage.
    return frame_count


_IMAGE_CONVERT_CACHE_VERSION = "v9-icc"
"""Bump when conversion output semantics change (modes, transparency, sizing)
so stale cache entries produced by older code are never served."""

MODEL_IMAGE_PNG_FALLBACK_MAX_BYTES = 1024 * 1024
"""PNG outputs larger than this are flattened onto white and re-encoded as JPEG."""


def normalize_model_image_max_size(value: object) -> int:
    """Normalize the model image longest-edge cap.

    Accepts ints, integer-like floats, and integer strings. Booleans,
    non-finite numbers, unparseable values, and values below the smallest
    usable montage edge (the grid size) fall back to the default with a
    warning, so every entry shares one effective-size semantic and a
    configured cap is actually honored by the produced montage.

    Args:
        value: Raw configured cap. ``None`` means unset and silently uses
            the default.

    Returns:
        The effective longest-edge cap in pixels.
    """
    normalized: int | None = None
    if isinstance(value, bool):
        normalized = None
    elif value is None:
        normalized = None
    elif isinstance(value, int):
        normalized = value
    elif isinstance(value, float):
        normalized = int(value) if math.isfinite(value) and value.is_integer() else None
    elif isinstance(value, str):
        try:
            normalized = int(value.strip())
        except ValueError:
            normalized = None
    if normalized is None or normalized < ANIMATED_MONTAGE_GRID:
        if value is not None:
            logger.warning(
                "Invalid model image max size %r; falling back to %d.",
                value,
                IMAGE_COMPRESS_DEFAULT_MAX_SIZE,
            )
        return IMAGE_COMPRESS_DEFAULT_MAX_SIZE
    return normalized


def _encode_image_frame_bytes(
    image: PILImage.Image,
    max_size: int | None = None,
    quality: int = IMAGE_COMPRESS_DEFAULT_QUALITY,
) -> bytes:
    """Encode a display-oriented frame as JPEG, or PNG when it carries transparency.

    Args:
        image: Opened source frame, which is never mutated.
        max_size: Optional longest-edge limit; smaller images are not enlarged.
        quality: JPEG output quality in the range 1-100.

    Returns:
        Encoded single-frame JPEG or PNG bytes. A PNG larger than
        MODEL_IMAGE_PNG_FALLBACK_MAX_BYTES is flattened onto white and
        re-encoded as JPEG to bound the request payload size. High-bit-depth
        samples are min-max normalized to 8-bit before JPEG encoding.
    """
    oriented = ImageOps.exif_transpose(image)
    prepared = oriented
    extras: list[PILImage.Image] = []
    try:
        has_alpha = _image_has_alpha(oriented)
        if has_alpha:
            # Promote RGB/L color-key transparency as well as palette alpha.
            prepared = oriented.convert("RGBA")
        elif oriented.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
            prepared = oriented.convert("RGB")
        if max_size is not None and max(prepared.size) > max_size:
            # Pillow forces nearest-neighbor resampling for palette and binary
            # images, regardless of the requested filter.
            if prepared.mode in {"P", "1"}:
                prepared = prepared.convert("RGB" if prepared.mode == "P" else "L")
            # Pillow's integer pre-reduction does not support I;16, although
            # direct Lanczos resampling does. Keep the original bit depth.
            prepared.thumbnail(
                (max_size, max_size),
                PILImage.Resampling.LANCZOS,
                reducing_gap=None if prepared.mode == "I;16" else 2.0,
            )
        # JPEG saving does not auto-embed the source ICC profile like PNG does;
        # attach it explicitly, but only when the color space survived intact
        # (conversions like CMYK -> RGB invalidate the source profile).
        icc_profile = image.info.get("icc_profile")
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1", "I", "I;16"}:
            icc_profile = None
        save_kwargs = {"icc_profile": icc_profile} if icc_profile else {}
        if has_alpha:
            buffer = io.BytesIO()
            prepared.save(buffer, "PNG", **save_kwargs)
            data = buffer.getvalue()
            if len(data) <= MODEL_IMAGE_PNG_FALLBACK_MAX_BYTES:
                return data
            # Oversized PNG flattens onto white and re-encodes as JPEG.
            rgba = prepared if prepared.mode == "RGBA" else prepared.convert("RGBA")
            extras.append(rgba)
            flattened = PILImage.new("RGB", rgba.size, (255, 255, 255))
            extras.append(flattened)
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            prepared = flattened
        elif prepared.mode in {"I", "I;16"}:
            # convert() clips high-bit-depth samples at 255 instead of scaling;
            # min-max normalize to 8-bit so the content survives JPEG.
            low, high = prepared.getextrema()
            if high > low:
                scaled = prepared.point(lambda v: (v - low) * (255.0 / (high - low)))
                extras.append(scaled)
                prepared = scaled
            prepared = prepared.convert("L")
        elif prepared.mode not in {"RGB", "L"}:
            # JPEG output supports only RGB and L among the remaining modes.
            prepared = prepared.convert("L" if prepared.mode == "1" else "RGB")
        buffer = io.BytesIO()
        prepared.save(buffer, "JPEG", quality=quality, optimize=True, **save_kwargs)
        return buffer.getvalue()
    finally:
        for extra in extras:
            extra.close()
        if prepared is not oriented and prepared not in extras:
            prepared.close()
        oriented.close()


def _read_valid_cached_image_bytes(output_path: Path) -> bytes | None:
    """Read a cache entry, treating any miss or corruption as absent.

    The temp cleaner may delete cache entries concurrently: a missing,
    unreadable, empty, or undecodable entry is treated as a cache miss and
    the caller rebuilds from the source bytes it already holds.

    Args:
        output_path: Candidate cache file path.

    Returns:
        Decodable cached bytes, or ``None`` when the entry is unusable.
    """
    try:
        data = output_path.read_bytes()
    except OSError as exc:
        if not is_recoverable_image_error(exc):
            raise
        return None
    if not data:
        return None
    try:
        frame_count = _inspect_image(data)
        with PILImage.open(io.BytesIO(data)) as image:
            if (
                image.format not in {"PNG", "JPEG"}
                or frame_count != 1
                or image.getexif().get(274, 1) != 1
            ):
                return None
    except Exception as exc:
        if not is_recoverable_image_error(exc):
            raise
        return None
    return data


def _publish_image_cache_atomic(
    output_path: Path,
    data: bytes,
) -> None:
    """Publish converted bytes via a unique temp file and atomic replace.

    Concurrent readers never observe a partially written cache file. Cache
    publication failures (e.g. the cache directory vanished mid-write) are
    non-fatal: callers already hold the complete encoded bytes.

    Args:
        output_path: Final cache entry path.
        data: Complete encoded bytes to publish.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    except OSError as exc:
        if not is_recoverable_image_error(exc):
            raise
        logger.debug("Failed to create image cache temp file: %s", exc)
        return
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(data)
        os.replace(tmp_path, output_path)
    except OSError as exc:
        if not is_recoverable_image_error(exc):
            raise
        logger.debug("Failed to publish image cache %s: %s", output_path, exc)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as exc:
            if not is_recoverable_image_error(exc):
                raise
            logger.debug("Failed to clean up image cache temp file: %s", exc)


def _convert_image_bytes_sync(
    source_bytes: bytes, max_size: int, quality: int
) -> bytes:
    """Normalize a validated still image with an optional derived cache.

    Args:
        source_bytes: Encoded source bytes already checked by _inspect_image.
        max_size: Longest-edge limit in pixels.
        quality: JPEG output quality in the range 1-100.

    Returns:
        Single-frame JPEG or PNG bytes. An oriented JPEG or PNG within the size
        and reuse-byte limits is reused unchanged; anything else is re-encoded.
    """
    with PILImage.open(io.BytesIO(source_bytes)) as image:
        if (
            image.format in {"PNG", "JPEG"}
            and image.getexif().get(274, 1) == 1
            and max(image.size) <= max_size
            and len(source_bytes) <= MODEL_IMAGE_REUSE_MAX_BYTES
        ):
            return source_bytes
        cache_key = _image_convert_cache_key(
            source_bytes,
            f"{_IMAGE_CONVERT_CACHE_VERSION}|convert|s={max_size}|q={quality}",
        )
        output_path = _image_convert_cache_dir() / (cache_key + ".img")
        cached = _read_valid_cached_image_bytes(output_path)
        if cached is not None:
            return cached
        encoded = _encode_image_frame_bytes(image, max_size=max_size, quality=quality)
    _publish_image_cache_atomic(output_path, encoded)
    return encoded


def _even_frame_indices(total_frames: int, max_frames: int) -> list[int]:
    """Pick evenly spaced animation frames, including both endpoints.

    Args:
        total_frames: Number of animation frames, excluding an independent cover.
        max_frames: Maximum number of frames to select.

    Returns:
        Ascending, unique frame indices.
    """
    count = min(max_frames, total_frames)
    if count <= 1:
        return [0]
    return sorted({round(i * (total_frames - 1) / (count - 1)) for i in range(count)})


def _extract_animation_montage_sync(
    source_bytes: bytes,
    max_size: int,
    quality: int,
) -> tuple[bytes, bool]:
    """Tile evenly spaced frames of an animated image into one grid montage.

    Frame sampling includes the first and last frames; grid cells beyond the
    available frames stay blank (white). The montage is flattened onto a white
    background and its longest edge is capped at ``max_size`` (never upscaled).

    Args:
        source_bytes: Encoded animated image bytes.
        max_size: Longest edge of the montage in pixels.

    Returns:
        Tuple of the montage bytes and whether extraction actually ran
        (``False`` when a valid cached montage was served).
    """
    cache_key = _image_convert_cache_key(
        source_bytes,
        f"{_IMAGE_CONVERT_CACHE_VERSION}|montage|s={max_size}|q={quality}",
    )
    output_path = _image_convert_cache_dir() / (cache_key + ".img")
    cached = _read_valid_cached_image_bytes(output_path)
    if cached is not None:
        return cached, False
    with PILImage.open(io.BytesIO(source_bytes)) as image:
        total_frames = getattr(image, "n_frames", 1)
        # APNG's independent default image is a cover, not an animation frame.
        first_frame = 1 if image.info.get("default_image", False) else 0
        frame_indices = [
            first_frame + index
            for index in _even_frame_indices(
                total_frames - first_frame, ANIMATED_MONTAGE_FRAME_COUNT
            )
        ]
        image.seek(first_frame)
        with ImageOps.exif_transpose(image) as oriented:
            display_size = oriented.size
        # Floor the per-cell scale so the montage never exceeds max_size.
        longest_edge = max(display_size) * ANIMATED_MONTAGE_GRID
        scale = min(1.0, max(max_size, 1) / longest_edge)
        cell_size = (
            max(1, int(display_size[0] * scale)),
            max(1, int(display_size[1] * scale)),
        )
        canvas = PILImage.new(
            "RGB",
            (
                cell_size[0] * ANIMATED_MONTAGE_GRID,
                cell_size[1] * ANIMATED_MONTAGE_GRID,
            ),
            (255, 255, 255),
        )
        try:
            for out_index, frame_index in enumerate(frame_indices):
                image.seek(frame_index)
                with (
                    ImageOps.exif_transpose(image) as oriented,
                    oriented.convert("RGBA") as frame,
                ):
                    resized = frame
                    try:
                        if frame.size != cell_size:
                            resized = frame.resize(
                                cell_size, PILImage.Resampling.LANCZOS
                            )
                        # Alpha shows the white canvas through transparent pixels.
                        canvas.paste(
                            resized,
                            (
                                (out_index % ANIMATED_MONTAGE_GRID) * cell_size[0],
                                (out_index // ANIMATED_MONTAGE_GRID) * cell_size[1],
                            ),
                            resized,
                        )
                    finally:
                        if resized is not frame:
                            resized.close()
            encoded = _encode_image_frame_bytes(canvas, quality=quality)
        finally:
            canvas.close()
    _publish_image_cache_atomic(output_path, encoded)
    return encoded, True


async def prepare_model_image(
    image_ref: str,
    *,
    max_size: int,
    output_dir: Path,
    quality: int = IMAGE_COMPRESS_DEFAULT_QUALITY,
    montage_max_size: int | None = None,
) -> str | None:
    """Prepare a single local model-ready image for the caller to own until consumption.

    Args:
        image_ref: Source reference accepted by MediaResolver.
        max_size: Validated longest-edge limit for stills.
        output_dir: Directory for independent request-owned working files.
        quality: JPEG output quality in the range 1-100.
        montage_max_size: Optional longest-edge limit for animation montages.
            CUA sessions lift the still-image cap to keep pixel coordinates 1:1,
            but montages are never used for coordinates, so callers pass the
            configured limit here to keep the 3x3 canvas bounded. Defaults to
            ``max_size``.

    Returns:
        An existing JPEG or PNG path, or None for a recoverable input or write
        failure.
        The caller owns this file; shared cache entries are never returned.
    """
    try:
        async with MediaResolver(image_ref, media_type="image").as_path() as source:
            input_size = source.path.stat().st_size
            if input_size > MODEL_IMAGE_MAX_INPUT_BYTES:
                logger.warning(
                    "Skipping oversized image input (%d bytes): %s",
                    input_size,
                    source.path,
                )
                return None
            image_bytes = await asyncio.to_thread(source.read_bytes)
        frame_count = await asyncio.to_thread(_inspect_image, image_bytes)
        if frame_count > 1:
            converted_bytes, _ = await asyncio.to_thread(
                _extract_animation_montage_sync,
                image_bytes,
                montage_max_size if montage_max_size is not None else max_size,
                quality,
            )
        else:
            converted_bytes = await asyncio.to_thread(
                _convert_image_bytes_sync, image_bytes, max_size, quality
            )
        # Publish the working file synchronously after encoding, so cancellation
        # cannot leave an untracked background write alive after this call.
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".jpg" if converted_bytes.startswith(b"\xff\xd8") else ".png"
        fd, name = tempfile.mkstemp(
            prefix="model_image_", suffix=suffix, dir=output_dir
        )
        output_path = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(converted_bytes)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
        return str(output_path)
    except Exception as exc:
        if not is_recoverable_image_error(exc):
            raise
        logger.warning(
            "Model image preparation failed; skipping image (%s).", type(exc).__name__
        )
        return None


async def resolve_audio_ref_to_base64_data(
    audio_ref: MediaRefStr,
    *,
    preserve_mp3: bool = False,
    target_format: str | None = None,
) -> ResolvedMediaData:
    """Resolve an audio reference to base64 data.

    Audio is converted to WAV by default. Pass preserve_mp3=True for legacy
    provider payloads that intentionally keep MP3 input unchanged.
    ``target_format`` overrides both defaults when provided.
    """

    audio_data = await MediaResolver(
        audio_ref,
        media_type="audio",
        default_suffix=".wav",
    ).to_base64_data(
        target_format=target_format,
        preserve_mp3=preserve_mp3,
        strict=True,
    )
    if audio_data is None:
        raise ValueError(f"Invalid audio data: {describe_media_ref(audio_ref)}")
    return audio_data


async def resolve_media_ref_to_base64_data(
    media_ref: MediaRefStr | bytes,
    *,
    media_type: str,
    strict: bool = False,
    image_options: ImagePreparationOptions | None = None,
    default_mime_type: str | None = "image/jpeg",
) -> ResolvedMediaData | None:
    """Resolve a media reference to base64 data through one shared entrypoint.

    This helper keeps provider sources from knowing whether a reference is local,
    HTTP(S), ``base64://``, a data URI, or a legacy bare base64 payload.
    """

    if media_type == "image":
        return await resolve_image_ref_to_base64_data(
            media_ref,
            strict=strict,
            default_mime_type=default_mime_type,
            options=image_options,
        )
    if media_type == "audio":
        return await resolve_audio_ref_to_base64_data(media_ref)

    return await MediaResolver(
        media_ref,
        media_type=media_type,
    ).to_base64_data(
        strict=strict,
    )


async def get_media_duration(file_path: str) -> int | None:
    """Probe media duration with ffprobe.

    Args:
        file_path: Local media file path.

    Returns:
        Duration in milliseconds, or ``None`` when probing fails.
    """
    try:
        # Probe duration with ffprobe.
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0 and stdout:
            duration_seconds = float(stdout.decode().strip())
            duration_ms = int(duration_seconds * 1000)
            logger.debug("Media duration detected: %sms", duration_ms)
            return duration_ms
        else:
            logger.warning("Failed to get media duration: %s", file_path)
            return None

    except FileNotFoundError:
        logger.warning(
            "ffprobe is not installed or not in PATH. "
            "Install ffmpeg: https://ffmpeg.org/"
        )
        return None
    except Exception as e:
        logger.warning("Error while probing media duration: %s", e)
        return None


async def convert_audio_to_opus(audio_path: str, output_path: str | None = None) -> str:
    """Convert an audio file to Opus format.

    Args:
        audio_path: Source audio file path.
        output_path: Optional output file path. When omitted, a temporary path is
            created under AstrBot's temp directory.

    Returns:
        The converted Opus file path.
    """
    return await convert_audio_format(
        audio_path=audio_path,
        output_format="opus",
        output_path=output_path,
    )


async def convert_video_format(
    video_path: str, output_format: str = "mp4", output_path: str | None = None
) -> str:
    """Convert a video file with ffmpeg.

    Args:
        video_path: Source video file path.
        output_format: Target format, such as ``mp4``.
        output_path: Optional output file path. When omitted, a temporary path is
            created under AstrBot's temp directory.

    Returns:
        The converted video file path.

    Raises:
        Exception: Raised when ffmpeg is unavailable or conversion fails.
    """
    # Return early when the source already appears to be in the target format.
    if video_path.lower().endswith(f".{output_format}"):
        return video_path

    # Create an output path when the caller does not provide one.
    if output_path is None:
        temp_dir = get_astrbot_temp_path()
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(
            temp_dir,
            f"media_video_{generate_timestamp_id()}.{output_format}",
        )

    try:
        # Convert the video with ffmpeg.
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # Remove a partial output file created by a failed ffmpeg run.
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    logger.debug(
                        "Removed failed %s output file: %s",
                        output_format,
                        output_path,
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to remove failed %s output file: %s",
                        output_format,
                        e,
                    )

            error_msg = stderr.decode() if stderr else "unknown error"
            logger.error("ffmpeg video conversion failed: %s", error_msg)
            raise Exception(f"ffmpeg conversion failed: {error_msg}")

        logger.debug(
            "Video converted successfully: %s -> %s",
            video_path,
            output_path,
        )
        return output_path

    except FileNotFoundError:
        logger.error(
            "ffmpeg is not installed or not in PATH. "
            "Install ffmpeg: https://ffmpeg.org/"
        )
        raise Exception("ffmpeg not found")
    except Exception as e:
        logger.error("Error while converting video format: %s", e)
        raise


async def convert_audio_format(
    audio_path: str,
    output_format: str = "amr",
    output_path: str | None = None,
) -> str:
    """Convert an audio file to the requested format with ffmpeg.

    Args:
        audio_path: Source audio file path.
        output_format: Target format, such as ``amr``, ``ogg``, ``opus``, or
            ``wav``.
        output_path: Optional output file path. When omitted, a temporary path is
            created under AstrBot's temp directory.

    Returns:
        The converted audio file path.

    Raises:
        Exception: Raised when ffmpeg is unavailable or conversion fails.
    """
    source_path = Path(audio_path)
    if source_path.suffix.lower() == f".{output_format}" and (
        not source_path.exists() or _get_audio_magic_type(audio_path) == output_format
    ):
        return audio_path

    if output_path is None:
        temp_dir = Path(get_astrbot_temp_path())
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            temp_dir / f"media_audio_{generate_timestamp_id()}.{output_format}"
        )

    args = ["ffmpeg", "-y", "-i", audio_path]
    if output_format == "amr":
        args.extend(
            [
                "-ac",
                "1",
                "-ar",
                "8000",
                "-ab",
                "12.2k",
                "-af",
                (
                    "highpass=f=310:poles=2,"
                    "lowpass=f=3720:poles=2,"
                    "equalizer=f=3150:width_type=h:width=1000:g=7.5,"
                    "loudnorm=I=-18.5:TP=-1.5:LRA=6,"
                    "aresample=8000"
                ),
            ]
        )
    elif output_format == "ogg":
        args.extend(["-acodec", "libopus", "-ac", "1", "-ar", "16000"])
    elif output_format == "opus":
        args.extend(["-acodec", "libopus", "-ac", "1", "-ar", "16000"])
    args.append(output_path)

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError as e:
                    logger.warning(
                        "Failed to remove failed audio output file: %s",
                        e,
                    )
            error_msg = stderr.decode() if stderr else "unknown error"
            raise Exception(f"ffmpeg conversion failed: {error_msg}")
        logger.debug(
            "Audio converted successfully: %s -> %s",
            audio_path,
            output_path,
        )
        return output_path
    except FileNotFoundError:
        raise Exception("ffmpeg not found")


async def convert_audio_to_amr(audio_path: str, output_path: str | None = None) -> str:
    """Convert an audio file to AMR format.

    Args:
        audio_path: Source audio file path.
        output_path: Optional output file path. When omitted, a temporary path is
            created under AstrBot's temp directory.

    Returns:
        The converted AMR file path.
    """
    return await convert_audio_format(
        audio_path=audio_path,
        output_format="amr",
        output_path=output_path,
    )


async def convert_audio_to_wav(audio_path: str, output_path: str | None = None) -> str:
    """Convert an audio file to WAV format.

    Args:
        audio_path: Source audio file path.
        output_path: Optional output file path. When omitted, a temporary path is
            created under AstrBot's temp directory.

    Returns:
        The converted WAV file path.
    """
    return await convert_audio_format(
        audio_path=audio_path,
        output_format="wav",
        output_path=output_path,
    )


async def ensure_wav(audio_path: str, output_path: str | None = None) -> str:
    """Ensure the audio path points to wav format by extension/guess and convert when needed.

    If the file appears to already be WAV, return it directly to avoid extra
    conversion. If the file does not exist yet, return the original path so
    upstream retry logic can handle platform races.

    Args:
        audio_path: Local audio path to inspect and convert when needed.
        output_path: Optional destination path. When omitted, conversion helpers
            create a temporary file under AstrBot's temp directory.

    Returns:
        The original path when it is already WAV or unavailable; otherwise the
        converted WAV path.

    Raises:
        Exception: Raised by the underlying conversion helper when conversion
            fails.
    """

    if not audio_path:
        return audio_path

    if not os.path.exists(audio_path):
        # File not available yet (e.g. napcat race condition);
        # return the path as-is so upstream retry logic can handle it later.
        return audio_path

    audio_type = _get_audio_magic_type(audio_path)
    if audio_type == "wav":
        return audio_path

    if audio_type == "silk":
        if output_path is None:
            temp_dir = get_astrbot_temp_path()
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(
                temp_dir, f"media_audio_{generate_timestamp_id()}.wav"
            )
        return await tencent_silk_to_wav(audio_path, output_path)

    return await convert_audio_to_wav(audio_path, output_path)


async def ensure_jpeg(image_path: str, output_path: str | None = None) -> str:
    """Ensure JPEG-compatible still images point to a JPEG file.

    Args:
        image_path: Local image path to inspect and convert when needed.
        output_path: Optional destination path. When omitted, a temporary file under
            AstrBot's temp directory is created for converted JPEG output.

    Returns:
        The original path when the source is already a JPEG file with a jpg/jpeg
        suffix, cannot be found, has alpha transparency, or is animated. JPEG
        files with another suffix are copied without re-encoding; other still
        images are converted to JPEG.

    Raises:
        Exception: Raised by Pillow when the source file cannot be opened or saved as
            an image.
    """

    if not image_path:
        return image_path

    source_path = Path(image_path)
    if not source_path.exists():
        return image_path

    with PILImage.open(source_path) as opened_img:
        image_format = str(opened_img.format or "").upper()
        image_has_alpha = opened_img.mode in {"RGBA", "LA"} or (
            opened_img.mode == "P" and "transparency" in opened_img.info
        )
        image_is_animated = (
            getattr(opened_img, "is_animated", False)
            or getattr(
                opened_img,
                "n_frames",
                1,
            )
            > 1
        )

    if image_format == "JPEG" and source_path.suffix.lower() in {".jpg", ".jpeg"}:
        return image_path

    if image_has_alpha or image_is_animated:
        return image_path

    if output_path is None:
        temp_dir = Path(get_astrbot_temp_path())
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(temp_dir / f"media_image_{generate_timestamp_id()}.jpg")
    jpeg_output_path = output_path

    try:
        if image_format == "JPEG":
            await asyncio.to_thread(shutil.copyfile, source_path, jpeg_output_path)
            return jpeg_output_path
    except Exception:
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logger.warning(
                    "Failed to remove failed image output file: %s",
                    e,
                )
        raise

    def convert_image_to_jpeg() -> str:
        converted_img: PILImage.Image | None = None

        with PILImage.open(image_path) as opened_img:
            try:
                working_img: PILImage.Image = opened_img
                if opened_img.mode != "RGB":
                    converted_img = opened_img.convert("RGB")
                    working_img = converted_img

                working_img.save(
                    jpeg_output_path,
                    "JPEG",
                    quality=IMAGE_COMPRESS_DEFAULT_QUALITY,
                    subsampling=0,
                )
                return jpeg_output_path
            finally:
                if converted_img is not None:
                    converted_img.close()

    try:
        return await asyncio.to_thread(convert_image_to_jpeg)
    except Exception:
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logger.warning(
                    "Failed to remove failed image output file: %s",
                    e,
                )
        raise


def _get_audio_magic_type(audio_path: str) -> str:
    """Detect common audio formats from magic bytes.

    Args:
        audio_path: Local audio path to inspect.

    Returns:
        A normalized format name such as ``wav``, ``mp3``, ``opus``, ``silk``, or
        an empty string when the type cannot be detected.
    """
    try:
        with open(audio_path, "rb") as f:
            header = f.read(64)
    except FileNotFoundError:
        logger.warning("WAV probe file not found: %s", audio_path)
        return ""
    except Exception as e:
        logger.warning(
            "WAV probe failed: %s, error: %s",
            audio_path,
            e,
        )
        return ""

    if len(header) < 12:
        return ""

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"

    if header[:4] == b"#!AM":
        return "amr"

    if header[:4] == b"OggS":
        if b"OpusHead" in header:
            return "opus"
        return "ogg"

    if header[:3] == b"fLa":
        return "flac"

    if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
        return "mp3"

    if header[:4] == b"ftyp" and b"mp4" in header[:8]:
        return "mp4"

    if header.startswith(b"#!SILK_V3"):
        return "silk"

    # Tencent SILK: leading \x02 byte before #!SILK_V3
    if header.startswith(b"\x02#!SILK_V3"):
        return "silk"

    return ""


async def extract_video_cover(
    video_path: str,
    output_path: str | None = None,
) -> str:
    """Extract a JPEG cover frame from a video.

    Args:
        video_path: Source video file path.
        output_path: Optional output image path. When omitted, a temporary JPEG
            path is created under AstrBot's temp directory.

    Returns:
        The extracted JPEG cover path.

    Raises:
        Exception: Raised when ffmpeg is unavailable or cover extraction fails.
    """
    if output_path is None:
        temp_dir = Path(get_astrbot_temp_path())
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(temp_dir / f"media_cover_{generate_timestamp_id()}.jpg")

    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            "00:00:00",
            "-frames:v",
            "1",
            output_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError as e:
                    logger.warning(
                        "Failed to remove failed video cover file: %s",
                        e,
                    )
            error_msg = stderr.decode() if stderr else "unknown error"
            raise Exception(f"ffmpeg extract cover failed: {error_msg}")
        return output_path
    except FileNotFoundError:
        raise Exception("ffmpeg not found")


def _resize_alpha_in_strips(
    source: PILImage.Image, size: tuple[int, int]
) -> PILImage.Image:
    """Resize transparency with bounded premultiplication buffers.

    Pillow premultiplies a complete RGBA image before filtering. Processing
    overlapping strips keeps those temporary buffers proportional to width,
    while retaining the Lanczos support around each output strip.

    Args:
        source: Caller-owned image with an alpha channel.
        size: Output dimensions.

    Returns:
        A caller-owned resized image.

    Raises:
        OSError: Pixel decoding or resizing fails.
        MemoryError: Output or temporary pixel allocation fails.
    """
    output = PILImage.new(source.mode, size)
    try:
        ratio = source.height / size[1]
        halo = math.ceil(3 * max(1, ratio)) + 1
        # Align strip boundaries with source rows whenever the rational scale
        # permits it, avoiding floating-point phase changes at those boundaries.
        alignment = size[1] // math.gcd(source.height, size[1])
        strip_height = max(alignment, 32 // alignment * alignment)
        if strip_height > 64:
            strip_height = 32
        for top in range(0, size[1], strip_height):
            bottom = min(top + strip_height, size[1])
            start = max(0, math.floor(top * ratio) - halo)
            end = min(source.height, math.ceil(bottom * ratio) + halo)
            with source.crop((0, start, source.width, end)) as strip:
                with strip.resize(
                    (size[0], bottom - top),
                    PILImage.Resampling.LANCZOS,
                    box=(0, top * ratio - start, source.width, bottom * ratio - start),
                ) as resized:
                    output.paste(resized, (0, top))
        return output
    except BaseException:
        output.close()
        raise


def _compress_image_sync(
    source: bytes | Path,
    temp_dir: Path,
    max_size: int,
    quality: int,
    optimize: bool,
    max_encoded_bytes: int = IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES,
    *,
    preserve_dimensions: bool = False,
) -> str | None:
    """Prepare one image without allocating base64 for candidate measurements.

    Args:
        source: Encoded bytes or a local file opened inside the worker.
        temp_dir: Directory for the caller-owned prepared file.
        max_size: Maximum edge length when resizing is allowed.
        quality: Initial JPEG quality, between 1 and 100.
        optimize: Whether to optimize the encoder output.
        max_encoded_bytes: Maximum base64 payload size, excluding its URI header.
        preserve_dimensions: Preserve screenshot coordinates, even over max_size.

    Returns:
        A caller-owned output path, or None to preserve the original bytes.

    Raises:
        ImagePayloadTooLargeError: No candidate meets the encoded-byte budget.
        ValueError: A size or quality option is invalid.
        OSError: Reading, decoding or writing the image fails.
    """
    if max_size < 1 or max_encoded_bytes < 1 or not 1 <= quality <= 100:
        raise ValueError("Image dimensions, byte budget and quality must be positive")
    source_bytes = len(source) if isinstance(source, bytes) else source.stat().st_size
    encoded_size = 4 * ((source_bytes + 2) // 3)
    direct_webp = None
    if isinstance(source, bytes):
        is_webp = source[:4] == b"RIFF" and source[8:12] == b"WEBP"
    else:
        with source.open("rb") as source_stream:
            header = source_stream.read(12)
        is_webp = (
            len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
        )
    if is_webp:
        direct_webp = _open_static_webp(
            source,
            max_size,
            max_encoded_bytes,
            preserve_dimensions=preserve_dimensions,
        )
        if direct_webp is _WEBP_PRESERVE:
            return None
    fp = io.BytesIO(source) if isinstance(source, bytes) else source
    opened = direct_webp if direct_webp is not None else PILImage.open(fp)
    with opened:
        animated = getattr(opened, "n_frames", 1) > 1
        # This baseline preserves animation; do not flatten it to fit a budget.
        if animated:
            if encoded_size > max_encoded_bytes:
                raise ImagePayloadTooLargeError(
                    f"Animated image exceeds the {max_encoded_bytes}-byte encoding limit"
                )
            return None
        if encoded_size <= max_encoded_bytes and (
            preserve_dimensions or max(opened.size) <= max_size
        ):
            return None

        temp_dir.mkdir(parents=True, exist_ok=True)
        path: Path | None = None
        best_size: int | None = None
        success = False
        # Resize before EXIF handling loads the pixels. JPEG thumbnailing can
        # then use decoder-level downsampling instead of a full-size RGB buffer.
        # A square bound is unchanged by EXIF rotations and reflections.
        if (
            opened.format == "JPEG"
            and not preserve_dimensions
            and max(opened.size) > max_size
        ):
            opened.thumbnail(
                (max_size, max_size), PILImage.Resampling.LANCZOS, reducing_gap=1.0
            )
        # In-place orientation avoids holding a second full oriented image.
        ImageOps.exif_transpose(opened, in_place=True)
        has_alpha = opened.mode in {"RGBA", "LA"} or (
            opened.mode == "P" and "transparency" in opened.info
        )
        target_mode = "RGBA" if has_alpha else "RGB"
        converted = opened.convert(target_mode) if opened.mode != target_mode else None
        working = converted if converted is not None else opened
        try:
            if not preserve_dimensions and max(working.size) > max_size:
                working.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            qualities = (
                [quality]
                if has_alpha
                else sorted(
                    {
                        quality,
                        *[value for value in (85, 70, 55, 40) if value < quality],
                    },
                    reverse=True,
                )
            )
            source_format = str(opened.format or "").upper()
            # If the encoded source is already more than twice the budget, a
            # same-size candidate cannot be a useful memory-saving first step.
            # Resize once before encoding so the source pixels and encoder
            # buffers are not resident together at the original dimensions.
            skip_current_candidate = (
                not preserve_dimensions
                and encoded_size > max_encoded_bytes * 2
                and max(working.size) >= max_size
            )
            while True:
                if not skip_current_candidate:
                    if has_alpha:
                        formats = [("PNG", ".png", None)]
                    else:
                        formats = (
                            [("PNG", ".png", None)] if source_format == "PNG" else []
                        ) + [
                            ("JPEG", ".jpg", candidate_quality)
                            for candidate_quality in qualities
                        ]
                    for output_format, suffix, candidate_quality in formats:
                        candidate_path: Path | None = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                dir=temp_dir,
                                prefix="compressed_",
                                suffix=suffix,
                                delete=False,
                            ) as output:
                                candidate_path = Path(output.name)
                            kwargs = {"optimize": optimize}
                            if candidate_quality is not None:
                                kwargs["quality"] = candidate_quality
                            working.save(candidate_path, output_format, **kwargs)
                            candidate_size = candidate_path.stat().st_size
                            encoded_size = 4 * ((candidate_size + 2) // 3)
                            if encoded_size <= max_encoded_bytes and (
                                best_size is None or candidate_size < best_size
                            ):
                                if path is not None:
                                    path.unlink(missing_ok=True)
                                path = candidate_path
                                best_size = candidate_size
                                candidate_path = None
                        finally:
                            if candidate_path is not None:
                                candidate_path.unlink(missing_ok=True)
                        # Compare a lossless PNG with the highest fitting JPEG
                        # quality; do not lower quality merely to minimize bytes.
                        if output_format == "JPEG" and path is not None:
                            break
                skip_current_candidate = False
                if path is not None:
                    success = True
                    return str(path)
                if preserve_dimensions or working.size == (1, 1):
                    raise ImagePayloadTooLargeError(
                        f"Image cannot fit the {max_encoded_bytes}-byte encoding limit"
                    )
                # One current candidate, replaced on disk; never retain all encodings.
                target = (
                    max(1, working.width * 3 // 4),
                    max(1, working.height * 3 // 4),
                )
                if has_alpha:
                    # Match thumbnail's aspect-ratio rounding. Rounding each
                    # edge independently can stretch narrow or odd-sized images.
                    width, height = target
                    aspect = working.width / working.height
                    if width / height >= aspect:
                        width = max(
                            min(
                                math.floor(height * aspect),
                                math.ceil(height * aspect),
                                key=lambda value: abs(aspect - value / height),
                            ),
                            1,
                        )
                    else:
                        height = max(
                            min(
                                math.floor(width / aspect),
                                math.ceil(width / aspect),
                                key=lambda value: (
                                    0 if value == 0 else abs(aspect - width / value)
                                ),
                            ),
                            1,
                        )
                    target = (width, height)
                    resized = _resize_alpha_in_strips(working, target)
                    working.close()
                    working = converted = resized
                else:
                    working.thumbnail(target, PILImage.Resampling.LANCZOS)
        finally:
            if converted is not None:
                converted.close()
            if not success and path is not None:
                path.unlink(missing_ok=True)


async def compress_image(
    url_or_path: str,
    max_size: int = IMAGE_COMPRESS_DEFAULT_MAX_SIZE,
    quality: int = IMAGE_COMPRESS_DEFAULT_QUALITY,
    max_encoded_bytes: int = IMAGE_COMPRESS_DEFAULT_MAX_ENCODED_BYTES,
    *,
    optimize: bool = IMAGE_COMPRESS_DEFAULT_OPTIMIZE,
    preserve_dimensions: bool = False,
) -> str:
    """Prepare a local image, preserving compliant bytes.

    Args:
        url_or_path: Local path or inline image; remote URLs remain unresolved.
        max_size: Maximum edge length when resizing is allowed.
        quality: Initial JPEG quality.
        max_encoded_bytes: Maximum base64 payload size.
        preserve_dimensions: Preserve oriented screenshot dimensions.

    Returns:
        The original reference or a caller-owned prepared file path.

    Raises:
        ImagePayloadTooLargeError: The image cannot meet the byte limit.
        OSError: Image decoding or filesystem access fails.
    """
    if url_or_path.startswith(("http://", "https://")):
        return url_or_path
    if url_or_path.startswith("data:image"):
        _, encoded = url_or_path.split(",", 1)
        image_source: bytes | Path = _decode_base64_payload(
            encoded, error_message="invalid image data URI payload"
        )
    else:
        image_source = Path(url_or_path)
        if not image_source.exists():
            return url_or_path

    worker = asyncio.create_task(
        asyncio.to_thread(
            _compress_image_sync,
            image_source,
            Path(get_astrbot_temp_path()),
            max(int(max_size), 1),
            min(max(int(quality), 1), 100),
            optimize,
            max(int(max_encoded_bytes), 1),
            preserve_dimensions=preserve_dimensions,
        )
    )
    try:
        compressed_path = await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Cancellation cannot stop Pillow in a thread. Retain ownership until
        # the worker finishes, then release its output without deleting inputs.
        def cleanup_finished(done: asyncio.Task) -> None:
            try:
                output = done.result()
                if output is not None:
                    Path(output).unlink(missing_ok=True)
            except Exception:
                logger.warning("Cancelled image preparation cleanup failed")

        worker.add_done_callback(cleanup_finished)
        raise
    return compressed_path or url_or_path


async def prepare_image_source(
    image_ref: MediaRefStr | bytes | ImagePreparationInput,
    *,
    options: ImagePreparationOptions | None = None,
    default_mime_type: str | None = "image/jpeg",
) -> ResolvedMediaData:
    """Resolve and prepare any image reference for a provider request.

    Args:
        image_ref: Local path, HTTP(S) URL, data URI, base64 reference, bare
            base64 payload, or an ``ImagePreparationInput`` descriptor.
        options: Optional preparation limits. ``None`` uses the standard budget.
        default_mime_type: Fallback MIME type for otherwise unidentified images.

    Returns:
        Provider-ready base64 data and its detected MIME type.

    Raises:
        ImagePayloadTooLargeError: The image cannot fit the configured budget.
        OSError: The source cannot be read or decoded.
        ValueError: The source is not a valid image.
    """
    selected = options or ImagePreparationOptions()
    preparation_input = (
        image_ref
        if isinstance(image_ref, ImagePreparationInput)
        else ImagePreparationInput(image_ref)
    )
    source_ref = preparation_input.value

    async def _prepare() -> ResolvedMediaData:
        owned_source: Path | None = None
        try:
            resolved_source = source_ref
            if isinstance(source_ref, bytes):
                owned_source = _temp_media_path("image", ".bin")
                await asyncio.to_thread(owned_source.write_bytes, source_ref)
                resolved_source = str(owned_source)
            async with MediaResolver(
                resolved_source, media_type="image", default_suffix=".bin"
            ).as_path() as resolved:
                if not selected.enabled:
                    source_size = resolved.path.stat().st_size
                    if (
                        selected.max_encoded_bytes is not None
                        and 4 * ((source_size + 2) // 3) > selected.max_encoded_bytes
                    ):
                        raise ImagePayloadTooLargeError(
                            f"Image exceeds the {selected.max_encoded_bytes}-byte encoding limit"
                        )
                    mime_type = await detect_image_mime_type_async(
                        resolved.path, default_mime_type=None
                    )
                    if not mime_type:
                        raise ValueError(
                            f"Invalid image file: {describe_media_ref(resolved_source)}"
                        )
                    image_size = source_size
                    return ResolvedMediaData(
                        base64_data=await asyncio.to_thread(
                            _encode_file_to_base64, resolved.path
                        ),
                        mime_type=mime_type,
                        byte_size=image_size,
                    )
                try:
                    prepared_path = await compress_image(
                        str(resolved.path),
                        max_size=selected.max_size,
                        quality=selected.quality,
                        max_encoded_bytes=(
                            selected.max_encoded_bytes
                            if selected.max_encoded_bytes is not None
                            else 2**63 - 1
                        ),
                        optimize=selected.optimize,
                        preserve_dimensions=selected.preserve_dimensions,
                    )
                except UnidentifiedImageError as exc:
                    raise ValueError(
                        f"Invalid image file: {describe_media_ref(source_ref)}"
                    ) from exc
                output_path = Path(prepared_path)
                try:
                    mime_type = await detect_image_mime_type_async(
                        output_path, default_mime_type=None
                    )
                    image_size = output_path.stat().st_size
                    encoded_data = await asyncio.to_thread(
                        _encode_file_to_base64, output_path
                    )
                finally:
                    if output_path != resolved.path:
                        output_path.unlink(missing_ok=True)
                if not mime_type:
                    mime_type = resolved.mime_type or default_mime_type
                if not mime_type:
                    raise ValueError(
                        f"Invalid image file: {describe_media_ref(resolved_source)}"
                    )
                return ResolvedMediaData(
                    base64_data=encoded_data,
                    mime_type=mime_type,
                    byte_size=image_size,
                )
        finally:
            if owned_source is not None:
                owned_source.unlink(missing_ok=True)
            for cleanup_path in preparation_input.cleanup_paths:
                cleanup_path.unlink(missing_ok=True)

    worker = asyncio.create_task(_prepare())
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # The worker owns the resolver context until Pillow and file reads exit.
        worker.add_done_callback(
            lambda done: done.exception() if not done.cancelled() else None
        )
        raise

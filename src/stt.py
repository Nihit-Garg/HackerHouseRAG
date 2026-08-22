"""
stt.py — Sarvam AI Speech-to-Text wrapper.

Endpoint: POST https://api.sarvam.ai/speech-to-text
Auth:     api-subscription-key header (SARVAM_API_KEY env var)

# TODO: verify exact request/response schema against current Sarvam docs
# Docs: https://docs.sarvam.ai/api-reference-docs/endpoints/stt
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from config import (
    SARVAM_API_KEY,
    SARVAM_LANGUAGE_CODE,
    SARVAM_MODEL,
    SARVAM_STT_URL,
    STT_MAX_RETRIES,
    STT_RETRY_BASE_DELAY_S,
)

logger = logging.getLogger(__name__)

# Supported audio formats (Sarvam accepts wav, mp3, ogg, flac, m4a)
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}


class STTError(Exception):
    """Raised when STT fails after all retries."""


def _build_headers() -> dict[str, str]:
    if not SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is not set. Export it as an environment variable.")
    return {"api-subscription-key": SARVAM_API_KEY}  # TODO: verify header name in Sarvam docs


def _validate_audio_path(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    return path


def _call_sarvam_api(audio_path: Path) -> str:
    """
    Make a single HTTP call to the Sarvam STT endpoint.

    # TODO: verify exact multipart field names against current Sarvam docs
    Returns:
        Transcribed text string.

    Raises:
        requests.HTTPError: On non-2xx response.
        STTError: On empty transcription.
    """
    headers = _build_headers()

    with open(audio_path, "rb") as audio_file:
        mime_type = _MIME_TYPES.get(audio_path.suffix.lower(), "audio/wav")
        files = {"file": (audio_path.name, audio_file, mime_type)}
        data = {
            "model": SARVAM_MODEL,
            "language_code": SARVAM_LANGUAGE_CODE,
        }
        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )

    response.raise_for_status()
    payload = response.json()

    # TODO: verify response field name ('transcript' vs 'text') against Sarvam docs
    transcript: str = payload.get("transcript") or payload.get("text") or ""
    transcript = transcript.strip()

    if not transcript:
        raise STTError("Sarvam API returned an empty transcription.")

    return transcript


def transcribe(audio_path: str | Path) -> str:
    """
    Transcribe an audio file using Sarvam AI STT with exponential backoff retries.

    Args:
        audio_path: Path to the audio file (wav, mp3, ogg, flac, m4a, webm).

    Returns:
        Transcribed text string.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: If audio format is unsupported.
        STTError: If transcription fails after all retries.
    """
    path = _validate_audio_path(audio_path)
    last_error: Exception | None = None

    for attempt in range(STT_MAX_RETRIES + 1):
        try:
            logger.info("STT attempt %d/%d for: %s", attempt + 1, STT_MAX_RETRIES + 1, path.name)
            transcript = _call_sarvam_api(path)
            logger.info("Transcription successful (%d chars).", len(transcript))
            return transcript

        except requests.HTTPError as exc:
            last_error = exc
            # Don't retry on client errors (4xx) — they won't resolve
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                logger.error("Client error from Sarvam API (no retry): %s", exc)
                raise STTError(f"Sarvam API client error: {exc}") from exc
            logger.warning("HTTP error on attempt %d: %s", attempt + 1, exc)

        except STTError as exc:
            last_error = exc
            logger.warning("STT error on attempt %d: %s", attempt + 1, exc)

        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Request exception on attempt %d: %s", attempt + 1, exc)

        # Exponential backoff: 1s, 2s, ...
        if attempt < STT_MAX_RETRIES:
            delay = STT_RETRY_BASE_DELAY_S * (2 ** attempt)
            logger.info("Retrying in %.1f s…", delay)
            time.sleep(delay)

    raise STTError(
        f"STT failed after {STT_MAX_RETRIES + 1} attempts. Last error: {last_error}"
    ) from last_error

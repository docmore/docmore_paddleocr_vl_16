"""Load images from the unified-parse ``source`` objects.

Two source types are supported:

* ``{"type": "base64", "data": ...}`` — base64, optionally wrapped in a
  ``data:`` URI (the format OpenAI clients send for inline images).
* ``{"type": "url", "url": ...}`` — an ``http(s)`` URL fetched by the service.

Remote fetching is opt-in per deployment (``ALLOW_URL_FETCH``) and rejects
non-public addresses unless ``ALLOW_PRIVATE_URLS`` is enabled, so the endpoint
cannot be used to probe internal networks by default. Redirects are followed
manually so every hop is re-validated.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

from PIL import Image

_ALLOWED_SCHEMES = {"http", "https"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class ImageSourceError(ValueError):
    """Raised when an image source cannot be decoded or fetched."""


class UrlFetchError(ImageSourceError):
    """Raised when a URL source is refused or cannot be fetched."""


def _open_rgb(raw: bytes) -> Image.Image:
    """Decode bytes into an RGB PIL image, normalising every failure."""
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise ImageSourceError(f"cannot decode image: {exc}") from exc


def _check_host(host: str, *, allow_private: bool) -> None:
    """Reject hosts that resolve to non-global addresses (SSRF guard)."""
    if not host:
        raise UrlFetchError("url has no host")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlFetchError(f"cannot resolve host {host!r}: {exc}") from exc
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            continue
        if not ip.is_global:
            raise UrlFetchError(
                f"refusing to fetch non-public address {address}; set "
                "PADDLEOCR_VL_16_SERVER_ALLOW_PRIVATE_URLS=1 to allow intranet URLs"
            )


def _validate_url(url: str, *, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UrlFetchError(f"unsupported URL scheme: {parsed.scheme!r}")
    _check_host(parsed.hostname or "", allow_private=allow_private)


def fetch_image_url(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    allow_private: bool = False,
    max_redirects: int = 5,
) -> bytes:
    """Download ``url`` with a size cap and per-hop SSRF validation.

    Args:
        url: absolute ``http(s)`` URL.
        max_bytes: maximum accepted body size in bytes.
        timeout: per-request timeout in seconds.
        allow_private: permit loopback / RFC1918 / link-local targets.
        max_redirects: maximum redirect hops before giving up.

    Returns:
        The response body as bytes.

    Raises:
        UrlFetchError: on a disallowed scheme/address, HTTP error, oversized
            body, or too many redirects.
    """
    import httpx

    current = url
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for _ in range(max_redirects + 1):
            _validate_url(current, allow_private=allow_private)
            try:
                with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise UrlFetchError(
                                f"HTTP {response.status_code} without Location "
                                f"fetching {current}"
                            )
                        current = urljoin(str(response.url), location)
                        continue
                    if response.status_code != 200:
                        raise UrlFetchError(
                            f"HTTP {response.status_code} fetching {current}"
                        )

                    declared = response.headers.get("content-length")
                    if declared:
                        try:
                            if int(declared) > max_bytes:
                                raise UrlFetchError(
                                    f"remote image exceeds {max_bytes} bytes"
                                )
                        except ValueError:
                            pass

                    buffer = bytearray()
                    for chunk in response.iter_bytes():
                        buffer.extend(chunk)
                        if len(buffer) > max_bytes:
                            raise UrlFetchError(
                                f"remote image exceeds {max_bytes} bytes"
                            )
            except httpx.HTTPError as exc:
                raise UrlFetchError(f"failed to fetch {current}: {exc}") from exc

            if not buffer:
                raise UrlFetchError(f"empty response body fetching {current}")
            return bytes(buffer)

    raise UrlFetchError(f"too many redirects fetching {url}")


def decode_base64_image(payload: str, *, max_bytes: int) -> Image.Image:
    """Decode a base64 payload (optionally a ``data:`` URI) into an RGB image."""
    raw_b64 = payload
    if payload.startswith("data:"):
        try:
            _, raw_b64 = payload.split(",", 1)
        except ValueError as exc:
            raise ImageSourceError("invalid data URI") from exc

    try:
        raw = base64.b64decode(raw_b64, validate=False)
    except Exception as exc:  # noqa: BLE001 - binascii raises several types
        raise ImageSourceError(f"invalid base64 image: {exc}") from exc
    if len(raw) > max_bytes:
        raise ImageSourceError(f"image exceeds {max_bytes} bytes")
    return _open_rgb(raw)


def load_image_source(
    *,
    data: str | None = None,
    url: str | None = None,
    max_bytes: int,
    timeout: float = 30.0,
    allow_private: bool = False,
    max_redirects: int = 5,
    allow_url: bool = True,
) -> Image.Image:
    """Load an image from either a base64 payload or an http(s) URL.

    ``data`` takes precedence when both are present. ``url`` also accepts a
    ``data:`` URI (OpenAI chat clients use that shape for ``image_url``).
    """
    if data:
        return decode_base64_image(data, max_bytes=max_bytes)
    if url:
        if url.startswith("data:"):
            return decode_base64_image(url, max_bytes=max_bytes)
        if not allow_url:
            raise UrlFetchError(
                "URL sources are disabled; set PADDLEOCR_VL_16_SERVER_ALLOW_URL_FETCH=1"
            )
        raw = fetch_image_url(
            url,
            max_bytes=max_bytes,
            timeout=timeout,
            allow_private=allow_private,
            max_redirects=max_redirects,
        )
        return _open_rgb(raw)
    raise ImageSourceError("image source requires base64 data or a url")
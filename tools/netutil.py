"""Safe outbound HTTP helpers with SSRF and response-size protections."""
from __future__ import annotations

import ipaddress
import os
import socket
import threading
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_MAX_BYTES = int(os.environ.get("AGENT_HTTP_MAX_BYTES", str(2 * 1024 * 1024)))
DEFAULT_TIMEOUT = float(os.environ.get("AGENT_HTTP_TIMEOUT", "12"))
USER_AGENT = os.environ.get(
    "AGENT_HTTP_USER_AGENT",
    "LocalAgent/1.0 (+https://github.com/ronasimi/agent)",
)


def _is_public_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def validate_public_url(url: str, allow_private: bool = False) -> str:
    """Validate an HTTP(S) URL and reject local/private destinations by default."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname:
        raise ValueError("URL must contain a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not allowed.")

    host = parsed.hostname
    if allow_private:
        return parsed.geturl()

    try:
        direct = ipaddress.ip_address(host)
        if not _is_public_ip(str(direct)):
            raise ValueError("Requests to private, loopback, link-local, multicast, or reserved IPs are blocked.")
    except ValueError as exc:
        if "Requests to" in str(exc):
            raise
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as dns_error:
            raise ValueError(f"Hostname could not be resolved: {dns_error}") from dns_error
        addresses = {item[4][0] for item in infos if item[4]}
        if not addresses:
            raise ValueError("Hostname did not resolve to an IP address.")
        blocked = [addr for addr in addresses if not _is_public_ip(addr)]
        if blocked:
            raise ValueError("The URL resolves to a private or otherwise local address and is blocked.")
    return parsed.geturl()


def fetch_bytes(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_types: Optional[set[str]] = None,
    max_redirects: int = 3,
    allow_private: bool = False,
) -> tuple[str, requests.Response, bytes]:
    """Fetch a URL with bounded redirects and response size."""
    timeout = max(0.1, float(timeout))
    deadline = time.monotonic() + timeout
    current = validate_public_url(url, allow_private=allow_private)
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}

    session = requests.Session()
    response = None
    deadline_fired = threading.Event()

    def abort_at_deadline() -> None:
        deadline_fired.set()
        try:
            if response is not None:
                response.close()
        except Exception:
            pass
        try:
            session.close()
        except Exception:
            pass

    deadline_timer = threading.Timer(timeout, abort_at_deadline)
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        for _ in range(max_redirects + 1):
            current = validate_public_url(current, allow_private=allow_private)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"HTTP fetch exceeded total timeout of {timeout:g}s.")
            response = session.get(
                current,
                headers=headers,
                timeout=max(0.1, remaining),
                stream=True,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    break
                current = urljoin(current, location)
                response.close()
                continue
            break

        if response is None:
            raise requests.RequestException("No HTTP response received.")
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if allowed_types and content_type not in allowed_types:
            raise ValueError(f"Unexpected content type: {content_type or 'unknown'}")

        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > max_bytes:
                    raise ValueError(f"Remote response is larger than the {max_bytes} byte limit.")
            except ValueError as exc:
                if "larger than" in str(exc):
                    raise

        chunks = []
        total = 0
        try:
            stream = response.iter_content(chunk_size=65536)
            for chunk in stream:
                if deadline_fired.is_set() or time.monotonic() > deadline:
                    raise TimeoutError(f"HTTP response body exceeded total timeout of {timeout:g}s.")
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Response exceeded the {max_bytes} byte limit.")
                chunks.append(chunk)
        except TimeoutError:
            raise
        except Exception as exc:
            if deadline_fired.is_set() or time.monotonic() > deadline:
                raise TimeoutError(f"HTTP response body exceeded total timeout of {timeout:g}s.") from exc
            raise
        if deadline_fired.is_set() or time.monotonic() > deadline:
            raise TimeoutError(f"HTTP response body exceeded total timeout of {timeout:g}s.")
        return current, response, b"".join(chunks)
    finally:
        deadline_timer.cancel()
        if response is not None:
            response.close()
        session.close()


def fetch_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_types: Optional[set[str]] = None,
    max_redirects: int = 3,
    allow_private: bool = False,
) -> tuple[str, str, str]:
    """Fetch text and return final URL, content type, and decoded body."""
    final_url, response, body = fetch_bytes(
        url,
        timeout=timeout,
        max_bytes=max_bytes,
        allowed_types=allowed_types or {
            "text/html",
            "text/plain",
            "application/xhtml+xml",
            "application/json",
            "application/xml",
            "text/xml",
        },
        max_redirects=max_redirects,
        allow_private=allow_private,
    )
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    encoding = response.encoding or "utf-8"
    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    return final_url, content_type, text

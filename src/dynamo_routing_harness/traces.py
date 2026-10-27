from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Callable

from .constants import TRACE_SOURCES, WORKLOADS

Downloader = Callable[[str], bytes]


class TraceDownloadError(RuntimeError):
    pass


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned HTTPS URL
        return response.read()


def trace_path(trace_dir: Path, workload: str) -> Path:
    return trace_dir / f"{workload}_trace.jsonl"


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _verify(workload: str, path: Path, expected_sha256: str) -> None:
    actual = _sha256(path.read_bytes())
    if actual != expected_sha256:
        raise TraceDownloadError(
            f"{workload} trace at {path} failed checksum: expected "
            f"{expected_sha256}, observed {actual}"
        )


def ensure_traces(
    trace_dir: Path,
    workloads: tuple[str, ...] = WORKLOADS,
    downloader: Downloader = _http_get,
) -> dict[str, Path]:
    """Ensure every configured Mooncake trace exists in ``trace_dir``.

    Existing files that already match their pinned checksum are left untouched.
    Missing traces are downloaded from their configured source and verified.
    Workloads without a configured source are left to the caller to provide.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for workload in workloads:
        path = trace_path(trace_dir, workload)
        source = TRACE_SOURCES.get(workload)
        if source is None:
            if not path.is_file():
                raise TraceDownloadError(
                    f"no download source configured for workload {workload!r}; "
                    f"place {path.name} in {trace_dir}"
                )
            resolved[workload] = path
            continue
        expected = source["sha256"]
        if path.is_file() and _sha256(path.read_bytes()) == expected:
            resolved[workload] = path
            continue
        data = downloader(source["url"])
        actual = _sha256(data)
        if actual != expected:
            raise TraceDownloadError(
                f"{workload} trace from {source['url']} failed checksum: "
                f"expected {expected}, observed {actual}"
            )
        path.write_bytes(data)
        _verify(workload, path, expected)
        resolved[workload] = path
    return resolved

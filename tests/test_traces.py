from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.constants import TRACE_SOURCES
from dynamo_routing_harness.traces import (
    TraceDownloadError,
    ensure_traces,
    trace_path,
)


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


class TraceTests(unittest.TestCase):
    def test_downloads_missing_traces_and_verifies(self) -> None:
        payloads = {
            workload: f"{workload}-trace".encode() for workload in TRACE_SOURCES
        }
        url_to_workload = {
            source["url"]: workload
            for workload, source in TRACE_SOURCES.items()
        }
        originals = {
            workload: source["sha256"]
            for workload, source in TRACE_SOURCES.items()
        }
        calls: list[str] = []

        def fake_downloader(url: str) -> bytes:
            calls.append(url)
            return payloads[url_to_workload[url]]

        try:
            for workload, data in payloads.items():
                TRACE_SOURCES[workload]["sha256"] = _sha256(data)
            with tempfile.TemporaryDirectory() as tmp:
                trace_dir = Path(tmp)
                resolved = ensure_traces(
                    trace_dir,
                    downloader=fake_downloader,
                )
                for workload in TRACE_SOURCES:
                    path = trace_path(trace_dir, workload)
                    self.assertEqual(resolved[workload], path)
                    self.assertEqual(path.read_bytes(), payloads[workload])
                self.assertEqual(len(calls), len(TRACE_SOURCES))
        finally:
            for workload, sha in originals.items():
                TRACE_SOURCES[workload]["sha256"] = sha

    def test_existing_valid_trace_is_not_redownloaded(self) -> None:
        originals = {
            workload: source["sha256"]
            for workload, source in TRACE_SOURCES.items()
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace_dir = Path(tmp)
                for workload in TRACE_SOURCES:
                    data = f"{workload}-bytes".encode()
                    TRACE_SOURCES[workload]["sha256"] = _sha256(data)
                    trace_path(trace_dir, workload).write_bytes(data)

                def fail_downloader(url: str) -> bytes:
                    raise AssertionError(f"unexpected download: {url}")

                resolved = ensure_traces(
                    trace_dir,
                    downloader=fail_downloader,
                )
                self.assertEqual(len(resolved), len(TRACE_SOURCES))
        finally:
            for workload, sha in originals.items():
                TRACE_SOURCES[workload]["sha256"] = sha

    def test_checksum_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            workload = next(iter(TRACE_SOURCES))
            with self.assertRaises(TraceDownloadError):
                ensure_traces(
                    trace_dir,
                    workloads=(workload,),
                    downloader=lambda url: b"corrupted",
                )

    def test_unconfigured_workload_requires_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            with self.assertRaises(TraceDownloadError):
                ensure_traces(
                    trace_dir,
                    workloads=("unconfigured",),
                    downloader=lambda url: b"",
                )


if __name__ == "__main__":
    unittest.main()

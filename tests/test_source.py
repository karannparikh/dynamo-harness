from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dynamo_routing_harness.source import (
    SourceValidationError,
    inspect_source,
    selected_source_paths,
)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], str]) -> None:
        self.responses = responses

    def run(self, args, **kwargs):
        key = tuple(os.fspath(arg) for arg in args)
        return SimpleNamespace(
            stdout=self.responses[key],
            stderr="",
            returncode=0,
        )


class SourceTests(unittest.TestCase):
    def test_defaults_are_sibling_checkouts(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            dynamo, fake_sglang = selected_source_paths()
        self.assertEqual(dynamo.name, "dynamo")
        self.assertEqual(fake_sglang.name, "fake-sglang")
        self.assertEqual(dynamo.parent, fake_sglang.parent)

    def test_environment_overrides_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "DYNAMO_SOURCE": os.fspath(root / "d"),
                    "FAKE_SGLANG_SOURCE": os.fspath(root / "f"),
                },
                clear=True,
            ):
                dynamo, fake_sglang = selected_source_paths()
        self.assertEqual(dynamo, (root / "d").resolve())
        self.assertEqual(fake_sglang, (root / "f").resolve())

    def test_inspect_clean_mock_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            prefix = ("git", "-C", os.fspath(path))
            runner = FakeRunner(
                {
                    (*prefix, "rev-parse", "--is-inside-work-tree"): "true\n",
                    (*prefix, "rev-parse", "HEAD"): "a" * 40 + "\n",
                    (
                        *prefix,
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=normal",
                    ): "",
                }
            )
            source = inspect_source(path, runner)
        self.assertEqual(source.head, "a" * 40)
        self.assertFalse(source.dirty)
        self.assertEqual(source.status, ())
        self.assertEqual(len(source.fingerprint), 64)

    def test_missing_checkout_fails_before_git(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "does not exist"):
            inspect_source(Path("/definitely/missing"))


if __name__ == "__main__":
    unittest.main()

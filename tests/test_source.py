from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dynamo_routing_harness.source import (
    SourceSpec,
    SourceValidationError,
    inspect_source,
    materialize_source,
    parse_source_spec,
    selected_source_specs,
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


class RemoteRunner:
    def __init__(self, remote_url: str) -> None:
        self.remote_url = remote_url
        self.commands: list[tuple[str, ...]] = []

    def run(self, args, **kwargs):
        command = tuple(os.fspath(arg) for arg in args)
        self.commands.append(command)
        if command[:3] == ("git", "clone", "--filter=blob:none"):
            Path(command[-1], ".git").mkdir(parents=True)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "remote" in command and "get-url" in command:
            return SimpleNamespace(
                stdout=self.remote_url + "\n", stderr="", returncode=0
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)


class SourceTests(unittest.TestCase):
    def test_defaults_are_sibling_checkouts(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            dynamo, fake_sglang = selected_source_specs()
        self.assertEqual(dynamo.local_path.name, "dynamo")
        self.assertEqual(fake_sglang.local_path.name, "fake-sglang")
        self.assertEqual(dynamo.local_path.parent, fake_sglang.local_path.parent)

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
                dynamo, fake_sglang = selected_source_specs()
        self.assertEqual(dynamo.local_path, (root / "d").resolve())
        self.assertEqual(fake_sglang.local_path, (root / "f").resolve())

    def test_github_locations_are_normalized(self) -> None:
        cases = (
            (
                "karannparikh/fake-sglang",
                "https://github.com/karannparikh/fake-sglang.git",
            ),
            (
                "github:karannparikh/fake-sglang",
                "https://github.com/karannparikh/fake-sglang.git",
            ),
            (
                "https://github.com/karannparikh/fake-sglang",
                "https://github.com/karannparikh/fake-sglang.git",
            ),
            (
                "git@github.com:karannparikh/fake-sglang.git",
                "git@github.com:karannparikh/fake-sglang.git",
            ),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                spec = parse_source_spec("FAKE_SGLANG", value, "0ff63336")
                self.assertTrue(spec.is_remote)
                self.assertEqual(spec.remote_url, expected)
                self.assertEqual(spec.ref, "0ff63336")

    def test_non_github_remote_is_rejected(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "GitHub"):
            parse_source_spec("DYNAMO", "https://example.com/org/dynamo.git")

    def test_local_checkout_ref_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SourceValidationError, "never mutated"):
                parse_source_spec("DYNAMO", directory, "main")

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

    def test_remote_ref_materializes_in_ignored_cache(self) -> None:
        remote = "https://github.com/ai-dynamo/dynamo.git"
        spec = SourceSpec(
            name="DYNAMO",
            requested="ai-dynamo/dynamo",
            ref="abc123",
            remote_url=remote,
        )
        runner = RemoteRunner(remote)
        with tempfile.TemporaryDirectory() as directory, patch(
            "dynamo_routing_harness.source.LOCAL_ROOT", Path(directory)
        ):
            path = materialize_source(spec, runner)
        self.assertEqual(path.name, "dynamo-68bac3a31ab9")
        fetch = next(command for command in runner.commands if "fetch" in command)
        checkout = next(
            command for command in runner.commands if "checkout" in command
        )
        self.assertEqual(Path(fetch[2]).resolve(), path)
        self.assertEqual(fetch[3:], ("fetch", "--depth=1", "origin", "abc123"))
        self.assertEqual(Path(checkout[2]).resolve(), path)
        self.assertEqual(checkout[3:], ("checkout", "--detach", "FETCH_HEAD"))


if __name__ == "__main__":
    unittest.main()

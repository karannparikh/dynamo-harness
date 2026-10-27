from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dynamo_routing_harness.build import ImageBuilder
from dynamo_routing_harness.source import SourceMetadata, SourcePair


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, args, **kwargs):
        command = tuple(os.fspath(arg) for arg in args)
        self.commands.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)


class BuildTests(unittest.TestCase):
    def test_build_consumes_both_local_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dynamo = root / "dynamo"
            fake = root / "fake-sglang"
            (dynamo / "deploy" / "operator").mkdir(parents=True)
            (dynamo / "container" / "compliance").mkdir(parents=True)
            fake.mkdir()
            (dynamo / "deploy" / "operator" / "Dockerfile").write_text("FROM scratch\n")
            sources = SourcePair(
                SourceMetadata(
                    os.fspath(dynamo), "d" * 40, False, (), "1" * 64
                ),
                SourceMetadata(os.fspath(fake), "f" * 40, False, (), "2" * 64),
            )
            runner = RecordingRunner()
            ImageBuilder(runner).build(sources)
        serving = runner.commands[0]
        self.assertIn(f"dynamo={dynamo}", serving)
        self.assertIn(f"fake_sglang={fake}", serving)
        self.assertIn("DYNAMO_COMMIT=" + "d" * 40, serving)
        self.assertIn("FAKE_SGLANG_COMMIT=" + "f" * 40, serving)
        planner = runner.commands[1]
        self.assertIn("planner", planner)
        operator = runner.commands[2]
        self.assertEqual(operator[-1], os.fspath(dynamo / "deploy" / "operator"))


if __name__ == "__main__":
    unittest.main()

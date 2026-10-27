from __future__ import annotations

import unittest

from dynamo_routing_harness.constants import H200_PROFILE
from dynamo_routing_harness.manifests import render_resources
from dynamo_routing_harness.source import SourceMetadata, SourcePair


def sources() -> SourcePair:
    return SourcePair(
        dynamo=SourceMetadata("/dynamo", "d" * 40, False, (), "1" * 64),
        fake_sglang=SourceMetadata(
            "/fake-sglang", "f" * 40, True, (" M file",), "2" * 64
        ),
    )


class ManifestTests(unittest.TestCase):
    def _dgd(self, mode: str = "kv") -> dict[str, object]:
        return next(
            resource
            for resource in render_resources(
                "local/image:test",
                "local/planner:test",
                sources(),
                mode,
            )
            if resource["kind"] == "DynamoGraphDeployment"
        )

    def test_validated_topology(self) -> None:
        components = self._dgd()["spec"]["components"]
        observed = {
            component["type"]: component["replicas"] for component in components
        }
        self.assertEqual(
            observed,
            {"frontend": 1, "planner": 1, "prefill": 4, "decode": 4},
        )

    def test_workers_inherit_h200_defaults(self) -> None:
        components = self._dgd()["spec"]["components"]
        forbidden = {
            "--fake-num-kv-blocks",
            "--page-size",
            "--context-length",
            "--kv-cache-dtype",
            "--max-running-requests",
        }
        for component in components:
            if component["type"] not in {"prefill", "decode"}:
                continue
            args = component["podTemplate"]["spec"]["containers"][0]["args"]
            self.assertIn("--dp-size", args)
            self.assertEqual(args[args.index("--dp-size") + 1], "10")
            self.assertIn("--enable-dp-attention", args)
            self.assertFalse(forbidden.intersection(args))
        self.assertEqual(
            H200_PROFILE["kv_blocks_per_rank"]
            * H200_PROFILE["tokens_per_page"],
            H200_PROFILE["token_slots_per_rank"],
        )

    def test_frontend_uses_requested_cli_mode(self) -> None:
        for mode in ("kv", "load-aware", "power-of-two"):
            frontend = self._dgd(mode)["spec"]["components"][0]
            args = frontend["podTemplate"]["spec"]["containers"][0]["args"]
            self.assertEqual(args[args.index("--router-mode") + 1], mode)

    def test_source_commits_are_annotated(self) -> None:
        annotations = self._dgd()["metadata"]["annotations"]
        self.assertEqual(annotations["dynamo-sim.local/dynamo-commit"], "d" * 40)
        self.assertEqual(
            annotations["dynamo-sim.local/fake-sglang-commit"], "f" * 40
        )

    def test_planner_uses_local_source_image(self) -> None:
        components = self._dgd()["spec"]["components"]
        planner = next(item for item in components if item["type"] == "planner")
        image = planner["podTemplate"]["spec"]["containers"][0]["image"]
        self.assertEqual(image, "local/planner:test")


if __name__ == "__main__":
    unittest.main()

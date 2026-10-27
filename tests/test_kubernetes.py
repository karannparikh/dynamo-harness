from __future__ import annotations

import unittest

from dynamo_routing_harness.kubernetes import (
    FleetValidationError,
    parse_fleet,
    validate_fleet,
)


def pod(name: str, role: str, index: int, *, ready: bool = True):
    args = []
    if role in {"prefill", "decode"}:
        args = [
            "--dp-size",
            "10",
            "--enable-dp-attention",
            "--model-path",
            "/opt/fake-sglang/models/qwen3-0.6b",
            "--served-model-name",
            "fake-model",
        ]
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{index}",
            "labels": {"nvidia.com/dynamo-component-type": role},
        },
        "spec": {
            "nodeName": "worker",
            "containers": [{"name": "main", "args": args}],
        },
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"name": "main", "restartCount": 0}],
        },
    }


def complete_payload():
    items = [pod("frontend-0", "frontend", 0), pod("planner-0", "planner", 1)]
    items.extend(pod(f"prefill-{i}", "prefill", 10 + i) for i in range(4))
    items.extend(pod(f"decode-{i}", "decode", 20 + i) for i in range(4))
    return {"items": items}


class KubernetesTests(unittest.TestCase):
    def test_complete_fleet_validates(self) -> None:
        snapshot = parse_fleet(complete_payload())
        validate_fleet(snapshot)
        self.assertEqual(len(snapshot.pods), 10)
        self.assertEqual(len(snapshot.by_role("prefill")), 4)

    def test_not_ready_pod_fails(self) -> None:
        payload = complete_payload()
        payload["items"][2]["status"]["conditions"][0]["status"] = "False"
        with self.assertRaisesRegex(FleetValidationError, "ready=False"):
            validate_fleet(parse_fleet(payload))

    def test_profile_override_fails(self) -> None:
        payload = complete_payload()
        payload["items"][2]["spec"]["containers"][0]["args"].extend(
            ["--page-size", "256"]
        )
        with self.assertRaisesRegex(FleetValidationError, "overridden"):
            validate_fleet(parse_fleet(payload))

    def test_stability_includes_uid_and_restarts(self) -> None:
        baseline = parse_fleet(complete_payload())
        changed = complete_payload()
        changed["items"][0]["metadata"]["uid"] = "replacement"
        self.assertFalse(parse_fleet(changed).stable_against(baseline))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.aiperf import (
    AIPerfValidationError,
    aiperf_command,
    validate_outputs,
)


class AIPerfTests(unittest.TestCase):
    def _artifacts(self, root: Path, count: int = 2) -> Path:
        payload = {
            "is_complete": True,
            "completed_request_count": {"avg": count},
            "request_count": {"avg": count},
            "request_error_rate": {"avg": 0},
            "error_summary": None,
            "osl_mismatch_count": {"avg": 0},
        }
        (root / "profile_export_aiperf.json").write_text(json.dumps(payload))
        rows = [
            {
                "metadata": {
                    "request_start_ns": index * 1_000_000_000,
                    "request_end_ns": index * 1_000_000_000 + 20_000_000,
                    "was_cancelled": False,
                    "context_overflow_skip": False,
                },
                "metrics": {
                    "time_to_first_token": {
                        "value": 10,
                        "unit": "ms",
                    }
                },
            }
            for index in range(count)
        ]
        (root / "profile_export.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return root

    def test_strict_outputs_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_outputs(self._artifacts(Path(directory)), 2)
        self.assertTrue(result["strict_completion"])
        self.assertEqual(result["successful"], 2)
        self.assertEqual(result["latency_ms_successful"]["p99"], 20)

    def test_request_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._artifacts(Path(directory), 1)
            with self.assertRaisesRegex(AIPerfValidationError, "mismatch"):
                validate_outputs(artifacts, 2)

    def test_command_uses_power_independent_trace_contract(self) -> None:
        command = aiperf_command(
            Path("/aiperf"),
            Path("/trace.jsonl"),
            Path("/artifacts"),
            20,
            100,
        )
        self.assertIn("--custom-dataset-type", command)
        self.assertIn("mooncake_trace", command)
        self.assertIn("--no-fixed-schedule", command)
        self.assertIn("--use-server-token-count", command)


if __name__ == "__main__":
    unittest.main()

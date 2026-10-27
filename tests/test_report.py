from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dynamo_routing_harness.constants import H200_PROFILE, expected_cases, topology
from dynamo_routing_harness.report import generate_report


class ReportTests(unittest.TestCase):
    def test_complete_matrix_generates_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sources = {
                "dynamo": {
                    "path": "/dynamo",
                    "head": "d" * 40,
                    "dirty": False,
                },
                "fake_sglang": {
                    "path": "/fake-sglang",
                    "head": "f" * 40,
                    "dirty": True,
                },
            }
            matrix = {
                "offered_rps": 20,
                "h200_profile_from_application_defaults": H200_PROFILE,
                "topology": topology(),
                "runtime_image": "local/image:test",
                "planner_image": "local/planner:test",
                "sources": sources,
                "cases_finished": True,
                "comparison_complete": True,
            }
            (output / "matrix.json").write_text(json.dumps(matrix))
            for case in expected_cases():
                case_dir = output / case["case"]
                case_dir.mkdir()
                payload = {
                    **case,
                    "strict_completion": True,
                    "successful": 1,
                    "errors": {},
                    "dispatch_rps": 20,
                    "fleet_stable": True,
                    "prefill_cache": {"token_hit_fraction": 0.5},
                    "latency_ms_successful": {"p99": 10},
                    "ttft_ms_successful": {"p99": 2},
                }
                (case_dir / "summary.json").write_text(json.dumps(payload))
            summary = generate_report(output)
            markdown = (output / "report.md").read_text()
        self.assertTrue(summary["matrix_complete"])
        self.assertIn("`" + "d" * 40 + "`", markdown)
        self.assertIn("50.00%", markdown)
        self.assertIn("power-of-two", markdown)


if __name__ == "__main__":
    unittest.main()

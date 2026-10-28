from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import expected_cases


class ReportValidationError(RuntimeError):
    pass


def _load_cases(output: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for expected in expected_cases():
        path = output / expected["case"] / "summary.json"
        if not path.is_file():
            raise ReportValidationError(f"missing case summary: {path}")
        payload = json.loads(path.read_text())
        for field in ("workload", "routing_cli_mode", "routing_label", "case"):
            if payload.get(field) != expected[field]:
                raise ReportValidationError(
                    f"{path} has invalid {field}: {payload.get(field)!r}"
                )
        cases.append(payload)
    return cases


def _value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for item in path:
        if not isinstance(current, dict):
            return None
        current = current.get(item)
    return current


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    sources = summary["sources"]
    lines = [
        "# Dynamo routing benchmark",
        "",
        f"- Offered RPS: `{summary['offered_rps']}`",
        "- Workloads: `conversation`, `toolagent`",
        "- Routing: `kv`, `load-aware`, `power-of-two`",
        "- Topology: 1 frontend, 1 Planner, 4 prefill processes x 10 ranks, "
        "4 decode processes x 10 ranks",
        "- H200 profile: 15,509 blocks/rank, 256 tokens/page, "
        "3,970,304 slots/rank, 1,048,576 context, FP8 E4M3, "
        "32 running requests/rank",
        "",
        "## Local sources",
        "",
        "| Source | HEAD | Dirty | Path |",
        "|---|---|---:|---|",
    ]
    for name in ("dynamo", "fake_sglang"):
        source = sources[name]
        lines.append(
            f"| {name.replace('_', '-')} | `{source['head']}` | "
            f"{'yes' if source['dirty'] else 'no'} | `{source['path']}` |"
        )
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Workload | Routing | Successful | Errors | Dispatch RPS | "
            "Cache hit | Latency p99 ms | TTFT p99 ms | Stable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        errors = case.get("failed")
        if errors is None:
            errors = len(case.get("errors", {}))
        hit_fraction = _value(case, "prefill_cache", "token_hit_fraction")
        hit = "n/a" if hit_fraction is None else f"{100 * float(hit_fraction):.2f}%"
        lines.append(
            "| {workload} | {routing} | {successful} | {errors} | {rps} | "
            "{hit} | {latency} | {ttft} | {stable} |".format(
                workload=case["workload"],
                routing=case["routing_label"],
                successful=case.get("successful", 0),
                errors=errors,
                rps=_number(case.get("dispatch_rps")),
                hit=hit,
                latency=_number(
                    _value(case, "latency_ms_successful", "p99")
                ),
                ttft=_number(_value(case, "ttft_ms_successful", "p99")),
                stable="yes" if case.get("fleet_stable") else "no",
            )
        )
    return "\n".join(lines) + "\n"


def generate_report(output: Path) -> dict[str, Any]:
    matrix_path = output / "matrix.json"
    if not matrix_path.is_file():
        raise ReportValidationError(f"missing benchmark matrix: {matrix_path}")
    matrix = json.loads(matrix_path.read_text())
    cases = _load_cases(output)
    matrix_complete = (
        matrix.get("cases_finished") is True
        and matrix.get("comparison_complete") is True
        and len(cases) == len(expected_cases())
        and all(case.get("strict_completion") is True for case in cases)
    )
    summary = {
        "matrix_complete": matrix_complete,
        "offered_rps": matrix["offered_rps"],
        "h200_profile_from_application_defaults": matrix[
            "h200_profile_from_application_defaults"
        ],
        "topology": matrix["topology"],
        "runtime_image": matrix["runtime_image"],
        "planner_image": matrix["planner_image"],
        "sources": matrix["sources"],
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(build_markdown(summary))
    return summary

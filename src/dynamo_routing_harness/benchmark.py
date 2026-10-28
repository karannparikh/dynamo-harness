from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .aiperf import run_aiperf
from .build import Images
from .constants import (
    DEFAULT_REQUEST_RATE,
    EXPECTED_REPLICAS,
    H200_PROFILE,
    RANKS_PER_PROCESS,
    ROUTING_MODES,
    SETTLE_SECONDS,
    WORKLOADS,
    case_name,
    expected_cases,
    topology,
)
from .kubernetes import Kubernetes, PortForward, validate_fleet
from .metrics import (
    cache_delta,
    wait_cache_snapshot,
    write_cache_snapshot,
)
from .report import generate_report
from .source import SourcePair


class BenchmarkFailed(RuntimeError):
    pass


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case_base(
    case_name: str,
    workload: str,
    routing_mode: str,
    request_rate: float,
    sources: SourcePair,
) -> dict[str, object]:
    return {
        "case": case_name,
        "workload": workload,
        "routing": ROUTING_MODES[routing_mode],
        "routing_label": ROUTING_MODES[routing_mode],
        "routing_cli_mode": routing_mode,
        "offered_rps": request_rate,
        "sources": sources.as_dict(),
        "h200_profile_from_application_defaults": H200_PROFILE,
        "replicas": EXPECTED_REPLICAS,
        "ranks_per_process": {
            "prefill": RANKS_PER_PROCESS,
            "decode": RANKS_PER_PROCESS,
        },
        "logical_workers": {
            "prefill": EXPECTED_REPLICAS["prefill"] * RANKS_PER_PROCESS,
            "decode": EXPECTED_REPLICAS["decode"] * RANKS_PER_PROCESS,
        },
    }


def run_matrix(
    kubernetes: Kubernetes,
    sources: SourcePair,
    images: Images,
    output: Path,
    trace_dir: Path,
    aiperf: Path,
    request_rate: float = DEFAULT_REQUEST_RATE,
) -> dict[str, object]:
    if request_rate <= 0:
        raise ValueError("request rate must be positive")
    traces = {
        workload: trace_dir / f"{workload}_trace.jsonl" for workload in WORKLOADS
    }
    missing = [path for path in traces.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Mooncake traces: " + ", ".join(str(path) for path in missing)
        )
    if not aiperf.is_file():
        raise FileNotFoundError(
            f"AIPerf executable not found: {aiperf}; run benchmark-setup first"
        )
    output.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, object] = {
        "offered_rps": request_rate,
        "h200_profile_from_application_defaults": H200_PROFILE,
        "topology": topology(),
        "runtime_image": images.serving,
        "planner_image": images.planner,
        "operator_image": images.operator,
        "sources": sources.as_dict(),
        "expected_cases": expected_cases(),
        "cases_finished": False,
        "comparison_complete": False,
    }
    _write_json(output / "matrix.json", matrix)
    previous_uids: set[str] = set()
    failures: list[str] = []

    for workload in WORKLOADS:
        for routing_mode in ROUTING_MODES:
            current_case = case_name(workload, routing_mode)
            case_dir = output / current_case
            case_dir.mkdir(parents=True, exist_ok=True)
            summary = _case_base(
                current_case,
                workload,
                routing_mode,
                request_rate,
                sources,
            )
            try:
                kubernetes.deploy_graph(
                    images.serving,
                    images.planner,
                    sources,
                    routing_mode,
                )
                kubernetes.wait_graph()
                current = kubernetes.fleet()
                previous_uids.update(current.uids)
                fresh = kubernetes.recreate_fleet(previous_uids)
                previous_uids.update(fresh.uids)
                time.sleep(SETTLE_SECONDS)
                baseline = kubernetes.fleet()
                validate_fleet(baseline)
                if not baseline.stable_against(fresh):
                    raise BenchmarkFailed(
                        "fleet changed during the mandatory 15-second settle"
                    )
                contract = kubernetes.validate_profile_contract(baseline)
                with PortForward(
                    kubernetes.kubeconfig,
                    "observability",
                    "service/prometheus",
                    "9090:9090",
                ):
                    before = wait_cache_snapshot(
                        "http://127.0.0.1:9090",
                        baseline,
                    )
                write_cache_snapshot(
                    before,
                    case_dir / "cache-metrics-before.json",
                )
                aiperf_summary, observations = run_aiperf(
                    kubernetes,
                    baseline,
                    aiperf,
                    traces[workload],
                    case_dir / "aiperf",
                    request_rate,
                )
                final = kubernetes.fleet()
                validate_fleet(final)
                fleet_stable = final.stable_against(baseline) and all(
                    observation["stable"] for observation in observations
                )
                if not fleet_stable:
                    raise BenchmarkFailed(
                        "fleet changed after traffic completed"
                    )
                with PortForward(
                    kubernetes.kubeconfig,
                    "observability",
                    "service/prometheus",
                    "9090:9090",
                ):
                    after = wait_cache_snapshot(
                        "http://127.0.0.1:9090",
                        final,
                    )
                write_cache_snapshot(
                    after,
                    case_dir / "cache-metrics-after.json",
                )
                summary.update(aiperf_summary)
                summary.update(
                    {
                        "fleet_stable": True,
                        "fleet": {
                            "baseline": baseline.as_dict(),
                            "final": final.as_dict(),
                            "observations": observations,
                        },
                        "effective_profile_contract": contract,
                        "prefill_cache": cache_delta(
                            before,
                            after,
                            fleet_stable=True,
                        )
                    }
                )
            except Exception as error:
                failures.append(f"{current_case}: {error}")
                summary.update(
                    {
                        "is_complete": False,
                        "strict_completion": False,
                        "fleet_stable": False,
                        "successful": 0,
                        "errors": {
                            "harness": {
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        },
                        "error_summary": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            _write_json(case_dir / "summary.json", summary)

    matrix["cases_finished"] = True
    matrix["comparison_complete"] = not failures
    _write_json(output / "matrix.json", matrix)
    report = generate_report(output)
    if failures:
        raise BenchmarkFailed("; ".join(failures))
    return report

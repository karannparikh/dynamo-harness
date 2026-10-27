from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .command import Runner
from .constants import (
    FLEET_POLL_SECONDS,
    LOCAL_ROOT,
    MODEL_NAME,
    REPOSITORY_ROOT,
    TOKENIZER,
    TOKENIZER_REVISION,
)
from .kubernetes import FleetSnapshot, Kubernetes, validate_fleet


class AIPerfValidationError(RuntimeError):
    pass


def setup_aiperf(runner: Runner | None = None) -> Path:
    active_runner = runner or Runner()
    venv = LOCAL_ROOT / "aiperf-venv"
    python = venv / "bin" / "python"
    if not python.exists():
        active_runner.run(("python3", "-m", "venv", venv))
    active_runner.run(
        (
            python,
            "-m",
            "pip",
            "install",
            "--requirement",
            REPOSITORY_ROOT / "benchmark" / "aiperf-requirements.txt",
        )
    )
    cache = LOCAL_ROOT / "aiperf-hf"
    cache.mkdir(parents=True, exist_ok=True)
    active_runner.run(
        (
            python,
            "-c",
            (
                "from transformers import AutoTokenizer;"
                f"AutoTokenizer.from_pretrained({TOKENIZER!r},"
                f" revision={TOKENIZER_REVISION!r}, cache_dir={os.fspath(cache)!r})"
            ),
        )
    )
    return venv / "bin" / "aiperf"


def trace_request_count(trace: Path) -> int:
    count = sum(1 for line in trace.read_text().splitlines() if line.strip())
    if count <= 0:
        raise AIPerfValidationError(f"trace has no requests: {trace}")
    return count


def aiperf_command(
    executable: Path,
    trace: Path,
    artifact_dir: Path,
    request_rate: float,
    request_count: int,
) -> tuple[str, ...]:
    return (
        "/usr/bin/time",
        "-l",
        os.fspath(executable),
        "profile",
        "--model",
        MODEL_NAME,
        "--url",
        "http://127.0.0.1:8000",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        TOKENIZER,
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--input-file",
        os.fspath(trace),
        "--custom-dataset-type",
        "mooncake_trace",
        "--no-fixed-schedule",
        "--request-count",
        str(request_count),
        "--num-dataset-entries",
        str(request_count),
        "--dataset-sampling-strategy",
        "sequential",
        "--request-rate",
        str(request_rate),
        "--arrival-pattern",
        "constant",
        "--isl-block-size",
        "512",
        "--use-server-token-count",
        "--extra-inputs",
        '{"temperature":0,"ignore_eos":true}',
        "--request-timeout-seconds",
        "120",
        "--random-seed",
        "42",
        "--workers-max",
        "4",
        "--ui-type",
        "none",
        "--artifact-dir",
        os.fspath(artifact_dir),
    )


def _metric_avg(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, dict) or "avg" not in value:
        raise AIPerfValidationError(f"AIPerf summary is missing {name}.avg")
    return float(value["avg"])


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p99": None, "max": None}
    ordered = sorted(values)

    def nearest(percentile: float) -> float:
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    return {
        "p50": nearest(0.50),
        "p99": nearest(0.99),
        "max": max(ordered),
    }


def validate_outputs(
    artifact_dir: Path,
    expected: int,
) -> dict[str, object]:
    summary_path = artifact_dir / "profile_export_aiperf.json"
    rows_path = artifact_dir / "profile_export.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        raise AIPerfValidationError(
            "AIPerf did not produce profile_export_aiperf.json and "
            "profile_export.jsonl"
        )
    payload = json.loads(summary_path.read_text())
    completed = _metric_avg(payload, "completed_request_count")
    requested = _metric_avg(payload, "request_count")
    error_rate = _metric_avg(payload, "request_error_rate")
    if payload.get("is_complete") is not True:
        raise AIPerfValidationError("AIPerf marked the profile incomplete")
    if completed != expected or requested != expected:
        raise AIPerfValidationError(
            f"AIPerf completion mismatch: expected={expected}, "
            f"completed={completed}, requested={requested}"
        )
    if error_rate != 0:
        raise AIPerfValidationError(f"AIPerf request_error_rate={error_rate}")
    if payload.get("error_summary") not in (None, {}, [], ""):
        raise AIPerfValidationError(
            f"AIPerf error_summary is non-empty: {payload['error_summary']!r}"
        )
    if "osl_mismatch_count" in payload:
        if _metric_avg(payload, "osl_mismatch_count") != 0:
            raise AIPerfValidationError("AIPerf reported output-length mismatches")

    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise AIPerfValidationError(
            f"expected {expected} profiling rows, observed {len(rows)}"
        )
    latency_ms: list[float] = []
    ttft_ms: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, row in enumerate(rows):
        if row.get("error") not in (None, "", {}, []):
            raise AIPerfValidationError(
                f"profiling row {index} contains an error: {row['error']!r}"
            )
        start = int(row["request_start_ns"])
        end = int(row["request_end_ns"])
        if end < start:
            raise AIPerfValidationError(
                f"profiling row {index} ends before it starts"
            )
        starts.append(start)
        ends.append(end)
        latency_ms.append((end - start) / 1_000_000)
        first_token = row.get("first_token_ns")
        if first_token is not None:
            ttft_ms.append((int(first_token) - start) / 1_000_000)
    dispatch_window = (
        (max(starts) - min(starts)) / 1_000_000_000 if len(starts) > 1 else 0.0
    )
    request_window = (max(ends) - min(starts)) / 1_000_000_000
    return {
        "expected": expected,
        "recorded": len(rows),
        "completed": int(completed),
        "successful": len(rows),
        "error_rate_pct": error_rate * 100,
        "errors": {},
        "error_summary": payload.get("error_summary"),
        "is_complete": True,
        "strict_completion": True,
        "dispatch_window_seconds": dispatch_window,
        "dispatch_rps": len(rows) / dispatch_window if dispatch_window > 0 else None,
        "request_window_seconds": request_window,
        "latency_ms_successful": _percentiles(latency_ms),
        "ttft_ms_successful": _percentiles(ttft_ms),
        "aiperf_summary": payload,
    }


def run_aiperf(
    kubernetes: Kubernetes,
    baseline: FleetSnapshot,
    executable: Path,
    trace: Path,
    artifact_dir: Path,
    request_rate: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    request_count = trace_request_count(trace)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = aiperf_command(
        executable,
        trace,
        artifact_dir,
        request_rate,
        request_count,
    )
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": os.fspath(LOCAL_ROOT / "aiperf-hf"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "900",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "960",
        }
    )
    timeout = max(1800.0, 960.0 + request_count / request_rate + 240.0)
    observations: list[dict[str, object]] = []

    def stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    with (artifact_dir / "stdout.log").open("w") as stdout, (
        artifact_dir / "stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"AIPerf exceeded {timeout:.0f} seconds")
                snapshot = kubernetes.fleet()
                validate_fleet(snapshot)
                stable = snapshot.stable_against(baseline)
                observations.append(
                    {
                        "timestamp": time.time(),
                        "stable": stable,
                        "fleet": snapshot.as_dict(),
                    }
                )
                if not stable:
                    raise AIPerfValidationError(
                        "fleet names, UIDs, readiness, roles, or restart counts "
                        "changed during traffic"
                    )
                time.sleep(FLEET_POLL_SECONDS)
        finally:
            stop_process(process)
        if process.returncode:
            raise AIPerfValidationError(
                f"AIPerf exited with status {process.returncode}; "
                f"see {artifact_dir / 'stderr.log'}"
            )
    return validate_outputs(artifact_dir, request_count), observations

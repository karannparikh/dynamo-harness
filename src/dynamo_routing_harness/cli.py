from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aiperf import setup_aiperf
from .benchmark import run_matrix
from .build import ImageBuilder, image_names
from .constants import (
    DEFAULT_KUBECONFIG,
    DEFAULT_REQUEST_RATE,
    LOCAL_ROOT,
)
from .kubernetes import Kubernetes, validate_fleet
from .manifests import render_resources, write_json_documents
from .report import generate_report
from .source import inspect_sources, write_source_record


def _default_trace_dir() -> Path:
    return LOCAL_ROOT / "mooncake-100rps"


def _default_aiperf() -> Path:
    return LOCAL_ROOT / "aiperf-venv" / "bin" / "aiperf"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="dynamo-routing-lab",
        description="Apple Silicon CPU-only local Dynamo routing harness",
    )
    root.add_argument(
        "--kubeconfig",
        type=Path,
        default=DEFAULT_KUBECONFIG,
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("up")
    commands.add_parser("rebuild")
    deploy = commands.add_parser("deploy")
    deploy.add_argument(
        "--routing-mode",
        choices=("kv", "load-aware", "power-of-two"),
        default="kv",
    )
    setup = commands.add_parser("benchmark-setup")
    setup.add_argument("trace_dir", nargs="?", type=Path, default=_default_trace_dir())
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--request-rate",
        type=float,
        default=DEFAULT_REQUEST_RATE,
    )
    benchmark.add_argument(
        "--trace-dir",
        type=Path,
        default=_default_trace_dir(),
    )
    benchmark.add_argument("--aiperf", type=Path, default=_default_aiperf())
    report = commands.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    commands.add_parser("status")
    commands.add_parser("stop")
    return root


def _sources_json(sources: object) -> str:
    return json.dumps(sources.as_dict(), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    kubernetes = Kubernetes(kubeconfig=args.kubeconfig)

    if args.command == "doctor":
        kubernetes.require_apple_silicon()
        kubernetes.require_tools()
        sources = inspect_sources()
        print(_sources_json(sources))
        return 0
    if args.command == "stop":
        kubernetes.stop()
        return 0
    if args.command == "status":
        sys.stdout.write(kubernetes.status())
        return 0
    if args.command == "benchmark-setup":
        missing = [
            args.trace_dir / name
            for name in ("conversation_trace.jsonl", "toolagent_trace.jsonl")
            if not (args.trace_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "place both Mooncake trace files in the trace directory; missing: "
                + ", ".join(str(path) for path in missing)
            )
        executable = setup_aiperf()
        print(executable)
        return 0
    if args.command == "report":
        summary = generate_report(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["matrix_complete"] else 1

    sources = inspect_sources()
    images = image_names(sources)
    write_source_record(sources, LOCAL_ROOT / "sources.json")

    if args.command == "render":
        args.output.mkdir(parents=True, exist_ok=True)
        for routing_mode in ("kv", "load-aware", "power-of-two"):
            write_json_documents(
                render_resources(
                    images.serving,
                    images.planner,
                    sources,
                    routing_mode,
                ),
                args.output / f"dgd-{routing_mode}.json",
            )
        print(args.output)
        return 0
    if args.command == "up":
        kubernetes.start_cluster()
        images = ImageBuilder().build(sources)
        kubernetes.install_platform(sources, images)
        kubernetes.install_observability()
        kubernetes.deploy_graph(images.serving, images.planner, sources)
        kubernetes.wait_graph()
        fleet = kubernetes.wait_fleet()
        validate_fleet(fleet)
        kubernetes.validate_profile_contract(fleet)
        print(kubernetes.status(), end="")
        return 0
    if args.command == "rebuild":
        images = ImageBuilder().build(sources)
        kubernetes.install_platform(sources, images)
        kubernetes.deploy_graph(images.serving, images.planner, sources)
        kubernetes.wait_graph()
        print(json.dumps(images.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "deploy":
        kubernetes.deploy_graph(
            images.serving,
            images.planner,
            sources,
            args.routing_mode,
        )
        kubernetes.wait_graph()
        fleet = kubernetes.wait_fleet()
        validate_fleet(fleet)
        kubernetes.validate_profile_contract(fleet)
        return 0
    if args.command == "benchmark":
        report = run_matrix(
            kubernetes,
            sources,
            images,
            args.output,
            args.trace_dir,
            args.aiperf,
            args.request_rate,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

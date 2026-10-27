from __future__ import annotations

from pathlib import Path

CLUSTER_NAME = "dynamo-routing"
COLIMA_PROFILE = "dynamo-routing"
KUBE_CONTEXT = f"kind-{CLUSTER_NAME}"
SERVING_NAMESPACE = "dynamo-sim"
OBSERVABILITY_NAMESPACE = "observability"
SYSTEM_NAMESPACE = "dynamo-system"
DGD_NAME = "fake-sglang"
MODEL_NAME = "fake-model"
MODEL_PATH = "/opt/fake-sglang/models/qwen3-0.6b"
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

DEFAULT_REQUEST_RATE = 20.0
SETTLE_SECONDS = 15
FLEET_POLL_SECONDS = 5
METRICS_TIMEOUT_SECONDS = 90

EXPECTED_REPLICAS = {
    "frontend": 1,
    "planner": 1,
    "prefill": 4,
    "decode": 4,
}
RANKS_PER_PROCESS = 10

H200_PROFILE = {
    "kv_blocks_per_rank": 15_509,
    "tokens_per_page": 256,
    "token_slots_per_rank": 3_970_304,
    "context_length": 1_048_576,
    "kv_cache_dtype": "fp8_e4m3",
    "max_running_requests_per_rank": 32,
}

ROUTING_MODES = {
    "kv": "KV-aware",
    "load-aware": "load-aware",
    "power-of-two": "PO2C",
}
WORKLOADS = ("conversation", "toolagent")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPOSITORY_ROOT / ".local"
DEFAULT_KUBECONFIG = LOCAL_ROOT / "kubeconfig"


def case_name(workload: str, routing_mode: str) -> str:
    suffix = "po2c" if routing_mode == "power-of-two" else routing_mode
    return f"{workload}-{suffix}"


def expected_cases() -> list[dict[str, str]]:
    return [
        {
            "workload": workload,
            "routing_cli_mode": mode,
            "routing_label": label,
            "case": case_name(workload, mode),
        }
        for workload in WORKLOADS
        for mode, label in ROUTING_MODES.items()
    ]


def topology() -> dict[str, int]:
    return {
        "frontend_processes": EXPECTED_REPLICAS["frontend"],
        "planner_processes": EXPECTED_REPLICAS["planner"],
        "prefill_processes": EXPECTED_REPLICAS["prefill"],
        "decode_processes": EXPECTED_REPLICAS["decode"],
        "ranks_per_worker_process": RANKS_PER_PROCESS,
    }

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .constants import (
    DGD_NAME,
    MODEL_NAME,
    MODEL_PATH,
    OBSERVABILITY_NAMESPACE,
    RANKS_PER_PROCESS,
    REPOSITORY_ROOT,
    SERVING_NAMESPACE,
)
from .source import SourcePair


def _env(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def _common_env(service_name: str) -> list[dict[str, object]]:
    return [
        _env("DYN_EVENT_PLANE", "nats"),
        _env("DYN_REQUEST_PLANE", "tcp"),
        _env("DYN_SYSTEM_PORT", "9090"),
        _env("OTEL_EXPORT_ENABLED", "true"),
        _env(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            f"http://otel-collector.{OBSERVABILITY_NAMESPACE}.svc.cluster.local:4317",
        ),
        _env("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"),
        _env(
            "OTEL_RESOURCE_ATTRIBUTES",
            "deployment.environment.name=local-simulation",
        ),
        _env("OTEL_SERVICE_NAME", service_name),
    ]


def _resources(memory_request: str, memory_limit: str) -> dict[str, object]:
    return {
        "requests": {
            "cpu": "100m",
            "memory": memory_request,
            "nvidia.com/gpu": "1",
        },
        "limits": {
            "cpu": "2",
            "memory": memory_limit,
            "nvidia.com/gpu": "1",
        },
    }


def _worker_component(
    role: str,
    image: str,
    router_policy_sha: str,
) -> dict[str, object]:
    env: list[dict[str, object]] = _common_env(f"dynamo-{role}")
    env.extend(
        [
            _env("DYN_DYNAMO_REQUEST_QUEUE_LIMIT", "20"),
            {
                "name": "SGLANG_HOST_IP",
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            },
        ]
    )
    args = [
        "--model-path",
        MODEL_PATH,
        "--served-model-name",
        MODEL_NAME,
        "--skip-tokenizer-init",
        "--host",
        "0.0.0.0",
        "--disaggregation-mode",
        role,
        "--router-mode",
        "kv",
        "--router-kv-events",
        "--disaggregation-bootstrap-port",
        "8998",
        "--dp-size",
        str(RANKS_PER_PROCESS),
        "--enable-dp-attention",
        "--fake-kv-generation-ms-per-1k-tokens",
        "50.0",
        "--kv-events-config",
        '{"publisher":"zmq","endpoint":"tcp://*:5557"}',
        "--enable-metrics",
        "--enable-forward-pass-metrics",
        "--router-policy-config",
        "/etc/lab-router/router-policy.yaml",
    ]
    return {
        "name": role,
        "type": role,
        "replicas": 4,
        "podTemplate": {
            "metadata": {
                "labels": {
                    "dynamo-sim.local/engine": "fake-sglang",
                    "dynamo-sim.local/router-policy-sha256": router_policy_sha,
                }
            },
            "spec": {
                "nodeSelector": {
                    "dynamo-sim.local/gpu-capacity": "simulated",
                },
                "terminationGracePeriodSeconds": 30,
                "securityContext": {
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumes": [
                    {
                        "name": "router-policy",
                        "configMap": {"name": "lab-router-policy"},
                    }
                ],
                "containers": [
                    {
                        "name": "main",
                        "image": image,
                        "imagePullPolicy": "Never",
                        "command": ["python", "-m", "dynamo.sglang"],
                        "args": args,
                        "env": env,
                        "ports": [
                            {
                                "name": "system",
                                "containerPort": 9090,
                                "protocol": "TCP",
                            }
                        ],
                        "resources": _resources("256Mi", "1Gi"),
                        "volumeMounts": [
                            {
                                "name": "router-policy",
                                "mountPath": "/etc/lab-router",
                                "readOnly": True,
                            }
                        ],
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": "system"},
                            "periodSeconds": 5,
                            "timeoutSeconds": 4,
                            "failureThreshold": 2,
                            "successThreshold": 1,
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/live", "port": "system"},
                            "periodSeconds": 30,
                            "timeoutSeconds": 10,
                            "failureThreshold": 3,
                            "successThreshold": 1,
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                        },
                    }
                ],
            },
        },
    }


def planner_config() -> dict[str, object]:
    return {
        "environment": "kubernetes",
        "backend": "sglang",
        "mode": "disagg",
        "model_name": MODEL_NAME,
        "optimization_target": "load",
        "pre_deployment_sweeping_mode": "none",
        "enable_throughput_scaling": False,
        "enable_load_scaling": True,
        "prefill_scale_up_queue_tokens": 8192,
        "prefill_scale_down_queue_tokens": 1024,
        "decode_scale_up_kv_rate": 70,
        "decode_scale_down_kv_rate": 20,
        "min_endpoint": 4,
        "max_gpu_budget": 8,
        "load_adjustment_interval_seconds": 10,
        "load_predictor": "constant",
    }


def render_resources(
    image: str,
    planner_image: str,
    sources: SourcePair,
    routing_mode: str = "kv",
) -> list[dict[str, object]]:
    policy = (REPOSITORY_ROOT / "config" / "router-policy.yaml").read_text()
    policy_sha = hashlib.sha256(policy.encode()).hexdigest()
    frontend_args = [
        "--http-port",
        "8000",
        "--router-mode",
        routing_mode,
        "--active-decode-blocks-threshold",
        "0.85",
        "--router-track-output-blocks",
        "--router-policy-config",
        "/etc/lab-router/router-policy.yaml",
    ]
    if routing_mode == "kv":
        frontend_args.append("--router-kv-events")
    components: list[dict[str, object]] = [
        {
            "name": "Frontend",
            "type": "frontend",
            "replicas": 1,
            "podTemplate": {
                "metadata": {
                    "labels": {
                        "dynamo-sim.local/router-policy-sha256": policy_sha,
                    }
                },
                "spec": {
                    "volumes": [
                        {
                            "name": "router-policy",
                            "configMap": {"name": "lab-router-policy"},
                        }
                    ],
                    "containers": [
                        {
                            "name": "main",
                            "image": image,
                            "imagePullPolicy": "Never",
                            "command": ["python", "-m", "dynamo.frontend"],
                            "args": frontend_args,
                            "env": _common_env("dynamo-frontend"),
                            "ports": [
                                {
                                    "name": "http",
                                    "containerPort": 8000,
                                    "protocol": "TCP",
                                }
                            ],
                            "volumeMounts": [
                                {
                                    "name": "router-policy",
                                    "mountPath": "/etc/lab-router",
                                    "readOnly": True,
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "2", "memory": "1Gi"},
                            },
                        }
                    ],
                },
            },
        },
        {
            "name": "Planner",
            "type": "planner",
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "image": planner_image,
                            "imagePullPolicy": "Never",
                            "command": ["python", "-m", "dynamo.planner"],
                            "args": [
                                "--config",
                                json.dumps(planner_config(), separators=(",", ":")),
                            ],
                            "env": _common_env("dynamo-planner"),
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "512Mi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                        }
                    ]
                }
            },
        },
        _worker_component("prefill", image, policy_sha),
        _worker_component("decode", image, policy_sha),
    ]
    return [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": SERVING_NAMESPACE},
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "lab-router-policy",
                "namespace": SERVING_NAMESPACE,
            },
            "data": {"router-policy.yaml": policy},
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {
                "name": DGD_NAME,
                "namespace": SERVING_NAMESPACE,
                "annotations": {
                    "dynamo-sim.local/dynamo-commit": sources.dynamo.head,
                    "dynamo-sim.local/fake-sglang-commit": sources.fake_sglang.head,
                },
            },
            "spec": {
                "backendFramework": "sglang",
                "components": components,
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "inference", "namespace": SERVING_NAMESPACE},
            "spec": {
                "type": "NodePort",
                "selector": {
                    "nvidia.com/dynamo-graph-deployment-name": DGD_NAME,
                    "nvidia.com/dynamo-component-type": "frontend",
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 8000,
                        "targetPort": 8000,
                        "nodePort": 30800,
                    }
                ],
            },
        },
    ]


def write_json_documents(resources: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n---\n".join(json.dumps(resource, indent=2) for resource in resources) + "\n"
    )


def render_operator_values(
    operator_repository: str,
    operator_tag: str,
    output: Path,
) -> None:
    template = (
        REPOSITORY_ROOT / "config" / "operator-values.yaml.tmpl"
    ).read_text()
    rendered = template.replace("__OPERATOR_REPOSITORY__", operator_repository)
    rendered = rendered.replace("__OPERATOR_TAG__", operator_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)

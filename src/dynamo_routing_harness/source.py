from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .command import CommandError, Runner
from .constants import REPOSITORY_ROOT


class SourceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SourceMetadata:
    path: str
    head: str
    dirty: bool
    status: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = list(self.status)
        return value


@dataclass(frozen=True)
class SourcePair:
    dynamo: SourceMetadata
    fake_sglang: SourceMetadata

    def as_dict(self) -> dict[str, object]:
        return {
            "dynamo": self.dynamo.as_dict(),
            "fake_sglang": self.fake_sglang.as_dict(),
        }


def default_source_path(name: str) -> Path:
    return REPOSITORY_ROOT.parent / name


def selected_source_paths(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    values = os.environ if environ is None else environ
    dynamo = Path(
        values.get("DYNAMO_SOURCE", os.fspath(default_source_path("dynamo")))
    )
    fake_sglang = Path(
        values.get(
            "FAKE_SGLANG_SOURCE", os.fspath(default_source_path("fake-sglang"))
        )
    )
    return dynamo.expanduser().resolve(), fake_sglang.expanduser().resolve()


def _git(runner: Runner, path: Path, *args: str) -> str:
    try:
        return runner.run(("git", "-C", path, *args)).stdout.strip()
    except CommandError as error:
        raise SourceValidationError(
            f"{path} is not a usable git checkout: {error}"
        ) from error


def _fingerprint(runner: Runner, path: Path, head: str, status: str) -> str:
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(b"\0")
    digest.update(status.encode())
    digest.update(b"\0")
    if status:
        diff = _git(runner, path, "diff", "--binary", "HEAD")
        digest.update(diff.encode())
        untracked = _git(
            runner,
            path,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        for relative in sorted(filter(None, untracked.split("\0"))):
            digest.update(relative.encode())
            digest.update(b"\0")
            file_path = path / relative
            if file_path.is_file():
                digest.update(file_path.read_bytes())
    return digest.hexdigest()


def inspect_source(path: Path, runner: Runner | None = None) -> SourceMetadata:
    active_runner = runner or Runner()
    if not path.is_dir():
        raise SourceValidationError(f"source checkout does not exist: {path}")
    inside = _git(active_runner, path, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise SourceValidationError(f"source path is not a git work tree: {path}")
    head = _git(active_runner, path, "rev-parse", "HEAD")
    status_text = _git(
        active_runner,
        path,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    status = tuple(line for line in status_text.splitlines() if line)
    return SourceMetadata(
        path=os.fspath(path),
        head=head,
        dirty=bool(status),
        status=status,
        fingerprint=_fingerprint(active_runner, path, head, status_text),
    )


def inspect_sources(runner: Runner | None = None) -> SourcePair:
    dynamo_path, fake_sglang_path = selected_source_paths()
    active_runner = runner or Runner()
    return SourcePair(
        dynamo=inspect_source(dynamo_path, active_runner),
        fake_sglang=inspect_source(fake_sglang_path, active_runner),
    )


def write_source_record(sources: SourcePair, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sources.as_dict(), indent=2, sort_keys=True) + "\n")

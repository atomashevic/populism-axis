"""Run provenance for the pipeline scripts.

Every script that writes results calls `write_run_manifest` as its last
action. The manifest records the code version (git commit and whether the
working tree was dirty), the command, the machine, library versions, and the
SHA-256 of every input file, so a reported number can be traced to the exact
code, data and anchors that produced it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "populism_axis.v4.run_manifest.v1"
TRACKED_PACKAGES = ("populism-axis", "numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "pyarrow")
REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, branch and dirty flag; values are None when git is unavailable."""

    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    # Git may search upward into an unrelated enclosing repository after installation.
    if not (Path(repo) / "pyproject.toml").is_file() or not (Path(repo) / "src" / "populism_axis" / "run_manifest.py").is_file():
        return {"commit": None, "branch": None, "dirty": None}
    if _run("rev-parse", "--show-toplevel") != str(Path(repo).resolve()):
        return {"commit": None, "branch": None, "dirty": None}
    status = _run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def run_hash(input_hashes: Mapping[str, str], argv: Sequence[str], encoder_id: str | None, extra: Mapping) -> str:
    """Hash of what determines the output: inputs, arguments, encoder, extra config."""
    payload = {
        "inputs": dict(sorted(input_hashes.items())),
        "argv": list(argv),
        "encoder_id": encoder_id,
        "extra": extra,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def package_version() -> str:
    try:
        return importlib.metadata.version("populism-axis")
    except importlib.metadata.PackageNotFoundError:
        from . import __version__
        return __version__


def build_run_manifest(
    *,
    script: str,
    argv: Sequence[str],
    inputs: Mapping[str, Path] | None = None,
    encoder_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    repo: Path = REPO_ROOT,
    timestamp: str | None = None,
) -> dict[str, Any]:
    inputs = inputs or {}
    extra = dict(extra or {})
    input_hashes = {name: file_sha256(Path(path)) for name, path in inputs.items()}
    return {
        "schema": SCHEMA,
        "package_version": package_version(),
        "script": script,
        "argv": list(argv),
        "timestamp_utc": timestamp
        or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "git": git_commit(repo),
        "packages": package_versions(),
        "encoder_id": encoder_id,
        "inputs": {
            name: {"path": str(Path(path)), "sha256": input_hashes[name]} for name, path in inputs.items()
        },
        "extra": extra,
        "run_hash": run_hash(input_hashes, argv, encoder_id, extra),
    }


def write_run_manifest(
    out_dir: Path,
    *,
    script: str,
    argv: Sequence[str] | None = None,
    inputs: Mapping[str, Path] | None = None,
    encoder_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    manifest = build_run_manifest(
        script=script,
        argv=list(sys.argv[1:] if argv is None else argv),
        inputs=inputs,
        encoder_id=encoder_id,
        extra=extra,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

#!/usr/bin/env python3
"""Read-only frozen runtime completeness check; standard library only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/frozen_reference/runtime_manifest.json"
SUMS = "results/frozen_reference/SHA256SUMS"
QUANTITIES = ("PE_axis", "P_axis", "E_axis", "PE_nli", "P_nli", "E_nli")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid manifest path: {name}")
    return root / relative


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"size mismatch: {path}")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_mean(path: Path, reference: dict) -> None:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("reference mean is not an NPY file")
        version = tuple(handle.read(2))
        if version not in ((1, 0), (2, 0)):
            raise ValueError(f"unsupported reference NPY version: {version}")
        size_format = "<H" if version == (1, 0) else "<I"
        size = struct.unpack(size_format, handle.read(struct.calcsize(size_format)))[0]
        if size > 65536:
            raise ValueError("reference NPY header is too large")
        header = ast.literal_eval(handle.read(size).decode("latin1"))
        expected_shape = tuple(reference["shape"])
        if (header.get("shape") != expected_shape or
                header.get("descr") != reference["dtype"] or header.get("fortran_order") is not False):
            raise ValueError("reference mean shape/dtype/order mismatch")
        payload = handle.read()
        if len(payload) != expected_shape[0] * 4:
            raise ValueError("reference mean payload length mismatch")
        if not all(math.isfinite(v) for v in struct.unpack(f"<{expected_shape[0]}f", payload)):
            raise ValueError("reference mean contains nonfinite values")


def verify_stats(path: Path, reference: dict) -> None:
    stats = json.loads(path.read_text())
    for key in ("n_documents", "n_windows"):
        if stats[key] != reference[key]:
            raise ValueError(f"reference statistics mismatch: {key}")
    for channel in ("P", "E"):
        mean, sd = stats["proj"][channel]
        if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"invalid projection scale: {channel}")
        activation = stats["act"][channel]
        if not math.isfinite(activation) or activation <= 0:
            raise ValueError(f"invalid activation scale: {channel}")
    for rule in reference["pooling_rules"]:
        for quantity in QUANTITIES:
            mean, sd = stats["doc"][rule][quantity]
            if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
                raise ValueError(f"invalid document scale: {rule}/{quantity}")


def verify(root: Path, model_cache: Path | None = None) -> dict:
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest["schema"] != "populism_axis.frozen_runtime.v1":
        raise ValueError("unsupported runtime manifest schema")
    entries = manifest["assets"] + manifest["source_files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate runtime manifest paths")
    expected_sums = "".join(entry["sha256"] + "  " + entry["path"] + "\n" for entry in entries)
    if (root / SUMS).read_text() != expected_sums:
        raise ValueError("SHA256SUMS disagrees with runtime manifest")
    for entry in entries:
        verify_file(local_path(root, entry["path"]), entry)
    verify_mean(root / "results/frozen_reference/reference_mean_embedding.npy", manifest["reference"])
    verify_stats(root / "results/frozen_reference/reference_stats.json", manifest["reference"])
    model_files = 0
    if model_cache is not None:
        for model in manifest["models"]:
            directory = model_cache / ("models--" + model["model_id"].replace("/", "--"))
            revision = model["pinned_revision"]
            snapshot = local_path(directory / "snapshots", revision)
            if not snapshot.is_dir():
                raise ValueError(f"pinned snapshot not in cache: {model['model_id']}@{revision}")
            for entry in model["files"]:
                verify_file(local_path(snapshot, entry["name"]), entry)
                model_files += 1
    return {"assets": len(manifest["assets"]), "source_files": len(manifest["source_files"]),
            "model_files_checked": model_files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout root (default: script's parent)")
    parser.add_argument("--model-cache", type=Path, help="also check the cached bytes of the pinned model snapshots; never download")
    args = parser.parse_args()
    try:
        result = verify(args.root, args.model_cache)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, struct.error) as error:
        print(f"FAIL: {error}\nRestore the documented package bytes; do not regenerate reference assets as a repair.", file=sys.stderr)
        return 1
    print(f"OK: {result['assets']} frozen assets and {result['source_files']} source/dependency files match.")
    if args.model_cache is None:
        print("Model cache not checked. Installed dependencies and inference were not tested.")
    else:
        print(f"OK: {result['model_files_checked']} cached model files match the pinned snapshots.")
        print("Historical GPU checkpoint identity, installed dependencies and inference were not tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

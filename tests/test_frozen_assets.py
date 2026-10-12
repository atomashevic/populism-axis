"""Lightweight package checks; no model downloads or scientific recomputation."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_frozen_assets", ROOT / "scripts/verify_frozen_assets.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class FrozenAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="frozen-package-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = json.loads((ROOT / checker.MANIFEST).read_text())
        paths = [e["path"] for e in self.manifest["assets"] + self.manifest["source_files"]]
        paths += [checker.MANIFEST, checker.SUMS, "scripts/verify_frozen_assets.py"]
        for name in paths:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

    def update_fixture_checksums(self, name: str) -> None:
        for entry in self.manifest["assets"]:
            if entry["path"] == name:
                path = self.root / name
                entry.update(bytes=path.stat().st_size, sha256=checker.sha256(path))
        (self.root / checker.MANIFEST).write_text(json.dumps(self.manifest))
        (self.root / checker.SUMS).write_text("".join(
            e["sha256"] + "  " + e["path"] + "\n"
            for e in self.manifest["assets"] + self.manifest["source_files"]))

    def test_minimal_checkout_runs_with_standard_library_only(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", str(self.root / "scripts/verify_frozen_assets.py")],
            cwd=self.root.parent, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 frozen assets", result.stdout)
        self.assertIn("Model cache not checked", result.stdout)
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / "results/embeddings").exists())

    def test_missing_or_corrupt_mean_fails(self) -> None:
        path = self.root / "results/frozen_reference/reference_mean_embedding.npy"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing required file"):
            checker.verify(self.root)
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            checker.verify(self.root)

    def test_mean_header_validated_even_with_matching_checksum(self) -> None:
        name = "results/frozen_reference/reference_mean_embedding.npy"
        path = self.root / name
        path.write_bytes(path.read_bytes().replace(b"<f4", b"<i4", 1))
        self.update_fixture_checksums(name)
        with self.assertRaisesRegex(ValueError, "shape/dtype/order"):
            checker.verify(self.root)

    def test_invalid_reference_scale_fails(self) -> None:
        name = "results/frozen_reference/reference_stats.json"
        path = self.root / name
        stats = json.loads(path.read_text())
        stats["doc"]["k:8"]["PE_axis"][1] = 0
        path.write_text(json.dumps(stats))
        self.update_fixture_checksums(name)
        with self.assertRaisesRegex(ValueError, "invalid document scale"):
            checker.verify(self.root)

    def test_missing_source_file_and_checksum_list_drift_fail(self) -> None:
        path = self.root / "scripts/score_texts.py"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing required file"):
            checker.verify(self.root)
        path.write_bytes(original)
        (self.root / checker.SUMS).write_text("")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS disagrees"):
            checker.verify(self.root)

    def test_model_cache_is_optional_and_explicit_check_fails_when_missing(self) -> None:
        self.assertEqual(checker.verify(self.root)["model_files_checked"], 0)
        with self.assertRaisesRegex(ValueError, "pinned snapshot not in cache"):
            checker.verify(self.root, self.root / "empty-cache")

    def test_cache_check_uses_declared_revision_and_file_bytes(self) -> None:
        model = {"model_id": "fixture/model", "pinned_revision": "fixture-revision",
                 "files": [{"name": "weights.bin", "bytes": 4, "sha256": ""}]}
        cache = self.root / "cache"
        directory = cache / "models--fixture--model"
        # A moving main ref that points elsewhere does not matter: loading is pinned.
        (directory / "refs").mkdir(parents=True)
        (directory / "refs/main").write_text("other-revision")
        snapshot = directory / "snapshots/fixture-revision"
        snapshot.mkdir(parents=True)
        (snapshot / "weights.bin").write_bytes(b"test")
        model["files"][0]["sha256"] = checker.sha256(snapshot / "weights.bin")
        self.manifest["models"] = [model]
        (self.root / checker.MANIFEST).write_text(json.dumps(self.manifest))
        self.assertEqual(checker.verify(self.root, cache)["model_files_checked"], 1)
        (snapshot / "weights.bin").write_bytes(b"fail")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            checker.verify(self.root, cache)

    def test_only_runtime_mean_is_exempt_from_cache_ignore(self) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "results/frozen_reference/reference_mean_embedding.npy"],
            cwd=ROOT, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "results/frozen_reference/other.npy",
             "results/embeddings/example.npz"], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("other.npy", result.stdout)
        self.assertIn("example.npz", result.stdout)


if __name__ == "__main__":
    unittest.main()

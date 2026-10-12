"""Exercise the shell helpers with local fixtures, without SSH or GPU jobs."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RemoteHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="populism-helper-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.origin = self.base / "origin.git"
        self.seed = self.base / "seed"
        self.repo = self.base / "checkout with spaces"
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Helper test",
            "GIT_AUTHOR_EMAIL": "helper-test@example.invalid",
            "GIT_COMMITTER_NAME": "Helper test",
            "GIT_COMMITTER_EMAIL": "helper-test@example.invalid",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.git(self.base, "init", "--bare", str(self.origin))
        self.git(self.base, "init", "-b", "work", str(self.seed))
        (self.seed / "scripts").mkdir()
        for name in ("remote_job.sh", "remote_pull.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.seed / "scripts" / name)
        (self.seed / "tracked.txt").write_text("initial\n")
        (self.seed / ".gitignore").write_text("*.npy\nlogs/\n")
        self.git(self.seed, "add", ".")
        self.git(self.seed, "commit", "-m", "fixture")
        self.git(self.seed, "remote", "add", "origin", str(self.origin))
        self.git(self.seed, "push", "origin", "work")
        self.git(self.base, "clone", "--branch", "work", str(self.origin), str(self.repo))

    def git(self, cwd: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd, env=self.env, text=True,
            capture_output=True, check=True, timeout=15,
        )
        return result.stdout.strip()

    def run_helper(self, name: str, *args: str, env: dict | None = None):
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / name), *args],
            cwd=self.base, env=self.env if env is None else env,
            text=True, capture_output=True, timeout=15,
        )

    def advance_origin(self, name: str = "new.txt", text: str = "upstream\n") -> str:
        (self.seed / name).write_text(text)
        self.git(self.seed, "add", "--force", "--", name)
        self.git(self.seed, "commit", "-m", "upstream change")
        self.git(self.seed, "push", "origin", "work")
        return self.git(self.seed, "rev-parse", "HEAD")

    def test_pull_resolves_root_and_fast_forwards(self) -> None:
        expected = self.advance_origin()
        result = self.run_helper("remote_pull.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), expected)
        self.assertEqual((self.repo / "new.txt").read_text(), "upstream\n")

    def test_pull_preserves_unrelated_untracked_and_ignored_files(self) -> None:
        (self.repo / "abstract.md").write_text("local draft\n")
        (self.repo / "cache.npy").write_bytes(b"local cache")
        expected = self.advance_origin()
        result = self.run_helper("remote_pull.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), expected)
        self.assertEqual((self.repo / "abstract.md").read_text(), "local draft\n")
        self.assertEqual((self.repo / "cache.npy").read_bytes(), b"local cache")

    def test_pull_refuses_staged_and_unstaged_tracked_edits(self) -> None:
        for staged in (False, True):
            with self.subTest(staged=staged):
                (self.repo / "tracked.txt").write_text("local edit\n")
                if staged:
                    self.git(self.repo, "add", "tracked.txt")
                before = self.git(self.repo, "status", "--porcelain")
                head = self.git(self.repo, "rev-parse", "HEAD")
                result = self.run_helper("remote_pull.sh")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tracked files have local changes", result.stderr)
                self.assertEqual(self.git(self.repo, "status", "--porcelain"), before)
                self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), head)
                self.assertEqual((self.repo / "tracked.txt").read_text(), "local edit\n")
                self.assertFalse((self.repo / ".git" / "FETCH_HEAD").exists())

    def test_pull_preserves_conflicting_untracked_and_ignored_files(self) -> None:
        for name in ("new local file.txt", "cache.npy"):
            with self.subTest(name=name):
                (self.repo / name).write_text("local artifact\n")
                self.advance_origin(name)
                head = self.git(self.repo, "rev-parse", "HEAD")
                before = self.git(self.repo, "status", "--porcelain")
                result = self.run_helper("remote_pull.sh")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("update stopped", result.stderr)
                self.assertIn("back up conflicting local files", result.stderr)
                self.assertEqual((self.repo / name).read_text(), "local artifact\n")
                self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), head)
                self.assertEqual(self.git(self.repo, "status", "--porcelain"), before)
                # Retire the fixture conflict to exercise the other case independently.
                (self.repo / name).unlink()

    def test_pull_refuses_divergence_without_resetting_local_commit(self) -> None:
        (self.repo / "local.txt").write_text("local work\n")
        self.git(self.repo, "add", "local.txt")
        self.git(self.repo, "commit", "-m", "local fixture change")
        head = self.git(self.repo, "rev-parse", "HEAD")
        self.advance_origin()
        result = self.run_helper("remote_pull.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resolve branch divergence manually", result.stderr)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), head)
        self.assertEqual((self.repo / "local.txt").read_text(), "local work\n")

    def test_pull_refuses_detached_head(self) -> None:
        self.git(self.repo, "checkout", "--detach")
        head = self.git(self.repo, "rev-parse", "HEAD")
        result = self.run_helper("remote_pull.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("detached HEAD", result.stderr)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), head)
        self.assertFalse((self.repo / ".git" / "FETCH_HEAD").exists())

    def test_job_resolves_root_before_gpu_busy_guard(self) -> None:
        bin_dir = self.base / "bin"
        bin_dir.mkdir()
        gpu_status = bin_dir / "nvidia-smi"
        gpu_status.write_text("#!/usr/bin/env bash\nprintf '4097\\n'\n")
        gpu_status.chmod(0o755)
        env = {
            **self.env, "PATH": f"{bin_dir}:{self.env['PATH']}",
            "GPU_MEM_LIMIT_MB": "4096",
        }
        result = self.run_helper("remote_job.sh", "fixture", "--", "false", env=env)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("GPU has 4097 MiB", result.stderr)
        self.assertTrue((self.repo / "logs").is_dir())
        self.assertFalse((self.base / "logs").exists())
        self.assertFalse((self.repo / "logs" / "fixture.pid").exists())


if __name__ == "__main__":
    unittest.main()

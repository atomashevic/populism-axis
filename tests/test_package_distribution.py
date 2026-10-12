"""Build and import distributions outside the repository without installing deps."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


class DistributionTests(unittest.TestCase):
    def test_wheel_and_sdist_have_only_verified_runtime_assets(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = work / "source"
            source.mkdir()
            for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
                shutil.copy2(root / name, source / name)
            (source / "docs").mkdir()
            shutil.copy2(root / "docs/custom-scales.md", source / "docs/custom-scales.md")
            (source / "examples").mkdir()
            for name in ("custom-scale-reference.csv", "custom-scale-texts.csv"):
                shutil.copy2(root / "examples" / name, source / "examples" / name)
            shutil.copytree(root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            dist = work / "dist"
            dist.mkdir()
            built = subprocess.run([sys.executable, "-c", "from setuptools.build_meta import build_wheel, build_sdist; import sys; out=sys.argv[1]; build_wheel(out); build_sdist(out)", str(dist)], cwd=source, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            wheel = next(dist.glob("*.whl"))
            self.assertTrue(list(dist.glob("*.tar.gz")), built.stdout + built.stderr + str(list(dist.iterdir())))
            sdist = next(dist.glob("*.tar.gz"))
            resource_root = "populism_axis/resources/"
            resource_names = {p.name for p in (root / "src/populism_axis/resources").iterdir() if p.is_file()}
            self.assertEqual(resource_names, {"proposition_anchors.json", "claim_hypotheses.json", "reference_stats.json", "reference_mean_embedding.npy"})
            installed = work / "installed"
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertEqual({name.removeprefix(resource_root) for name in names if name.startswith(resource_root)}, resource_names)
                for filename in resource_names:
                    self.assertEqual(archive.read(resource_root + filename), (root / "src/populism_axis/resources" / filename).read_bytes())
                self.assertFalse(any(name.endswith((".bin", ".safetensors", ".parquet")) for name in names))
            with tarfile.open(sdist) as archive:
                members = archive.getnames()
                prefix = members[0].split('/')[0] + '/'
                self.assertFalse(any(name.startswith(prefix + directory + '/') for name in members for directory in ('data', 'results', 'scored', 'research', 'scripts', 'tests')))
                self.assertEqual({name for name in members if name.startswith(prefix + 'docs/') and name.endswith('.md')}, {prefix + 'docs/custom-scales.md'})
                for filename in resource_names:
                    data = archive.extractfile(prefix + "src/" + resource_root + filename).read()
                    self.assertEqual(data, (root / "src/populism_axis/resources" / filename).read_bytes())
                unpacked = work / "sdist-source"
                archive.extractall(unpacked, filter="data")
            rebuilt = work / "rebuilt"
            rebuilt.mkdir()
            rebuild_check = subprocess.run([sys.executable, "-c", "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])", str(rebuilt)],
                                           cwd=unpacked / prefix.rstrip('/'), capture_output=True, text=True)
            self.assertEqual(rebuild_check.returncode, 0, rebuild_check.stdout + rebuild_check.stderr)
            with zipfile.ZipFile(next(rebuilt.glob("*.whl"))) as rebuilt_archive:
                for filename in resource_names:
                    self.assertEqual(rebuilt_archive.read(resource_root + filename), (root / "src/populism_axis/resources" / filename).read_bytes())
            installed_check = subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                                              "--target", str(installed), "--no-deps", "--offline", str(wheel)],
                                             cwd=work, capture_output=True, text=True)
            self.assertEqual(installed_check.returncode, 0, installed_check.stdout + installed_check.stderr)
            env = dict(os.environ, PYTHONPATH=str(installed))
            checked = subprocess.run([sys.executable, "-m", "populism_axis.cli", "verify-assets"], cwd=work, env=env, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(len(json.loads(checked.stdout)["hashes"]), 4)
            executable = subprocess.run([str(installed / "bin/populism-axis"), "template"], cwd=work, env=env, capture_output=True, text=True)
            self.assertEqual(executable.returncode, 0, executable.stderr)
            self.assertTrue(executable.stdout.startswith("component,pole,proposition\n"))
            entrypoint = (installed / "populism_axis-0.1.0.dist-info" / "entry_points.txt").read_text()
            self.assertIn("populism-axis = populism_axis.cli:main", entrypoint)


if __name__ == "__main__":
    unittest.main()

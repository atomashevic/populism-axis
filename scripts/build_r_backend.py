#!/usr/bin/env python3
"""Build the R package's immutable Python wheel and locked dependency profiles.

Run with the prepared project Python. Uses only local source, setuptools and the
existing uv lockfile; it neither resolves dependencies nor downloads models.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'r/populismaxis/inst/backend'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_frozen_assets.py')], check=True)
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    profiles = {}
    for flavor in ('cpu', 'cu126'):
        profiles[flavor] = subprocess.check_output(
            ['uv', 'export', '--frozen', '--offline', '--extra', flavor,
             '--no-hashes', '--no-annotate', '--no-header', '--no-emit-project',
             '--no-emit-package', 'torch'], cwd=ROOT, text=True)
        if 'extra-' in profiles[flavor]:
            raise RuntimeError('Unresolved uv-specific markers in dependency profile')
    wheels = {'cpu': {}, 'cu126': {}}
    platforms = {
        'manylinux_2_28_x86_64': 'Linux-x86_64',
        'manylinux_2_28_aarch64': 'Linux-aarch64',
        'win_amd64': 'Windows-x86_64', 'win_arm64': 'Windows-aarch64',
        'macosx_14_0_arm64': 'Darwin-aarch64',
    }
    for package in lock['package']:
        if package['name'] != 'torch':
            continue
        flavor = package['source'].get('registry', '').rsplit('/', 1)[-1]
        if flavor not in wheels:
            continue
        for wheel in package.get('wheels', []):
            for tag, platform in platforms.items():
                if wheel['url'].endswith(f'-cp312-cp312-{tag}.whl'):
                    wheels[flavor][platform] = wheel['url'] + '#' + wheel['hash'].replace(':', '=', 1)
    with tempfile.TemporaryDirectory(prefix='populismaxis-wheel-') as temp:
        work = Path(temp)
        source = work / 'source'
        source.mkdir()
        for name in ('pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE'):
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
        dist = work / 'dist'
        dist.mkdir()
        env = {**os.environ, 'SOURCE_DATE_EPOCH': '315532800'}
        built = subprocess.run([sys.executable, '-c',
                                'from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])',
                                str(dist)], cwd=source, env=env, capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stdout + built.stderr)
        wheel = next(dist.glob('*.whl'))
        data = wheel.read_bytes()
        wheel_hash = digest(data)
        wheel_rel = wheel_hash[:16] + '/' + wheel.name
        target = DEST / wheel_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        with zipfile.ZipFile(wheel) as archive:
            engine = {name.removeprefix('populism_axis/'): digest(archive.read(name))
                      for name in archive.namelist() if name.startswith('populism_axis/')}
    for flavor, content in profiles.items():
        (DEST / f'requirements-{flavor}.txt').write_text(content)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {
        'schema': 'populismaxis.backend.v1', 'package_version': project['version'],
        'python_version': '>=3.12,<3.13', 'torch_version': '2.14.0',
        'wheel': wheel_rel, 'wheel_sha256': wheel_hash, 'engine_files': engine,
        'requirements_sha256': {flavor: digest(content.encode()) for flavor, content in profiles.items()},
        'torch_wheels': wheels, 'source_base_commit': commit,
        'source_lock_sha256': digest((ROOT / 'uv.lock').read_bytes()),
        'build_command': 'uv run --extra cpu python scripts/build_r_backend.py',
    }
    (DEST / 'bundle.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'Bundled {wheel.name}: {len(data)} bytes, SHA256 {wheel_hash}')


if __name__ == '__main__':
    main()

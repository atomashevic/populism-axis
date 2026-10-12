"""Validate prepared environments without installing or importing model weights."""
from importlib import metadata
import sys


def validate_environment(requirements, torch_version):
    if sys.version_info < (3, 11):
        raise RuntimeError('populismaxis requires Python 3.11 or newer; managed setup uses 3.12.')
    from packaging.requirements import Requirement
    errors = []
    for line in requirements:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            errors.append(f'{req.name} is missing')
            continue
        if version not in req.specifier:
            errors.append(f'{req.name}=={version} does not satisfy {req.specifier}')
    try:
        version = metadata.version('torch')
    except metadata.PackageNotFoundError:
        version = 'missing'
    if version not in {torch_version, torch_version + '+cpu', torch_version + '+cu126'}:
        errors.append(f'torch=={version}; expected {torch_version} CPU or CUDA 12.6')
    if errors:
        raise RuntimeError('Incompatible Python dependency profile: ' + '; '.join(errors) +
                           '. Prepare python_requirements() in the selected environment and restart R.')
    return True

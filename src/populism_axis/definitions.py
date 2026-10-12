"""Authored proposition scales. Duplicate checks use NFKC and collapsed whitespace.

Text passed to models is preserved exactly; case is significant in duplicate checks.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping


def canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(payload):
    return hashlib.sha256(canonical(payload).encode('utf-8')).hexdigest()


def save_json(path, payload, overwrite=False):
    path = Path(path)
    with path.open('w' if overwrite else 'x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def validate_name(name):
    if not isinstance(name, str) or (not name or not name[0].isalpha() or any(not (c.isalnum() or c == '_') for c in name)) or '__' in name or name in {'id', 'n_windows'}:
        raise ValueError(f"Invalid name {name!r}: start with a letter, use letters, digits and single underscores; id and n_windows are reserved")


def _pole(values, component, pole):
    if isinstance(values, str) or values is None:
        raise ValueError(f"Component {component!r}: {pole} requires a sequence of nonempty propositions")
    values = tuple(values)
    seen = set()
    if not values:
        raise ValueError(f"Component {component!r}: {pole} requires at least one proposition")
    for text in values:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Component {component!r}: {pole} contains an empty or nonstring proposition")
        key = ' '.join(unicodedata.normalize('NFKC', text).split())
        if key in seen:
            raise ValueError(f"Component {component!r}: duplicate proposition in {pole}: {text!r}")
        seen.add(key)
    return values, seen


@dataclass(frozen=True)
class ComponentDefinition:
    name: str
    positive: tuple[str, ...]
    contrast: tuple[str, ...]

    def __post_init__(self):
        validate_name(self.name)
        positive, keys = _pole(self.positive, self.name, 'positive')
        contrast, other = _pole(self.contrast, self.name, 'contrast')
        if keys & other:
            raise ValueError(f"Component {self.name!r}: proposition appears in both positive and contrast")
        object.__setattr__(self, 'positive', positive)
        object.__setattr__(self, 'contrast', contrast)

    def to_dict(self):
        return {'name': self.name, 'positive': list(self.positive), 'contrast': list(self.contrast)}


@dataclass(frozen=True)
class ScaleDefinition:
    name: str
    components: tuple[ComponentDefinition, ...]
    combine: str | None = None
    preset: bool = False

    def __post_init__(self):
        validate_name(self.name)
        object.__setattr__(self, 'components', tuple(self.components))
        if not self.components or any(not isinstance(c, ComponentDefinition) for c in self.components):
            raise ValueError('components must contain ComponentDefinition values')
        if len({c.name for c in self.components}) != len(self.components):
            raise ValueError('Component names must be unique')
        if self.combine not in {None, 'mean', 'conjunction'}:
            raise ValueError("combine must be 'mean' or 'conjunction'")
        if len(self.components) > 1 and self.combine is None:
            raise ValueError("Multiple components require combine='mean' or combine='conjunction'")
        if not isinstance(self.preset, bool):
            raise ValueError('preset must be boolean')
        if self.preset:
            from .assets import load_preset_assets
            channels = load_preset_assets()['anchors']['channels']
            expected = tuple(ComponentDefinition(c, channels[c]['populist'], channels[c]['mainstream']) for c in ('P', 'E'))
            if self.name != 'populism' or self.combine != 'conjunction' or self.components != expected:
                raise ValueError('Preset definition does not match the frozen populism assets')

    def to_dict(self):
        return {'schema': 'populism_axis.scale.v1', 'name': self.name, 'components': [c.to_dict() for c in self.components], 'combine': self.combine, 'preset': self.preset}

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != 'populism_axis.scale.v1':
            raise ValueError('Unsupported scale schema')
        return cls(value['name'], tuple(ComponentDefinition(**c) for c in value['components']), value['combine'], value.get('preset', False))

    @property
    def identity(self):
        return identity(self.to_dict())

    def summary(self):
        return {**self.to_dict(), 'identity': self.identity, 'hypothesis_selection': 'frozen ordered hypotheses' if self.preset else 'axis alignment, min(5, positive count), authored-order ties; v1', 'validated_preset': self.preset}

    def save(self, path, overwrite=False):
        save_json(path, self.to_dict(), overwrite)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def define_scale(name, *, positive=None, contrast=None, components=None, combine=None):
    if components is not None:
        if positive is not None or contrast is not None:
            raise ValueError('Provide positive and contrast, or components; do not mix them')
        if not isinstance(components, Mapping):
            raise ValueError('components must be an ordered mapping')
        blocks = []
        for key, value in components.items():
            if not isinstance(value, Mapping) or set(value) != {'positive', 'contrast'}:
                raise ValueError(f"Component {key!r} requires exactly positive and contrast")
            blocks.append(ComponentDefinition(key, value['positive'], value['contrast']))
    else:
        blocks = [ComponentDefinition(name, positive, contrast)]
    return ScaleDefinition(name, tuple(blocks), combine)


def read_scale_csv(path, *, name, combine=None):
    components = {}
    seen = {}
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['component', 'pole', 'proposition']:
            raise ValueError('Definition CSV columns must be component,pole,proposition in that order')
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'CSV row {row_number}: malformed row')
            component, pole, text = row['component'], row['pole'], row['proposition']
            try:
                validate_name(component)
                if pole not in {'positive', 'contrast'} or not text.strip():
                    raise ValueError('requires pole positive or contrast and nonempty proposition')
                key = (component, ' '.join(unicodedata.normalize('NFKC', text).split()))
                if key in seen:
                    previous_pole, previous_row = seen[key]
                    raise ValueError(f'duplicate or opposing-pole overlap at rows {previous_row} and {row_number} ({previous_pole}, {pole})')
                seen[key] = (pole, row_number)
                components.setdefault(component, {'positive': [], 'contrast': []})[pole].append(text)
            except ValueError as error:
                raise ValueError(f'CSV row {row_number}, component {component!r}: {error}') from error
    return define_scale(name, components=components, combine=combine)


def populism_preset():
    from .assets import load_preset_assets
    channels = load_preset_assets()['anchors']['channels']
    return ScaleDefinition('populism', tuple(ComponentDefinition(c, channels[c]['populist'], channels[c]['mainstream']) for c in ('P', 'E')), 'conjunction', True)

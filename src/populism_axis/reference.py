"""Immutable, JSON-serializable fitted references and explicit threshold rules."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .definitions import ScaleDefinition, canonical, identity, save_json


@dataclass(frozen=True, init=False)
class FittedReference:
    _json: str

    def __init__(self, payload):
        value = json.loads(canonical(payload))
        if value.get('schema') != 'populism_axis.reference.v1':
            raise ValueError('Unsupported reference schema')
        scale = ScaleDefinition.from_dict(value['scale'])
        from .scoring import scoring_settings
        settings = value['settings']
        expected = scoring_settings(scale, **{k: settings[k] for k in ('unit', 'pooling', 'clean', 'claim_check')})
        if settings != expected:
            raise ValueError('Reference scoring settings/model revisions/algorithm are incompatible')
        components = [c.name for c in scale.components]
        mean = np.asarray(value['window']['mu'], dtype=float)
        if mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all():
            raise ValueError('Invalid reference embedding mean')
        if set(value['axes']) != set(components) or set(value['window']['proj']) != set(components):
            raise ValueError('Reference component axes/statistics mismatch')
        for component in components:
            axis = np.asarray(value['axes'][component], dtype=float)
            if axis.shape != mean.shape or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9):
                raise ValueError(f'Invalid axis for component {component!r}')
            self._check_stats(value['window']['proj'][component], component, scale.preset)
        if scale.combine == 'conjunction' and len(components) > 1:
            for component in components:
                activation = value['window']['act'][component]
                if not isinstance(activation, (int, float)) or not math.isfinite(activation) or activation <= 1e-12:
                    raise ValueError(f'Invalid activation scale for component {component!r}')
        from .scoring import quantity_names
        if set(value['document']) != set(quantity_names(scale, settings['claim_check'])):
            raise ValueError('Reference document quantities are incomplete or unexpected')
        for name, stats in value['document'].items():
            self._check_stats(stats, name, scale.preset)
        hypotheses = value['hypotheses']
        if not isinstance(hypotheses, list):
            raise ValueError('Invalid resolved hypotheses')
        if settings['claim_check']:
            if scale.preset:
                from .assets import load_preset_assets
                if hypotheses != load_preset_assets()['hypotheses']:
                    raise ValueError('Preset hypotheses differ from frozen assets')
            else:
                for c in scale.components:
                    subset = [h for h in hypotheses if h.get('component') == c.name]
                    if len(subset) != min(5, len(c.positive)):
                        raise ValueError(f'Invalid hypothesis count for component {c.name!r}')
                    ids = []
                    for h in subset:
                        index = h['anchor_index']
                        if not isinstance(index, int) or not 0 <= index < len(c.positive) or h['text'] != c.positive[index] or h['hypothesis_id'] != f'{c.name}:{index}' or h['selection_rule'] != 'axis_alignment_min5_v1':
                            raise ValueError(f'Invalid resolved hypothesis for component {c.name!r}')
                        ids.append(index)
                    if len(set(ids)) != len(ids):
                        raise ValueError('Duplicate resolved hypothesis')
                if len(hypotheses) != sum(min(5, len(c.positive)) for c in scale.components):
                    raise ValueError('Unexpected hypothesis components')
        elif hypotheses:
            raise ValueError('Disabled claim check must not contain hypotheses')
        if not isinstance(value['inputs'].get('n_documents'), int) or value['inputs']['n_documents'] < 30:
            raise ValueError('Fitted reference requires at least 30 documents')
        object.__setattr__(self, '_json', canonical(value))

    @staticmethod
    def _check_stats(stats, quantity, preset):
        if len(stats) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in stats) or stats[1] < 0 or (not preset and stats[1] <= 1e-12):
            raise ValueError(f'Invalid reference mean/SD for {quantity!r}; required SD must exceed 1e-12')

    def to_dict(self):
        return json.loads(self._json)

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        expected = payload.pop('identity', None)
        value = cls(payload)
        if expected is not None and expected != value.identity:
            raise ValueError('Reference identity does not match its contents')
        return value

    @property
    def identity(self):
        return identity(self.to_dict())

    @property
    def scale(self):
        return ScaleDefinition.from_dict(self.to_dict()['scale'])

    @property
    def settings(self):
        return self.to_dict()['settings']

    def summary(self):
        value = self.to_dict()
        return {'identity': self.identity, 'scale': self.scale.summary(), 'settings': value['settings'], 'hypotheses': value['hypotheses'], 'inputs': value['inputs']}

    def save(self, path, overwrite=False):
        save_json(path, {**self.to_dict(), 'identity': self.identity}, overwrite)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True)
class ThresholdRule:
    definition_identity: str
    reference_identity: str
    score: str
    threshold: float
    below: str
    at_or_above: str

    def __post_init__(self):
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) or not math.isfinite(self.threshold):
            raise ValueError('threshold must be a finite number')
        if any(not isinstance(label, str) or not label.strip() for label in (self.below, self.at_or_above)) or self.below == self.at_or_above:
            raise ValueError('below and at_or_above must be distinct nonempty string labels')
        if not self.score or not self.definition_identity or not self.reference_identity:
            raise ValueError('Threshold requires exact score, definition and reference identities')

    def to_dict(self):
        return {'schema': 'populism_axis.threshold.v1', **{k: getattr(self, k) for k in ('definition_identity', 'reference_identity', 'score', 'threshold', 'below', 'at_or_above')}}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop('schema', None) != 'populism_axis.threshold.v1':
            raise ValueError('Unsupported threshold schema')
        return cls(**value)

    @property
    def identity(self):
        return identity(self.to_dict())

    def summary(self):
        return {**self.to_dict(), 'identity': self.identity, 'equality': 'at_or_above'}

    def save(self, path, overwrite=False):
        save_json(path, self.to_dict(), overwrite)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def apply(self, result):
        from .scoring import ScoreResult
        if result.provenance.get('reference_identity') != self.reference_identity or result.provenance.get('definition_identity') != self.definition_identity or result.provenance.get('reference') == 'corpus':
            raise ValueError('Threshold definition/reference mismatch or corpus-relative result')
        if self.score not in result.scores or self.score in {'id', 'n_windows'}:
            raise ValueError(f'Score {self.score!r} is unavailable')
        label = f'{self.score}__label'
        if label in result.scores:
            raise ValueError(f'Label column {label!r} already exists')
        values = result.scores[self.score].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Cannot label nonfinite scores')
        scores = result.scores.copy()
        scores[label] = np.where(values >= self.threshold, self.at_or_above, self.below)
        provenance = json.loads(json.dumps(result.provenance))
        provenance['classifier'] = self.summary()
        return ScoreResult(scores, result.evidence, provenance)


def make_classifier(reference, *, threshold, below, at_or_above, score=None):
    if not isinstance(reference, FittedReference):
        raise ValueError('A reusable classifier requires a fitted reference')
    from .scoring import score_fields
    if score is None:
        if reference.scale.preset:
            raise ValueError('Preset classifier requires an explicit score field')
        score = reference.scale.name
    if score not in score_fields(reference.scale, reference.settings['claim_check']):
        raise ValueError(f'Score {score!r} is unavailable')
    return ThresholdRule(reference.scale.identity, reference.identity, score, threshold, below, at_or_above)

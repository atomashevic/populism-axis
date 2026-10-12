"""Synthetic serialization fixtures and mocked models; never real inference."""
from hashlib import sha256
from unittest.mock import patch
import numpy as np
import pandas as pd
from populism_axis import define_scale, FittedReference, ScoreResult
from populism_axis.scoring import scoring_settings
from populism_axis.definitions import canonical


def reference():
    scale = define_scale('trust', positive=['Јавне установе делују правично.'], contrast=['Јавне установе поступају неправично.'])
    return FittedReference({'schema': 'populism_axis.reference.v1', 'scale': scale.to_dict(),
        'settings': scoring_settings(scale), 'axes': {'trust': [1., 0.]},
        'window': {'mu': [0., 0.], 'proj': {'trust': [0., 1.]}, 'act': {}},
        'document': {'axis': [0., 1.]}, 'hypotheses': [],
        'inputs': {'n_documents': 30, 'n_windows': 3000000001, 'sha256': 'synthetic'}, 'execution': {}})


def result():
    ref = reference()
    values = [-.25, 0., .10000000000000002]
    return ScoreResult(pd.DataFrame({'id': ['001', 'NA', 'ж'], 'n_windows': [1, 3000000001, 1],
        'trust': values, 'trust__axis': values, 'trust__axis_z': values}),
        [{'id': '001', 'top_windows': [{'window_index': 0, 'window': 'Јавне установе', 'component_window_indices': {'trust': [0]}}]}],
        {'definition_identity': ref.scale.identity, 'reference_identity': ref.identity, 'reference': 'fitted',
         'n_documents': 3, 'n_windows': 3000000001})


def assert_equal_result(left, right):
    pd.testing.assert_frame_equal(left.scores, right.scores, check_exact=True)
    assert canonical(left.evidence) == canonical(right.evidence)
    assert canonical(left.provenance) == canonical(right.provenance)
    return True


def _factory(*args, **kwargs):
    def encode(texts):
        vectors = []
        for text in texts:
            digest = sha256(text.encode('utf-8')).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype(float) - 127.5
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors)
    return encode, lambda: None


class MockModels:
    def __init__(self):
        self.patches = [patch('populism_axis.scoring.make_encoder', _factory), patch('populism_axis.scoring.preflight_models')]

    def start(self):
        for p in self.patches:
            p.start()
        return self

    def stop(self):
        for p in reversed(self.patches):
            p.stop()

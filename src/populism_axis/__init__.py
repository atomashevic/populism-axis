"""Training-free proposition scales and the frozen populism preset."""
__version__ = "0.1.0"

from .definitions import ScaleDefinition, define_scale, read_scale_csv
from .reference import FittedReference, ThresholdRule, make_classifier
from .scoring import ScoreResult, fit_reference, score_texts


def fit_reference_csv(*args, **kwargs):
    from .io import fit_reference_csv as implementation
    return implementation(*args, **kwargs)


def score_csv(*args, **kwargs):
    from .io import score_csv as implementation
    return implementation(*args, **kwargs)


def write_scores(*args, **kwargs):
    from .io import write_scores as implementation
    return implementation(*args, **kwargs)


__all__ = ['__version__', 'ScaleDefinition', 'FittedReference', 'ThresholdRule', 'ScoreResult', 'define_scale', 'read_scale_csv', 'fit_reference', 'fit_reference_csv', 'score_texts', 'score_csv', 'make_classifier', 'write_scores']

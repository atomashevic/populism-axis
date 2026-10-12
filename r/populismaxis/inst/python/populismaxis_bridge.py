"""Serialization only: the canonical Python engine owns all scoring behavior."""
import json

from populism_axis import ScaleDefinition, FittedReference, ThresholdRule, ScoreResult
from populism_axis.definitions import canonical, identity

_TYPES = {"scale": ScaleDefinition, "reference": FittedReference, "classifier": ThresholdRule}


def encode_result(result):
    if not isinstance(result, ScoreResult):
        raise TypeError("Expected a Python ScoreResult")
    frame = result.scores
    payload = {
        "schema": "populismaxis.result.v1",
        "scores": {"columns": list(frame.columns), "rows": frame.to_numpy(dtype=object).tolist(),
                   "dtypes": [str(dtype) for dtype in frame.dtypes]},
        "evidence": result.evidence,
        "provenance": result.provenance,
    }
    return canonical(payload)


def decode_result(text):
    import pandas as pd
    value = json.loads(text)
    if value.get("schema") != "populismaxis.result.v1":
        raise ValueError("Unsupported R result snapshot schema")
    scores = value["scores"]
    columns, rows, dtypes = scores["columns"], scores["rows"], scores["dtypes"]
    if not isinstance(columns, list) or len(columns) != len(set(columns)) or any(not isinstance(c, str) for c in columns):
        raise ValueError("Invalid snapshot score columns")
    if len(dtypes) != len(columns) or any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        raise ValueError("Invalid snapshot score rows or dtypes")
    frame = pd.DataFrame(rows, columns=columns)
    for column, dtype in zip(columns, dtypes):
        frame[column] = frame[column].astype(dtype)
    return ScoreResult(frame, value["evidence"], value["provenance"])


def encode_object(obj):
    if isinstance(obj, ScoreResult):
        return encode_result(obj)
    if not isinstance(obj, tuple(_TYPES.values())):
        raise TypeError("Unsupported Python object for the R interface")
    return canonical(obj.to_dict())


def decode_object(kind, text, expected_identity=None):
    if kind == "result":
        obj = decode_result(text)
        actual = identity(json.loads(text))
    else:
        if kind not in _TYPES:
            raise ValueError("Unsupported R object kind")
        obj = _TYPES[kind].from_dict(json.loads(text))
        actual = obj.identity
    if expected_identity is not None and expected_identity != actual:
        raise ValueError("R object identity does not match its canonical JSON payload")
    return obj


def object_identity(kind, text):
    obj = decode_object(kind, text)
    return identity(json.loads(text)) if kind == "result" else obj.identity


def object_summary(kind, text):
    obj = decode_object(kind, text)
    if kind == "result":
        return canonical({"columns": list(obj.scores), "n_documents": len(obj.scores), "provenance": obj.provenance})
    return canonical(obj.summary())

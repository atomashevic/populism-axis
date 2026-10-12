"""Literal CSV adapters and explicit output writing."""
import csv
from hashlib import sha256
import json
from pathlib import Path


def ordered_input_hash(ids, texts) -> str:
    """SHA-256 of UTF-8 compact JSON ordered [id, text] pairs, without newline."""
    if len(ids) != len(texts):
        raise ValueError("IDs and texts must have the same length.")
    payload = json.dumps(list(zip(ids, texts)), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _read_rows(path, required):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError("CSV input must contain a header row.")
        if len(set(fields)) != len(fields):
            raise ValueError("CSV input has duplicate column names.")
        for name in required:
            if name not in fields:
                raise ValueError(f"CSV input is missing required column '{name}'.")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {row_number} does not match the header's column count.")
            for name in required:
                if not row[name].strip():
                    raise ValueError(f"CSV row {row_number} has an empty required cell in column '{name}'.")
            rows.append((row_number, row))
    if not rows:
        raise ValueError("CSV input must contain at least one data row.")
    return rows


def read_corpus_csv(path, *, text_column="text", id_column=None):
    required = [text_column] + ([id_column] if id_column is not None else [])
    rows = _read_rows(path, required)
    texts = [row[text_column] for _, row in rows]
    ids = [row[id_column] for _, row in rows] if id_column is not None else [str(i) for i in range(len(rows))]
    if len(set(ids)) != len(ids):
        raise ValueError("CSV IDs must be unique.")
    return texts, ids


def read_scale_csv(path, *, name, combine=None):
    from .definitions import read_scale_csv as read_definition
    return read_definition(path, name=name, combine=combine)


def fit_reference_csv(path, *, scale, text_column="text", id_column=None, **fit_options):
    from .scoring import fit_reference
    texts, ids = read_corpus_csv(path, text_column=text_column, id_column=id_column)
    return fit_reference(texts, scale=scale, ids=ids, **fit_options)


def score_csv(path, *, reference, scale=None, text_column="text", id_column=None, **scoring_options):
    from .scoring import score_texts
    input_identity = {"path": str(path), "sha256": sha256(Path(path).read_bytes()).hexdigest()}
    texts, ids = read_corpus_csv(path, text_column=text_column, id_column=id_column)
    result = score_texts(texts, reference=reference, scale=scale, ids=ids, **scoring_options)
    result.provenance.setdefault("inputs", {})["input"] = input_identity
    return result


def preflight_scores(out_dir, *, overwrite=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(out_dir)
    paths = [out_dir / name for name in ("scores.csv", "evidence.json", "run_manifest.json")]
    for path in paths:
        if path.exists() and (not overwrite or not path.is_file()):
            raise FileExistsError(f"Output already exists: {path}. Use overwrite=True to replace score outputs.")
    return paths


def write_scores(result, out_dir, *, overwrite=False) -> None:
    paths = preflight_scores(out_dir, overwrite=overwrite)
    # Serialize before touching outputs, so invalid evidence cannot leave partial files.
    evidence = json.dumps(result.evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    provenance = json.dumps(result.provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    scores = result.scores.to_csv(index=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths[0].write_text(scores, encoding="utf-8")
    paths[1].write_text(evidence, encoding="utf-8")
    paths[2].write_text(provenance, encoding="utf-8")

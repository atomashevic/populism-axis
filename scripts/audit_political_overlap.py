#!/usr/bin/env python3
"""Exact normalized text overlap with pinned PolNLI train premises; no inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from populism_axis.run_manifest import file_sha256, write_run_manifest

REVISION = '062724c0a0e601ffc5fba98a5546cc2237a766fa'
TRAIN_FILE = 'data/train-00000-of-00001.parquet'


def text_hash(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def audit(frame, text_column, train_hashes):
    if frame[text_column].isna().any():
        raise ValueError('Missing benchmark text')
    frame = frame.copy()
    frame['exact_overlap'] = frame[text_column].map(text_hash).isin(train_hashes)
    groups = frame.groupby('dataset', sort=True) if 'dataset' in frame else [('all', frame)]
    return {str(name): {'rows': len(rows), 'overlapping_rows': int(rows['exact_overlap'].sum()),
                        'unique_texts': int(rows[text_column].map(text_hash).nunique()),
                        'unique_overlapping_texts': int(rows.loc[rows['exact_overlap'], text_column].map(text_hash).nunique())}
            for name, rows in groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', type=Path, required=True)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--training-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.training_cache.exists():
        args.training_cache.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/mlburnham/Pol_NLI/resolve/{REVISION}/{TRAIN_FILE}'
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        temporary = args.training_cache.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(args.training_cache)
    training = pd.read_parquet(args.training_cache, columns=['premise', 'dataset'])
    if training['premise'].isna().any():
        raise ValueError('Missing training premise')
    hashes = set(training['premise'].map(text_hash))
    result = {'training_dataset': 'mlburnham/Pol_NLI', 'revision': REVISION, 'split': 'train',
              'training_file': TRAIN_FILE, 'training_sha256': file_sha256(args.training_cache),
              'normalization': 'Unicode NFKC, casefold, collapse whitespace; exact SHA-256 equality',
              'training_rows': len(training), 'unique_training_premises': len(hashes),
              'training_source_rows': {str(k): int(v) for k, v in training['dataset'].value_counts().items()},
              'documents': audit(pd.read_parquet(args.documents), 'text', hashes),
              'limitations': ['Exact normalized matches only; no substring, near-duplicate or translation matching.',
                             'Does not audit underlying universal-classifier training or language-model pretraining.',
                             'Dataset current pinned revision may differ from the training-time snapshot.']}
    inputs = {'documents': args.documents, 'training': args.training_cache, 'script': Path(__file__)}
    if args.requests:
        paths = sorted(args.requests.glob('requests_*.parquet')) if args.requests.is_dir() else [args.requests]
        if not paths:
            raise ValueError('No request parquet files')
        result['requests'] = {}
        for path in paths:
            requests = pd.read_parquet(path)
            text_column = 'premise' if 'premise' in requests else 'text'
            result['requests'][path.stem] = audit(requests, text_column, hashes)
            inputs[path.stem] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    write_run_manifest(args.output.parent, script='scripts/audit_political_overlap.py', inputs=inputs,
                       extra={'training_revision': REVISION}, filename='overlap_run_manifest.json')
    print(json.dumps({k: result[k] for k in ('documents', 'requests') if k in result}, indent=2))


if __name__ == '__main__':
    main()

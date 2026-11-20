#!/usr/bin/env python3
"""Pinned APSI comparator; no configuration is fitted to benchmark outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DEFAULT_CONFIG = ROOT / 'anchors/apsi_comparator.json'


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    payload = {k: config[k] for k in ('hypotheses', 'topic_hypotheses')}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()
    if digest != config['hypotheses_sha256']:
        raise ValueError('Pinned APSI hypothesis checksum does not match')
    return config


def score_probabilities(topic_probs, probabilities, config):
    """Reproduce upstream numeric score and all three abstention gates."""
    topic = float(np.max(topic_probs))
    result = {'score': float('nan'), 'topic_probability': topic,
              'abstention_reason': '', 'passed_precheck': False}
    if topic < config['topic_threshold']:
        return dict(result, abstention_reason='topic')
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (len(config['hypotheses']),) or not np.isfinite(probs).all():
        raise ValueError('Missing or invalid substantive probabilities')
    threshold = config['threshold']
    if not np.any(probs > threshold):
        return dict(result, abstention_reason='no_hypothesis_above_threshold')
    sides = {'populist': [], 'pluralist': []}
    for p, h in zip(probs, config['hypotheses']):
        sides[h['direction']].append(float(p * h['weight']) if p >= config['prob_floor'] else 0.)
    sides = {key: sorted(values, reverse=True) for key, values in sides.items()}
    if all(np.mean(values[:2]) < threshold for values in sides.values()):
        return dict(result, abstention_reason='weak_top_two')
    k = max(4, int(np.sum(probs > threshold)) + 2)
    score = np.clip(5 + 5 * (np.mean(sides['populist'][:k]) - np.mean(sides['pluralist'][:k])), 0, 10)
    return dict(result, score=float(score), passed_precheck=True)


def aggregate_chunks(results):
    scores = [r['score'] for r in results if np.isfinite(r['score'])]
    return {'score': float(np.mean(scores)) if scores else float('nan'),
            'topic_probability': max(r['topic_probability'] for r in results),
            'n_chunks': len(results), 'n_scored_chunks': len(scores),
            'abstention_reason': '' if scores else 'all_chunks_abstained'}


def chunk_text(text, tokenizer, premise_budget):
    """Nonoverlapping token-budget chunks, preserving original text via offsets."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                         truncation=False)
    offsets = encoding['offset_mapping']
    if len(offsets) <= premise_budget:
        return [text]
    # Include all original whitespace and characters exactly once.
    starts = [0] + [offsets[i][0] for i in range(premise_budget, len(offsets), premise_budget)]
    ends = starts[1:] + [len(text)]
    return [text[start:end] for start, end in zip(starts, ends)]


def infer_pairs(model, tokenizer, texts, hypotheses, batch_size, device):
    import torch
    out = np.empty((len(texts), len(hypotheses)), dtype=np.float32)
    labels = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    entailment = labels.get('entailment', labels.get('entail'))
    if entailment is None:
        raise ValueError('Model has no explicit entailment label')
    total = len(texts) * len(hypotheses)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            indices = range(start, min(start + batch_size, total))
            pairs = [divmod(i, len(hypotheses)) for i in indices]
            inputs = tokenizer([texts[i] for i, _ in pairs], [hypotheses[j] for _, j in pairs],
                               return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            probs = torch.softmax(model(**inputs).logits.float(), dim=-1)[:, entailment].cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError('NLI inference produced nonfinite probabilities')
            out.reshape(-1)[start:start + len(probs)] = probs
    return out


def main():
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from populism_axis.run_manifest import file_sha256, write_run_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-size', choices=['large', 'base'], required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--unit-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
    parser.add_argument('--modes', default='native,chunk')
    args = parser.parse_args()
    if args.batch_size < 1 or args.unit_batch_size < 1:
        parser.error('batch sizes must be positive')
    modes = args.modes.split(',')
    if not set(modes) <= {'native', 'chunk'}:
        parser.error('modes must be native and/or chunk')
    config = load_config(args.config)
    spec = config['models'][args.model_size]
    df = pd.read_parquet(args.input) if args.input.suffix == '.parquet' else pd.read_csv(args.input, dtype=str)
    if not {'dataset', 'id', 'text'} <= set(df):
        raise ValueError('Input requires dataset,id,text')
    if df[['dataset', 'id', 'text']].isna().any().any() or df.duplicated(['dataset', 'id']).any():
        raise ValueError('Missing input values or duplicate document identifiers')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'probabilities'
    cache_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec['id'], revision=spec['revision'], torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    hypotheses = [h['text'] for h in config['hypotheses']]
    all_hypotheses = config['topic_hypotheses'] + hypotheses
    longest = max(len(tokenizer(h, add_special_tokens=False)['input_ids']) for h in all_hypotheses)
    budget = 512 - longest - tokenizer.num_special_tokens_to_add(pair=True)
    units, mapping = [], []
    for row in df.itertuples(index=False):
        native_idx = len(units)
        units.append(str(row.text))
        chunks = chunk_text(str(row.text), tokenizer, budget) if 'chunk' in modes else []
        chunk_ids = [native_idx] if len(chunks) == 1 else list(range(len(units), len(units) + len(chunks)))
        if len(chunks) > 1:
            units.extend(chunks)
        mapping.append((str(row.dataset), str(row.id), native_idx, chunk_ids))
    identity = {'input_sha256': file_sha256(args.input), 'config_sha256': file_sha256(args.config),
                'script_sha256': file_sha256(Path(__file__)),
                'model': spec, 'modes': modes, 'unit_batch_size': args.unit_batch_size,
                'batch_size': args.batch_size, 'premise_budget': budget, 'dtype': args.dtype,
                'torch_cpu_threads': 2,
                'chunk_aggregation': 'unweighted mean of non-abstained chunk scores; NA if none',
                'chunk_boundaries': 'original-text offsets every premise_budget tokens; no overlap; '
                                    'retokenized chunks retain native max_length=512 truncation',
                'native_truncation': 'longest_first, max_length=512',
                'n_documents': len(df), 'n_inference_units': len(units)}
    run_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    results = []
    for start in range(0, len(units), args.unit_batch_size):
        texts = units[start:start + args.unit_batch_size]
        cache = cache_dir / f'{run_key}_{start:08d}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                topic, substantive = data['topic'], data['substantive']
        else:
            topic = infer_pairs(model, tokenizer, texts, config['topic_hypotheses'], args.batch_size, args.device)
            relevant = np.max(topic, axis=1) >= config['topic_threshold']
            substantive = np.full((len(texts), len(hypotheses)), np.nan, dtype=np.float32)
            if np.any(relevant):
                substantive[relevant] = infer_pairs(model, tokenizer, [t for t, r in zip(texts, relevant) if r],
                                                   hypotheses, args.batch_size, args.device)
            tmp = cache.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, topic=topic, substantive=substantive)
            tmp.replace(cache)
        if (topic.shape != (len(texts), len(config['topic_hypotheses']))
                or substantive.shape != (len(texts), len(hypotheses))
                or not np.isfinite(topic).all()):
            raise ValueError(f'Invalid probability cache: {cache}')
        results.extend(score_probabilities(t, p, config) for t, p in zip(topic, substantive))
        print(f'{args.model_size}: {min(start + len(texts), len(units))}/{len(units)} units', flush=True)
    output = []
    for dataset, doc_id, native_idx, chunk_ids in mapping:
        for mode in modes:
            if mode == 'native':
                result = dict(results[native_idx], n_chunks=1,
                              n_scored_chunks=int(np.isfinite(results[native_idx]['score'])))
            else:
                result = aggregate_chunks([results[i] for i in chunk_ids])
            result.pop('passed_precheck', None)
            output.append(dict(dataset=dataset, id=doc_id, model_size=args.model_size, mode=mode, **result))
    pd.DataFrame(output).to_csv(args.output_dir / 'scores.csv', index=False)
    identity['elapsed_seconds'] = time.monotonic() - started
    identity['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(args.device)
                                             if args.device.startswith('cuda') else None)
    write_run_manifest(args.output_dir, script='scripts/apsi_comparator.py',
                       inputs={'input': args.input, 'config': args.config, 'script': Path(__file__)},
                       encoder_id=spec['id'], extra=identity)


if __name__ == '__main__':
    main()
